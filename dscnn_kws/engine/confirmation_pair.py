"""Two-frame training objective aligned with streaming KWS confirmation.

The deployed detector confirms a wake only when two adjacent windows both
cross its threshold.  This module deliberately requires explicit adjacent
windows; shuffled examples from an ordinary classification batch are not a
valid substitute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dscnn_kws.data.confirmation_pair import extract_adjacent_window_pairs


def wake_logit_margin(pair_logits: torch.Tensor, *, positive_class: int = 0) -> torch.Tensor:
    """Return wake-vs-rest logit margins for ``[batch, frames, classes]``."""

    if pair_logits.ndim != 3 or pair_logits.shape[1] != 2 or pair_logits.shape[2] < 2:
        raise ValueError("pair_logits must have shape [batch, 2, classes] with at least two classes")
    class_count = pair_logits.shape[2]
    if not 0 <= positive_class < class_count:
        raise ValueError("positive_class is outside the logits class dimension")
    positive = pair_logits[..., positive_class]
    negative_indexes = [index for index in range(class_count) if index != positive_class]
    negative = torch.logsumexp(pair_logits[..., negative_indexes], dim=-1)
    return positive - negative


def softmin_bounds(frame_margins: torch.Tensor, *, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound the exact two-frame minimum with differentiable soft minima.

    Returns ``(lower, upper)`` such that
    ``lower <= amin(frame_margins) <= upper``.  The lower bound makes the
    positive constraint conservative; the upper bound does the same for the
    negative false-confirmation constraint.
    """

    if frame_margins.ndim != 2 or frame_margins.shape[1] != 2:
        raise ValueError("frame_margins must have shape [batch, 2]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    lower = -float(temperature) * torch.logsumexp(-frame_margins / float(temperature), dim=1)
    upper = lower + float(temperature) * math.log(frame_margins.shape[1])
    return lower, upper


def _tail_mean(values: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    if values.ndim != 1:
        raise ValueError("CVaR values must be one-dimensional")
    if values.numel() == 0:
        return values.sum(), 0
    count = max(1, math.ceil(float(fraction) * values.numel()))
    return torch.topk(values, k=count, largest=True, sorted=False).values.mean(), count


def _tail_group_mean(
    values: Sequence[torch.Tensor],
    member_counts: Sequence[int],
    fraction: float,
) -> tuple[torch.Tensor, int, int]:
    """Average the worst group losses and retain their underlying member count."""

    if not values or len(values) != len(member_counts):
        raise ValueError("group values and member counts must be non-empty and aligned")
    stacked = torch.stack(list(values))
    selected_count = max(1, math.ceil(float(fraction) * stacked.numel()))
    selected = torch.topk(stacked, k=selected_count, largest=True, sorted=False)
    selected_indexes = selected.indices.detach().cpu().tolist()
    return selected.values.mean(), sum(member_counts[index] for index in selected_indexes), selected_count


def source_balanced_cvar(
    violations: torch.Tensor,
    *,
    fraction: float,
    source_fraction: float = 1.0,
    source_ids: Sequence[Hashable] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    """Average worst-tail violations, weighting each source equally.

    Without ``source_ids`` this is ordinary batch-local CVaR.  With source IDs,
    CVaR is computed inside each source first and the worst ``source_fraction``
    of source losses are averaged.  This prevents one long noise recording
    from dominating while preserving pressure on the hardest sources.
    """

    if violations.ndim != 1:
        raise ValueError("violations must be one-dimensional")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not math.isfinite(source_fraction) or not 0.0 < source_fraction <= 1.0:
        raise ValueError("source_fraction must be in (0, 1]")
    if source_ids is None:
        value, count = _tail_mean(violations, fraction)
        return value, count, int(violations.numel() > 0)
    if isinstance(source_ids, torch.Tensor):
        if source_ids.ndim != 1:
            raise ValueError("source_ids tensor must be one-dimensional")
        source_values: Sequence[Hashable] = source_ids.detach().cpu().tolist()
    else:
        source_values = list(source_ids)
    if len(source_values) != violations.numel():
        raise ValueError("source_ids must contain one value per violation")

    grouped: dict[Hashable, list[int]] = {}
    for index, source_id in enumerate(source_values):
        grouped.setdefault(source_id, []).append(index)
    if not grouped:
        return violations.sum(), 0, 0
    source_losses: list[torch.Tensor] = []
    source_tail_counts: list[int] = []
    for indexes in grouped.values():
        index_tensor = torch.tensor(indexes, device=violations.device, dtype=torch.long)
        source_loss, source_tail_count = _tail_mean(violations.index_select(0, index_tensor), fraction)
        source_losses.append(source_loss)
        source_tail_counts.append(source_tail_count)
    return _tail_group_mean(source_losses, source_tail_counts, source_fraction)


def source_domain_balanced_cvar(
    violations: torch.Tensor,
    *,
    fraction: float,
    source_fraction: float = 1.0,
    source_ids: Sequence[Hashable] | torch.Tensor,
    domain_ids: Sequence[Hashable] | torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    """Average source-balanced CVaR within domains, then average domains.

    Each source has equal weight inside its domain and each domain has equal
    weight in the final loss.  Consequently, adding more sources from a large
    domain cannot dilute a hard source from a smaller domain.
    """

    if violations.ndim != 1:
        raise ValueError("violations must be one-dimensional")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if not math.isfinite(source_fraction) or not 0.0 < source_fraction <= 1.0:
        raise ValueError("source_fraction must be in (0, 1]")

    def identifier_values(
        identifiers: Sequence[Hashable] | torch.Tensor,
        *,
        name: str,
    ) -> list[Hashable]:
        if isinstance(identifiers, torch.Tensor):
            if identifiers.ndim != 1:
                raise ValueError(f"{name} tensor must be one-dimensional")
            values = identifiers.detach().cpu().tolist()
        else:
            if isinstance(identifiers, (str, bytes)):
                raise ValueError(f"{name} must contain one value per violation")
            values = list(identifiers)
        if len(values) != violations.numel():
            raise ValueError(f"{name} must contain one value per violation")
        return values

    source_values = identifier_values(source_ids, name="source_ids")
    domain_values = identifier_values(domain_ids, name="domain_ids")
    grouped: dict[Hashable, dict[Hashable, list[int]]] = {}
    for index, (source_id, domain_id) in enumerate(zip(source_values, domain_values)):
        grouped.setdefault(domain_id, {}).setdefault(source_id, []).append(index)
    if not grouped:
        return violations.sum(), 0, 0, 0

    domain_losses: list[torch.Tensor] = []
    tail_count = 0
    source_count = 0
    for domain_sources in grouped.values():
        source_losses: list[torch.Tensor] = []
        source_tail_counts: list[int] = []
        for indexes in domain_sources.values():
            index_tensor = torch.tensor(indexes, device=violations.device, dtype=torch.long)
            source_loss, source_tail_count = _tail_mean(
                violations.index_select(0, index_tensor), fraction
            )
            source_losses.append(source_loss)
            source_tail_counts.append(source_tail_count)
        domain_loss, domain_tail_count, selected_sources = _tail_group_mean(
            source_losses,
            source_tail_counts,
            source_fraction,
        )
        domain_losses.append(domain_loss)
        tail_count += domain_tail_count
        source_count += selected_sources
    return torch.stack(domain_losses).mean(), tail_count, source_count, len(domain_losses)


@dataclass(frozen=True)
class ConfirmationPairLossTerms:
    total: torch.Tensor
    frame_ce: torch.Tensor
    positive_hinge: torch.Tensor
    negative_cvar: torch.Tensor
    negative_frame_cvar: torch.Tensor
    tail_ranking: torch.Tensor
    positive_pairs: int
    positive_hinge_tail_pairs: int
    positive_tail_pairs: int
    negative_pairs: int
    negative_tail_pairs: int
    negative_frame_tail_pairs: int
    negative_sources: int
    negative_domains: int


class ConfirmationPairLoss(nn.Module):
    """CE plus threshold-aligned positive hinge and negative tail risk."""

    def __init__(
        self,
        *,
        runtime_threshold: float,
        temperature: float = 0.25,
        positive_margin: float = 0.0,
        negative_margin: float = 0.0,
        frame_ce_weight: float = 1.0,
        positive_weight: float = 1.0,
        positive_hinge_tail_fraction: float = 1.0,
        negative_weight: float = 1.0,
        negative_cvar_fraction: float = 0.1,
        negative_source_cvar_fraction: float = 1.0,
        negative_frame_target_probability: float = 0.5,
        negative_frame_weight: float = 0.0,
        tail_ranking_weight: float = 0.0,
        tail_ranking_margin: float = 0.0,
        positive_tail_fraction: float = 0.1,
        positive_class: int = 0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < runtime_threshold < 1.0:
            raise ValueError("runtime_threshold must be in (0, 1)")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not 0.0 < negative_cvar_fraction <= 1.0:
            raise ValueError("negative_cvar_fraction must be in (0, 1]")
        if not 0.0 < positive_hinge_tail_fraction <= 1.0:
            raise ValueError("positive_hinge_tail_fraction must be in (0, 1]")
        if not 0.0 < negative_source_cvar_fraction <= 1.0:
            raise ValueError("negative_source_cvar_fraction must be in (0, 1]")
        if not 0.0 < negative_frame_target_probability < 1.0:
            raise ValueError("negative_frame_target_probability must be in (0, 1)")
        if not 0.0 < positive_tail_fraction <= 1.0:
            raise ValueError("positive_tail_fraction must be in (0, 1]")
        for name, value in (
            ("positive_margin", positive_margin),
            ("negative_margin", negative_margin),
            ("frame_ce_weight", frame_ce_weight),
            ("positive_weight", positive_weight),
            ("negative_weight", negative_weight),
            ("negative_frame_weight", negative_frame_weight),
            ("tail_ranking_weight", tail_ranking_weight),
            ("tail_ranking_margin", tail_ranking_margin),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.runtime_logit_margin = math.log(runtime_threshold / (1.0 - runtime_threshold))
        self.temperature = float(temperature)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.frame_ce_weight = float(frame_ce_weight)
        self.positive_weight = float(positive_weight)
        self.positive_hinge_tail_fraction = float(positive_hinge_tail_fraction)
        self.negative_weight = float(negative_weight)
        self.negative_cvar_fraction = float(negative_cvar_fraction)
        self.negative_source_cvar_fraction = float(negative_source_cvar_fraction)
        self.negative_frame_target_logit_margin = math.log(
            negative_frame_target_probability / (1.0 - negative_frame_target_probability)
        )
        self.negative_frame_target_probability = float(negative_frame_target_probability)
        self.negative_frame_weight = float(negative_frame_weight)
        self.tail_ranking_weight = float(tail_ranking_weight)
        self.tail_ranking_margin = float(tail_ranking_margin)
        self.positive_tail_fraction = float(positive_tail_fraction)
        self.positive_class = int(positive_class)
        self.frame_criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    def terms(
        self,
        pair_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        negative_source_ids: Sequence[Hashable] | torch.Tensor | None = None,
        negative_domain_ids: Sequence[Hashable] | torch.Tensor | None = None,
    ) -> ConfirmationPairLossTerms:
        if labels.ndim != 1 or labels.shape[0] != pair_logits.shape[0]:
            raise ValueError("labels must have shape [batch] and align with pair_logits")
        margins = wake_logit_margin(pair_logits.float(), positive_class=self.positive_class)
        lower_softmin, upper_softmin = softmin_bounds(margins, temperature=self.temperature)
        positive_mask = labels == self.positive_class
        negative_mask = ~positive_mask
        zero = margins.sum() * 0.0

        positive_violations = F.relu(
            self.runtime_logit_margin + self.positive_margin - lower_softmin[positive_mask]
        )
        positive_hinge, positive_hinge_tail_count = _tail_mean(
            positive_violations,
            self.positive_hinge_tail_fraction,
        )
        negative_violations = F.relu(
            upper_softmin[negative_mask] - (self.runtime_logit_margin - self.negative_margin)
        )

        def filter_negative_identifiers(
            identifiers: Sequence[Hashable] | torch.Tensor | None,
            *,
            name: str,
        ) -> Sequence[Hashable] | torch.Tensor | None:
            if identifiers is None:
                return None
            if isinstance(identifiers, torch.Tensor):
                if identifiers.ndim != 1 or identifiers.shape[0] != labels.shape[0]:
                    raise ValueError(f"{name} must align with the full pair batch")
                return identifiers[negative_mask.to(identifiers.device)]
            if isinstance(identifiers, (str, bytes)):
                raise ValueError(f"{name} must align with the full pair batch")
            values = list(identifiers)
            if len(values) != labels.shape[0]:
                raise ValueError(f"{name} must align with the full pair batch")
            mask_values = negative_mask.detach().cpu().tolist()
            return [value for value, keep in zip(values, mask_values) if keep]

        filtered_source_ids = filter_negative_identifiers(
            negative_source_ids, name="negative_source_ids"
        )
        filtered_domain_ids = filter_negative_identifiers(
            negative_domain_ids, name="negative_domain_ids"
        )
        def balanced_cvar(violations: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
            if filtered_domain_ids is None:
                value, selected_count, selected_sources = source_balanced_cvar(
                    violations,
                    fraction=self.negative_cvar_fraction,
                    source_fraction=self.negative_source_cvar_fraction,
                    source_ids=filtered_source_ids,
                )
                return value, selected_count, selected_sources, int(selected_sources > 0)
            if filtered_source_ids is None:
                raise ValueError("negative_source_ids are required when negative_domain_ids are provided")
            return source_domain_balanced_cvar(
                violations,
                fraction=self.negative_cvar_fraction,
                source_fraction=self.negative_source_cvar_fraction,
                source_ids=filtered_source_ids,
                domain_ids=filtered_domain_ids,
            )

        negative_cvar, tail_count, source_count, domain_count = balanced_cvar(negative_violations)

        # Pair confirmation uses the lower of two adjacent frame scores, so a
        # pair-only loss can leave isolated high-scoring negative frames
        # untouched. Penalizing the maximum frame margin closes that gap and
        # directly optimizes the requested low maximum score on every source.
        negative_frame_violations = F.relu(
            margins[negative_mask].amax(dim=1) - self.negative_frame_target_logit_margin
        )
        (
            negative_frame_cvar,
            negative_frame_tail_count,
            frame_source_count,
            frame_domain_count,
        ) = balanced_cvar(negative_frame_violations)
        if frame_source_count != source_count or frame_domain_count != domain_count:
            raise AssertionError("negative pair and frame risk grouping must match")

        # Rank conservative bounds on the exact deployment margin without
        # fixing a runtime threshold. The weakest positive lower-bound tail
        # defines the reference, while negatives use an upper bound. ReLU is
        # applied before domain averaging so an easy domain cannot cancel a
        # hard one.
        positive_pair_margins = lower_softmin[positive_mask]
        if self.tail_ranking_weight > 0.0 and (
            positive_pair_margins.numel() == 0 or not bool(negative_mask.any())
        ):
            raise ValueError("tail ranking requires positive and negative pairs in every batch")
        if positive_pair_margins.numel():
            positive_tail_count = max(
                1,
                math.ceil(self.positive_tail_fraction * positive_pair_margins.numel()),
            )
            positive_tail_margin = torch.topk(
                positive_pair_margins,
                k=positive_tail_count,
                largest=False,
                sorted=False,
            ).values.mean()
            ranking_violations = F.relu(
                upper_softmin[negative_mask] + self.tail_ranking_margin - positive_tail_margin
            )
            tail_ranking, _, _, _ = balanced_cvar(ranking_violations)
        else:
            positive_tail_count = 0
            tail_ranking = zero

        expanded_labels = labels[:, None].expand(-1, 2).reshape(-1)
        frame_ce = self.frame_criterion(pair_logits.float().reshape(-1, pair_logits.shape[-1]), expanded_labels)
        total = (
            self.frame_ce_weight * frame_ce
            + self.positive_weight * positive_hinge
            + self.negative_weight * negative_cvar
            + self.negative_frame_weight * negative_frame_cvar
            + self.tail_ranking_weight * tail_ranking
        )
        return ConfirmationPairLossTerms(
            total=total,
            frame_ce=frame_ce,
            positive_hinge=positive_hinge,
            negative_cvar=negative_cvar,
            negative_frame_cvar=negative_frame_cvar,
            tail_ranking=tail_ranking,
            positive_pairs=int(positive_mask.sum().item()),
            positive_hinge_tail_pairs=positive_hinge_tail_count,
            positive_tail_pairs=positive_tail_count,
            negative_pairs=int(negative_mask.sum().item()),
            negative_tail_pairs=tail_count,
            negative_frame_tail_pairs=negative_frame_tail_count,
            negative_sources=source_count,
            negative_domains=domain_count,
        )

    def forward(
        self,
        pair_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        negative_source_ids: Sequence[Hashable] | torch.Tensor | None = None,
        negative_domain_ids: Sequence[Hashable] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.terms(
            pair_logits,
            labels,
            negative_source_ids=negative_source_ids,
            negative_domain_ids=negative_domain_ids,
        ).total
