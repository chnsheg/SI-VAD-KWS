"""Fail-closed deployment quality gates and deterministic checkpoint ranking.

The training loss/validation F1 is not a deployment objective: a candidate may
score well on isolated one-second examples while producing unacceptable
streaming false wakes.  This module keeps the deployment acceptance policy
small, serializable, and independent from the trainer/evaluator so both can
use exactly the same predicates.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class DeploymentEvidence:
    """Deployment-style evidence for one immutable model checkpoint."""

    positive_recall: float
    captured_triggers: int
    false_wake_triggers: int
    max_tau_fah_upper: float
    # Optional one-sided FAH upper bounds for captured and false-wake streams.
    # Trigger-count limits remain available for short fixed corpora where a
    # duration-normalized rate would have poor resolution.
    captured_fah_upper: float | None = None
    false_wake_fah_upper: float | None = None
    # Optional stricter metric: recall measured with the deployed consecutive
    # confirmation rule.  If configured by the gate, it is mandatory.
    consecutive_recall: float | None = None

    def validate(self) -> "DeploymentEvidence":
        if not _finite(self.positive_recall) or not 0.0 <= float(self.positive_recall) <= 1.0:
            raise ValueError("positive_recall must be finite and in [0, 1]")
        _nonnegative_int(self.captured_triggers, "captured_triggers")
        _nonnegative_int(self.false_wake_triggers, "false_wake_triggers")
        if not _finite(self.max_tau_fah_upper) or float(self.max_tau_fah_upper) < 0.0:
            raise ValueError("max_tau_fah_upper must be finite and non-negative")
        for name, value in (("captured_fah_upper", self.captured_fah_upper), ("false_wake_fah_upper", self.false_wake_fah_upper)):
            if value is not None and (not _finite(value) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when supplied")
        if self.consecutive_recall is not None and (
            not _finite(self.consecutive_recall)
            or not 0.0 <= float(self.consecutive_recall) <= 1.0
        ):
            raise ValueError("consecutive_recall must be finite and in [0, 1]")
        return self


@dataclass(frozen=True)
class DevelopmentDeploymentGate:
    """Explicit acceptance policy used during development checkpoint selection.

    Defaults are intentionally conservative: no captured/false-wake trigger
    is accepted and the TAU one-sided 95% FAH upper bound must be <= 1/h.
    Projects with a different exposure or product requirement should serialize
    their limits in a gate JSON instead of changing evaluator code.
    """

    min_positive_recall: float = 0.94
    max_captured_triggers: int = 0
    max_false_wake_triggers: int = 0
    max_tau_fah_upper: float = 1.0
    max_captured_fah_upper: float | None = None
    max_false_wake_fah_upper: float | None = None
    min_consecutive_recall: float | None = None

    def __post_init__(self) -> None:
        if not _finite(self.min_positive_recall) or not 0.0 <= float(self.min_positive_recall) <= 1.0:
            raise ValueError("min_positive_recall must be finite and in [0, 1]")
        _nonnegative_int(self.max_captured_triggers, "max_captured_triggers")
        _nonnegative_int(self.max_false_wake_triggers, "max_false_wake_triggers")
        if not _finite(self.max_tau_fah_upper) or float(self.max_tau_fah_upper) < 0.0:
            raise ValueError("max_tau_fah_upper must be finite and non-negative")
        for name, value in (("max_captured_fah_upper", self.max_captured_fah_upper), ("max_false_wake_fah_upper", self.max_false_wake_fah_upper)):
            if value is not None and (not _finite(value) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when supplied")
        if self.min_consecutive_recall is not None and (
            not _finite(self.min_consecutive_recall)
            or not 0.0 <= float(self.min_consecutive_recall) <= 1.0
        ):
            raise ValueError("min_consecutive_recall must be finite and in [0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> "DevelopmentDeploymentGate":
        """Load a gate policy; a missing file means the documented defaults.

        Missing configuration is useful for backwards-compatible callers, but
        malformed or non-object JSON fails closed rather than silently falling
        back to permissive behavior.
        """
        source = Path(path)
        if not source.exists():
            return cls()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid deployment gate JSON: {source}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("deployment gate JSON must contain an object")
        allowed = {
            "min_positive_recall",
            "max_captured_triggers",
            "max_false_wake_triggers",
            "max_tau_fah_upper",
            "max_captured_fah_upper",
            "max_false_wake_fah_upper",
            "min_consecutive_recall",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown deployment gate fields: {unknown}")
        return cls(**{key: payload[key] for key in allowed if key in payload})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, evidence: DeploymentEvidence) -> tuple[bool, tuple[str, ...]]:
        """Return ``(eligible, reasons)`` without raising for metric failures."""
        try:
            evidence.validate()
        except ValueError as error:
            return False, (f"invalid_evidence:{error}",)
        reasons: list[str] = []
        if float(evidence.positive_recall) < float(self.min_positive_recall):
            reasons.append("positive_recall_below_gate")
        if evidence.captured_triggers > self.max_captured_triggers:
            reasons.append("captured_triggers_above_gate")
        if evidence.false_wake_triggers > self.max_false_wake_triggers:
            reasons.append("false_wake_triggers_above_gate")
        if float(evidence.max_tau_fah_upper) > float(self.max_tau_fah_upper):
            reasons.append("tau_fah_upper_above_gate")
        if self.max_captured_fah_upper is not None:
            if evidence.captured_fah_upper is None:
                reasons.append("missing_captured_fah_upper")
            elif float(evidence.captured_fah_upper) > float(self.max_captured_fah_upper):
                reasons.append("captured_fah_upper_above_gate")
        if self.max_false_wake_fah_upper is not None:
            if evidence.false_wake_fah_upper is None:
                reasons.append("missing_false_wake_fah_upper")
            elif float(evidence.false_wake_fah_upper) > float(self.max_false_wake_fah_upper):
                reasons.append("false_wake_fah_upper_above_gate")
        if self.min_consecutive_recall is not None:
            if evidence.consecutive_recall is None:
                reasons.append("missing_consecutive_recall")
            elif float(evidence.consecutive_recall) < float(self.min_consecutive_recall):
                reasons.append("consecutive_recall_below_gate")
        return not reasons, tuple(reasons)

    def eligible(self, evidence: DeploymentEvidence) -> bool:
        return self.check(evidence)[0]

    def rejection_reasons(self, evidence: DeploymentEvidence) -> tuple[str, ...]:
        return self.check(evidence)[1]


def checkpoint_rank(
    evidence: DeploymentEvidence,
    macro_f1: float,
    negative_recall: float,
) -> tuple[float, float, float]:
    """Rank eligible checkpoints: lower TAU FAH first, then F1 and specificity."""
    evidence.validate()
    if not _finite(macro_f1) or not _finite(negative_recall):
        raise ValueError("macro_f1 and negative_recall must be finite")
    return (-float(evidence.max_tau_fah_upper), float(macro_f1), float(negative_recall))


__all__ = ["DeploymentEvidence", "DevelopmentDeploymentGate", "checkpoint_rank"]
