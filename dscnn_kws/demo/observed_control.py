"""Pure reducer for observed controls plus documented PC assumptions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cascade import (
    CascadeEvent,
    CascadeState,
    KwsExecutionLane,
    KwsScoreRunner,
    PcmRingBuffer,
    VadFeatureCache,
    VadScoreRunner,
    _rms_dbfs,
)
from .contracts import ObservedControlContract, ThresholdComparator, TimingSchedule, VadContract
from .kws_executor import KwsJob, KwsResult


NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
MIN_KWS_LOOKBACK_MS = 1_000
MAX_KWS_LOOKBACK_MS = 2_000
# This buffer never contributes ordinary history windows.  It only keeps a
# little extra continuous PCM so priority phase jobs can still materialize
# their +/- windows after a bounded worker/queue delay.
PHASE_ASYNC_HISTORY_MS = MAX_KWS_LOOKBACK_MS + 1_000
PHASE_PROBE_OFFSET_MS = 16
PHASE_GREY_MARGIN = 0.15
PHASE_CONFIRMATION_DELAY_MS = 24
PHASE_REFINEMENT_SEED_OFFSETS_MS = (-48, 8, 16)
PHASE_REFINEMENT_NEIGHBOR_DELTA_MS = 4
PHASE_REFINEMENT_CANDIDATE_FLOOR = 0.30
PHASE_REFINEMENT_VALLEY_FLOOR = 0.15
PHASE_REFINEMENT_CANDIDATE_CEILING = 0.70
PHASE_STRONG_SCORE = 0.80
PHASE_STRONG_MIN_LIFT = 0.20
ENERGY_HOP_SAMPLES = 160
ENERGY_VALLEY_MIN_DROP_DB = 7.0
ENERGY_VALLEY_PEAK_MAX_DBFS = -48.0


def conditional_phase_probe_offsets(
    *, primary_score: float, kws_threshold: float, waveform: np.ndarray
) -> tuple[int, ...]:
    """Return bounded phase probes for one logical 96 ms KWS slot.

    The forward probe is the corpus-supported default for a score in the grey
    band.  The reverse probe is only enabled when the one-second waveform has
    a voiced/quiet/voiced valley, a cheap, model-independent cue for the
    ``hi`` to ``xiao`` syllable boundary.  The caller must merge all probe
    scores into the single logical slot score.
    """

    if not math.isfinite(float(primary_score)) or not math.isfinite(float(kws_threshold)):
        raise ValueError("phase compensation scores must be finite")
    values = np.asarray(waveform)
    if values.dtype != np.float32 or values.shape != (16_000,) or not np.all(np.isfinite(values)):
        raise ValueError("phase compensation waveform must be finite float32 [16000]")
    if not kws_threshold - PHASE_GREY_MARGIN <= primary_score < kws_threshold:
        return ()

    offsets: list[int] = [PHASE_PROBE_OFFSET_MS]
    if _has_voiced_quiet_voiced_valley(values):
        offsets.append(-PHASE_PROBE_OFFSET_MS)
    return tuple(offsets)


def phase_probe_plan(
    *, primary_score: float, kws_threshold: float, waveform: np.ndarray
) -> tuple[int, ...]:
    """Select a bounded fine-phase probe only for plausible keyword windows.

    Main-grid scores below the usual grey band can still be an unlucky sample
    of the sharp KWS phase peak.  The wider candidate band is bounded above
    and below, while lower scores require the same voiced/quiet/voiced cue
    used by the existing rescue path.
    """

    if not math.isfinite(float(primary_score)) or not math.isfinite(float(kws_threshold)):
        raise ValueError("phase compensation scores must be finite")
    values = np.asarray(waveform)
    if values.dtype != np.float32 or values.shape != (16_000,) or not np.all(np.isfinite(values)):
        raise ValueError("phase compensation waveform must be finite float32 [16000]")
    upper = max(float(kws_threshold), PHASE_REFINEMENT_CANDIDATE_CEILING)
    if PHASE_REFINEMENT_CANDIDATE_FLOOR <= primary_score <= upper:
        return PHASE_REFINEMENT_SEED_OFFSETS_MS
    if (
        PHASE_REFINEMENT_VALLEY_FLOOR <= primary_score < PHASE_REFINEMENT_CANDIDATE_FLOOR
        and _has_voiced_quiet_voiced_valley(values)
    ):
        return PHASE_REFINEMENT_SEED_OFFSETS_MS
    return ()


def _has_voiced_quiet_voiced_valley(waveform: np.ndarray) -> bool:
    frame_count = waveform.size // ENERGY_HOP_SAMPLES
    if frame_count < 11:
        return False
    frames = waveform[: frame_count * ENERGY_HOP_SAMPLES].reshape(frame_count, ENERGY_HOP_SAMPLES)
    dbfs = 20.0 * np.log10(np.sqrt(np.maximum(1e-12, np.mean(frames * frames, axis=1))))
    for index in range(5, frame_count - 5):
        valley = float(dbfs[index])
        left_peak = float(np.max(dbfs[index - 5 : index]))
        right_peak = float(np.max(dbfs[index + 1 : index + 6]))
        if (
            valley <= left_peak - ENERGY_VALLEY_MIN_DROP_DB
            and valley <= right_peak - ENERGY_VALLEY_MIN_DROP_DB
            and min(left_peak, right_peak) > ENERGY_VALLEY_PEAK_MAX_DBFS
            and valley <= float(np.min(dbfs[index - 2 : index + 3]))
        ):
            return True
    return False


@dataclass(frozen=True)
class ObservedControlGatePlan:
    """A model-work decision emitted by the reducer for one controller boundary."""

    run_vad: bool = False
    run_kws: bool = False
    skip_reason: str | None = None
    events: tuple[CascadeEvent, ...] = ()


@dataclass
class _PendingPhaseConfirmation:
    source_sequence: int
    source_window_end_ns: int
    phase_offset_ms: int
    confirmation_window_end_ns: int
    submitted: bool = False


@dataclass
class _PendingPhaseRefinement:
    source_sequence: int
    source_window_end_ns: int
    source_result: KwsResult
    seed_offsets_ms: tuple[int, ...]
    refinement_window_end_ns: int
    submitted: bool = False


@dataclass(frozen=True)
class ObservedControlRuntimeConfig:
    """Validated dashboard-controlled values for one PC controller session."""

    vad_threshold: float
    kws_threshold: float
    energy_threshold_dbfs: float
    energy_enabled: bool
    vad_enabled: bool
    schedule: TimingSchedule

    @classmethod
    def from_contract(cls, contract: ObservedControlContract) -> "ObservedControlRuntimeConfig":
        return cls.create(
            contract=contract,
            vad_threshold=contract.vad_threshold,
            kws_threshold=contract.kws_threshold,
            energy_threshold_dbfs=contract.energy_threshold_dbfs,
            energy_enabled=True,
            vad_enabled=True,
            vad_period_ms=contract.vad_period_ms,
            kws_period_ms=contract.kws_period_ms,
    )

    @classmethod
    def create(
        cls,
        *,
        contract: ObservedControlContract,
        vad_threshold: float,
        kws_threshold: float,
        energy_threshold_dbfs: int | float | None = None,
        energy_enabled: bool,
        vad_enabled: bool,
        vad_period_ms: int,
        kws_period_ms: int,
    ) -> "ObservedControlRuntimeConfig":
        schedule = TimingSchedule.from_periods(
            vad_period_ms, kws_period_ms, sample_rate_hz=contract.sample_rate_hz
        )
        if schedule.kws_period_ms % schedule.vad_period_ms != 0:
            raise ValueError("kws_period_ms must be an integer multiple of vad_period_ms")
        return cls(
            vad_threshold=_require_probability("vad_threshold", vad_threshold),
            kws_threshold=_require_probability("kws_threshold", kws_threshold),
            energy_threshold_dbfs=_require_finite_number(
                "energy_threshold_dbfs",
                contract.energy_threshold_dbfs
                if energy_threshold_dbfs is None
                else energy_threshold_dbfs,
            ),
            energy_enabled=_require_bool("energy_enabled", energy_enabled),
            vad_enabled=_require_bool("vad_enabled", vad_enabled),
            schedule=schedule,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "vad_threshold": self.vad_threshold,
            "kws_threshold": self.kws_threshold,
            "energy_enabled": self.energy_enabled,
            "vad_enabled": self.vad_enabled,
            "energy_period_ms": self.schedule.energy_period_ms,
            "vad_period_ms": self.schedule.vad_period_ms,
            "kws_period_ms": self.schedule.kws_period_ms,
        }


class ObservedControlStateMachine:
    """Apply observed controls plus documented PC assumptions to supplied inputs."""

    def __init__(
        self,
        contract: ObservedControlContract,
        runtime_config: ObservedControlRuntimeConfig | None = None,
    ) -> None:
        contract.validate()
        self.contract = contract
        self.runtime_config = runtime_config or ObservedControlRuntimeConfig.from_contract(contract)
        self.state = CascadeState.IDLE
        self.vad_streak = 0
        self._candidate_silence_streak = 0
        self.kws_streak = 0
        self.last_vad_positive_ns: int | None = None
        self.kws_next_ns: int | None = None
        self.kws_energy_hangover_deadline_ns: int | None = None
        self._last_seen_ns: int | None = None
        self._active_tick_ns: int | None = None
        self._energy_dbfs: float | None = None
        self._energy_pass = False
        self._link: int | None = None
        self._planned_vad_ns: int | None = None
        self._planned_kws_ns: int | None = None
        self._no_speech_expiry_pending = False

    @property
    def phase(self) -> CascadeState:
        """Alias for callers that describe the controller state as a phase."""

        return self.state

    @property
    def hangover_deadline_ns(self) -> int | None:
        return self.kws_energy_hangover_deadline_ns

    @property
    def next_kws_captured_ns(self) -> int | None:
        return self.kws_next_ns

    @property
    def effective_schedule(self) -> dict[str, int]:
        return {
            "energy_period_ms": self.runtime_config.schedule.energy_period_ms,
            "vad_period_ms": self.runtime_config.schedule.vad_period_ms,
            "kws_period_ms": self.runtime_config.schedule.kws_period_ms,
        }

    @property
    def energy_hangover_remaining_ms(self) -> int:
        captured_ns = self._active_tick_ns
        if captured_ns is None:
            captured_ns = self._last_seen_ns
        return 0 if captured_ns is None else self._hangover_remaining_ms(captured_ns)

    @property
    def kws_gate_open(self) -> bool:
        captured_ns = self._active_tick_ns
        if captured_ns is None:
            captured_ns = self._last_seen_ns
        return (
            self.state is CascadeState.KWS_ACTIVE
            and captured_ns is not None
            and self._energy_permits_processing(captured_ns)
        )

    @property
    def vad_positive_count(self) -> int:
        return self.vad_streak

    @property
    def vad_silence_count(self) -> int:
        return self._candidate_silence_streak

    @property
    def kws_positive_count(self) -> int:
        return self.kws_streak

    @property
    def vad_no_speech_remaining_ms(self) -> int:
        if self.state is not CascadeState.KWS_ACTIVE or self.last_vad_positive_ns is None:
            return 0
        captured_ns = self._active_tick_ns
        if captured_ns is None:
            captured_ns = self._last_seen_ns
        if captured_ns is None:
            return self.contract.no_speech_timeout_ms
        elapsed_ms = max(0, (captured_ns - self.last_vad_positive_ns) // NS_PER_MS)
        return max(0, self.contract.no_speech_timeout_ms - elapsed_ms)

    def force_kws_active(self, *, captured_ns: int) -> None:
        """Enter KWS phase for deterministic trace replay without model work."""

        self._require_timestamp(captured_ns)
        if self._last_seen_ns is not None and captured_ns < self._last_seen_ns:
            raise ValueError("captured_ns must not move backward")

        self._last_seen_ns = captured_ns
        self._active_tick_ns = None
        self.state = CascadeState.KWS_ACTIVE
        self.vad_streak = self.contract.vad_confirmations
        self._candidate_silence_streak = 0
        self._reset_kws_confirmation()
        self.last_vad_positive_ns = captured_ns
        self.kws_next_ns = captured_ns
        self.kws_energy_hangover_deadline_ns = None
        self._energy_dbfs = None
        self._energy_pass = False
        self._link = None
        self._planned_vad_ns = None
        self._planned_kws_ns = None
        self._no_speech_expiry_pending = False

    def begin_tick(
        self,
        *,
        captured_ns: int,
        energy_dbfs: int | float,
        link: int | None = None,
    ) -> list[CascadeEvent]:
        """Record one VAD-period boundary and refresh energy-derived state."""

        self._require_timestamp(captured_ns)
        if self._last_seen_ns is not None and captured_ns < self._last_seen_ns:
            raise ValueError("captured_ns must not move backward")
        energy = _require_finite_number("energy_dbfs", energy_dbfs)

        self._last_seen_ns = captured_ns
        self._active_tick_ns = captured_ns
        self._planned_vad_ns = None
        self._planned_kws_ns = None
        self._no_speech_expiry_pending = False
        self._energy_dbfs = energy
        self._energy_pass = _compares(
            self._energy_dbfs,
            self.runtime_config.energy_threshold_dbfs,
            self.contract.energy_comparator,
        )
        self._link = link
        if self._energy_pass and self.runtime_config.energy_enabled:
            self.kws_energy_hangover_deadline_ns = (
                captured_ns + self.contract.kws_energy_hangover_ms * NS_PER_MS
            )
        elif not self.runtime_config.energy_enabled:
            self.kws_energy_hangover_deadline_ns = None

        events: list[CascadeEvent] = []
        if self.state is CascadeState.KWS_ACTIVE and self._no_speech_expired(captured_ns):
            self._no_speech_expiry_pending = True

        events.insert(
            0,
            self._event(
                "energy_decision",
                captured_ns,
                gate="energy",
                energy_dbfs=self._energy_dbfs,
                energy_gate=self._energy_pass,
            ),
        )
        return events

    def plan_vad(self, *, captured_ns: int) -> ObservedControlGatePlan:
        """Decide whether VAD inference may run at the current controller boundary."""

        self._require_current_tick(captured_ns)
        if self.state is CascadeState.WAKE_LATCHED:
            return self._skip_plan(captured_ns, gate="vad", reason="wake_latched")
        if not self.runtime_config.vad_enabled:
            return self._plan_kws_without_vad(captured_ns)
        if not self._energy_permits_processing(captured_ns) and self.state not in (
            CascadeState.VAD_CANDIDATE,
            CascadeState.KWS_ACTIVE,
        ):
            return self._skip_plan(captured_ns, gate="vad", reason="energy")
        events: list[CascadeEvent] = []
        if self.state is CascadeState.IDLE:
            self._candidate_silence_streak = 0
            self._transition(
                CascadeState.VAD_CANDIDATE,
                captured_ns,
                events,
                gate="vad",
                reason="energy_gate_open",
            )
        self._planned_vad_ns = captured_ns
        return ObservedControlGatePlan(run_vad=True, events=tuple(events))

    def commit_vad(self, *, captured_ns: int, score: int | float) -> list[CascadeEvent]:
        """Commit one VAD score supplied by the adapter after a run plan."""

        self._require_current_tick(captured_ns)
        if self._planned_vad_ns != captured_ns:
            raise RuntimeError("commit_vad requires a successful VAD plan for this tick")
        score_value = _require_probability("score", score)
        self._planned_vad_ns = None
        positive = _compares(
            score_value, self.runtime_config.vad_threshold, self.contract.vad_probability_comparator
        )
        if positive:
            self.vad_streak += 1
            self.last_vad_positive_ns = captured_ns
        else:
            self.vad_streak = 0
        events = [
            self._event(
                "vad_decision",
                captured_ns,
                gate="vad",
                score=float(score),
                positive=positive,
            )
        ]

        if self.state is CascadeState.VAD_CANDIDATE:
            if positive:
                self._candidate_silence_streak = 0
                if self.vad_streak >= self.contract.vad_confirmations:
                    self._reset_kws_confirmation()
                    self.kws_next_ns = captured_ns
                    self._transition(
                        CascadeState.KWS_ACTIVE,
                        captured_ns,
                        events,
                        gate="vad",
                        reason="vad_confirmed",
                    )
            else:
                self._candidate_silence_streak += 1
                if self._candidate_silence_streak >= self.contract.vad_confirmations:
                    self._candidate_silence_streak = 0
                    self._transition(
                        CascadeState.IDLE,
                        captured_ns,
                        events,
                        gate="vad",
                        reason="vad_candidate_silence_confirmed",
                    )
        elif self.state is CascadeState.KWS_ACTIVE and self._no_speech_expiry_pending:
            if positive:
                self._no_speech_expiry_pending = False
            else:
                self._close_kws_for_no_speech(captured_ns, events)
        return events

    def plan_kws(self, *, captured_ns: int) -> ObservedControlGatePlan:
        """Decide whether KWS can run on this KWS-period boundary."""

        self._require_current_tick(captured_ns)
        if self.state is not CascadeState.KWS_ACTIVE:
            return self._skip_plan(captured_ns, gate="kws", reason="phase")
        if self._no_speech_expiry_pending:
            return self._skip_plan(captured_ns, gate="kws", reason="vad_timeout_pending")
        if self.kws_next_ns is None:
            self.kws_next_ns = captured_ns
        if captured_ns < self.kws_next_ns:
            return self._skip_plan(captured_ns, gate="kws", reason="cadence")

        period_ns = self.runtime_config.schedule.kws_period_ms * NS_PER_MS
        while self.kws_next_ns < captured_ns:
            self.kws_next_ns += period_ns
        if self.kws_next_ns != captured_ns:
            return self._skip_plan(captured_ns, gate="kws", reason="cadence")
        self.kws_next_ns += period_ns

        if self._energy_permits_processing(captured_ns):
            self._planned_kws_ns = captured_ns
            return ObservedControlGatePlan(run_kws=True)
        return self._skip_plan(captured_ns, gate="kws", reason="energy_hangover_expired")

    def commit_kws(self, *, captured_ns: int, score: int | float) -> list[CascadeEvent]:
        """Commit one KWS score immediately and emit a wake after two positives."""

        self._require_current_tick(captured_ns)
        if self.state is not CascadeState.KWS_ACTIVE:
            raise RuntimeError("KWS scores can only be committed while KWS is active")
        if self._planned_kws_ns != captured_ns:
            raise RuntimeError("commit_kws requires a successful KWS plan for this tick")
        events = self._commit_kws_value(
            captured_ns=captured_ns,
            score=score,
            primary_score=score,
            count_as_confirmation=True,
        )
        self._planned_kws_ns = None
        return events

    def commit_authorized_kws(
        self,
        *,
        captured_ns: int,
        score: int | float,
        primary_score: int | float | None = None,
        count_as_confirmation: bool = True,
    ) -> list[CascadeEvent]:
        """Commit a score for a KWS window authorized at an earlier PCM boundary."""

        self._require_timestamp(captured_ns)
        if self.state is not CascadeState.KWS_ACTIVE:
            return []
        return self._commit_kws_value(
            captured_ns=captured_ns,
            score=score,
            primary_score=score if primary_score is None else primary_score,
            count_as_confirmation=count_as_confirmation,
        )

    def _commit_kws_value(
        self,
        *,
        captured_ns: int,
        score: int | float,
        primary_score: int | float,
        count_as_confirmation: bool,
    ) -> list[CascadeEvent]:
        """Apply KWS confirmation with a primary-phase quality requirement."""

        score_value = _require_probability("score", score)
        _require_probability("primary_score", primary_score)
        count_confirmation = _require_bool("count_as_confirmation", count_as_confirmation)
        positive = _compares(
            score_value, self.runtime_config.kws_threshold, self.contract.kws_probability_comparator
        )
        if positive and count_confirmation:
            self.kws_streak += 1
        else:
            if not positive:
                self._reset_kws_confirmation()

        events = [
            self._event(
                "kws_call",
                captured_ns,
                gate="kws",
                score=float(score),
                positive=positive,
            )
        ]
        if positive and count_confirmation and self.kws_streak >= self.contract.kws_confirmations:
            events.append(
                self._event(
                    "wake",
                    captured_ns,
                    gate="kws",
                    score=float(score),
                    confirmations=self.kws_streak,
                )
            )
            self._transition(
                CascadeState.WAKE_LATCHED,
                captured_ns,
                events,
                gate="kws",
                reason="kws_confirmed",
            )
        return events

    def _skip_plan(self, captured_ns: int, *, gate: str, reason: str) -> ObservedControlGatePlan:
        return ObservedControlGatePlan(
            skip_reason=reason,
            events=(self._event(f"{gate}_skip", captured_ns, gate=gate, skip_reason=reason),),
        )

    def _plan_kws_without_vad(self, captured_ns: int) -> ObservedControlGatePlan:
        events: list[CascadeEvent] = []
        if self.state is CascadeState.IDLE and self._energy_permits_processing(captured_ns):
            self.vad_streak = 0
            self._candidate_silence_streak = 0
            self._reset_kws_confirmation()
            self.kws_next_ns = captured_ns
            self._transition(
                CascadeState.KWS_ACTIVE,
                captured_ns,
                events,
                gate="vad",
                reason="vad_bypassed",
            )
        elif self.state is CascadeState.KWS_ACTIVE and not self._energy_permits_processing(
            captured_ns
        ):
            self._close_kws_for_energy(captured_ns, events)
        events.append(self._event("vad_skip", captured_ns, gate="vad", skip_reason="disabled"))
        return ObservedControlGatePlan(skip_reason="disabled", events=tuple(events))

    def _transition(
        self,
        state: CascadeState,
        captured_ns: int,
        events: list[CascadeEvent],
        *,
        gate: str,
        reason: str,
    ) -> None:
        previous = self.state
        if previous is state:
            return
        self.state = state
        events.append(
            self._event(
                "state_transition",
                captured_ns,
                gate=gate,
                from_state=previous.value,
                to_state=state.value,
                reason=reason,
            )
        )

    def _close_kws_for_no_speech(self, captured_ns: int, events: list[CascadeEvent]) -> None:
        self._reset_kws_confirmation()
        self.kws_next_ns = None
        self._no_speech_expiry_pending = False
        self._transition(
            CascadeState.IDLE,
            captured_ns,
            events,
            gate="vad",
            reason="no_speech_timeout",
        )

    def _close_kws_for_energy(self, captured_ns: int, events: list[CascadeEvent]) -> None:
        self._reset_kws_confirmation()
        self.kws_next_ns = None
        self._transition(
            CascadeState.IDLE,
            captured_ns,
            events,
            gate="energy",
            reason="energy_hangover_expired",
        )

    def _reset_kws_confirmation(self) -> None:
        self.kws_streak = 0

    def _event(self, kind: str, captured_ns: int, *, gate: str, **fields: object) -> CascadeEvent:
        return CascadeEvent(
            kind,
            captured_ns,
            {
                "profile": self.contract.contract_id,
                "gate": gate,
                "state": self.state.value,
                "vad_streak": self.vad_streak,
                "kws_streak": self.kws_streak,
                "hangover_deadline_ns": self.kws_energy_hangover_deadline_ns,
                "hangover_remaining_ms": self._hangover_remaining_ms(captured_ns),
                "hangover_active": self._hangover_active(captured_ns),
                "skip_reason": None,
                "link": self._link,
                "link_gate_mode": self.contract.link_gate_mode.value,
                "energy_enabled": self.runtime_config.energy_enabled,
                "vad_enabled": self.runtime_config.vad_enabled,
                **fields,
            },
        )

    def _energy_permits_processing(self, captured_ns: int) -> bool:
        return (
            not self.runtime_config.energy_enabled
            or self._energy_pass
            or self._hangover_active(captured_ns)
        )

    def _hangover_active(self, captured_ns: int) -> bool:
        return (
            self.kws_energy_hangover_deadline_ns is not None
            and captured_ns <= self.kws_energy_hangover_deadline_ns
        )

    def _hangover_remaining_ms(self, captured_ns: int) -> int:
        if not self._hangover_active(captured_ns):
            return 0
        assert self.kws_energy_hangover_deadline_ns is not None
        return (self.kws_energy_hangover_deadline_ns - captured_ns) // NS_PER_MS

    def _no_speech_expired(self, captured_ns: int) -> bool:
        return (
            self.last_vad_positive_ns is not None
            and captured_ns - self.last_vad_positive_ns
            >= self.contract.no_speech_timeout_ms * NS_PER_MS
        )

    def _require_current_tick(self, captured_ns: int) -> None:
        self._require_timestamp(captured_ns)
        if self._active_tick_ns != captured_ns:
            raise RuntimeError("begin_tick must be called before planning VAD or KWS work")

    @staticmethod
    def _require_timestamp(captured_ns: int) -> None:
        if isinstance(captured_ns, bool) or not isinstance(captured_ns, int) or captured_ns < 0:
            raise ValueError("captured_ns must be a nonnegative integer")


class ObservedControlEngine:
    """Synchronously adapt PC PCM and ONNX runners to the observed-control reducer.

    This owns only PC-side buffering and model invocation.  It aligns the
    observed control contract without claiming hardware or model parity.
    """

    def __init__(
        self,
        *,
        contract: ObservedControlContract,
        vad_contract: VadContract,
        vad_runner: VadScoreRunner,
        kws_runner: KwsScoreRunner,
        vad_feature_cache: VadFeatureCache,
        kws_executor: KwsExecutionLane | None = None,
        kws_lookback_ms: int = 1_500,
        runtime_config: ObservedControlRuntimeConfig | None = None,
    ) -> None:
        contract.validate()
        if vad_contract.sample_rate != contract.sample_rate_hz:
            raise ValueError("observed control sample rate must match the VAD contract")
        if vad_feature_cache is None:
            raise ValueError("vad_feature_cache is required for observed control")
        if not callable(getattr(vad_runner, "score_chunk", None)):
            raise ValueError("observed control requires a VAD runner with score_chunk")
        self.contract = contract
        self.vad_contract = vad_contract
        self._vad_runner = vad_runner
        self._kws_runner = kws_runner
        self._vad_feature_cache = vad_feature_cache
        self._window_samples = contract.sample_rate_hz * contract.window_ms // 1000
        if self._window_samples != 16_000:
            raise ValueError("observed control requires a 16000-sample PCM window")
        self._kws_lookback_ms = _require_kws_lookback_ms(
            kws_lookback_ms, sample_rate_hz=contract.sample_rate_hz
        )

        self._ring = PcmRingBuffer(self._window_samples)
        # Keep pre-conditioning PCM for energy gating when the model uses AGC.
        self._energy_ring = PcmRingBuffer(self._window_samples)
        self._kws_history = PcmRingBuffer(
            contract.sample_rate_hz * self._kws_lookback_ms // 1000
        )
        self._phase_async_history = PcmRingBuffer(
            contract.sample_rate_hz * PHASE_ASYNC_HISTORY_MS // 1000
        )
        self._runtime_config = runtime_config or ObservedControlRuntimeConfig.from_contract(contract)
        self._configure_timing(self._runtime_config)
        self._machine = ObservedControlStateMachine(contract, self._runtime_config)
        self._samples_until_tick = self._window_samples
        self._last_input_captured_ns: int | None = None
        self._kws_executor = kws_executor
        self._kws_generation = 0
        self._kws_sequence = 0
        self._kws_slot_sequence = 0
        self._next_kws_window_end_ns: int | None = None
        self._kws_phase_origin_ns: int | None = None
        self._kws_history_floor_ns: int | None = None
        self._pending_kws_results: list[KwsResult] = []
        self._pending_phase_confirmation: _PendingPhaseConfirmation | None = None
        self._pending_phase_refinement: _PendingPhaseRefinement | None = None
        self._vad_cache_active = False
        self._latest_vad_score: float | None = None
        self._latest_kws_score: float | None = None
        self._vad_call_count = 0
        self._kws_call_count = 0

    @property
    def state(self) -> CascadeState:
        return self._machine.state

    @property
    def phase(self) -> CascadeState:
        return self._machine.phase

    @property
    def latest_vad_score(self) -> float | None:
        return self._latest_vad_score

    @property
    def latest_kws_score(self) -> float | None:
        return self._latest_kws_score

    @property
    def vad_call_count(self) -> int:
        return self._vad_call_count

    @property
    def kws_call_count(self) -> int:
        return self._kws_call_count

    @property
    def effective_schedule(self) -> dict[str, int]:
        """Return the cadence currently used by the controller and dashboard."""

        return self._machine.effective_schedule

    @property
    def energy_hangover_remaining_ms(self) -> int:
        return self._machine.energy_hangover_remaining_ms

    @property
    def kws_gate_open(self) -> bool:
        return self._machine.kws_gate_open

    @property
    def vad_positive_count(self) -> int:
        return self._machine.vad_positive_count

    @property
    def vad_silence_count(self) -> int:
        return self._machine.vad_silence_count

    @property
    def kws_positive_count(self) -> int:
        return self._machine.kws_positive_count

    @property
    def vad_no_speech_remaining_ms(self) -> int:
        return self._machine.vad_no_speech_remaining_ms

    @property
    def effective_runtime_config(self) -> dict[str, object]:
        """Return the six dashboard-controlled values used by the next tick."""

        return {**self._runtime_config.as_dict(), "kws_lookback_ms": self._kws_lookback_ms}

    @property
    def kws_lookback_ms(self) -> int:
        """Return the retained PCM history available to KWS scheduling."""

        return self._kws_lookback_ms

    def apply_runtime_config(
        self,
        *,
        vad_threshold: float,
        kws_threshold: float,
        energy_enabled: bool,
        vad_enabled: bool,
        vad_period_ms: int,
        kws_period_ms: int,
        kws_lookback_ms: int | None = None,
        captured_ns: int,
    ) -> list[CascadeEvent]:
        """Install a complete dashboard configuration and clear causal state."""

        runtime_config = ObservedControlRuntimeConfig.create(
            contract=self.contract,
            vad_threshold=vad_threshold,
            kws_threshold=kws_threshold,
            energy_threshold_dbfs=self._runtime_config.energy_threshold_dbfs,
            energy_enabled=energy_enabled,
            vad_enabled=vad_enabled,
            vad_period_ms=vad_period_ms,
            kws_period_ms=kws_period_ms,
        )
        self._runtime_config = runtime_config
        self._configure_timing(runtime_config)
        if kws_lookback_ms is not None:
            self._kws_lookback_ms = _require_kws_lookback_ms(
                kws_lookback_ms, sample_rate_hz=self.contract.sample_rate_hz
            )
            self._kws_history = PcmRingBuffer(
                self.contract.sample_rate_hz * self._kws_lookback_ms // 1000
            )
            self._phase_async_history = PcmRingBuffer(
                self.contract.sample_rate_hz * PHASE_ASYNC_HISTORY_MS // 1000
            )
        return self.reset(captured_ns)

    def push_pcm(
        self,
        pcm: np.ndarray,
        captured_ns: int,
        *,
        energy_pcm: np.ndarray | None = None,
    ) -> list[CascadeEvent]:
        """Process every completed sample-time controller boundary synchronously."""

        values = _require_pcm(pcm)
        energy_values = values if energy_pcm is None else _require_pcm(energy_pcm)
        if energy_values.size != values.size:
            raise ValueError("energy_pcm must contain the same number of samples as pcm")
        ObservedControlStateMachine._require_timestamp(captured_ns)
        if (
            self._last_input_captured_ns is not None
            and captured_ns < self._last_input_captured_ns
        ):
            raise ValueError("captured_ns must not move backward")

        source_start_ns = captured_ns - values.size * NS_PER_SECOND // self.contract.sample_rate_hz
        cursor = 0
        events = self.drain_kws_results()
        if (
            self._machine.state is not CascadeState.KWS_ACTIVE
            or not self._machine.kws_gate_open
        ):
            self._next_kws_window_end_ns = None
            self._pending_phase_confirmation = None
            self._pending_phase_refinement = None
        while cursor < values.size:
            current_end_ns = source_start_ns + cursor * NS_PER_SECOND // self.contract.sample_rate_hz
            if self._kws_schedule_is_due(current_end_ns):
                events.extend(self._submit_due_kws_windows(captured_ns=current_end_ns))
                continue
            samples_until_kws = self._samples_until_next_kws_window(current_end_ns)
            take = min(
                self._samples_until_tick,
                values.size - cursor,
                samples_until_kws if samples_until_kws is not None else values.size - cursor,
            )
            appended = values[cursor : cursor + take]
            energy_appended = energy_values[cursor : cursor + take]
            self._ring.append(appended)
            self._energy_ring.append(energy_appended)
            self._kws_history.append(appended)
            self._phase_async_history.append(appended)
            if self._vad_cache_active:
                self._vad_feature_cache.append_pcm(appended)
            cursor += take
            self._samples_until_tick -= take
            if self._samples_until_tick != 0:
                continue
            if not self._ring.is_full:
                raise RuntimeError("observed control tick occurred before the PCM window filled")
            boundary_ns = source_start_ns + cursor * NS_PER_SECOND // self.contract.sample_rate_hz
            events.extend(self._on_vad_tick(boundary_ns))
            self._samples_until_tick = self._vad_tick_samples

        events.extend(self._submit_due_kws_windows(captured_ns=captured_ns))

        self._last_input_captured_ns = captured_ns
        return events

    def reset(self, captured_ns: int) -> list[CascadeEvent]:
        """Discard buffered PCM, causal feature state, and runner state."""

        ObservedControlStateMachine._require_timestamp(captured_ns)
        self._ring.clear()
        self._energy_ring.clear()
        self._kws_history.clear()
        self._phase_async_history.clear()
        self._machine = ObservedControlStateMachine(self.contract, self._runtime_config)
        self._samples_until_tick = self._window_samples
        self._last_input_captured_ns = captured_ns
        self._kws_generation += 1
        self._kws_sequence = 0
        self._kws_slot_sequence = 0
        self._next_kws_window_end_ns = None
        self._kws_phase_origin_ns = None
        self._kws_history_floor_ns = None
        self._pending_kws_results.clear()
        self._pending_phase_confirmation = None
        self._pending_phase_refinement = None
        if self._kws_executor is not None:
            self._kws_executor.invalidate(generation=self._kws_generation)
        self._vad_cache_active = False
        self._vad_feature_cache.reset()
        _reset_if_supported(self._vad_runner)
        _reset_if_supported(self._kws_runner)
        self._latest_vad_score = None
        self._latest_kws_score = None
        self._vad_call_count = 0
        self._kws_call_count = 0
        return []

    def rearm(self, captured_ns: int) -> list[CascadeEvent]:
        """Start a fresh recognition session while retaining the continuous PCM timeline.

        AUTO_REARM is a controller transition, not a capture restart.  Keep both
        PCM rings so the next VAD tick and KWS window can use already-received
        samples, but invalidate model state and jobs from the previous session.
        """

        ObservedControlStateMachine._require_timestamp(captured_ns)
        if self._last_input_captured_ns is not None and captured_ns < self._last_input_captured_ns:
            raise ValueError("captured_ns must not move backward")
        self._machine = ObservedControlStateMachine(self.contract, self._runtime_config)
        self._samples_until_tick = (
            self._vad_tick_samples if self._ring.is_full else self._window_samples - self._ring.sample_count
        )
        self._last_input_captured_ns = captured_ns
        self._kws_generation += 1
        self._kws_sequence = 0
        self._kws_slot_sequence = 0
        self._next_kws_window_end_ns = None
        self._kws_history_floor_ns = captured_ns
        self._pending_kws_results.clear()
        self._pending_phase_confirmation = None
        self._pending_phase_refinement = None
        if self._kws_executor is not None:
            self._kws_executor.invalidate(generation=self._kws_generation)
        self._vad_cache_active = False
        self._vad_feature_cache.reset()
        _reset_if_supported(self._vad_runner)
        _reset_if_supported(self._kws_runner)
        self._latest_vad_score = None
        self._latest_kws_score = None
        self._vad_call_count = 0
        self._kws_call_count = 0
        return []

    def latest_window(self) -> np.ndarray:
        return self._ring.latest()

    def drain_kws_results(self, *, wait: bool = False, timeout: float = 0.0) -> list[CascadeEvent]:
        """Commit completed KWS evidence in model-window time order."""

        if self._kws_executor is None:
            return []
        events: list[CascadeEvent] = []
        for _ in range(2 if wait else 1):
            if wait and not self._kws_executor.wait_for_generation(
                generation=self._kws_generation, timeout=timeout
            ):
                raise RuntimeError("timed out waiting for KWS analysis results")
            scheduled_phase_work = self._collect_kws_results(events)
            if not wait or not scheduled_phase_work:
                break
        return events

    def _collect_kws_results(self, events: list[CascadeEvent]) -> bool:
        """Buffer completed work and preserve a rescue confirmation's time slot."""

        if self._kws_executor is None:
            return False
        confirmation_collector = getattr(self._kws_executor, "collect_phase_confirmations", None)
        if callable(confirmation_collector):
            self._pending_kws_results.extend(
                confirmation_collector(generation=self._kws_generation)
            )
        self._pending_kws_results.extend(
            self._kws_executor.collect_ordered(generation=self._kws_generation)
        )
        scheduled_phase_work = False
        while self._pending_kws_results:
            self._pending_kws_results.sort(key=_kws_result_window_key)
            result = self._pending_kws_results[0]
            pending = self._pending_phase_confirmation
            if (
                pending is not None
                and result.decision_kind not in {"phase_confirmation", "phase_refinement"}
                and _result_window_end_ns(result) > pending.confirmation_window_end_ns
            ):
                break
            refinement = self._pending_phase_refinement
            if (
                refinement is not None
                and result.decision_kind != "phase_refinement"
                and _result_window_end_ns(result) > refinement.source_window_end_ns
            ):
                break
            self._pending_kws_results.pop(0)
            window_end_ns = _result_window_end_ns(result)
            if result.error is not None or result.score is None:
                events.append(
                    CascadeEvent(
                        "kws_error",
                        result.captured_ns,
                        {"error": result.error or "KWS score missing", "window_end_ns": window_end_ns},
                    )
                )
                if result.decision_kind == "phase_confirmation":
                    self._pending_phase_confirmation = None
                elif result.decision_kind == "phase_refinement":
                    self._pending_phase_refinement = None
                continue
            if result.decision_kind == "phase_refinement":
                self._pending_phase_refinement = None
                refined_phase_peak = _supported_strong_phase_peak(
                    result,
                    threshold=self._runtime_config.kws_threshold,
                )
                committed = self._commit_phase_refinement(result)
                if not committed:
                    continue
                self._latest_kws_score = result.score
                self._kws_call_count += 1
                _attach_kws_evidence(
                    committed,
                    window_end_ns=window_end_ns,
                    window_ms=self.contract.window_ms,
                    primary_score=result.primary_score,
                    phase_scores=result.phase_scores,
                    decision_kind=result.decision_kind,
                    phase_refinement_of_window_end_ns=result.phase_refinement_of_window_end_ns,
                    selected_phase_offset_ms=(
                        refined_phase_peak[0] if refined_phase_peak is not None else None
                    ),
                )
                events.extend(committed)
                continue
            strong_phase_peak = _supported_strong_phase_peak(
                result,
                threshold=self._runtime_config.kws_threshold,
            )
            primary_score = result.score if result.primary_score is None else result.primary_score
            defer_phase_rescue_confirmation = (
                self.contract.kws_confirmations > 1
                and self._machine.kws_positive_count >= self.contract.kws_confirmations - 1
                and _is_phase_rescue(result, threshold=self._runtime_config.kws_threshold)
            )
            committed = self._machine.commit_authorized_kws(
                captured_ns=result.captured_ns,
                score=result.score,
                primary_score=primary_score,
                count_as_confirmation=not defer_phase_rescue_confirmation,
            )
            if not committed:
                continue
            self._latest_kws_score = result.score
            self._kws_call_count += 1
            _attach_kws_evidence(
                committed,
                window_end_ns=window_end_ns,
                window_ms=self.contract.window_ms,
                primary_score=result.primary_score,
                phase_scores=result.phase_scores,
                decision_kind=result.decision_kind,
                phase_confirmation_of_window_end_ns=result.phase_confirmation_of_window_end_ns,
                phase_confirmation_phase_offset_ms=result.phase_confirmation_phase_offset_ms,
                selected_phase_offset_ms=(
                    strong_phase_peak[0] if strong_phase_peak is not None else None
                ),
            )
            events.extend(committed)
            if result.decision_kind == "phase_confirmation":
                self._pending_phase_confirmation = None
                continue
            if any(event.kind == "wake" for event in committed):
                continue
            if not _is_phase_rescue(result, threshold=self._runtime_config.kws_threshold):
                if self._schedule_phase_refinement(result):
                    scheduled_phase_work = self._submit_pending_phase_refinement(
                        captured_ns=self._last_input_captured_ns or result.captured_ns
                    ) or scheduled_phase_work
                continue
            # A phase-rescued score can start a confirmation streak, but its
            # second evidence must come from the next regular 96-ms slot.
            # An off-grid confirmation reuses almost all of the same PCM and
            # can select another narrow false peak.
        return scheduled_phase_work

    def _on_vad_tick(self, captured_ns: int) -> list[CascadeEvent]:
        waveform = self._ring.latest()
        energy_waveform = self._energy_ring.latest()
        energy_tail = energy_waveform[-self._vad_tick_samples :]
        energy_dbfs = _rms_dbfs(energy_tail)
        events = self._machine.begin_tick(captured_ns=captured_ns, energy_dbfs=energy_dbfs)

        vad_plan = self._machine.plan_vad(captured_ns=captured_ns)
        events.extend(vad_plan.events)
        if vad_plan.run_vad:
            was_vad_phase_active = self._machine.state in (
                CascadeState.VAD_CANDIDATE,
                CascadeState.KWS_ACTIVE,
            )
            features, bootstrap = self._features_for_vad(waveform)
            vad_score = self._score_vad_control_tick(features, bootstrap=bootstrap)
            vad_events = self._machine.commit_vad(captured_ns=captured_ns, score=vad_score)
            self._latest_vad_score = float(vad_score)
            self._vad_call_count += 1
            events.extend(vad_events)
            if was_vad_phase_active and self._machine.state is CascadeState.IDLE:
                self._deactivate_vad_cache()
        elif vad_plan.skip_reason == "energy":
            self._deactivate_vad_cache()
        kws_plan = self._machine.plan_kws(captured_ns=captured_ns)
        events.extend(kws_plan.events)
        if self._kws_executor is not None:
            if self._machine.kws_gate_open:
                events.extend(self._start_kws_history_schedule(captured_ns=captured_ns))
            else:
                self._next_kws_window_end_ns = None
                self._pending_phase_confirmation = None
                self._pending_phase_refinement = None
        elif kws_plan.run_kws:
            kws_score = self._kws_runner.score(waveform)
            kws_events = self._machine.commit_kws(captured_ns=captured_ns, score=kws_score)
            self._latest_kws_score = float(kws_score)
            self._kws_call_count += 1
            events.extend(kws_events)
        return events

    def _start_kws_history_schedule(self, *, captured_ns: int) -> list[CascadeEvent]:
        """Start a left-edge phased KWS cursor when the PC gate first opens."""

        if self._kws_executor is None or self._next_kws_window_end_ns is not None:
            return []
        available_samples = self._kws_history.sample_count
        if available_samples < self._window_samples:
            return []
        trailing_samples = available_samples - self._window_samples
        first_available_window_end_ns = (
            captured_ns - trailing_samples * NS_PER_SECOND // self.contract.sample_rate_hz
        )
        if self._kws_history_floor_ns is not None:
            if self._kws_phase_origin_ns is not None:
                self._next_kws_window_end_ns = self._first_phase_slot_after(
                    self._kws_history_floor_ns
                )
            else:
                period_ns = self._runtime_config.schedule.kws_period_ms * NS_PER_MS
                self._next_kws_window_end_ns = max(
                    first_available_window_end_ns,
                    self._kws_history_floor_ns + period_ns,
                )
        else:
            self._next_kws_window_end_ns = first_available_window_end_ns
            self._kws_phase_origin_ns = first_available_window_end_ns
        return self._submit_due_kws_windows(captured_ns=captured_ns)

    def _first_phase_slot_after(self, boundary_ns: int) -> int:
        """Return the first existing KWS phase slot strictly after a rearm."""

        assert self._kws_phase_origin_ns is not None
        period_ns = self._runtime_config.schedule.kws_period_ms * NS_PER_MS
        offset_ns = max(0, boundary_ns - self._kws_phase_origin_ns)
        return self._kws_phase_origin_ns + (offset_ns // period_ns + 1) * period_ns

    def _submit_due_kws_windows(self, *, captured_ns: int) -> list[CascadeEvent]:
        """Submit all complete KWS windows already present on the PCM timeline."""

        if (
            self._kws_executor is None
            or self._next_kws_window_end_ns is None
            or self._machine.state is not CascadeState.KWS_ACTIVE
            or not self._machine.kws_gate_open
        ):
            return []
        self._submit_pending_phase_confirmation(captured_ns=captured_ns)
        self._submit_pending_phase_refinement(captured_ns=captured_ns)
        events: list[CascadeEvent] = []
        period_ns = self._runtime_config.schedule.kws_period_ms * NS_PER_MS
        while (
            self._next_kws_window_end_ns is not None
            and self._next_kws_window_end_ns <= captured_ns
            and self._machine.state is CascadeState.KWS_ACTIVE
            and self._machine.kws_gate_open
        ):
            window_end_ns = self._next_kws_window_end_ns
            waveform = self._kws_window_ending_at(
                window_end_ns=window_end_ns, available_end_ns=captured_ns
            )
            if waveform is None:
                # A timestamp-rounded or evicted window can never become
                # reconstructible at this cursor.  Advance it so push_pcm()
                # cannot spin forever on the same due time.
                self._next_kws_window_end_ns = window_end_ns + period_ns
                continue
            probe_waveforms: list[tuple[int, np.ndarray]] = []
            # Preserve the established grey-band rescue as the primary job's
            # low-cost fast path.  Wider refinement is scheduled only after
            # this logical score has been committed.
            probe_offsets = conditional_phase_probe_offsets(
                primary_score=self._runtime_config.kws_threshold - PHASE_GREY_MARGIN,
                kws_threshold=self._runtime_config.kws_threshold,
                waveform=waveform,
            )
            for offset_ms in probe_offsets:
                probe = self._kws_window_ending_at(
                    window_end_ns=window_end_ns + offset_ms * NS_PER_MS,
                    available_end_ns=captured_ns,
                )
                if probe is not None:
                    probe_waveforms.append((offset_ms, probe))
            self._kws_executor.submit(
                KwsJob(
                    sequence=self._kws_sequence,
                    generation=self._kws_generation,
                    captured_ns=captured_ns,
                    waveform=waveform,
                    window_end_ns=window_end_ns,
                    phase_probe_waveforms=tuple(probe_waveforms),
                    phase_probe_lower=self._runtime_config.kws_threshold - PHASE_GREY_MARGIN,
                    phase_probe_threshold=self._runtime_config.kws_threshold,
                    slot_sequence=self._kws_slot_sequence,
                    phase_offset_ms=0,
                )
            )
            self._kws_sequence += 1
            self._kws_slot_sequence += 1
            self._next_kws_window_end_ns = window_end_ns + period_ns
        return events

    def _kws_schedule_is_due(self, captured_ns: int) -> bool:
        return (
            self._kws_executor is not None
            and self._next_kws_window_end_ns is not None
            and self._next_kws_window_end_ns <= captured_ns
        )

    def _samples_until_next_kws_window(self, captured_ns: int) -> int | None:
        if (
            self._kws_executor is None
            or self._next_kws_window_end_ns is None
        ):
            return None
        remaining_ns = self._next_kws_window_end_ns - captured_ns
        if remaining_ns <= 0:
            return 0
        return max(1, remaining_ns * self.contract.sample_rate_hz // NS_PER_SECOND)

    def _kws_window_ending_at(
        self, *, window_end_ns: int, available_end_ns: int
    ) -> np.ndarray | None:
        return self._window_from_history(
            self._kws_history,
            window_end_ns=window_end_ns,
            available_end_ns=available_end_ns,
        )

    def _phase_window_ending_at(
        self, *, window_end_ns: int, available_end_ns: int
    ) -> np.ndarray | None:
        """Read a phase-only window without widening normal KWS lookback."""

        return self._window_from_history(
            self._phase_async_history,
            window_end_ns=window_end_ns,
            available_end_ns=available_end_ns,
        )

    def _window_from_history(
        self,
        history_ring: PcmRingBuffer,
        *,
        window_end_ns: int,
        available_end_ns: int,
    ) -> np.ndarray | None:
        try:
            history = history_ring.snapshot()
        except RuntimeError:
            return None
        trailing_ns = available_end_ns - window_end_ns
        if trailing_ns < 0:
            return None
        trailing_samples = trailing_ns * self.contract.sample_rate_hz // NS_PER_SECOND
        end_index = history.size - trailing_samples
        start_index = end_index - self._window_samples
        if start_index < 0 or end_index > history.size:
            return None
        waveform = history[start_index:end_index]
        if waveform.shape != (self._window_samples,):
            return None
        return waveform.copy()

    def _submit_pending_phase_confirmation(self, *, captured_ns: int) -> bool:
        pending = self._pending_phase_confirmation
        if self._kws_executor is None or pending is None or pending.submitted:
            return False
        if pending.confirmation_window_end_ns > captured_ns:
            return False
        waveform = self._kws_window_ending_at(
            window_end_ns=pending.confirmation_window_end_ns, available_end_ns=captured_ns
        )
        if waveform is None:
            return False
        self._kws_executor.submit(
            KwsJob(
                sequence=pending.source_sequence,
                generation=self._kws_generation,
                captured_ns=captured_ns,
                waveform=waveform,
                window_end_ns=pending.confirmation_window_end_ns,
                decision_kind="phase_confirmation",
                phase_confirmation_of_window_end_ns=pending.source_window_end_ns,
                phase_confirmation_phase_offset_ms=pending.phase_offset_ms,
            )
        )
        pending.submitted = True
        return True

    def _schedule_phase_refinement(self, result: KwsResult) -> bool:
        """Defer bounded fine-phase work until every future seed window exists."""

        if self._pending_phase_refinement is not None or result.primary_score is None:
            return False
        source_end_ns = _result_window_end_ns(result)
        source_waveform = self._phase_window_ending_at(
            window_end_ns=source_end_ns,
            available_end_ns=self._last_input_captured_ns or result.captured_ns,
        )
        if source_waveform is None:
            return False
        seed_offsets_ms = phase_probe_plan(
            primary_score=result.primary_score,
            kws_threshold=self._runtime_config.kws_threshold,
            waveform=source_waveform,
        )
        if not seed_offsets_ms:
            return False
        available_end_ns = self._last_input_captured_ns or result.captured_ns
        earliest_probe_offset_ms = min(seed_offsets_ms) - PHASE_REFINEMENT_NEIGHBOR_DELTA_MS
        if self._phase_window_ending_at(
            window_end_ns=source_end_ns + earliest_probe_offset_ms * NS_PER_MS,
            available_end_ns=available_end_ns,
        ) is None:
            return False
        self._pending_phase_refinement = _PendingPhaseRefinement(
            source_sequence=result.sequence,
            source_window_end_ns=source_end_ns,
            source_result=result,
            seed_offsets_ms=seed_offsets_ms,
            refinement_window_end_ns=(
                source_end_ns
                + (max(seed_offsets_ms) + PHASE_REFINEMENT_NEIGHBOR_DELTA_MS) * NS_PER_MS
            ),
        )
        return True

    def _submit_pending_phase_refinement(self, *, captured_ns: int) -> bool:
        pending = self._pending_phase_refinement
        if self._kws_executor is None or pending is None or pending.submitted:
            return False
        if pending.refinement_window_end_ns > captured_ns:
            return False
        seed_waveforms: list[tuple[int, np.ndarray]] = []
        neighbors: list[tuple[int, tuple[tuple[int, np.ndarray], ...]]] = []
        for offset_ms in pending.seed_offsets_ms:
            seed = self._phase_window_ending_at(
                window_end_ns=pending.source_window_end_ns + offset_ms * NS_PER_MS,
                available_end_ns=captured_ns,
            )
            if seed is None:
                return False
            seed_waveforms.append((offset_ms, seed))
            neighbor_waveforms: list[tuple[int, np.ndarray]] = []
            for neighbor_offset_ms in (
                offset_ms - PHASE_REFINEMENT_NEIGHBOR_DELTA_MS,
                offset_ms + PHASE_REFINEMENT_NEIGHBOR_DELTA_MS,
            ):
                neighbor = self._phase_window_ending_at(
                    window_end_ns=pending.source_window_end_ns + neighbor_offset_ms * NS_PER_MS,
                    available_end_ns=captured_ns,
                )
                if neighbor is None:
                    return False
                neighbor_waveforms.append((neighbor_offset_ms, neighbor))
            neighbors.append((offset_ms, tuple(neighbor_waveforms)))
        primary_waveform = self._phase_window_ending_at(
            window_end_ns=pending.refinement_window_end_ns,
            available_end_ns=captured_ns,
        )
        if primary_waveform is None:
            return False
        self._kws_executor.submit(
            KwsJob(
                sequence=pending.source_sequence,
                generation=self._kws_generation,
                captured_ns=captured_ns,
                waveform=primary_waveform,
                window_end_ns=pending.refinement_window_end_ns,
                phase_probe_waveforms=tuple(seed_waveforms),
                phase_probe_threshold=self._runtime_config.kws_threshold,
                decision_kind="phase_refinement",
                phase_refinement_primary_score=pending.source_result.primary_score,
                phase_refinement_neighbors=tuple(neighbors),
                phase_refinement_of_window_end_ns=pending.source_window_end_ns,
            )
        )
        pending.submitted = True
        return True

    def _commit_phase_refinement(self, result: KwsResult) -> list[CascadeEvent]:
        """Accept only a supported strong peak; otherwise preserve normal confirmation."""

        if result.score is None:
            return []
        selected_peak = _supported_strong_phase_peak(
            result,
            threshold=self._runtime_config.kws_threshold,
        )
        if selected_peak is None:
            return []
        selected_offset_ms, selected_score = selected_peak
        return self._machine.commit_authorized_kws(
            captured_ns=result.captured_ns,
            score=selected_score,
            primary_score=result.score if result.primary_score is None else result.primary_score,
            count_as_confirmation=True,
        )

    def _features_for_vad(self, waveform: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._vad_cache_active:
            _reset_if_supported(self._vad_runner)
            features = _require_features(self._vad_feature_cache.rebuild(waveform))
            self._vad_cache_active = True
            return features, True
        return _require_features(self._vad_feature_cache.take_new_features()), False

    def _score_vad_control_tick(self, features: np.ndarray, *, bootstrap: bool) -> float:
        """Use the tick maximum when the causal runner exposes every posterior."""

        scores = np.asarray(self._vad_runner.score_chunk(features))  # type: ignore[attr-defined]
        if scores.dtype != np.float32 or scores.ndim != 1 or scores.size == 0:
            raise ValueError("VAD score chunk must be a nonempty float32 vector")
        if scores.size != features.shape[0]:
            raise ValueError("VAD score chunk must provide one posterior per VAD feature frame")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("VAD score chunk must contain probabilities")
        if bootstrap:
            # The causal pre-roll evolves model state; only its final control tick decides.
            scores = scores[-self._bootstrap_tick_posterior_count :]
        return float(np.max(scores))

    def _deactivate_vad_cache(self) -> None:
        if not self._vad_cache_active:
            return
        self._vad_feature_cache.reset()
        _reset_if_supported(self._vad_runner)
        self._vad_cache_active = False

    def _configure_timing(self, runtime_config: ObservedControlRuntimeConfig) -> None:
        self._vad_tick_samples = (
            self.contract.sample_rate_hz * runtime_config.schedule.vad_period_ms // 1000
        )
        if self._vad_tick_samples <= 0:
            raise ValueError("observed control requires a positive VAD tick size")
        frame_samples = round(self.vad_contract.sample_rate * self.vad_contract.frame_ms / 1000.0)
        hop_samples = round(self.vad_contract.sample_rate * self.vad_contract.hop_ms / 1000.0)
        frame_count = (self._window_samples - frame_samples) // hop_samples + 1
        tick_start_sample = self._window_samples - self._vad_tick_samples
        self._bootstrap_tick_posterior_count = sum(
            frame_index * hop_samples + frame_samples > tick_start_sample
            for frame_index in range(frame_count)
        )
        if self._bootstrap_tick_posterior_count <= 0:
            raise ValueError("observed control bootstrap has no VAD posterior in its first tick")


def _require_kws_lookback_ms(value: object, *, sample_rate_hz: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("kws_lookback_ms must be an integer")
    if not MIN_KWS_LOOKBACK_MS <= value <= MAX_KWS_LOOKBACK_MS:
        raise ValueError(
            f"kws_lookback_ms must be in [{MIN_KWS_LOOKBACK_MS}, {MAX_KWS_LOOKBACK_MS}]"
        )
    if value < 1_000:
        raise ValueError("kws_lookback_ms must retain a complete KWS window")
    if (sample_rate_hz * value) % 1000:
        raise ValueError("kws_lookback_ms must resolve to an integer PCM sample count")
    return value


def _result_window_end_ns(result: KwsResult) -> int:
    return result.captured_ns if result.window_end_ns is None else result.window_end_ns


def _kws_result_window_key(result: KwsResult) -> tuple[int, int, int]:
    return (
        _result_window_end_ns(result),
        0 if result.decision_kind == "logical_slot" else 1,
        result.sequence,
    )


def _is_phase_rescue(result: KwsResult, *, threshold: float) -> bool:
    if result.primary_score is None or result.primary_score >= threshold:
        return False
    if not result.phase_scores:
        return False
    offset_ms, selected_score = max(result.phase_scores, key=lambda item: item[1])
    return offset_ms != 0 and selected_score >= threshold


def _is_supported_strong_phase_peak(result: KwsResult, *, threshold: float) -> bool:
    """Return whether an off-grid high peak has local waveform-phase support."""

    return _supported_strong_phase_peak(result, threshold=threshold) is not None


def _supported_strong_phase_peak(
    result: KwsResult, *, threshold: float
) -> tuple[int, float] | None:
    """Select a highest supported phase peak, including tied seed candidates."""

    if result.primary_score is None or not result.phase_scores:
        return None
    candidates = sorted(
        result.phase_scores,
        key=lambda item: (-item[1], abs(item[0]), item[0]),
    )
    for selected_offset_ms, selected_score in candidates:
        if (
            selected_offset_ms == 0
            or selected_score < PHASE_STRONG_SCORE
            or selected_score - result.primary_score < PHASE_STRONG_MIN_LIFT
        ):
            continue
        if any(
            neighbor_score >= threshold
            for offset_ms, neighbor_score in result.phase_scores
            if offset_ms not in {0, selected_offset_ms}
            and abs(offset_ms - selected_offset_ms) <= PHASE_REFINEMENT_NEIGHBOR_DELTA_MS
        ):
            return selected_offset_ms, selected_score
    return None


def _attach_kws_evidence(
    events: list[CascadeEvent],
    *,
    window_end_ns: int,
    window_ms: int,
    primary_score: float | None,
    phase_scores: tuple[tuple[int, float], ...],
    decision_kind: str,
    phase_confirmation_of_window_end_ns: int | None = None,
    phase_confirmation_phase_offset_ms: int | None = None,
    phase_refinement_of_window_end_ns: int | None = None,
    selected_phase_offset_ms: int | None = None,
) -> None:
    fields: dict[str, object] = {
        "window_end_ns": window_end_ns,
        "model_input_start_ns": window_end_ns - window_ms * NS_PER_MS,
        "model_input_end_ns": window_end_ns,
        "primary_score": primary_score,
        "phase_scores": {str(offset): score for offset, score in phase_scores},
        "selected_phase_offset_ms": (
            selected_phase_offset_ms
            if selected_phase_offset_ms is not None
            else max(phase_scores, key=lambda item: item[1])[0] if phase_scores else 0
        ),
        "decision_kind": decision_kind,
    }
    if phase_confirmation_of_window_end_ns is not None:
        fields["phase_confirmation_of_window_end_ns"] = phase_confirmation_of_window_end_ns
        fields["phase_confirmation_phase_offset_ms"] = phase_confirmation_phase_offset_ms
    if phase_refinement_of_window_end_ns is not None:
        fields["phase_refinement_of_window_end_ns"] = phase_refinement_of_window_end_ns
    for event in events:
        event.fields.update(fields)


def _compares(value: float, threshold: int | float, comparator: ThresholdComparator) -> bool:
    if comparator is ThresholdComparator.GT:
        return value > threshold
    if comparator is ThresholdComparator.GE:
        return value >= threshold
    raise ValueError(f"unsupported comparator: {comparator}")


def _require_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_probability(name: str, value: object) -> float:
    result = _require_finite_number(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_pcm(pcm: np.ndarray) -> np.ndarray:
    values = np.asarray(pcm)
    if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
        raise ValueError("PCM must be a nonempty float32 mono array")
    if not np.all(np.isfinite(values)):
        raise ValueError("PCM must contain only finite values")
    return values


def _require_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[0] == 0:
        raise RuntimeError("VAD feature cache did not produce complete float32 frames")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("VAD feature cache did not produce finite frames")
    return values


def _reset_if_supported(component: object) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()
