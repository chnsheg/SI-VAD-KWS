"""Command-line entry point for reproducible two-recording VAD/KWS tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from dscnn_kws.ONNX.export_vad_stateful import export_stateful_vad

from .contracts import sha256_file
from .input_frontend import FrontendProfile
from .optimization import (
    Candidate,
    CandidateEvaluation,
    CandidateRejection,
    CascadeTuning,
    FrontendTuning,
    ModelPaths,
    RecordingEvaluator,
    ReplayMetrics,
    SourceSpec,
    search_candidates,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.vad_model is None or args.kws_model is None or args.output is None or args.csv_output is None:
        print("--vad-model, --kws-model, --output, and --csv-output are required", file=sys.stderr)
        return 2
    if len(args.source) != 2:
        print("exactly two --source WAV paths are required", file=sys.stderr)
        return 2
    if args.budget < len(FrontendProfile):
        print(f"--budget must be at least {len(FrontendProfile)}", file=sys.stderr)
        return 2

    sources = tuple(SourceSpec(f"source_{index + 1}", path) for index, path in enumerate(args.source))
    try:
        for source in sources:
            source.load_pcm()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    try:
        result, baseline = _run_search(
            sources=sources,
            vad_model=args.vad_model,
            kws_model=args.kws_model,
            seed=args.seed,
            budget=args.budget,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"optimization failed: {error}", file=sys.stderr)
        return 1

    payload = {
        "schema_version": 1,
        "sources": [
            {"name": source.name, "path": str(source.path), "sha256": sha256_file(source.path)}
            for source in sources
        ],
        "models": {
            "vad_model": str(args.vad_model),
            "vad_sha256": sha256_file(args.vad_model),
            "kws_model": str(args.kws_model),
            "kws_sha256": sha256_file(args.kws_model),
        },
        "seed": args.seed,
        "budget": args.budget,
        "baseline": {source: _metrics_dict(metrics) for source, metrics in baseline.items()},
        "selected": _candidate_dict(result.selected) if result.selected is not None else None,
        "accepted": [_evaluation_dict(evaluation) for evaluation in result.accepted],
        "rejected": [
            {
                "candidate": _candidate_dict(rejection.candidate),
                "reports": {source: _metrics_dict(metrics) for source, metrics in rejection.reports.items()},
                "reason": rejection.reason,
            }
            for rejection in result.rejected
        ],
    }
    _write_json(args.output, payload)
    _write_csv(args.csv_output, baseline, result.accepted, result.rejected)
    if result.selected is None:
        print("No stable winner: all candidates regressed on at least one source.")
    else:
        print(f"Selected stable candidate: {result.selected.name}")
    return 0


def _run_search(
    *,
    sources: tuple[SourceSpec, SourceSpec],
    vad_model: Path,
    kws_model: Path,
    seed: int,
    budget: int,
):
    with TemporaryDirectory(prefix="vad-kws-optimization-") as directory:
        stateful_vad = Path(directory) / "vad_stateful.onnx"
        export_stateful_vad(vad_model, stateful_vad)
        evaluator = RecordingEvaluator(sources, ModelPaths(vad_model, kws_model, stateful_vad))
        baseline_candidate = Candidate(
            "baseline-raw",
            FrontendTuning(FrontendProfile.RAW, -20.0),
            CascadeTuning(-33.0, 1_000, 1_000, 32, 96, 0.80, 3, 0.75),
        )
        baseline = evaluator.evaluate(baseline_candidate)
        return search_candidates(evaluator, baseline, seed=seed, budget=budget), baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad-model", type=Path)
    parser.add_argument("--kws-model", type=Path)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--seed", type=int, default=20_260_805)
    parser.add_argument("--budget", type=int, default=24)
    return parser


def _metrics_dict(metrics: ReplayMetrics) -> dict[str, int | float | None]:
    values = asdict(metrics)
    latency = values["p95_wake_latency_ms"]
    if isinstance(latency, float) and not math.isfinite(latency):
        values["p95_wake_latency_ms"] = None
    return values


def _candidate_dict(candidate: Candidate | None) -> dict[str, object] | None:
    if candidate is None:
        return None
    return {
        "name": candidate.name,
        "frontend": {
            "profile": candidate.frontend.profile.value,
            "target_rms_dbfs": candidate.frontend.target_rms_dbfs,
        },
        "cascade": asdict(candidate.cascade),
    }


def _evaluation_dict(evaluation: CandidateEvaluation) -> dict[str, object]:
    candidate = _candidate_dict(evaluation.candidate)
    assert candidate is not None
    return {
        "candidate": candidate,
        "reports": {source: _metrics_dict(metrics) for source, metrics in evaluation.reports.items()},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(
    path: Path,
    baseline: Mapping[str, ReplayMetrics],
    accepted: Sequence[CandidateEvaluation],
    rejected: Sequence[CandidateRejection],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source, metrics in baseline.items():
        rows.append({"candidate": "baseline-raw", "status": "baseline", "source": source, **_metrics_dict(metrics)})
    for evaluation in accepted:
        for source, metrics in evaluation.reports.items():
            rows.append({"candidate": evaluation.candidate.name, "status": "accepted", "source": source, **_metrics_dict(metrics)})
    for rejection in rejected:
        candidate = rejection.candidate
        reports = rejection.reports
        for source, metrics in reports.items():
            rows.append({"candidate": candidate.name, "status": "rejected", "source": source, **_metrics_dict(metrics)})
    fields = ["candidate", "status", "source", *asdict(ReplayMetrics(0, 0, 0)).keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
