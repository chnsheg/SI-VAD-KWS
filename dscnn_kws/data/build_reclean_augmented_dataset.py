from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reclean.catalog import CAPTURED_SCENES, acquire_and_normalize_rirs, fast_inventory_sources, inventory_sources
from .reclean.generator import generate_shard, iter_generation_requests, rank_device_from_environment, stratified_pilot_requests
from .reclean.recipes import build_generation_requests, plan_output_quotas, write_generation_requests
from .reclean.prepare import prepare_sources
from .reclean.validate import build_dataset_manifests, validate_dataset


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible KWS recleaned-corpus builder")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="audit all source datasets before generation")
    inventory.add_argument("--mobvoi-manifest-root", type=_path, required=True, help="Directory containing Mobvoi JSONL manifests")
    inventory.add_argument("--tau-root", type=_path, required=True, help="TAU root containing all ten named acoustic scenes")
    inventory.add_argument("--kindgarden-root", type=_path, required=True, help="Captured kindgarden noise root")
    inventory.add_argument("--livingroom-root", type=_path, required=True, help="Captured livingroom noise root")
    inventory.add_argument("--pub-root", type=_path, required=True, help="Captured pub noise root")
    inventory.add_argument("--road-root", type=_path, required=True, help="Captured road noise root")
    inventory.add_argument("--wind-root", type=_path, required=True, help="Captured wind-noise root (catalogued as 风噪)")
    inventory.add_argument("--false-wake-root", type=_path, required=True, help="Continuous similar-phrase recordings")
    inventory.add_argument("--rir-root", type=_path, required=True, help="Normalized public RIR catalog root")
    inventory.add_argument("--output-root", type=_path, required=True, help="Destination corpus root")
    inventory.add_argument("--require-free-gib", type=float, default=100.0, help="Minimum free destination space")
    inventory.add_argument("--false-wake-seed", type=int, default=42, help="Source-level false-wake split seed")
    fast_inventory = commands.add_parser(
        "fast-inventory", help="write a lightweight source inventory from verified downloads without source hashing"
    )
    fast_inventory.add_argument("--mobvoi-resource-root", type=_path, required=True, help="Official Mobvoi p_*.json and n_*.json directory")
    fast_inventory.add_argument("--mobvoi-audio-root", type=_path, required=True, help="Flat Mobvoi WAV directory addressed by utt_id")
    fast_inventory.add_argument("--target-keyword-id", type=int, default=0, help="Mobvoi target keyword ID; 0 is hi_xiaowen")
    fast_inventory.add_argument("--tau-root", type=_path, required=True, help="TAU root containing all ten named acoustic scenes")
    fast_inventory.add_argument("--kindgarden-root", type=_path, required=True, help="Captured kindgarden noise root")
    fast_inventory.add_argument("--livingroom-root", type=_path, required=True, help="Captured livingroom noise root")
    fast_inventory.add_argument("--pub-root", type=_path, required=True, help="Captured pub noise root")
    fast_inventory.add_argument("--road-root", type=_path, required=True, help="Captured road noise root")
    fast_inventory.add_argument("--wind-root", type=_path, required=True, help="Captured wind-noise root (catalogued as 风噪)")
    fast_inventory.add_argument("--false-wake-root", type=_path, required=True, help="Continuous similar-phrase recordings")
    fast_inventory.add_argument("--rir-root", type=_path, required=True, help="Normalized public RIR catalog root")
    fast_inventory.add_argument("--output-root", type=_path, required=True, help="Destination corpus root")
    fast_inventory.add_argument("--require-free-gib", type=float, default=100.0, help="Minimum free destination space")
    fast_inventory.add_argument("--false-wake-seed", type=int, default=42, help="Source-level false-wake split seed")
    rir = commands.add_parser("download-rir", help="download, verify, and normalize public RIRS_NOISES impulses")
    rir.add_argument("--url", required=True, help="OpenSLR RIRS_NOISES archive URL")
    rir.add_argument("--expected-sha256", required=True, help="Required archive SHA-256")
    rir.add_argument("--archive-path", type=_path, required=True, help="Verified archive destination")
    rir.add_argument("--extraction-root", type=_path, required=True, help="Archive extraction directory")
    rir.add_argument("--normalized-root", type=_path, required=True, help="Normalized 16 kHz RIR catalog directory")
    plan = commands.add_parser("plan", help="calculate locked class quotas before synthesis")
    plan.add_argument("--positive-source-count", type=int, default=21825, help="Audited positive-source count")
    plan.add_argument("--variants-per-positive", type=int, default=48, help="Fixed coverage slots per positive source")
    plan.add_argument("--global-seed", type=int, default=42, help="Global deterministic recipe seed")
    plan.add_argument("--output-root", type=_path, required=True, help="Destination corpus root")
    plan.add_argument("--prepared-manifest", type=_path, default=None, help="Prepared-source JSONL; defaults below output root")
    prepare = commands.add_parser("prepare", help="canonicalize sources, find positive boundaries, and segment false wakes")
    prepare.add_argument("--inventory-json", type=_path, required=True, help="Inventory JSON written by the inventory command")
    prepare.add_argument("--output-root", type=_path, required=True, help="Destination corpus root")
    for command_name, help_text in (
        ("pilot", "generate a fixed-size four-rank throughput and integrity pilot"),
        ("generate", "generate the complete four-rank corpus from a frozen request manifest"),
    ):
        generation = commands.add_parser(command_name, help=help_text)
        generation.add_argument("--recipe-manifest", type=_path, default=None, help="Frozen generation-request JSONL; defaults below output root")
        generation.add_argument("--output-root", type=_path, required=True, help="Destination corpus root")
        generation.add_argument("--allow-cpu-debug", action="store_true", help="Permit a single-rank CPU debug run; never use for corpus generation")
        if command_name == "pilot":
            generation.add_argument("--examples", type=int, default=10000, help="Total examples across all four ranks")
    validate = commands.add_parser("validate", help="validate generated audio format, metadata, and output hashes")
    validate.add_argument("--output-root", type=_path, required=True, help="Generated corpus root")
    validate.add_argument("--prepared-manifest", type=_path, default=None, help="Prepared-source JSONL used for clean validation/test manifests")
    validate.add_argument("--no-verify-hashes", action="store_true", help="Skip output hash verification for a format-only diagnostic")
    return parser


def run_inventory(args: argparse.Namespace) -> dict[str, object]:
    captured = {
        "kindgarden": args.kindgarden_root,
        "livingroom": args.livingroom_root,
        "pub": args.pub_root,
        "road": args.road_root,
        "风噪": args.wind_root,
    }
    if set(captured) != set(CAPTURED_SCENES):
        raise RuntimeError("Captured-scene CLI mapping does not match the locked V1 catalog")
    return inventory_sources(
        mobvoi_manifest_root=args.mobvoi_manifest_root,
        tau_root=args.tau_root,
        captured_scene_roots=captured,
        false_wake_root=args.false_wake_root,
        rir_root=args.rir_root,
        output_root=args.output_root,
        require_free_gib=args.require_free_gib,
        false_wake_seed=args.false_wake_seed,
    )


def run_fast_inventory(args: argparse.Namespace) -> dict[str, object]:
    captured = {
        "kindgarden": args.kindgarden_root,
        "livingroom": args.livingroom_root,
        "pub": args.pub_root,
        "road": args.road_root,
        "风噪": args.wind_root,
    }
    if set(captured) != set(CAPTURED_SCENES):
        raise RuntimeError("Captured-scene CLI mapping does not match the locked V1 catalog")
    return fast_inventory_sources(
        mobvoi_resource_root=args.mobvoi_resource_root,
        mobvoi_audio_root=args.mobvoi_audio_root,
        target_keyword_id=args.target_keyword_id,
        tau_root=args.tau_root,
        captured_scene_roots=captured,
        false_wake_root=args.false_wake_root,
        rir_root=args.rir_root,
        output_root=args.output_root,
        require_free_gib=args.require_free_gib,
        false_wake_seed=args.false_wake_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inventory":
        report = run_inventory(args)
        print(json.dumps({key: report[key] for key in ("output_root", "available_space_bytes", "noise_scene_file_counts")}, ensure_ascii=False))
        return 0
    if args.command == "fast-inventory":
        report = run_fast_inventory(args)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("output_root", "available_space_bytes", "noise_scene_file_counts", "mobvoi_target_keyword_id")
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "download-rir":
        rows = acquire_and_normalize_rirs(
            url=args.url,
            expected_sha256=args.expected_sha256,
            archive_path=args.archive_path,
            extraction_root=args.extraction_root,
            normalized_root=args.normalized_root,
        )
        print(json.dumps({"normalized_rirs": len(rows), "normalized_root": str(args.normalized_root)}, ensure_ascii=False))
        return 0
    if args.command == "plan":
        prepared_manifest = args.prepared_manifest or args.output_root / "prepared" / "prepared_sources.jsonl"
        if prepared_manifest.is_file():
            prepared_rows = [json.loads(line) for line in prepared_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            requests = build_generation_requests(prepared_rows, args.global_seed, args.variants_per_positive)
            positive_source_count = sum(
                row.get("source_kind") == "speech" and row.get("source_label") == "positive" and row.get("source_split") == "train"
                for row in prepared_rows
            )
        else:
            requests = []
            positive_source_count = args.positive_source_count
        quotas = plan_output_quotas(positive_source_count, args.variants_per_positive)
        args.output_root.mkdir(parents=True, exist_ok=True)
        if requests:
            written_request_count = write_generation_requests(requests, args.output_root / "generation_requests.jsonl")
            if written_request_count != sum(quotas.values()):
                raise RuntimeError("Generation-request manifest does not match planned quotas")
        (args.output_root / "plan_summary.json").write_text(
            json.dumps(
                {
                    "global_seed": args.global_seed,
                    "positive_source_count": positive_source_count,
                    "variants_per_positive": args.variants_per_positive,
                    "quotas": quotas,
                    "total": sum(quotas.values()),
                    "generation_request_manifest": str(args.output_root / "generation_requests.jsonl") if requests else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"quotas": quotas, "total": sum(quotas.values())}, ensure_ascii=False))
        return 0
    if args.command == "prepare":
        rows = prepare_sources(args.inventory_json, args.output_root / "prepared")
        print(json.dumps({"prepared_rows": len(rows), "prepared_root": str(args.output_root / "prepared")}, ensure_ascii=False))
        return 0
    if args.command in {"pilot", "generate"}:
        rank, world_size, device = rank_device_from_environment()
        if not args.allow_cpu_debug and (world_size != 4 or device.type != "cuda"):
            raise RuntimeError(
                "Corpus generation requires torchrun with exactly four CUDA ranks; use --allow-cpu-debug only for a small local check"
            )
        manifest_path = args.recipe_manifest or args.output_root / "generation_requests.jsonl"
        requests = iter_generation_requests(manifest_path)
        if args.command == "pilot":
            if args.examples <= 0:
                raise ValueError("--examples must be positive")
            requests = stratified_pilot_requests(manifest_path, args.examples)
        telemetry = generate_shard(requests, args.output_root, rank=rank, world_size=world_size, device=device)
        print(json.dumps(telemetry, ensure_ascii=False))
        return 0
    if args.command == "validate":
        report = validate_dataset(args.output_root, verify_hashes=not args.no_verify_hashes)
        manifest_counts = None
        prepared_manifest = args.prepared_manifest or args.output_root / "prepared" / "prepared_sources.jsonl"
        if report.ok and prepared_manifest.is_file():
            prepared_rows = [json.loads(line) for line in prepared_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            manifest_counts = build_dataset_manifests(args.output_root, prepared_rows)
        print(json.dumps({"ok": report.ok, "errors": report.errors, "counts": report.counts, "manifest_counts": manifest_counts}, ensure_ascii=False))
        return 0 if report.ok else 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
