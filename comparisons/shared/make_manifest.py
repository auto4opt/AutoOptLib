"""Write a machine-local manifest for the built official external methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from comparisons.shared.store import atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_native_provenance(runner: Path) -> dict[str, str]:
    lock_path = Path(__file__).resolve().parents[1] / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["sources"]
    build_path = runner.resolve().parent / "build-manifest.json"
    if not build_path.is_file():
        raise FileNotFoundError(
            f"Native runner has no verifiable build manifest: {build_path}"
        )
    build = json.loads(build_path.read_text(encoding="utf-8"))
    expected = {
        "binary": str(runner.resolve()),
        "binary_sha256": _sha256(runner),
        "paradiseo_commit": str(lock["paradiseo"]["commit"]),
        "iohexperimenter_commit": str(lock["iohexperimenter"]["commit"]),
    }
    for name, value in expected.items():
        if build.get(name) != value:
            raise ValueError(
                f"Native build manifest field {name!r} is {build.get(name)!r}; "
                f"expected {value!r}. Rebuild from the pinned sources."
            )
    return {
        **expected,
        "native_build_manifest": str(build_path),
    }


def make_manifest(
    output: Path,
    runner: Path,
    *,
    python: str,
    rscript: str,
    sparkle: str,
    workers: int,
) -> dict:
    if not runner.is_file():
        raise FileNotFoundError(runner)
    native_provenance = _verified_native_provenance(runner)
    lock = json.loads(
        (Path(__file__).resolve().parents[1] / "sources.lock.json").read_text(
            encoding="utf-8"
        )
    )["sources"]
    common_run = [
        str(runner.resolve()),
        "--config",
        "{artifact}",
        "--suite",
        "{suite}",
        "--function",
        "{function}",
        "--dimension",
        "{dimension}",
        "--instance",
        "{instance}",
        "--seed",
        "{seed}",
        "--budget",
        "{budget}",
        "--json",
    ]
    manifest = {
        "paradiseo_irace": {
            "implementation": "ParadisEO eoFastGA configured by official irace",
            "design_command": [
                python,
                "-m",
                "comparisons.paradiseo.configure",
                "--request",
                "{request}",
                "--output",
                "{artifact}",
                "--runner",
                str(runner.resolve()),
                "--work",
                "{artifact}.irace-work",
                "--rscript",
                rscript,
                "--workers",
                str(max(1, workers)),
            ],
            "command": common_run,
            "provenance": {
                **native_provenance,
                "irace_commit": str(lock["irace"]["commit"]),
            },
        },
        "sparkle_smac3": {
            "implementation": "Official Sparkle platform with SMAC3 over eoFastGA",
            "design_command": [
                python,
                "-m",
                "comparisons.sparkle.configure",
                "--request",
                "{request}",
                "--output",
                "{artifact}",
                "--runner",
                str(runner.resolve()),
                "--work",
                "{artifact}.sparkle-work",
                "--sparkle",
                sparkle,
                "--workers",
                "1",
            ],
            "command": common_run,
            "provenance": {
                **native_provenance,
                "sparkle_commit": str(lock["sparkle"]["commit"]),
                "smac3_commit": str(lock["smac3"]["commit"]),
            },
        },
    }
    atomic_json(output, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--sparkle", default="sparkle")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for irace; Sparkle uses its official sequential CLI.",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            make_manifest(
                args.output,
                args.runner,
                python=args.python,
                rscript=args.rscript,
                sparkle=args.sparkle,
                workers=args.workers,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
