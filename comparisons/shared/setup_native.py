"""Build official ParadisEO, the common native runner, and a local manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from comparisons.shared.make_manifest import make_manifest
from comparisons.shared.native.build import build as build_runner


def setup(
    source_root: Path,
    output_root: Path,
    manifest: Path,
    *,
    cmake: str,
    jobs: int,
    python: str,
    rscript: str,
    sparkle: str,
    workers: int,
) -> dict:
    paradiseo = source_root / "paradiseo"
    ioh = source_root / "iohexperimenter"
    for required in (paradiseo, ioh):
        if not (required / ".git").is_dir():
            raise FileNotFoundError(
                f"Missing official checkout {required}; run comparisons.shared.bootstrap first."
            )
    paradiseo_build = output_root / "paradiseo-build"
    native_build = output_root / "native-build"
    subprocess.run(
        [
            cmake,
            "-S",
            str(paradiseo.resolve()),
            "-B",
            str(paradiseo_build.resolve()),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(paradiseo_build), "--parallel", str(max(1, jobs))],
        check=True,
    )
    native = build_runner(
        source_root,
        paradiseo_build,
        native_build,
        cmake=cmake,
        jobs=jobs,
    )
    local_manifest = make_manifest(
        manifest,
        Path(native["binary"]),
        python=python,
        rscript=rscript,
        sparkle=sparkle,
        workers=workers,
    )
    return {
        "paradiseo_build": str(paradiseo_build.resolve()),
        "native": native,
        "external_manifest": str(manifest.resolve()),
        "methods": sorted(local_manifest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", type=int, default=2)
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
            setup(
                args.source_root,
                args.output_root,
                args.manifest,
                cmake=args.cmake,
                jobs=args.jobs,
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
