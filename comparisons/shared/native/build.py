"""Build the common official-ParadisEO eoFastGA native runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    source_root: Path,
    paradiseo_build: Path,
    output: Path,
    *,
    cmake: str,
    jobs: int,
) -> dict[str, str]:
    here = Path(__file__).resolve().parent
    paradiseo = source_root / "paradiseo"
    ioh = source_root / "iohexperimenter"
    for required in (paradiseo, paradiseo_build, ioh):
        if not required.exists():
            raise FileNotFoundError(required)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            cmake,
            "-S",
            str(here),
            "-B",
            str(output),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPARADISEO_ROOT={paradiseo.resolve()}",
            f"-DPARADISEO_BUILD={paradiseo_build.resolve()}",
            f"-DIOH_ROOT={ioh.resolve()}",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(output), "--parallel", str(max(1, jobs))],
        check=True,
    )
    binary = output / "autooptlib-fastga-runner"
    if not binary.is_file():
        raise RuntimeError(f"build succeeded without creating {binary}")
    manifest = {
        "schema": "autooptlib.native-runner-build",
        "schema_version": 1,
        "binary": str(binary.resolve()),
        "binary_sha256": _sha256(binary),
        "paradiseo_commit": _git_commit(paradiseo),
        "iohexperimenter_commit": _git_commit(ioh),
    }
    (output / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--paradiseo-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                args.source_root,
                args.paradiseo_build,
                args.output,
                cmake=args.cmake,
                jobs=args.jobs,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
