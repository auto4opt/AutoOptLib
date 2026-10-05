"""Fetch and verify the exact official external-platform revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = HERE / "sources.lock.json"
GIT_SOURCES = ("iohexperimenter", "paradiseo", "irace", "sparkle", "smac3")


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        arguments,
        cwd=cwd,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _checkout(source: dict[str, Any], target: Path) -> dict[str, str]:
    repository = str(source["repository"])
    commit = str(source["commit"])
    if not target.exists():
        target.mkdir(parents=True)
        _run(["git", "init"], cwd=target)
        _run(["git", "remote", "add", "origin", repository], cwd=target)
    if not (target / ".git").is_dir():
        raise RuntimeError(f"Refusing to overwrite non-git directory: {target}")
    origin = _run(["git", "remote", "get-url", "origin"], cwd=target)
    if origin.rstrip("/") != repository.rstrip("/"):
        raise RuntimeError(
            f"Origin mismatch for {target}: expected {repository}, found {origin}."
        )
    try:
        _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=target)
    except subprocess.CalledProcessError:
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=target)
    _run(["git", "checkout", "--detach", commit], cwd=target)
    _run(
        [
            "git",
            "-c",
            "url.https://github.com/.insteadOf=git@github.com:",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        cwd=target,
    )
    actual = _run(["git", "rev-parse", "HEAD"], cwd=target)
    if actual != commit:
        raise RuntimeError(f"Checkout verification failed for {target}.")
    return {"repository": repository, "commit": actual, "path": str(target.resolve())}


def bootstrap(root: Path, lock_path: Path, selected: set[str]) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sources = lock["sources"]
    root.mkdir(parents=True, exist_ok=True)
    installed = {}
    for name in GIT_SOURCES:
        if name in selected:
            installed[name] = _checkout(sources[name], root / name)
    manifest = {
        "schema": "autooptlib.external-source-checkout",
        "schema_version": 1,
        "lock": str(lock_path.resolve()),
        "sources": installed,
    }
    output = root / "checkout-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--sources",
        default=",".join(GIT_SOURCES),
        help=(
            "comma-separated subset of iohexperimenter,paradiseo,irace,sparkle,smac3"
        ),
    )
    args = parser.parse_args(argv)
    selected = {token.strip() for token in args.sources.split(",") if token.strip()}
    unknown = selected - set(GIT_SOURCES)
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")
    print(json.dumps(bootstrap(args.root, args.lock, selected), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
