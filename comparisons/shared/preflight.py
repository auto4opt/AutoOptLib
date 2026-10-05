"""Preflight the machine-local external-method toolchain before formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from problems import IOHInstance, known_ioh_optimum_raw


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, timeout=120
    )
    return {
        "command": command,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-2000:],
    }


def _option(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Manifest command has no {name} value: {command}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sparkle_python(sparkle: str) -> list[str]:
    executable = Path(shutil.which(sparkle) or sparkle)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    first = executable.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if not first.startswith("#!"):
        raise ValueError(f"Cannot identify Sparkle's Python interpreter: {executable}")
    return shlex.split(first[2:].strip())


def preflight(manifest_path: Path, *, native_only: bool) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(
        (Path(__file__).resolve().parents[1] / "sources.lock.json").read_text(
            encoding="utf-8"
        )
    )["sources"]
    paradiseo = manifest["paradiseo_irace"]
    sparkle_entry = manifest["sparkle_smac3"]
    paradiseo_runner = Path(paradiseo["command"][0])
    sparkle_runner = Path(sparkle_entry["command"][0])
    if paradiseo_runner.resolve() != sparkle_runner.resolve():
        raise ValueError("Both platforms must use the same native eoFastGA runner.")
    if not paradiseo_runner.is_file():
        raise FileNotFoundError(paradiseo_runner)
    actual_binary_hash = _sha256(paradiseo_runner)
    for method, expected_pins in (
        (
            "paradiseo_irace",
            {
                "paradiseo_commit": lock["paradiseo"]["commit"],
                "iohexperimenter_commit": lock["iohexperimenter"]["commit"],
                "irace_commit": lock["irace"]["commit"],
            },
        ),
        (
            "sparkle_smac3",
            {
                "paradiseo_commit": lock["paradiseo"]["commit"],
                "iohexperimenter_commit": lock["iohexperimenter"]["commit"],
                "sparkle_commit": lock["sparkle"]["commit"],
                "smac3_commit": lock["smac3"]["commit"],
            },
        ),
    ):
        provenance = manifest[method].get("provenance", {})
        for name, value in expected_pins.items():
            if provenance.get(name) != value:
                raise ValueError(
                    f"{method} provenance {name!r} does not match sources.lock.json."
                )
        if provenance.get("binary_sha256") != actual_binary_hash:
            raise ValueError(
                f"{method} native binary hash does not match its manifest."
            )
        build_manifest = Path(str(provenance.get("native_build_manifest", "")))
        if not build_manifest.is_file():
            raise FileNotFoundError(
                f"{method} has no verifiable native build manifest: {build_manifest}"
            )
        build = json.loads(build_manifest.read_text(encoding="utf-8"))
        if build.get("binary_sha256") != actual_binary_hash:
            raise ValueError(
                "Native build manifest hash does not match the executable."
            )
        for name in ("paradiseo_commit", "iohexperimenter_commit"):
            if build.get(name) != expected_pins[name]:
                raise ValueError(
                    f"Native build manifest {name!r} does not match the pinned source."
                )
    checks: dict[str, Any] = {
        "parameter_space": _run([str(paradiseo_runner), "--describe-space"])
    }
    default = Path(__file__).resolve().parent / "native" / "default-config.json"
    native_runs = []
    for suite, dimension in (("bbob", 2), ("pbo", 16)):
        command = [
            str(paradiseo_runner),
            "--config",
            str(default),
            "--suite",
            suite,
            "--function",
            "1",
            "--dimension",
            str(dimension),
            "--instance",
            "1",
            "--seed",
            "7",
            "--budget",
            "40",
            "--json",
        ]
        result = _run(command)
        payload = json.loads(result["stdout"])
        if int(payload["evaluations"]) != 40:
            raise ValueError(f"Native {suite} preflight did not use exactly 40 FE.")
        native_runs.append(payload)
    checks["native_runs"] = native_runs
    f22_runs = []
    for instance in (1, 2, 33):
        dimension = 100
        command = [
            str(paradiseo_runner),
            "--config",
            str(default),
            "--suite",
            "pbo",
            "--function",
            "22",
            "--dimension",
            str(dimension),
            "--instance",
            str(instance),
            "--seed",
            "7",
            "--budget",
            "40",
            "--json",
        ]
        payload = json.loads(_run(command)["stdout"])
        expected = known_ioh_optimum_raw(
            "pbo", 22, IOHInstance(dimension=dimension, instance=instance)
        )
        if abs(float(payload["optimum_raw"]) - expected) > 1e-8 * max(
            1.0, abs(expected)
        ):
            raise ValueError(
                f"Native PBO F22 optimum differs from Python for instance {instance}."
            )
        if int(payload["evaluations"]) != 40:
            raise ValueError("Native PBO F22 preflight did not use exactly 40 FE.")
        f22_runs.append({"instance": instance, "optimum_raw": payload["optimum_raw"]})
    checks["pbo_f22_optimum_parity"] = f22_runs
    if not native_only:
        irace_command = paradiseo["design_command"]
        rscript = _option(irace_command, "--rscript")
        checks["irace"] = _run(
            [rscript, "-e", 'cat(as.character(packageVersion("irace")))']
        )
        if checks["irace"]["stdout"] != str(lock["irace"]["version"]):
            raise ValueError(
                "Installed irace version does not match the pinned source."
            )
        sparkle_command = sparkle_entry["design_command"]
        sparkle = _option(sparkle_command, "--sparkle")
        # Sparkle 0.9.6 dispatches help through its subcommands and returns 255
        # for a top-level ``--help``. This command imports the real CLI and
        # exits without creating a workspace.
        checks["sparkle"] = _run([sparkle, "about"])
        if f"Version: {lock['sparkle']['version']}" not in checks["sparkle"]["stdout"]:
            raise ValueError(
                "Installed Sparkle version does not match the pinned source."
            )
        checks["smac3"] = _run(
            _sparkle_python(sparkle)
            + [
                "-c",
                ("from importlib.metadata import version; print(version('smac'))"),
            ]
        )
        if checks["smac3"]["stdout"] != str(lock["smac3"]["version"]):
            raise ValueError("Sparkle environment has the wrong SMAC3 version.")
    # Confirm that an artifact can be parsed from a regular temporary path.
    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "artifact.json"
        artifact.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(payload.get("configuration"), dict):
            raise ValueError("Native default artifact is invalid.")
    return {
        "schema": "autooptlib.external-preflight",
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "native_only": native_only,
        "ok": True,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.manifest, native_only=args.native_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
