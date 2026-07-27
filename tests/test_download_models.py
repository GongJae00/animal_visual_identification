from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "download_models.py"


def _run_downloader(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CVI_MODELS_DIR"] = str(tmp_path / "models")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_distinguishes_operation_statuses(tmp_path: Path) -> None:
    completed = _run_downloader(tmp_path, "--help")

    assert completed.returncode == 0
    assert "supported  Automatic operation" in completed.stdout
    assert "manual     Instructions only" in completed.stdout
    assert "disabled   Unavailable or unadmitted" in completed.stdout
    assert "currently none" in completed.stdout
    assert not (tmp_path / "models").exists()


def test_list_reports_all_selectors_disabled_without_admitting_cached_files(
    tmp_path: Path,
) -> None:
    completed = _run_downloader(tmp_path, "--list")

    assert completed.returncode == 0
    for name in ("dogflw-landmark", "miewid", "superanimal"):
        assert f"[disabled ] {name}" in completed.stdout
    assert "Supported automatic operations: none" in completed.stdout
    assert "Manual operations: none" in completed.stdout
    assert not (tmp_path / "models").exists()


def test_miewid_fails_before_network_or_model_imports(tmp_path: Path) -> None:
    guarded = """
import builtins
import runpy
import sys

blocked = {
    "huggingface_hub",
    "onnx",
    "onnxruntime",
    "requests",
    "safetensors",
    "timm",
    "torch",
    "transformers",
}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"blocked import reached: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = ["tools/download_models.py", "--model", "miewid"]
runpy.run_path("tools/download_models.py", run_name="__main__")
"""
    environment = os.environ.copy()
    environment["CVI_MODELS_DIR"] = str(tmp_path / "models")
    completed = subprocess.run(
        [sys.executable, "-c", guarded],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "MiewID export is disabled and unadmitted" in completed.stderr
    assert "cvi.miewid_artifact_bundle.v1" in completed.stderr
    assert "genuine passing parity receipt" in completed.stderr
    assert "blocked import reached" not in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "models").exists()


def test_default_all_skips_every_disabled_selector(tmp_path: Path) -> None:
    completed = _run_downloader(tmp_path)

    assert completed.returncode == 0
    for name in ("dogflw-landmark", "miewid", "superanimal"):
        assert f"[SKIP disabled] {name}" in completed.stdout
    assert (
        "No supported automatic model operations are configured."
        in completed.stdout
    )
    assert "[DOWN]" not in completed.stdout
    assert not (tmp_path / "models").exists()
