"""Inspect model artifact acquisition status for IdentityEngine.

Operation statuses:
    supported  Automatic operation; included by the default ``all`` selector.
    manual     Instructions only; never included by ``all``.
    disabled   Unavailable or unadmitted; explicit selection fails closed.

There are currently no supported automatic or manual model operations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts.model_paths import (
    DOGFLW_LANDMARK_PATH,
    MIEWID_REID_ONNX_PATH,
    MODELS_DIR,
    SUPERANIMAL_QUADRUPED_PATH,
)

_SUPPORTED = "supported"
_MANUAL = "manual"
_DISABLED = "disabled"

_MODELS: dict[str, dict[str, object]] = {
    "dogflw-landmark": {
        "path": DOGFLW_LANDMARK_PATH,
        "status": _DISABLED,
        "desc": "DogFLW facial landmark model",
        "reason": (
            "DogFLW download is disabled: no publisher-authoritative artifact URL, "
            "checksum, and redistribution contract have been verified"
        ),
    },
    "superanimal": {
        "path": SUPERANIMAL_QUADRUPED_PATH,
        "status": _DISABLED,
        "desc": "SuperAnimal-Quadruped HRNet-W32 research weights",
        "reason": (
            "SuperAnimal is disabled: the official weights are non-commercial, "
            "and IdentityEngine has no verified official-architecture ONNX export contract"
        ),
    },
    "miewid": {
        "path": MIEWID_REID_ONNX_PATH,
        "status": _DISABLED,
        "desc": "MiewID-msv3 wildlife ReID export (unadmitted)",
        "reason": (
            "MiewID export is disabled and unadmitted: this downloader cannot "
            "produce the exact cvi.miewid_artifact_bundle.v1 runtime manifest "
            "(including external_data=False) and a genuine passing parity receipt"
        ),
    },
}


def _download_hf(repo: str, filename: str, dest: Path, desc: str = "") -> None:
    """Fail closed because no Hugging Face acquisition is currently admitted."""
    raise RuntimeError("No supported Hugging Face model download is configured")


def _convert_superanimal_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    raise RuntimeError(
        "SuperAnimal ONNX export is disabled. The removed exporter discarded "
        "the checkpoint and exported a newly initialized CNN. Re-enable only "
        "with the pinned official HRNet-W32 architecture, strict state loading, "
        "39-channel heatmap parity, and an approved weight license."
    )


def download_model(name: str) -> None:
    info = _MODELS.get(name)
    if info is None:
        print(f"Unknown: {name}. Available: {list(_MODELS)}")
        return

    status = info["status"]
    if status in {_DISABLED, _MANUAL}:
        raise RuntimeError(str(info["reason"]))
    raise RuntimeError(f"{name} has no supported automatic download handler")


def list_models() -> None:
    print("Operation status:")
    print("  supported  automatic operation; included by the default all selector")
    print("  manual     instructions only; skipped by the default all selector")
    print("  disabled   unavailable or unadmitted; explicit selection fails closed")
    print("Current selectors:")
    for name, info in _MODELS.items():
        path = info["path"]
        assert isinstance(path, Path)
        artifact = "present (not validated)" if path.exists() else "absent"
        print(
            f"  [{info['status']:9s}] {name:20s} {info['desc']} "
            f"-- artifact {artifact}"
        )
    supported = [
        name for name, info in _MODELS.items() if info["status"] == _SUPPORTED
    ]
    manual = [name for name, info in _MODELS.items() if info["status"] == _MANUAL]
    print(f"Supported automatic operations: {', '.join(supported) or 'none'}")
    print(f"Manual operations: {', '.join(manual) or 'none'}")


def _print_cache_summary() -> None:
    total = (
        sum(path.stat().st_size for path in MODELS_DIR.rglob("*") if path.is_file())
        if MODELS_DIR.exists()
        else 0
    )
    print(f"\nCache: {MODELS_DIR}")
    print(f"Total: {total / 2**20:.0f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The default all selector runs only supported automatic operations. "
            "It never attempts manual or disabled selectors."
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(_MODELS) + ["all"],
        default="all",
        help=(
            "operation to run (default: all supported automatic operations; "
            "currently none)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show each selector's supported, manual, or disabled status",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model != "all":
        try:
            download_model(args.model)
        except RuntimeError as exc:
            parser.error(str(exc))
        _print_cache_summary()
        return

    for name, info in _MODELS.items():
        status = info["status"]
        if status != _SUPPORTED:
            print(f"  [SKIP {status}] {name}: {info['reason']}")

    supported = [
        name for name, info in _MODELS.items() if info["status"] == _SUPPORTED
    ]
    if not supported:
        print("No supported automatic model operations are configured.")
    for name in supported:
        try:
            download_model(name)
        except RuntimeError as exc:
            parser.error(str(exc))
    _print_cache_summary()


if __name__ == "__main__":
    main()
