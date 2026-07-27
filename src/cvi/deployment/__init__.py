"""Deployment status facade.

No end-to-end CPU or CUDA deployment constructor is currently validated. The
former constructor names remain available lazily only to provide an explicit,
fail-closed compatibility error.
"""

from importlib import import_module
from typing import Any

_DISABLED_COMPAT_EXPORTS = {
    "CVIDeploymentCPU": ("cvi.deployment.cpu", "CVIDeploymentCPU"),
    "CVIDeploymentCUDA": ("cvi.deployment.cuda", "CVIDeploymentCUDA"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _DISABLED_COMPAT_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__: list[str] = []
