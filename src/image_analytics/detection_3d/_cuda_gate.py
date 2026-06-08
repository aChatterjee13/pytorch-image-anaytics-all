"""Shared gate for CUDA-only 3D methods (spconv / mmdet3d / MinkowskiEngine).

These packages have no CPU/macOS wheels, so the wrappers that use them check
for CUDA and the dependency up front and raise an actionable error rather than
failing obscurely. They are verified on a Linux/CUDA box; here they exist as
registered, import-safe stubs (importing the module never fails).
"""

from __future__ import annotations

import importlib


def require_cuda_packages(feature: str, packages: list[str], extra: str = "3d-cuda") -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{feature} requires CUDA — its backend ({', '.join(packages)}) is "
            f"GPU-only and unavailable on this machine (x86_64 macOS, CPU). Run "
            f"on a Linux/CUDA box with: pip install 'image-analytics[{extra}]'."
        )
    missing = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"{feature} needs {missing}, installed via the (Linux/CUDA-only) "
            f"'{extra}' extra: pip install 'image-analytics[{extra}]'."
        )
