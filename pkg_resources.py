"""Minimal pkg_resources compatibility shim for environments without setuptools.

This project only relies on ``pkg_resources.resource_filename`` via librosa.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Union


def resource_filename(package_or_requirement: Union[str, ModuleType], resource_name: str) -> str:
    """Return an absolute filesystem path for a resource within a package."""
    if isinstance(package_or_requirement, str):
        module = importlib.import_module(package_or_requirement)
    else:
        module = package_or_requirement

    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise FileNotFoundError(f"Cannot resolve resource path for module: {module}")

    return str((Path(module_file).resolve().parent / resource_name).resolve())


__all__ = ["resource_filename"]
