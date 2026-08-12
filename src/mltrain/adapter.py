from __future__ import annotations

import importlib
from typing import cast

from mltrain.config import adapter_spec, repository_root
from mltrain.contracts import ProjectAdapter


def load_adapter() -> ProjectAdapter:
    module_name, attribute = adapter_spec(repository_root()).split(":", 1)
    try:
        module = importlib.import_module(module_name)
        adapter = getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"cannot load project adapter {module_name}:{attribute}: {error}"
        ) from error
    required = ("config_model", "train", "evaluate", "validate")
    if not all(hasattr(adapter, name) for name in required):
        raise RuntimeError(
            "project adapter must expose config_model and train/evaluate/validate methods"
        )
    return cast(ProjectAdapter, adapter)
