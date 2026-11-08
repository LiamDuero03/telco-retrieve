from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable


def load_callable(spec: str) -> Callable:
    """Resolve 'package.module:function'; the current directory is importable."""
    module_name, _, function_name = spec.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"'{spec}' must look like package.module:function")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"cannot import '{module_name}': {error}") from error
    function = getattr(module, function_name, None)
    if callable(function):
        return function
    raise ValueError(f"'{module_name}' has no callable '{function_name}'")
