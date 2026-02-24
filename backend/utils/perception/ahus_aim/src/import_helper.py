import importlib
from typing import Any


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])
