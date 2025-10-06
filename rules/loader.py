import importlib
from types import ModuleType
from typing import Callable


def load_assign_rules(ruleset_name: str) -> Callable[[object, dict], None]:
    """
    Load the assign_rules function from a ruleset module.

    Args:
        ruleset_name (str): The name of the ruleset module under the `rules` package.

    Returns:
        Callable[[np.ndarray, dict], None]: The function that assigns rules into Ru.
    """
    module_name = f"rules.{ruleset_name}"
    module: ModuleType = importlib.import_module(module_name)
    if not hasattr(module, "assign_rules"):
        raise AttributeError(f"Ruleset '{ruleset_name}' does not define 'assign_rules' function")
    return getattr(module, "assign_rules")


__all__ = ["load_assign_rules"]


