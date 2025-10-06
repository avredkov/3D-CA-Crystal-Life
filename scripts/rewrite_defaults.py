from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import numpy as np

# Use the project service to persist recipe files in the new format
from gui.services.rules_service import save_recipe, get_recipe_parameters


def build_symbolic_recipe(module_name: str) -> Tuple[List[str], List[float], List[dict]]:
    mod = importlib.import_module(f"rules.{module_name}")
    if not hasattr(mod, "assign_rules"):
        raise AttributeError(f"rules.{module_name} has no assign_rules")
    params = get_recipe_parameters(module_name) or []

    # Create distinct sentinel values for each parameter key inside (0,1)
    key_list: List[str] = [str(p.get("key")) for p in params if p.get("key")]
    sent: Dict[str, float] = {}
    base = 0.6001
    delta = 0.0001
    for i, key in enumerate(key_list):
        v = base + i * delta
        if v >= 0.9999:
            v = 0.9998
        sent[key] = v

    # Evaluate recipe with sentinel parameters
    Ru = np.zeros(3 ** 7, dtype=np.float32)
    mod.assign_rules(Ru, sent)
    vals_sentinel = Ru.astype(float).tolist()

    # Classify into RULE_EXPR and RULE_VAL
    expr: List[str] = [""] * (3 ** 7)
    values: List[float] = [0.0] * (3 ** 7)
    eps = 1e-9
    for i, v in enumerate(vals_sentinel):
        if abs(v - 0.0) < eps:
            expr[i] = "0"
            values[i] = 0.0
        elif abs(v - 1.0) < eps:
            expr[i] = "1"
            values[i] = 0.0
        else:
            matched = False
            for k, s in sent.items():
                if abs(v - s) < 1e-7:
                    expr[i] = k
                    values[i] = 0.0
                    matched = True
                    break
            if not matched:
                expr[i] = ""
                values[i] = float(v)

    return expr, values, params


def rewrite(module_name: str) -> None:
    expr, values, params = build_symbolic_recipe(module_name)
    # Overwrite the recipe module in-place using our saver
    save_recipe(module_name, expr, values, overwrite=True, parameters=params)


def main() -> None:
    for name in ("Default", "Default_with_3D_nucleation"):
        rewrite(name)
        print(f"Rewrote rules/{name}.py")


if __name__ == "__main__":
    main()


