from __future__ import annotations

from typing import List, Tuple, Dict, Any
from pathlib import Path
import importlib
import numpy as np


# Order matches the kernel/README encoding:
# index = 729*self + 243*right + 81*down + 27*left + 9*up + 3*back + front
BASE_WEIGHTS: List[int] = [729, 243, 81, 27, 9, 3, 1]


def index_to_vector(index: int) -> List[int]:
    """
    Convert rule index (0..2186) into a 7-element base-3 vector
    [self, right, down, left, up, back, front].
    """
    if index < 0 or index >= 3 ** 7:
        raise ValueError("index must be in [0, 2186]")
    remaining = int(index)
    vec: List[int] = []
    for w in BASE_WEIGHTS:
        digit = remaining // w
        if digit not in (0, 1, 2):
            # Should not happen for valid indices, but guard anyway
            raise ValueError("invalid digit computed from index")
        vec.append(int(digit))
        remaining = remaining % w
    return vec


def vector_to_index(vec: List[int]) -> int:
    """
    Convert a 7-element base-3 vector [self, right, down, left, up, back, front]
    into the encoded rule index (0..2186).
    """
    if len(vec) != 7:
        raise ValueError("vector must have length 7")
    total = 0
    for v, w in zip(vec, BASE_WEIGHTS):
        if v not in (0, 1, 2):
            raise ValueError("vector digits must be in {0,1,2}")
        total += int(v) * w
    return total


__all__ = ["index_to_vector", "vector_to_index", "BASE_WEIGHTS"]


PROTECTED_MODULES = {"Default", "Default_with_3D_nucleation"}


def _get_recipe_parameters(mod: Any) -> List[Dict[str, Any]]:
    params = []
    if hasattr(mod, "get_parameters"):
        try:
            params = list(getattr(mod, "get_parameters")() or [])
        except Exception:
            params = []
    if not params and hasattr(mod, "RECIPE_PARAMETERS"):
        try:
            params = list(getattr(mod, "RECIPE_PARAMETERS") or [])
        except Exception:
            params = []
    return params


def get_recipe_parameters(module_name: str) -> List[Dict[str, Any]]:
    mod = importlib.import_module(f"rules.{module_name}")
    return _get_recipe_parameters(mod)


def load_recipe(module_name: str, recipe_values: Dict[str, float] | None = None) -> Tuple[List[str], List[float]]:
    """
    Import rules.<module_name>, run assign_rules to obtain Ru numeric array.
    Returns (expr, values) where expr holds symbolic keys when used, otherwise empty string.
    """
    mod = importlib.import_module(f"rules.{module_name}")
    if not hasattr(mod, "assign_rules"):
        raise AttributeError(f"rules.{module_name} has no assign_rules")
    # If module provides explicit RULE_EXPR and RULE_VAL arrays, prefer them (GUI fidelity)
    if hasattr(mod, "RULE_EXPR") and hasattr(mod, "RULE_VAL"):
        try:
            expr = list(getattr(mod, "RULE_EXPR"))
            vals = [float(x) for x in list(getattr(mod, "RULE_VAL"))]
            if len(expr) == 3 ** 7 and len(vals) == 3 ** 7:
                return expr, vals
        except Exception:
            pass
    params = _get_recipe_parameters(mod)
    # Build params dict with defaults overridden by recipe_values
    merged: Dict[str, float] = {}
    for p in params:
        key = str(p.get("key"))
        dv = float(p.get("default", 0.0))
        v = dv
        if recipe_values and key in recipe_values:
            v = float(recipe_values[key])
        # clamp to [0,1] for probabilities
        v = max(0.0, min(1.0, v))
        merged[key] = v
    Ru = np.zeros(3 ** 7, dtype=np.float32)
    # Call recipe with mapping
    mod.assign_rules(Ru, merged)
    vals = [float(x) for x in Ru.tolist()]
    # If RULE_EXPR present, it was returned earlier; otherwise, try to infer symbols using sentinels
    expr = [""] * len(vals)
    try:
        if params:
            # Build distinct sentinel values for each parameter in (0,1)
            sent: Dict[str, float] = {}
            base = 0.5001
            delta = 0.0001
            for i, p in enumerate(params):
                key = str(p.get("key"))
                val = base + i * delta
                if val >= 0.9999:
                    val = 0.9999
                sent[key] = val
            Ru2 = np.zeros(3 ** 7, dtype=np.float32)
            mod.assign_rules(Ru2, sent)
            eps = 1e-9
            for i in range(len(vals)):
                if abs(vals[i] - 0.0) < eps:
                    expr[i] = "0"
                elif abs(vals[i] - 1.0) < eps:
                    expr[i] = "1"
                else:
                    v2 = float(Ru2[i])
                    matched = False
                    for k, s in sent.items():
                        if abs(v2 - s) < 1e-7:
                            expr[i] = k
                            matched = True
                            break
                    if not matched:
                        expr[i] = ""
    except Exception:
        # Fallback: keep expr numeric-only
        pass
    return expr, vals


def inverted_self_vector(vec: List[int]) -> List[int]:
    """
    Return a copy of the 7-digit neighborhood vector with the Self atom inverted
    between mobile (1) and crystalline (2). If Self is 0, it is kept as 0.

    Args:
        vec (List[int]): Base-3 vector [self, right, down, left, up, back, front].

    Returns:
        List[int]: New vector with vec[0] inverted when in {1,2}.
    """
    if len(vec) != 7:
        raise ValueError("vector must have length 7")
    self_state = int(vec[0])
    if self_state == 1:
        new_self = 2
    elif self_state == 2:
        new_self = 1
    else:
        new_self = 0
    out = list(vec)
    out[0] = new_self
    return out


def invert_self_in_index(index: int) -> int:
    """
    Invert the Self atom state for a rule index and return the new index.
    If Self is 0, the same index is returned.

    Args:
        index (int): Original rule index (0..2186).

    Returns:
        int: Index corresponding to vector with inverted Self.
    """
    vec = index_to_vector(index)
    inv = inverted_self_vector(vec)
    return vector_to_index(inv)


def save_recipe_py(path: str, expr: List[str], values: List[float], parameters: List[Dict[str, Any]] | None = None) -> None:
    """
    Write a full, self-contained recipe module where each rule index is assigned individually.
    Emits:
      - RECIPE_PARAMETERS: parameter metadata list (probabilities 0..1; default only)
      - assign_rules(Ru, params): sets Ru[:] = 0.0 then Ru[index] per line using either 0, 1, params[key], or numeric literal
    """
    if len(values) != 3 ** 7 or len(expr) != 3 ** 7:
        raise ValueError("expr/values must have length 2187")
    parameters = list(parameters or [])
    
    clean_params: List[Dict[str, Any]] = []
    for p in parameters:
        try:
            clean_params.append({
                "key": str(p.get("key", "")),
                "label": str(p.get("label", "")),
                "description": str(p.get("description", "")),
                "default": float(p.get("default", 0.0)),
            })
        except Exception:
            continue
    # Build defaults dict literal for convenience in manual editing
    defaults_map = {p["key"]: float(p.get("default", 0.0)) for p in clean_params if p.get("key")}
    # Helper to compute vector label for comments
    def _vec_for_index(i: int) -> List[int]:
        return index_to_vector(i)
    lines: List[str] = []
    lines.append("from typing import List\n")
    lines.append("import numpy as np\n\n")
    lines.append("# Auto-generated by GUI: explicit per-index rule assignments\n")
    lines.append("RECIPE_PARAMETERS = [\n")
    for p in clean_params:
        lines.append("    {\n")
        for k in ("key","label","description"):
            lines.append(f"        \"{k}\": {repr(p.get(k, ''))},\n")
        lines.append(f"        \"default\": {float(p.get('default', 0.0)):.10g},\n")
        lines.append("    },\n")
    lines.append("]\n\n")
    # For GUI fidelity and easy diffing, also emit symbolic/number arrays
    def _emit_list_str(strs: List[str]) -> str:
        parts = []
        for s in strs:
            parts.append(repr(str(s or "")))
        return ", ".join(parts)
    def _emit_list_float(nums: List[float]) -> str:
        parts = []
        for x in nums:
            parts.append(f"{float(x):.10g}")
        return ", ".join(parts)
    lines.append("RULE_EXPR = [\n")
    for i in range(0, len(expr), 81):
        chunk = expr[i:i+81]
        lines.append("    " + _emit_list_str(chunk) + ",\n")
    lines.append("]\n\n")
    lines.append("RULE_VAL = [\n")
    for i in range(0, len(values), 81):
        chunk = values[i:i+81]
        lines.append("    " + _emit_list_float(chunk) + ",\n")
    lines.append("]\n\n")
    lines.append("def assign_rules(Ru: np.ndarray, params: dict) -> None:\n")
    lines.append("    \"\"\"Fill the 2187-length Ru array with probabilities per neighborhood rule.\n")
    lines.append("    params: mapping param_key -> float in [0,1]. Unknown keys fall back to default values declared above.\n")
    lines.append("    \"\"\"\n")
    lines.append(f"    _defaults = { {k: float(v) for k, v in defaults_map.items()} }\n")
    lines.append("    Ru[:] = 0.0\n")
    # Emit one line per index with either param reference or numeric literal
    for i, sym in enumerate(expr):
        vec = _vec_for_index(i)
        comment = f"# vec={vec}"
        sym = (sym or "").strip()
        if sym == '0':
            lines.append(f"    Ru[{i}] = 0.0  {comment}\n")
        elif sym == '1':
            lines.append(f"    Ru[{i}] = 1.0  {comment}\n")
        elif sym:
            keylit = repr(sym)
            # clamp at runtime to [0,1]
            lines.append(f"    Ru[{i}] = float(max(0.0, min(1.0, params.get({keylit}, _defaults.get({keylit}, 0.0)))))  {comment}\n")
        else:
            v = float(values[i])
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            lines.append(f"    Ru[{i}] = {v:.10g}  {comment}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def validate_recipe(module_name: str, recipe_values: Dict[str, float] | None = None) -> None:
    """
    Import rules.<module_name>, run assign_rules and ensure outputs are sane.
    Raises on any error.
    """
    mod = importlib.import_module(f"rules.{module_name}")
    if not hasattr(mod, "assign_rules"):
        raise AttributeError(f"rules.{module_name} has no assign_rules")
    Ru = np.zeros(3 ** 7, dtype=np.float32)
    params = _get_recipe_parameters(mod)
    merged: Dict[str, float] = {}
    for p in params:
        key = str(p.get("key"))
        dv = float(p.get("default", 0.0))
        v = dv
        if recipe_values and key in recipe_values:
            v = float(recipe_values[key])
        v = max(0.0, min(1.0, v))
        merged[key] = v
    mod.assign_rules(Ru, merged)
    if Ru.shape[0] != 3 ** 7:
        raise ValueError("Ru length invalid")
    if np.any(Ru < 0.0) or np.any(Ru > 1.0):
        raise ValueError("Ru contains values outside [0,1]")


def is_protected_target(target: str) -> bool:
    """
    True if attempting to overwrite a protected built-in recipe.
    Accepts a module name (without .py) or a filesystem path under rules/.
    """
    p = Path(target)
    name = p.stem if p.suffix == ".py" else str(target)
    return name in PROTECTED_MODULES


def save_recipe(path_or_module: str, expr: List[str], values: List[float], overwrite: bool = False, parameters: List[Dict[str, Any]] | None = None) -> str:
    """
    Save a recipe to rules/<name>.py, guarding built-ins. Returns saved path.
    If a module name is provided, resolves to rules/<module>.py.
    """
    # Resolve to path under rules/
    p = Path(path_or_module)
    if p.suffix != ".py":
        p = Path("rules") / f"{p.name}.py"
    # Guard protected modules
    if is_protected_target(p) and not overwrite:
        raise PermissionError("Cannot overwrite protected built-in recipe. Save under a different name.")
    # Ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)
    save_recipe_py(str(p), expr, values, parameters)
    return str(p)


def toggle_nucleation(values: List[float], enable: bool) -> None:
    """
    Toggle the special 3D nucleation rule to 1.0 or 0.0 in-place.
    Index = 729*1 + 243*1 + 81*1 + 27*1 + 9*1 + 3*1 + 1
    """
    if len(values) != 3 ** 7:
        raise ValueError("values must have length 2187")
    idx = 729 * 1 + 243 * 1 + 81 * 1 + 27 * 1 + 9 * 1 + 3 * 1 + 1
    values[idx] = 1.0 if enable else 0.0


def list_rulesets() -> List[str]:
    """
    Scan the rules/ folder for available ruleset modules (with assign_rules).
    Returns module basenames without .py.
    """
    # Resolve rules directory relative to project root (not CWD)
    try:
        rules_dir = Path(__file__).resolve().parents[2] / "rules"
    except Exception:
        rules_dir = Path("rules")
    names: List[str] = []
    if not rules_dir.exists():
        return names
    for p in rules_dir.glob("*.py"):
        name = p.stem
        try:
            mod = importlib.import_module(f"rules.{name}")
            if hasattr(mod, "assign_rules"):
                names.append(name)
        except Exception:
            continue
    # sort, keep Default first if present
    names = sorted(names)
    if "Default" in names:
        names.remove("Default"); names.insert(0, "Default")
    return names



