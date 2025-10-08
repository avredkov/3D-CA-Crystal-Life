"""
Configuration utilities for the 3D Cellular Automaton project.

This module provides functions for handling configuration data,
serialization, and metadata processing.
"""

from typing import Any, Dict, Optional
from .data_utils import write_json


def serialize_config_metadata(config_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize configuration metadata for persistence.

    Args:
        config_metadata (Optional[Dict[str, Any]]): Raw metadata passed from caller.

    Returns:
        Dict[str, Any]: Serializable metadata with initialization details expanded.
    """
    if not config_metadata:
        return {}

    params = dict(config_metadata.get("parameters", {}))
    init_from_params = params.pop("init", None)
    params.pop("rule_probabilities", None)
    params.pop("probabilities", None)
    params.pop("recipe_values", None)
    params.pop("ruleset", None)
    params.pop("ruleset_name", None)
    init_raw = config_metadata.get("init") or init_from_params

    init_section = None
    if isinstance(init_raw, dict):
        init_section = dict(init_raw)
    elif hasattr(init_raw, "model_dump"):
        init_section = init_raw.model_dump()

    payload: Dict[str, Any] = {
        "parameters": params,
    }

    ruleset = config_metadata.get("ruleset") or config_metadata.get("ruleset_name")
    if ruleset is not None:
        payload["ruleset"] = ruleset

    recipe_values = config_metadata.get("recipe_values")
    if recipe_values is not None:
        payload["recipe_values"] = recipe_values

    if init_section is not None:
        payload["init"] = init_section

    return payload
