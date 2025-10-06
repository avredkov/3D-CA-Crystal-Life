from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import SimulationConfig, load_config


PROTECTED_CONFIG_NAMES = {"config.example.json"}


class ConfigService:
    def __init__(self) -> None:
        self.current_path: Optional[Path] = None
        self.current: Optional[SimulationConfig] = None

    def load(self, path: str) -> SimulationConfig:
        p = Path(path)
        cfg = load_config(str(p))
        self.current_path = p
        self.current = cfg
        return cfg

    def is_protected(self, path: Optional[Path] = None) -> bool:
        p = path or self.current_path
        if p is None:
            return False
        return p.name in PROTECTED_CONFIG_NAMES

    def save(self, path: Optional[str] = None) -> None:
        if self.current is None:
            raise ValueError("No config loaded")
        target = Path(path) if path else (self.current_path or Path("config.json"))
        if target.name in PROTECTED_CONFIG_NAMES and target.exists():
            raise PermissionError(f"Protected config cannot be overwritten: {target.name}")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.current.model_dump(), f, ensure_ascii=False, indent=2)
        self.current_path = target

    def save_as(self, path: str) -> None:
        self.save(path)

    def validate(self, cfg: Optional[SimulationConfig] = None) -> SimulationConfig:
        data = (cfg or self.current)
        if data is None:
            raise ValueError("No config to validate")
        # round-trip through pydantic for validation
        return SimulationConfig(**data.model_dump())

    def reset(self) -> SimulationConfig:
        self.current = SimulationConfig(output_dir="./output/")
        self.current_path = None
        return self.current



