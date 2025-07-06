from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import yaml


class ThinkingHacks(TypedDict):
    enable_thinking: bool
    add_message: dict[str, str] | None
    content_parsers: list[str]


class Config:
    def __init__(self):
        default_path = Path(__file__).parent / "config.yaml"
        self.models: dict[str, ThinkingHacks] = {}
        self.enable_hacks = False
        self.load_config(default_path)

    def load_config(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            print("WARNING: config not found at {path}, no hacks for older models enabled.")
            self.enable_hacks = False
            self.models = {}
            return

        # Parse YAML
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        # Get defaults and hacks
        defaults = config.get("defaults", {})
        hacks = config.get("hacks", {})

        # Process each model configuration
        for model in config.get("models", []):
            name = model.get("name")
            if not name:
                continue  # Skip entries without a name

            # Merge defaults with model-specific config
            model_config = deepcopy(defaults)
            model_specific = {k: v for k, v in model.items() if k != "name"}
            model_config.update(model_specific)

            self.models[name] = model_config

        self.enable_hacks = hacks.get("enabled", False)

    def get_hacks_if_enabled(self, model: str) -> ThinkingHacks | None:
        if not self.enable_hacks:
            return None
        for key in self.models.keys():
            if model.startswith(key):
                return self.models[key]
        return None
