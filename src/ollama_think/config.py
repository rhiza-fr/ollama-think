"""Loading of per-model thinking hacks from config.yaml."""

from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import yaml


class ThinkingHacks(TypedDict):
    """Hacks applied to a model that does not natively separate thinking."""

    enable_thinking: bool
    add_message: dict[str, str] | None
    content_parsers: list[str]


class Config:
    """Thinking hacks for each model name prefix, loaded from a YAML file."""

    def __init__(self):
        """Load the default config.yaml bundled with the package."""
        default_path = Path(__file__).parent / "config.yaml"
        self.models: dict[str, ThinkingHacks] = {}
        self.enable_hacks = False
        self.load_config(default_path)

    def load_config(self, path: str | Path):
        """Load model hacks from the YAML file at `path`, merging each model over `defaults`.

        If the file does not exist, hacks are disabled and all models are cleared.
        """
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
        """Return the hacks for the first configured prefix of `model`, or None.

        Always returns None when hacks are disabled in the config.
        """
        if not self.enable_hacks:
            return None
        for key in self.models:
            if model.startswith(key):
                return self.models[key]
        return None
