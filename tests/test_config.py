import pytest
from pathlib import Path
from ollama_think.config import Config

@pytest.fixture
def default_config():
    """Returns a Config instance with default settings."""
    return Config()

def test_default_config_loading(default_config):
    """Tests that the default config is loaded correctly."""
    assert default_config.enable_hacks is True
    assert "cogito" in default_config.models
    assert "granite3.2" in default_config.models
    assert default_config.models["cogito"]["add_message"] == {
        "role": "system",
        "content": "Enable deep thinking subroutine.",
    }

def test_missing_config_file():
    """Tests that the system handles a missing config file gracefully."""
    config = Config()
    config.load_config("non_existent_file.yaml")
    assert config.enable_hacks is False
    assert config.models == {}

def test_custom_config_loading(tmp_path: Path):
    """Tests loading a custom configuration file."""
    custom_config_content = """
hacks:
  enabled: true
defaults:
  enable_thinking: true
  add_message: null
  content_parsers: ["default_parser"]
models:
  - name: my-model
    enable_thinking: false
    content_parsers: ["my_parser"]
"""
    config_path = tmp_path / "custom_config.yaml"
    config_path.write_text(custom_config_content)

    config = Config()
    config.load_config(config_path)

    assert config.enable_hacks is True
    assert "my-model" in config.models
    assert config.models["my-model"]["enable_thinking"] is False
    assert config.models["my-model"]["content_parsers"] == ["my_parser"]
    # Check that default values are inherited correctly
    assert config.models["my-model"]["add_message"] is None

def test_get_hacks_if_enabled(default_config):
    """Tests the get_hacks_if_enabled method."""
    # Test with a model that has specific hacks
    granite_hacks = default_config.get_hacks_if_enabled("granite3.2-instruct")
    assert granite_hacks is not None
    assert granite_hacks["add_message"]["role"] == "control"

    # Test with a model that uses default hacks
    phi_hacks = default_config.get_hacks_if_enabled("phi4-reasoning")
    assert phi_hacks is not None
    assert phi_hacks["enable_thinking"] is False # from default

    # Test with a model not in the config
    unknown_hacks = default_config.get_hacks_if_enabled("unknown-model")
    assert unknown_hacks is None

    # Test with hacks disabled
    default_config.enable_hacks = False
    disabled_hacks = default_config.get_hacks_if_enabled("granite3.2")
    assert disabled_hacks is None

def test_model_prefix_matching(default_config):
    """Tests that model names are matched by prefix."""
    hacks = default_config.get_hacks_if_enabled("cogito-pro-v2")
    assert hacks is not None
    assert hacks["add_message"]["content"] == "Enable deep thinking subroutine."