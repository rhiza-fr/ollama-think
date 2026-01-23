# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ollama-think is a Python library that wraps the ollama-python client to add:
- **Response caching** using diskcache
- **Thinking mode compatibility** for models that don't officially support it via "hacks"
- **Syntax sugar** for cleaner API usage (prompt parameter, response unpacking)
- **Type-safe separation** of streaming vs non-streaming interfaces

## Development Commands

### Setup
```bash
# Install with development dependencies
uv sync --extra dev
```

### Testing
```bash
# Run fast unit tests only (default)
uv run pytest

# Run all tests including slow integration tests (requires running Ollama instance)
uv run pytest -m "slow or not slow"

# Run tests against a specific host
uv run pytest -m "slow or not slow" --host http://localhost:11434

# Run tests for a specific model
uv run pytest -m "slow or not slow" --model "qwen3"
```

### Linting and Formatting
```bash
# Format code
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check . --fix
```

### Testing New Models
When adding support for a new model with thinking capabilities:
```bash
# 1. Edit src/ollama_think/config.yaml to add model configuration

# 2. Test the hacks for both streaming and non-streaming
uv run ./tests/test_hacks.py --host http://localhost:11434 --model "model_name"

# 3. Verify the model capabilities
uv run pytest ./tests/test_model_capabilities.py --host http://localhost:11434 -m "slow" --model "model_name"

# 4. Regenerate the model capabilities documentation
uv run tests/generate_model_capabilities_report.py
```

## Architecture

### Core Components

**Client Classes** (`src/ollama_think/client.py`):
- `Client` and `AsyncClient` extend `ollama.Client` and `ollama.AsyncClient`
- Add caching layer using diskcache with MD5-hashed request keys
- Provide `call()` and `stream()` methods that return `ThinkResponse` objects
- Apply model-specific "hacks" when needed based on config

**ThinkResponse** (`src/ollama_think/thinkresponse.py`):
- Extends `ollama.ChatResponse` with convenience properties
- Supports unpacking: `thinking, content = response`
- String representation returns just the content

**Thinking Hacks System** (`src/ollama_think/thinking_hacks.py` + `config.py`):
- Modifies requests/responses for models that don't officially support thinking
- `hack_request()`: Adds system/control messages, disables think parameter if needed
- `hack_response()`: Parses content using regex to extract thinking from content
- `setup_stream_parser()` + `hack_stream_chunk()`: Handles streaming responses with stateful parsing
- Configuration in `src/ollama_think/config.yaml` maps model name prefixes to hack rules

**Stream Parser** (`src/ollama_think/stream_parser.py`):
- Stateful parser for streaming responses that need thinking extraction
- Handles incomplete chunks and delimiter detection across chunk boundaries

### How Thinking Hacks Work

Models are matched by name prefix in config.yaml. Each model can specify:
- `enable_thinking`: Whether to send `think=True` to Ollama (false for models that error)
- `add_message`: A message to inject at the start (e.g., system message "Enable deep thinking subroutine" for cogito)
- `content_parsers`: List of regex patterns with named groups `(?P<thinking>...)` and `(?P<content>...)`

For streaming, the last parser in the list is used with the `StreamingParser` class.

### Cache System

- Cache keys are MD5 hashes of `ChatRequest.model_dump_json() + host`
- Cached by model tag for potential bulk clearing
- Streaming caches the entire list of `ThinkResponse` chunks
- Cache can be bypassed per-call with `use_cache=False`

## Important Files

- `src/ollama_think/config.yaml` - Model-specific thinking hack configurations
- `src/ollama_think/client.py` - Main Client and AsyncClient implementations
- `src/ollama_think/thinking_hacks.py` - Request/response transformation logic
- `tests/conftest.py` - Pytest fixtures for host and client configuration
- `model_capabilities.md` - Generated report of model thinking support

## Testing Patterns

Tests are marked with `@pytest.mark.slow` for integration tests requiring Ollama. Fast tests use mocks.

Fixtures:
- `host` - Configurable via `--host` flag (default: http://localhost:11434)
- `client` - Session-scoped Client instance using the host fixture

## Code Conventions

- Line length: 100 characters (ruff configured)
- Minimum Python version: 3.9
- Type hints used throughout with pydantic for ollama types
