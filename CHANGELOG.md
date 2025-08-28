# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [0.1.8] - 2025-08-28

### Changed
- Updated to ollama 0.5.3
- The `think` parameter now accepts `'low'`, `'medium'`, or `'high'` to support `gpt-oss` style thinking levels.

### Added
- Added tests for `gpt-oss` thinking levels.

# [0.1.7] - 2025-08-01

### Added
- Added ai_docs.md for compact LLM docs
- Added top level import for AsyncClient and ThinkResponse

# [0.1.6] - 2025-07-16

### Removed
- Removed the `rich` and `hatch` dependencies to simplify installation. 

### Added
- Added AsyncClient to mirror Ollama.AsyncClient

# [0.1.5] - 2025-07-14

### Changed
- Widened version ranges for `diskcache` and `pyyaml` dependencies.

### Removed
- Removed the `ollama-think` command-line interface and the `typer` dependency.

# [0.1.4] - 2025-07-09

- Reworked README.md and context7.json to make it easier for LLMs
- Added default import of Client for less boilerplate

## [0.1.3] - 2025-07-08

- Add `pytest-mock` and other dev dependencies to the Hatch test environment.
- Add `context7.json` for LLMs

## [0.1.2] - 2025-07-06

### Added

- `CHANGELOG.md` to document project changes.
- `LICENSE` file with the full MIT License text.
- `requirements.txt` lock file for reproducible development environments.
- `tests/conftest.py` to provide a parameterized `--host` option for tests.
- Comprehensive "Contributing" section to `README.md` with development setup and workflow instructions.
- `hatch` configuration in `pyproject.toml` for standardized task running.

### Changed

- **Packaging:** Moved `config.yaml` into the source directory to ensure it is included in the final package.
- **Configuration:** Centralized `pytest` and `ruff` configurations in `pyproject.toml`.
- **Testing:** Refactored integration tests to use a parameterized `client` fixture instead of a hardcoded host.
- **Dependencies:** Added `hatch` as a development dependency.

### Fixed

- Corrected various syntax and configuration issues in `pyproject.toml`.
- Fixed a broken link to `model_capabilities.md` in `README.md`.
- Resolved a `NameError` in the test suite by correctly passing fixtures.
- Addressed and resolved various shell and command execution issues during the interactive session.
