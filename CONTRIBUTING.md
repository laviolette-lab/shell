# Contributing

Thank you for your interest in contributing! This document provides guidelines for working with this project.

## Development Setup

1. **Clone the repository:**

   ```console
   git clone https://github.com/LavLabInfrastructure/shell.git
   cd shell
   ```

2. **Install Hatch** (build & environment manager):

   ```console
   pip install hatch
   ```

3. **Install pre-commit hooks:**

   ```console
   pip install pre-commit
   pre-commit install
   ```

## Common Tasks

| Task                    | Command                      |
|-------------------------|------------------------------|
| Run tests               | `hatch run test:test`        |
| Run tests with coverage | `hatch run test:cov`         |
| Lint code               | `hatch run lint:check`       |
| Format code             | `hatch run lint:format`      |
| Auto-fix lint issues    | `hatch run lint:fix`         |
| Run all lint checks     | `hatch run lint:all`         |
| Type check              | `hatch run types:check`      |
| Build docs              | `hatch run docs:build-docs`  |
| Serve docs locally      | `hatch run docs:serve-docs`  |
| Build wheel             | `hatch build`                |

## Code Style

- **Formatter & Linter:** [Ruff](https://docs.astral.sh/ruff/) handles both formatting and linting.
- **Line length:** 88 characters.
- **Docstrings:** Sphinx-style (`:param:`, `:type:`, `:return:`, `:rtype:`).
- **Type hints:** Encouraged. The project includes a `py.typed` marker for PEP 561 compliance.

## Project Architecture

1. **Library modules** (`src/shell/`) contain all business logic as importable functions.
2. **CLI** (`src/shell/cli.py`) is a thin wrapper that delegates to library functions.
3. **Tests** (`tests/`) import directly from the library.

## Submitting Changes

1. Create a branch from `main`.
2. Make your changes and ensure all checks pass.
3. Open a Pull Request against `main`.

## License

`shell` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
