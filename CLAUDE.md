# Project Context - llm-reward-repo

## Python Environment Management (uv)

This project uses **uv** for Python package and project management. uv is an extremely fast Python package and project manager written in Rust, designed to replace pip, pip-tools, poetry, and more.

### Setup

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux

# Add dependencies
uv add <package-name>

# Sync environment with lockfile
uv sync

# Run commands in the virtual environment
uv run <command>
```

### Key Commands

- `uv add <package>` - Add a dependency to pyproject.toml
- `uv remove <package>` - Remove a dependency
- `uv sync` - Sync virtual environment with lockfile
- `uv run <script>` - Run a script in the virtual environment
- `uv lock` - Update the lockfile without syncing

## Project Info

- Python version: 3.13 (managed via .python-version)
- Package manager: uv
- Project file: pyproject.toml
