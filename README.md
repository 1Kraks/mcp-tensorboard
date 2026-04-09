# MCP TensorBoard

A Model Context Protocol (MCP) server that exposes TensorBoard data through a standardized API. Built with FastMCP, this server enables AI coding agents to query and analyze TensorBoard experiment data programmatically.

## Features

- **Pure Python implementation** - No subprocess or external binaries required
- **Multiple transports** - stdio, Streamable HTTP, and SSE
- **Full TensorBoard support** - Scalars, tensors, histograms, distributions, and images
- **Structured output** - Pydantic models for type-safe, validated responses
- **AI-optimized** - Compact data formats ideal for LLM consumption

## Quickstart

### Run directly from GitHub (no installation)

```bash
uvx --from git+https://github.com/1Kraks/mcp-tensorboard mcp-tensorboard --logdir /path/to/logs
```

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/1Kraks/mcp-tensorboard
cd mcp-tensorboard

# Create virtual environment and install
uv venv
source .venv/bin/activate  # macOS/Linux
uv sync

# Run the server
uv run mcp-tensorboard --logdir /path/to/logs
```

### Install with pip

```bash
pip install -e .
mcp-tensorboard --logdir /path/to/logs
```

## Usage

### Command Line Options

```
mcp-tensorboard --logdir <path> [--transport stdio|http|sse] [--port PORT] [--host HOST] [--debug]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--logdir` | (required) | Path to TensorBoard logs directory |
| `--transport` | `stdio` | Transport protocol |
| `--port` | `8000` | Port for HTTP/SSE transport |
| `--host` | `0.0.0.0` | Host for HTTP/SSE transport |
| `--debug` | off | Enable debug logging |

### Environment Variables

- `TENSORBOARD_LOGDIR` - Default log directory (alternative to `--logdir`)
- `TENSORBOARD_LOGS` - Alternative log directory variable

## Available Tools

### Run Management

| Tool | Description |
|------|-------------|
| `tensorboard_list_runs` | List all runs in the log directory |

### Scalars

| Tool | Description |
|------|-------------|
| `tensorboard_list_scalar_tags` | List scalar tags for a run |
| `tensorboard_get_scalar_series` | Get time series for a scalar |
| `tensorboard_get_scalar_series_batch` | Get multiple scalars in one call |
| `tensorboard_get_scalar_last` | Get the most recent scalar value |

### Tensors

| Tool | Description |
|------|-------------|
| `tensorboard_list_tensor_tags` | List tensor tags for a run |
| `tensorboard_get_tensor_series` | Get time series for scalar tensors |

### Histograms & Distributions

| Tool | Description |
|------|-------------|
| `tensorboard_list_histogram_tags` | List histogram tags |
| `tensorboard_get_histogram_series` | Get raw histogram data |
| `tensorboard_list_distribution_tags` | List distribution tags (alias) |
| `tensorboard_get_distribution_series` | Get compressed distributions (recommended) |

### Images

| Tool | Description |
|------|-------------|
| `tensorboard_list_image_tags` | List image tags |
| `tensorboard_get_image_series` | Get image references (blob keys) |
| `tensorboard_get_image` | Fetch image by blob key (returns base64) |

## Integration with Coding Agents

### Claude Code

**Option 1: Run from git (no install)**

```bash
claude mcp add --transport http tensorboard-http \
  uvx --from git+https://github.com/1Kraks/mcp-tensorboard mcp-tensorboard --logdir /path/to/logs --transport http
```

**Option 2: Local installation**

```bash
# Install globally or in a shared venv
pip install -e /path/to/mcp-tensorboard

# Add to Claude Code
claude mcp add tensorboard mcp-tensorboard --logdir /path/to/logs
```

**Option 3: Via Claude Code settings.json**

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "tensorboard": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/1Kraks/mcp-tensorboard",
        "mcp-tensorboard",
        "--logdir",
        "/path/to/logs"
      ]
    }
  }
}
```

### GitHub Copilot / VS Code

Add to VS Code `settings.json`:

```json
{
  "github.copilot.chat.mcp.servers": {
    "tensorboard": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/1Kraks/mcp-tensorboard",
        "mcp-tensorboard",
        "--logdir",
        "/path/to/logs"
      ]
    }
  }
}
```

### Cline (VS Code Extension)

Add to Cline's MCP settings:

```json
{
  "mcpServers": {
    "tensorboard": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/1Kraks/mcp-tensorboard",
        "mcp-tensorboard",
        "--logdir",
        "/path/to/logs"
      ]
    }
  }
}
```

### Cursor

Add to Cursor's MCP configuration:

```json
{
  "mcpServers": {
    "tensorboard": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/1Kraks/mcp-tensorboard",
        "mcp-tensorboard",
        "--logdir",
        "/path/to/logs"
      ]
    }
  }
}
```

### Generic MCP Client (Streamable HTTP)

For HTTP transport, run the server:

```bash
mcp-tensorboard --logdir /path/to/logs --transport http --port 8000
```

Connect to `http://localhost:8000/mcp` from any MCP-compatible client.

## Example Usage

### List all runs

```json
{
  "method": "tools/call",
  "params": {
    "name": "tensorboard_list_runs",
    "arguments": {}
  }
}
```

### Get scalar training loss over time

```json
{
  "method": "tools/call",
  "params": {
    "name": "tensorboard_get_scalar_series",
    "arguments": {
      "run": ".",
      "tag": "loss",
      "max_points": 500
    }
  }
}
```

### Compare multiple metrics

```json
{
  "method": "tools/call",
  "params": {
    "name": "tensorboard_get_scalar_series_batch",
    "arguments": {
      "run": "experiment_1",
      "tags": ["loss", "accuracy", "val_loss", "val_accuracy"],
      "max_points": 200
    }
  }
}
```

### Get compressed distribution (AI-friendly)

```json
{
  "method": "tools/call",
  "params": {
    "name": "tensorboard_get_distribution_series",
    "arguments": {
      "run": ".",
      "tag": "weights",
      "max_points": 50
    }
  }
}
```

## Development

### Setup

```bash
# Clone and set up environment
git clone https://github.com/1Kraks/mcp-tensorboard
cd mcp-tensorboard
uv venv
source .venv/bin/activate
uv sync --all-extras
```

### Run tests

```bash
pytest
```

### Run with debug logging

```bash
mcp-tensorboard --logdir /path/to/logs --debug
```

### Code style

```bash
# Format code
ruff format .

# Lint
ruff check .
```

## Project Structure

```
mcp-tensorboard/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── src/mcp_tensorboard/
│   ├── __init__.py             # Package init
│   ├── __main__.py             # python -m entry point
│   ├── server.py               # FastMCP server & tools
│   ├── data_reader.py          # Pure Python event file reader
│   └── types.py                # Pydantic response models
└── tests/
    └── test_server.py          # Unit tests
```

## Troubleshooting

**No runs found**
- Ensure `--logdir` points to the directory containing TensorBoard event files
- Event files are typically named `events.out.tfevents.*`

**Import errors**
- Run `uv sync` or `pip install -e .` to install dependencies

**HTTP transport not connecting**
- Verify the server is running: `curl http://localhost:8000/mcp`
- Check firewall settings for the specified port

**Images not displaying**
- Image support requires Pillow: `pip install pillow`
- Some TensorBoard image formats may not be supported

## License

MIT License - See LICENSE file for details.
