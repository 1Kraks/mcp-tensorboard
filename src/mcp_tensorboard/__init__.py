"""MCP TensorBoard server package.

This package provides an MCP server that exposes TensorBoard data
through the FastMCP framework.
"""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("mcp-tensorboard")
except Exception:
    __version__ = "0.1.0"

__all__ = ["__version__"]
