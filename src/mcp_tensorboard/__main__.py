"""Entry point for running as python -m mcp_tensorboard."""

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
