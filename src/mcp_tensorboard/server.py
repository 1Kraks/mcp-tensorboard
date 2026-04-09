"""FastMCP server for TensorBoard data access.

This module provides an MCP server that exposes TensorBoard data through
a set of tools using the FastMCP framework.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .data_reader import TensorBoardDataReader
from .types import (
    DistributionSeriesResponse,
    ErrorResponse,
    HistogramSeriesResponse,
    ImagePoint,
    ImageResponse,
    ImageSeriesResponse,
    RunsResponse,
    RunInfo,
    ScalarLastResponse,
    ScalarPoint,
    ScalarSeriesBatchResponse,
    ScalarSeriesResponse,
    TagListResponse,
    TensorPoint,
    TensorSeriesResponse,
)

# Create FastMCP server
mcp = FastMCP(
    "TensorBoard",
    instructions="MCP server for querying TensorBoard data",
    dependencies=["tensorboard", "numpy", "pillow"],
)

# Global reader instance (initialized on first tool call)
_reader: TensorBoardDataReader | None = None


def _get_reader() -> TensorBoardDataReader:
    """Get or create the TensorBoard data reader."""
    global _reader
    if _reader is None:
        raise RuntimeError(
            "TensorBoard reader not initialized. "
            "Please set the TENSORBOARD_LOGDIR environment variable or pass --logdir."
        )
    return _reader


def _init_reader(logdir: Path) -> None:
    """Initialize the global reader."""
    global _reader
    _reader = TensorBoardDataReader(logdir)


def _handle_tool_error(func):
    """Decorator to handle tool errors and return ErrorResponse."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            return ErrorResponse(error=str(e))
        except Exception as e:
            return ErrorResponse(error=f"Internal error: {type(e).__name__}: {e}")

    return wrapper


# =============================================================================
# Run Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_runs() -> RunsResponse:
    """List all runs in the TensorBoard log directory.

    Returns a list of run names that can be used with other tools.
    """
    reader = _get_reader()
    runs = reader.list_runs()
    return RunsResponse(runs=[RunInfo(name=r) for r in runs])


@mcp.tool()
@_handle_tool_error
def tensorboard_refresh() -> dict[str, str]:
    """Refresh the data cache to pick up new TensorBoard events.

    Call this tool after new training runs have written event files
    to reload the data from disk.

    Returns:
        dict with 'status' and 'last_load_time' keys
    """
    reader = _get_reader()
    reader.refresh()
    return {
        "status": "Cache refreshed successfully",
        "last_load_time": str(datetime.fromtimestamp(reader.last_load_time)) if reader.last_load_time else "unknown",
    }


# =============================================================================
# Scalar Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_scalar_tags(run: str) -> TagListResponse | ErrorResponse:
    """List all scalar tags for a specific run.

    Args:
        run: The run name (from tensorboard_list_runs)
    """
    reader = _get_reader()
    tags = reader.list_scalar_tags(run)
    return TagListResponse(run=run, tags=tags)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_scalar_series(
    run: str, tag: str, max_points: int | None = 1000, include_wall_time: bool = True
) -> ScalarSeriesResponse | ErrorResponse:
    """Get time series data for a scalar tag.

    Args:
        run: The run name
        tag: The scalar tag name
        max_points: Maximum number of points to return (samples evenly if needed)
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points = reader.get_scalar_series(run, tag, max_points)

    return ScalarSeriesResponse(
        run=run,
        tag=tag,
        points=[
            ScalarPoint(
                step=p.step,
                value=p.value,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ],
    )


@mcp.tool()
@_handle_tool_error
def tensorboard_get_scalar_series_batch(
    run: str, tags: list[str], max_points: int | None = 1000, include_wall_time: bool = True
) -> ScalarSeriesBatchResponse | ErrorResponse:
    """Get time series data for multiple scalar tags in one call.

    Args:
        run: The run name
        tags: List of scalar tag names
        max_points: Maximum number of points per tag
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points_by_tag: dict[str, list[ScalarPoint]] = {}

    for tag in tags:
        points = reader.get_scalar_series(run, tag, max_points)
        points_by_tag[tag] = [
            ScalarPoint(
                step=p.step,
                value=p.value,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ]

    return ScalarSeriesBatchResponse(run=run, points_by_tag=points_by_tag)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_scalar_last(run: str, tag: str) -> ScalarLastResponse | ErrorResponse:
    """Get the most recent scalar value for a tag.

    Args:
        run: The run name
        tag: The scalar tag name
    """
    reader = _get_reader()
    datum = reader.get_scalar_last(run, tag)

    if datum is None:
        return ScalarLastResponse(run=run, tag=tag)

    return ScalarLastResponse(
        run=run,
        tag=tag,
        step=datum.step,
        value=datum.value,
        wall_time=datetime.fromtimestamp(datum.wall_time),
    )


# =============================================================================
# Tensor Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_tensor_tags(run: str) -> TagListResponse | ErrorResponse:
    """List all tensor tags for a specific run.

    Args:
        run: The run name
    """
    reader = _get_reader()
    tags = reader.list_tensor_tags(run)
    return TagListResponse(run=run, tags=tags)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_tensor_series(
    run: str, tag: str, max_points: int | None = 1000, include_wall_time: bool = True
) -> TensorSeriesResponse | ErrorResponse:
    """Get time series data for a scalar tensor tag.

    Only works with tensors that reduce to a single scalar value.

    Args:
        run: The run name
        tag: The tensor tag name
        max_points: Maximum number of points to return
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points = reader.get_tensor_series(run, tag, max_points)

    return TensorSeriesResponse(
        run=run,
        tag=tag,
        points=[
            TensorPoint(
                step=p.step,
                value=p.value,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ],
    )


# =============================================================================
# Histogram Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_histogram_tags(run: str) -> TagListResponse | ErrorResponse:
    """List all histogram tags for a specific run.

    Args:
        run: The run name
    """
    reader = _get_reader()
    tags = reader.list_histogram_tags(run)
    return TagListResponse(run=run, tags=tags)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_histogram_series(
    run: str, tag: str, max_points: int | None = 100, include_wall_time: bool = True
) -> HistogramSeriesResponse | ErrorResponse:
    """Get histogram time series data.

    Note: Histogram data can be large. Consider using tensorboard_get_distribution_series
    for a more compact representation.

    Args:
        run: The run name
        tag: The histogram tag name
        max_points: Maximum number of histogram snapshots to return
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points = reader.get_histogram_series(run, tag, max_points)

    return HistogramSeriesResponse(
        run=run,
        tag=tag,
        points=[
            HistogramPoint(
                step=p.step,
                values=p.values,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ],
    )


# =============================================================================
# Distribution Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_distribution_tags(run: str) -> TagListResponse | ErrorResponse:
    """List all distribution tags for a specific run (alias for histogram tags).

    Distributions are compressed histograms with fixed basis points.

    Args:
        run: The run name
    """
    return tensorboard_list_histogram_tags(run)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_distribution_series(
    run: str, tag: str, max_points: int | None = 100, include_wall_time: bool = True
) -> DistributionSeriesResponse | ErrorResponse:
    """Get distribution (compressed histogram) time series.

    Returns histograms compressed to fixed basis points for compact representation.
    This is ideal for AI consumption as it preserves distribution shape with minimal data.

    Args:
        run: The run name
        tag: The distribution tag name
        max_points: Maximum number of distribution snapshots to return
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points = reader.get_distribution_series(run, tag, max_points)

    return DistributionSeriesResponse(
        run=run,
        tag=tag,
        points=[
            DistributionPoint(
                step=p.step,
                basis_points=p.basis_points,
                values=p.values,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ],
    )


# =============================================================================
# Image Tools
# =============================================================================


@mcp.tool()
@_handle_tool_error
def tensorboard_list_image_tags(run: str) -> TagListResponse | ErrorResponse:
    """List all image tags for a specific run.

    Args:
        run: The run name
    """
    reader = _get_reader()
    tags = reader.list_image_tags(run)
    return TagListResponse(run=run, tags=tags)


@mcp.tool()
@_handle_tool_error
def tensorboard_get_image_series(
    run: str, tag: str, max_points: int | None = 50, include_wall_time: bool = True
) -> ImageSeriesResponse | ErrorResponse:
    """Get image series references.

    Returns blob keys that can be used with tensorboard_get_image to fetch actual images.
    This two-step approach keeps responses small.

    Args:
        run: The run name
        tag: The image tag name
        max_points: Maximum number of image references to return
        include_wall_time: Whether to include wall clock timestamps
    """
    reader = _get_reader()
    points = reader.get_image_series(run, tag, max_points)

    return ImageSeriesResponse(
        run=run,
        tag=tag,
        points=[
            ImagePoint(
                step=p.step,
                blob_key=p.blob_key,
                wall_time=datetime.fromtimestamp(p.wall_time) if include_wall_time else None,
            )
            for p in points
        ],
    )


@mcp.tool()
def tensorboard_get_image(blob_key: str) -> ImageResponse | ErrorResponse:
    """Get a single image by blob key.

    The blob_key comes from tensorboard_get_image_series response.
    Returns base64-encoded image data suitable for display.

    Args:
        blob_key: The image blob key (format: run::tag::step, uses :: delimiter)
    """
    reader = _get_reader()
    result = reader.get_image(blob_key)

    if result is None:
        return ErrorResponse(error=f"Image not found for blob_key: {blob_key}")

    mime_type, image_bytes = result
    return ImageResponse(
        blob_key=blob_key,
        mime_type=mime_type,
        data=base64.b64encode(image_bytes).decode("ascii"),
    )


# =============================================================================
# Main entry point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the mcp-tensorboard command."""
    parser = argparse.ArgumentParser(description="TensorBoard MCP Server")
    parser.add_argument(
        "--logdir",
        default=None,
        help="Path to TensorBoard log directory (or set TENSORBOARD_LOGDIR)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transport")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    # Determine logdir
    logdir_str = args.logdir or os.environ.get("TENSORBOARD_LOGDIR") or os.environ.get("TENSORBOARD_LOGS")

    if not logdir_str:
        print("ERROR: --logdir not set (or TENSORBOARD_LOGDIR not set).", file=__import__("sys").stderr)
        print("Usage: mcp-tensorboard --logdir /path/to/logs", file=__import__("sys").stderr)
        return 1

    logdir = Path(logdir_str).expanduser().resolve()

    if not logdir.exists():
        print(f"ERROR: logdir does not exist: {logdir}", file=__import__("sys").stderr)
        return 1

    if not logdir.is_dir():
        print(f"ERROR: logdir is not a directory: {logdir}", file=__import__("sys").stderr)
        return 1

    # Initialize reader
    _init_reader(logdir)

    # Run server
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
