"""Tests for the MCP TensorBoard server."""

from __future__ import annotations

import tempfile
from pathlib import Path


def create_test_event_file(logdir: Path) -> Path:
    """Create a test event file with scalar data using low-level API."""
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(logdir=str(logdir))

    # Create scalar summary
    scalar_summary = summary_pb2.Summary()
    scalar_summary.value.add(tag="loss", simple_value=0.5)

    scalar_event = event_pb2.Event()
    scalar_event.wall_time = 1700000000.0
    scalar_event.step = 1
    scalar_event.summary.CopyFrom(scalar_summary)
    writer.add_event(scalar_event)

    # Create another scalar
    scalar_summary2 = summary_pb2.Summary()
    scalar_summary2.value.add(tag="accuracy", simple_value=0.85)

    scalar_event2 = event_pb2.Event()
    scalar_event2.wall_time = 1700000001.0
    scalar_event2.step = 1
    scalar_event2.summary.CopyFrom(scalar_summary2)
    writer.add_event(scalar_event2)

    writer.flush()
    writer.close()

    # Find the created event file
    event_files = list(logdir.glob("events.out.tfevents.*"))
    return event_files[0] if event_files else None


def test_data_reader_scalars() -> None:
    """Test the data reader with scalar data."""
    from mcp_tensorboard.data_reader import TensorBoardDataReader

    with tempfile.TemporaryDirectory() as tmpdir:
        logdir = Path(tmpdir)
        create_test_event_file(logdir)

        reader = TensorBoardDataReader(logdir)

        # Test list runs
        runs = reader.list_runs()
        assert len(runs) > 0, "Should find at least one run"

        # Test list scalar tags
        run = runs[0]
        tags = reader.list_scalar_tags(run)
        assert "loss" in tags, "Should find loss tag"
        assert "accuracy" in tags, "Should find accuracy tag"

        # Test get scalar series
        points = reader.get_scalar_series(run, "loss")
        assert len(points) > 0, "Should find scalar points"
        assert points[0].step == 1
        assert abs(points[0].value - 0.5) < 0.001, "Value should be 0.5"


def test_server_import() -> None:
    """Test that the server module can be imported."""
    from mcp_tensorboard import server

    assert hasattr(server, "mcp")
    assert hasattr(server, "main")


def test_types_import() -> None:
    """Test that the types module can be imported."""
    from mcp_tensorboard import types

    # Check that key types exist
    assert hasattr(types, "RunsResponse")
    assert hasattr(types, "ScalarSeriesResponse")
    assert hasattr(types, "TagListResponse")
