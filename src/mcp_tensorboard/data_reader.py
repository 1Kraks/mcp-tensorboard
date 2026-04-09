"""Pure Python TensorBoard event file reader.

This module reads TensorBoard event files directly without requiring
the TensorBoard data server subprocess.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing.io_wrapper import IsSummaryEventsFile
from tensorboard.util import tensor_util


@dataclass
class ScalarDatum:
    """A single scalar data point."""

    step: int
    value: float
    wall_time: float


@dataclass
class TensorDatum:
    """A single tensor data point (scalar tensors)."""

    step: int
    value: float
    wall_time: float


@dataclass
class HistogramDatum:
    """A single histogram data point."""

    step: int
    values: list[float]
    wall_time: float


@dataclass
class DistributionDatum:
    """A single distribution (compressed histogram) data point."""

    step: int
    basis_points: list[int]
    values: list[float]
    wall_time: float


@dataclass
class ImageDatum:
    """A single image reference."""

    step: int
    blob_key: str
    wall_time: float
    encoded_image: bytes | None = None


@dataclass
class RunData:
    """Data for a single run."""

    name: str
    scalars: dict[str, list[ScalarDatum]] = field(default_factory=dict)
    tensors: dict[str, list[TensorDatum]] = field(default_factory=dict)
    histograms: dict[str, list[HistogramDatum]] = field(default_factory=dict)
    images: dict[str, list[ImageDatum]] = field(default_factory=dict)


class TensorBoardDataReader:
    """Read TensorBoard data from event files using pure Python."""

    # Default basis points for distribution compression
    DEFAULT_BASIS_POINTS: list[int] = [0, 668, 1587, 3085, 5000, 6915, 8413, 9332, 10000]

    def __init__(self, logdir: Path | str):
        self.logdir = Path(logdir).expanduser().resolve()
        self._runs: dict[str, RunData] | None = None
        self._last_load_time: float | None = None

    @property
    def runs(self) -> dict[str, RunData]:
        """Load and return all runs data."""
        if self._runs is None:
            self._runs = self._load_all_runs()
            self._last_load_time = datetime.now().timestamp()
        return self._runs

    def refresh(self) -> None:
        """Reload all runs data from disk, picking up new TensorBoard events.

        Call this method to refresh the cache and include any new event files
        that have been written since the initial load.
        """
        self._runs = self._load_all_runs()
        self._last_load_time = datetime.now().timestamp()

    @property
    def last_load_time(self) -> float | None:
        """Return the timestamp when the data was last loaded."""
        return self._last_load_time

    def _load_all_runs(self) -> dict[str, RunData]:
        """Load all runs from the logdir."""
        runs: dict[str, RunData] = {}

        # Find all event files
        event_files = list(self._discover_event_files())

        # Group by run (directory)
        files_by_run: dict[str, list[Path]] = {}
        for ef in event_files:
            run_name = str(ef.parent.relative_to(self.logdir))
            if run_name == ".":
                run_name = "."
            if run_name not in files_by_run:
                files_by_run[run_name] = []
            files_by_run[run_name].append(ef)

        # Load each run
        for run_name, files in sorted(files_by_run.items()):
            run_data = RunData(name=run_name)
            for event_file in sorted(files):
                self._load_event_file(run_data, event_file)
            runs[run_name] = run_data

        return runs

    def _discover_event_files(self) -> Iterator[Path]:
        """Discover all event files in the logdir."""
        for root, _, files in os.walk(self.logdir):
            for fname in files:
                if IsSummaryEventsFile(fname):
                    yield Path(root) / fname

    def _load_event_file(self, run_data: RunData, event_file: Path) -> None:
        """Load a single event file."""
        loader = event_file_loader.LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            wall_time = event.wall_time
            step = event.step

            if event.HasField("summary"):
                for value in event.summary.value:
                    tag = value.tag

                    # Scalars - check for simple_value field
                    if value.HasField("simple_value"):
                        if tag not in run_data.scalars:
                            run_data.scalars[tag] = []
                        run_data.scalars[tag].append(
                            ScalarDatum(step=step, value=value.simple_value, wall_time=wall_time)
                        )

                    # Tensors - check for tensor field
                    if value.HasField("tensor"):
                        tensor_proto = value.tensor

                        # Try to get plugin info
                        plugin_name = ""
                        if value.HasField("metadata") and value.metadata.HasField("plugin_data"):
                            plugin_name = value.metadata.plugin_data.plugin_name

                        # Histograms
                        if plugin_name == "histograms":
                            if tag not in run_data.histograms:
                                run_data.histograms[tag] = []
                            # Extract histogram from tensor proto
                            values = self._extract_histogram_from_tensor(tensor_proto)
                            run_data.histograms[tag].append(
                                HistogramDatum(step=step, values=values, wall_time=wall_time)
                            )

                        # Images
                        elif plugin_name == "images":
                            if tag not in run_data.images:
                                run_data.images[tag] = []
                            # Extract image data from tensor
                            image_bytes = self._extract_image_from_tensor(tensor_proto)
                            # Use :: delimiter to handle runs/tags with slashes
                            blob_key = f"{run_data.name}::{tag}::{step}"
                            run_data.images[tag].append(
                                ImageDatum(
                                    step=step,
                                    blob_key=blob_key,
                                    wall_time=wall_time,
                                    encoded_image=image_bytes,
                                )
                            )

                        # Scalar tensors (0-d or 1-element)
                        else:
                            try:
                                tensor_np = tensor_util.make_ndarray(tensor_proto)
                                if tensor_np.ndim == 0 or (tensor_np.ndim == 1 and tensor_np.size == 1):
                                    if tag not in run_data.tensors:
                                        run_data.tensors[tag] = []
                                    scalar_val = float(tensor_np.flatten()[0])
                                    run_data.tensors[tag].append(
                                        TensorDatum(step=step, value=scalar_val, wall_time=wall_time)
                                    )
                            except (ValueError, TypeError, AttributeError):
                                pass  # Skip non-numeric or unparseable tensors

    def _extract_histogram_from_tensor(self, tensor_proto: Any) -> list[float]:
        """Extract histogram bucket values from tensor proto."""
        try:
            tensor_np = tensor_util.make_ndarray(tensor_proto)
            return tensor_np.flatten().tolist()
        except Exception:
            return []

    def _extract_image_from_tensor(self, tensor_proto: Any) -> bytes | None:
        """Extract encoded image bytes from tensor proto."""
        try:
            # Try to get string/bytes content
            if tensor_proto.string_value:
                return tensor_proto.string_value
            tensor_np = tensor_util.make_ndarray(tensor_proto)
            if tensor_np.dtype == np.object_:
                return tensor_np.flatten()[0]
            return None
        except Exception:
            return None

    def list_runs(self) -> list[str]:
        """List all run names."""
        return sorted(self.runs.keys())

    def list_scalar_tags(self, run: str) -> list[str]:
        """List scalar tags for a run."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        return sorted(run_data.scalars.keys())

    def list_tensor_tags(self, run: str) -> list[str]:
        """List tensor tags for a run."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        return sorted(run_data.tensors.keys())

    def list_histogram_tags(self, run: str) -> list[str]:
        """List histogram tags for a run."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        return sorted(run_data.histograms.keys())

    def list_image_tags(self, run: str) -> list[str]:
        """List image tags for a run."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        return sorted(run_data.images.keys())

    def get_scalar_series(
        self, run: str, tag: str, max_points: int | None = None
    ) -> list[ScalarDatum]:
        """Get scalar time series."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        if tag not in run_data.scalars:
            raise ValueError(f"Tag not found: {tag}")

        points = sorted(run_data.scalars[tag], key=lambda x: x.step)
        if max_points is not None and len(points) > max_points:
            # Sample evenly
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        return points

    def get_tensor_series(
        self, run: str, tag: str, max_points: int | None = None
    ) -> list[TensorDatum]:
        """Get tensor time series (scalar tensors only)."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        if tag not in run_data.tensors:
            raise ValueError(f"Tag not found: {tag}")

        points = sorted(run_data.tensors[tag], key=lambda x: x.step)
        if max_points is not None and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        return points

    def get_histogram_series(
        self, run: str, tag: str, max_points: int | None = None
    ) -> list[HistogramDatum]:
        """Get histogram time series."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        if tag not in run_data.histograms:
            raise ValueError(f"Tag not found: {tag}")

        points = sorted(run_data.histograms[tag], key=lambda x: x.step)
        if max_points is not None and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        return points

    def get_distribution_series(
        self, run: str, tag: str, max_points: int | None = None
    ) -> list[DistributionDatum]:
        """Get distribution (compressed histogram) time series."""
        histograms = self.get_histogram_series(run, tag, max_points)

        distributions: list[DistributionDatum] = []
        for h in histograms:
            # Compress histogram to distribution using basis points
            values = self._compress_histogram(h.values)
            distributions.append(
                DistributionDatum(
                    step=h.step,
                    basis_points=self.DEFAULT_BASIS_POINTS.copy(),
                    values=values,
                    wall_time=h.wall_time,
                )
            )
        return distributions

    def _compress_histogram(self, values: list[float]) -> list[float]:
        """Compress histogram values to fixed basis points.

        This creates a cumulative distribution function (CDF) approximation.
        The basis points represent fixed thresholds on a normalized 0-10000 scale.
        For each basis point, we compute what percentage of values fall below that threshold.

        This matches TensorBoard's approach: normalize the data to 0-10000, then
        compute cumulative percentages at each basis point.
        """
        if not values:
            return [0.0] * len(self.DEFAULT_BASIS_POINTS)

        values_arr = np.array(values)
        if len(values_arr) == 0:
            return [0.0] * len(self.DEFAULT_BASIS_POINTS)

        # Handle all-same values: 100% of values are <= any threshold >= the value
        min_val, max_val = values_arr.min(), values_arr.max()
        if min_val == max_val:
            # All values are identical - they're all at the same point
            # Return 0% for basis points below the value, 100% for basis points at/above
            # Since we normalize to 0-10000, the single value maps to 5000 (middle)
            return [100.0 if bp >= 5000 else 0.0 for bp in self.DEFAULT_BASIS_POINTS]

        # Normalize values to 0-10000 scale
        normalized = (values_arr - min_val) / (max_val - min_val) * 10000

        # Compute cumulative distribution: % of values <= each basis point
        cumulative = []
        for bp in self.DEFAULT_BASIS_POINTS:
            count = np.sum(normalized <= bp)
            cumulative.append(float(count) / len(values_arr) * 100)  # Percentage

        return cumulative

    def get_image_series(
        self, run: str, tag: str, max_points: int | None = None
    ) -> list[ImageDatum]:
        """Get image series."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        if tag not in run_data.images:
            raise ValueError(f"Tag not found: {tag}")

        points = sorted(run_data.images[tag], key=lambda x: x.step)
        if max_points is not None and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        return points

    def get_image(self, blob_key: str) -> tuple[str, bytes] | None:
        """Get image data by blob key.

        Returns (mime_type, image_bytes) or None if not found.
        Blob key format: run::tag::step (uses :: delimiter to handle slashes in paths)
        """
        # Parse blob_key: run::tag::step
        parts = blob_key.split("::")
        if len(parts) < 3:
            return None

        run = parts[0]
        tag = "::".join(parts[1:-1])  # Handle tags with :: (unlikely but safe)
        try:
            step = int(parts[-1])
        except ValueError:
            return None

        run_data = self.runs.get(run)
        if run_data is None:
            return None

        if tag not in run_data.images:
            return None

        for img in run_data.images[tag]:
            if img.step == step and img.blob_key == blob_key:
                if img.encoded_image:
                    return ("image/png", img.encoded_image)
                break

        return None

    def get_scalar_last(self, run: str, tag: str) -> ScalarDatum | None:
        """Get the last scalar value."""
        run_data = self.runs.get(run)
        if run_data is None:
            raise ValueError(f"Run not found: {run}")
        if tag not in run_data.scalars:
            raise ValueError(f"Tag not found: {tag}")

        points = run_data.scalars[tag]
        if not points:
            return None
        return max(points, key=lambda x: x.step)
