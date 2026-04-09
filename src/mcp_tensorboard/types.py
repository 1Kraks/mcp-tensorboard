"""Pydantic models for structured tool responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunInfo(BaseModel):
    """Information about a single run."""

    name: str = Field(description="Run name/path relative to logdir")


class RunsResponse(BaseModel):
    """Response for list_runs tool."""

    runs: list[RunInfo] = Field(description="List of runs in the logdir")


class ScalarPoint(BaseModel):
    """A single scalar data point."""

    step: int = Field(description="Global step value")
    value: float = Field(description="Scalar value")
    wall_time: datetime | None = Field(default=None, description="Wall clock time")


class ScalarSeriesResponse(BaseModel):
    """Response for get_scalar_series tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Scalar tag name")
    points: list[ScalarPoint] = Field(description="Time series data points")


class ScalarSeriesBatchResponse(BaseModel):
    """Response for get_scalar_series_batch tool."""

    run: str = Field(description="Run name")
    points_by_tag: dict[str, list[ScalarPoint]] = Field(
        description="Time series data points per tag"
    )


class ScalarLastResponse(BaseModel):
    """Response for get_scalar_last tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Scalar tag name")
    step: int | None = Field(default=None, description="Step of last value")
    value: float | None = Field(default=None, description="Last scalar value")
    wall_time: datetime | None = Field(default=None, description="Wall clock time of last value")


class TensorPoint(BaseModel):
    """A single tensor data point (scalar tensors only)."""

    step: int = Field(description="Global step value")
    value: float = Field(description="Scalar value extracted from tensor")
    wall_time: datetime | None = Field(default=None, description="Wall clock time")


class TensorSeriesResponse(BaseModel):
    """Response for get_tensor_series tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Tensor tag name")
    points: list[TensorPoint] = Field(description="Time series data points")


class HistogramPoint(BaseModel):
    """A single histogram data point."""

    step: int = Field(description="Global step value")
    values: list[float] = Field(description="Histogram bucket values")
    wall_time: datetime | None = Field(default=None, description="Wall clock time")


class HistogramSeriesResponse(BaseModel):
    """Response for get_histogram_series tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Histogram tag name")
    points: list[HistogramPoint] = Field(description="Histogram time series data")


class DistributionPoint(BaseModel):
    """A single distribution (compressed histogram) data point."""

    step: int = Field(description="Global step value")
    basis_points: list[int] = Field(description="Basis points (x-axis labels)")
    values: list[float] = Field(description="Cumulative distribution values")
    wall_time: datetime | None = Field(default=None, description="Wall clock time")


class DistributionSeriesResponse(BaseModel):
    """Response for get_distribution_series tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Distribution tag name")
    points: list[DistributionPoint] = Field(description="Distribution time series data")


class ImagePoint(BaseModel):
    """A single image reference in a series."""

    step: int = Field(description="Global step value")
    blob_key: str = Field(description="Blob key for fetching the image")
    wall_time: datetime | None = Field(default=None, description="Wall clock time")


class ImageSeriesResponse(BaseModel):
    """Response for get_image_series tool."""

    run: str = Field(description="Run name")
    tag: str = Field(description="Image tag name")
    points: list[ImagePoint] = Field(description="Image series references")


class ImageResponse(BaseModel):
    """Response for get_image tool."""

    blob_key: str = Field(description="Blob key that was requested")
    mime_type: str = Field(description="MIME type of the image (e.g., image/png)")
    data: str = Field(description="Base64-encoded image data")


class TagListResponse(BaseModel):
    """Response for list_*_tags tools."""

    run: str = Field(description="Run name")
    tags: list[str] = Field(description="List of tag names")


class ErrorResponse(BaseModel):
    """Error response structure."""

    error: str = Field(description="Error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
