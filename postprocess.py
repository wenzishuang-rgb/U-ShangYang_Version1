#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Identify atmospheric-river candidates independently at each analysis time.

Input IVT, uIVT and vIVT fields are passed through a trained convolutional
autoencoder. The absolute first principal component of its latent features is
segmented using local or global Otsu thresholding. Subsequent filters retain
or remove complete connected objects without changing their pixel boundaries.

The default filter uses area-integrated local anomalies. Optional filters use
joint local-anomaly and global-IVT ranks or physical object area. A tropical
domain screen and an additional area filter are also available. These scores
are object-selection diagnostics, not calibrated probabilities.

For questions, feedback or collaboration, please contact Zishuang Wen at
wenzishuang@stu.ouc.edu.cn. See README.md for affiliation and usage details.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, hsv_to_rgb
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy.ndimage import label
from skimage.filters import rank, threshold_multiotsu, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import disk

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

# Default paths; override these with command-line arguments.
DEFAULT_DATA_ROOT = Path("/Global/Wen_AR_0.25degree/IVT")
DEFAULT_MODEL_PATH = Path(
    "/Global/Wen_AR_0.25degree/AR_Model/Model_Right/checkpoints/best_model.pth"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data2/Wenzishuang/data/U_ShangYang/Objective_test/local_otsu_r48_object_anomaly_filter"
)
DEFAULT_TIMES = ("2016-06-15T12:00:00",)
INPUT_VARIABLES = ("IVT", "uIVT", "vIVT")
EARTH_RADIUS_KM = 6371.0
EPS = np.finfo(np.float32).eps


# Model architecture shared with training
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SSSARStructureModel(nn.Module):
    """Use the same convolutional architecture as the training script."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract model weights and remove optional DataParallel key prefixes."""
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must be a dictionary containing model weights.")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    else:
        state = checkpoint
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


# PCA and Otsu segmentation
def absolute_first_principal_component(
    latent: np.ndarray,
    chunk_size: int = 65536,
) -> Tuple[np.ndarray, float]:
    """Compute abs(PC1) in chunks without a full float64 spatial-feature matrix."""
    array = np.asarray(latent, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 64:
        raise ValueError(f"Expected latent shape [64, H, W]; received {array.shape}")

    channels, height, width = array.shape
    flattened = array.reshape(channels, height * width)
    sample_count = int(flattened.shape[1])
    if sample_count <= 1:
        return np.zeros((height, width), dtype=np.float32), 0.0

    mean = np.mean(flattened, axis=1, dtype=np.float64)
    covariance_sum = np.zeros((channels, channels), dtype=np.float64)
    for start in range(0, sample_count, int(chunk_size)):
        stop = min(start + int(chunk_size), sample_count)
        centered = (
            flattened[:, start:stop].astype(np.float64, copy=False) - mean[:, None]
        )
        covariance_sum += centered @ centered.T

    covariance = covariance_sum / float(sample_count - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    component = eigenvectors[:, order[0]].astype(np.float64, copy=False)

    score = np.empty(sample_count, dtype=np.float32)
    for start in range(0, sample_count, int(chunk_size)):
        stop = min(start + int(chunk_size), sample_count)
        centered = (
            flattened[:, start:stop].astype(np.float64, copy=False) - mean[:, None]
        )
        score[start:stop] = (component @ centered).astype(np.float32)

    total_variance = float(eigenvalues.sum())
    explained_ratio = (
        float(eigenvalues[0] / total_variance) if total_variance > 0 else 0.0
    )
    return np.abs(score.reshape(height, width)).astype(np.float32), explained_ratio


def robust_unit_interval(
    field: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Scale to [0, 1] using percentiles for local histogram quantization."""
    array = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(array)
    output = np.zeros_like(array, dtype=np.float32)
    if not finite.any():
        return output
    values = array[finite]
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    if upper <= lower:
        return output
    output[finite] = np.clip((array[finite] - lower) / (upper - lower), 0.0, 1.0)
    return output


def local_otsu_mask(
    field: np.ndarray,
    radius_pixels: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Apply local Otsu with edge padding in latitude and periodic longitude.

    Return the median threshold, scaled threshold map, candidate mask and scaled
    PC1 field. A radius of 48 cells gives a disk diameter of 97 cells.
    """
    radius = int(radius_pixels)
    if radius < 1:
        raise ValueError("--local-otsu-radius must be at least 1.")

    array = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(array)
    empty_mask = np.zeros_like(array, dtype=np.float32)
    empty_threshold = np.full_like(array, np.nan, dtype=np.float32)
    scaled = robust_unit_interval(array)
    if not finite.any():
        return float("nan"), empty_threshold, empty_mask, scaled
    values = array[finite]
    if float(values.max()) <= float(values.min()):
        threshold = np.full_like(array, float(values.min()), dtype=np.float32)
        threshold[~finite] = np.nan
        return float(values.min()), threshold, empty_mask, scaled

    image = np.rint(scaled * 255.0).astype(np.uint8)
    image[~finite] = 0
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="wrap")
    padded = np.pad(padded, ((radius, radius), (0, 0)), mode="edge")
    padded_threshold = rank.otsu(padded, footprint=disk(radius))
    threshold_u8 = padded_threshold[radius:-radius, radius:-radius]
    threshold_scaled = threshold_u8.astype(np.float32) / 255.0
    threshold_scaled[~finite] = np.nan
    mask = (finite & (image > threshold_u8)).astype(np.float32)
    return float(np.nanmedian(threshold_scaled)), threshold_scaled, mask, scaled


def global_otsu_mask(
    field: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one global Otsu threshold to the complete abs(PC1) field.

    Return the same four items as local_otsu_mask. The threshold map is in raw
    abs(PC1) units; the fourth item is a scaled field for diagnostic output.
    """
    array = np.asarray(field, dtype=np.float32)
    finite = np.isfinite(array)
    empty_mask = np.zeros_like(array, dtype=np.float32)
    empty_threshold = np.full_like(array, np.nan, dtype=np.float32)
    scaled = robust_unit_interval(array)
    if not finite.any():
        return float("nan"), empty_threshold, empty_mask, scaled
    values = array[finite]
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        threshold_map = np.full_like(array, minimum, dtype=np.float32)
        threshold_map[~finite] = np.nan
        return minimum, threshold_map, empty_mask, scaled
    threshold = float(threshold_otsu(values))
    threshold_map = np.full_like(array, threshold, dtype=np.float32)
    threshold_map[~finite] = np.nan
    mask = (finite & (array > threshold)).astype(np.float32)
    return threshold, threshold_map, mask, scaled


def local_excess_map(
    scaled_pc1: np.ndarray,
    local_threshold: np.ndarray,
    core_mask: np.ndarray,
) -> np.ndarray:
    """Compute positive local-threshold excess within the candidate mask.

    The excess is max(scaled_abs_PC1 - local_threshold, 0). It is used as an
    object attribute; it does not define a second pixel-level segmentation.
    """
    scaled = np.asarray(scaled_pc1, dtype=np.float32)
    threshold = np.asarray(local_threshold, dtype=np.float32)
    core = np.asarray(core_mask > 0.5)
    valid = core & np.isfinite(scaled) & np.isfinite(threshold)
    output = np.zeros_like(scaled, dtype=np.float32)
    output[valid] = np.maximum(scaled[valid] - threshold[valid], 0.0)
    return output


def empirical_percentile_map(field: np.ndarray) -> np.ndarray:
    """Return global empirical IVT percentiles at the current analysis time.

    Values lie in [0, 1]. Equal values receive their empirical midrank, so ties
    are treated consistently without imposing a fixed IVT threshold.
    """
    array = np.asarray(field, dtype=np.float64)
    output = np.full(array.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(array)
    values = array[finite]
    if values.size == 0:
        return output
    ordered = np.sort(values)
    lower = np.searchsorted(ordered, values, side="left")
    upper = np.searchsorted(ordered, values, side="right")
    output[finite] = ((lower + upper) / (2.0 * values.size)).astype(np.float32)
    return output


def empirical_midrank(values: np.ndarray) -> np.ndarray:
    """Convert object attributes to empirical midranks in [0, 1]."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    ordered = np.sort(array)
    lower = np.searchsorted(ordered, array, side="left")
    upper = np.searchsorted(ordered, array, side="right")
    return (lower + upper) / (2.0 * array.size)


# Periodic connected objects and data-adaptive filtering
def periodic_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label eight-connected objects with periodic longitude.

    Merge horizontal and diagonal neighbours across the first and last columns.
    Latitude is not periodic, and opposite poles are never connected.
    """
    binary = np.asarray(mask) > 0.5
    initial, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(initial, dtype=np.int32), 0

    parent = np.arange(count + 1, dtype=np.int32)

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        if first == 0 or second == 0:
            return
        first_root, second_root = root(int(first)), root(int(second))
        if first_root != second_root:
            parent[second_root] = first_root

    height = int(binary.shape[0])
    for row in range(height):
        if not binary[row, 0]:
            continue
        # 8-neighbour periodic seam: west/east cells may meet horizontally or
        # diagonally across the longitude boundary.
        for neighbour_row in (row - 1, row, row + 1):
            if 0 <= neighbour_row < height and binary[neighbour_row, -1]:
                union(int(initial[row, 0]), int(initial[neighbour_row, -1]))

    output = np.zeros_like(initial, dtype=np.int32)
    remap: Dict[int, int] = {}
    next_id = 1
    for old_id in np.unique(initial):
        if old_id == 0:
            continue
        new_root = root(int(old_id))
        if new_root not in remap:
            remap[new_root] = next_id
            next_id += 1
        output[initial == old_id] = remap[new_root]
    return output, next_id - 1


def grid_cell_area_km2(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Compute spherical grid-cell areas in square kilometres.

    Latitude edges are coordinate midpoints clipped to the poles. Longitude
    widths use periodic midpoint boundaries and sum to 360 degrees, including
    when the final periodic gap differs from the interior grid spacing.
    """
    if latitude.size < 2 or longitude.size < 2:
        return np.ones((latitude.size, longitude.size), dtype=np.float64)

    lat = np.asarray(latitude, dtype=np.float64)
    edges = np.empty(lat.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    edges[0] = lat[0] + 0.5 * (lat[0] - lat[1])
    edges[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    edges = np.clip(edges, -90.0, 90.0)
    south = np.minimum(edges[:-1], edges[1:])
    north = np.maximum(edges[:-1], edges[1:])
    lon = np.asarray(longitude, dtype=np.float64)
    unwrapped_lon = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    forward_width = np.diff(
        np.concatenate((unwrapped_lon, np.asarray([unwrapped_lon[0] + 360.0])))
    )
    if np.all(np.isfinite(forward_width)) and np.all(forward_width > 0.0):
        # A cell occupies half the gap to each neighbouring longitude centre.
        # The widths sum exactly to 360° even if the storage grid has a wider
        # final periodic gap.
        longitude_width = 0.5 * (forward_width + np.roll(forward_width, 1))
    else:
        # This pipeline is global and periodic.  A duplicate or malformed
        # coordinate endpoint should not make the inferred global area vanish.
        longitude_width = np.full(lon.size, 360.0 / float(lon.size), dtype=np.float64)
    latitude_factor = np.abs(np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south)))
    band_area = (
        EARTH_RADIUS_KM**2
        * latitude_factor[:, None]
        * np.deg2rad(longitude_width)[None, :]
    )
    return np.asarray(band_area, dtype=np.float64)


def area_multiotsu_filter(
    candidate_mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    retain_policy: str = "large_only",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Classify whole objects by three-class Otsu on log10(area in km^2).

    The large_only policy retains the largest class; remove_small retains the
    medium and large classes. If three classes cannot be resolved, retain all
    objects and record the reason. Object boundaries are unchanged.
    """
    if retain_policy not in {"large_only", "remove_small"}:
        raise ValueError(f"Unknown area retention policy: {retain_policy}")
    labels, object_count = periodic_connected_components(candidate_mask)
    class_map = np.zeros(candidate_mask.shape, dtype=np.int8)
    score_map = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    info: Dict[str, Any] = {
        "method": f"physical_area_multiotsu_{retain_policy}",
        "candidate_objects": int(object_count),
        "retained_objects": int(object_count),
        "removed_objects": 0,
        "applied": False,
        "reason": "not_evaluated",
        "thresholds_log10_km2": None,
        "thresholds_km2": None,
    }
    regions = list(regionprops(labels))
    cell_area = grid_cell_area_km2(latitude, longitude)
    physical_areas = np.asarray(
        [
            float(np.sum(cell_area[region.coords[:, 0], region.coords[:, 1]]))
            for region in regions
        ],
        dtype=np.float64,
    )
    log_areas = np.log10(np.maximum(physical_areas, EPS))

    def retain_all(reason: str) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict[str, Any]],
        Dict[str, Any],
    ]:
        """Retain all objects when a three-class area split is unavailable.

        CSV rows explicitly use unclassified_retained. The display class indicates
        retention, not a successful assignment to the largest area class.
        """
        output = np.asarray(candidate_mask, dtype=np.float32)
        fallback_rows: List[Dict[str, Any]] = []
        for region, physical_area, log_area in zip(regions, physical_areas, log_areas):
            yy, xx = region.coords[:, 0], region.coords[:, 1]
            class_map[yy, xx] = 3
            score_map[yy, xx] = float(log_area)
            fallback_rows.append(
                {
                    "candidate_object_id": int(region.label),
                    "pixel_count": int(region.area),
                    "physical_area_km2": float(physical_area),
                    "log10_physical_area_km2": float(log_area),
                    "area_class": 3,
                    "area_class_name": "unclassified_retained",
                    "multiotsu_thresholds_log10_km2": "",
                    "multiotsu_thresholds_km2": "",
                    "area_retain_policy": retain_policy,
                    "retained_by_area_multiotsu": 1,
                }
            )
        info.update({"reason": reason})
        return output, class_map, score_map, fallback_rows, info

    if object_count < 3:
        return retain_all("too_few_candidate_objects_for_multiotsu")
    if np.unique(log_areas).size < 3:
        return retain_all("too_few_unique_object_areas_for_multiotsu")

    try:
        thresholds = np.asarray(
            threshold_multiotsu(log_areas, classes=3), dtype=np.float64
        )
    except ValueError:
        return retain_all("multiotsu_could_not_split_object_areas")

    area_classes = np.digitize(log_areas, thresholds, right=False)
    retained_classes = {2} if retain_policy == "large_only" else {1, 2}
    retained_labels = {
        int(region.label)
        for region, area_class in zip(regions, area_classes)
        if int(area_class) in retained_classes
    }
    if not retained_labels or len(retained_labels) == object_count:
        return retain_all("multiotsu_did_not_create_nonempty_retained_area_class")

    output = np.zeros_like(candidate_mask, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for region, physical_area, log_area, area_class in zip(
        regions, physical_areas, log_areas, area_classes
    ):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        class_map[yy, xx] = int(area_class) + 1
        score_map[yy, xx] = float(log_area)
        retained = int(region.label) in retained_labels
        if retained:
            output[yy, xx] = 1.0
        rows.append(
            {
                "candidate_object_id": int(region.label),
                "pixel_count": int(region.area),
                "physical_area_km2": float(physical_area),
                "log10_physical_area_km2": float(log_area),
                "area_class": int(area_class + 1),
                "area_class_name": ("small", "medium", "large")[int(area_class)],
                "multiotsu_thresholds_log10_km2": ";".join(
                    f"{x:.8g}" for x in thresholds
                ),
                "multiotsu_thresholds_km2": ";".join(
                    f"{10.0 ** x:.8g}" for x in thresholds
                ),
                "area_retain_policy": retain_policy,
                "retained_by_area_multiotsu": int(retained),
            }
        )
    info.update(
        {
            "retained_objects": int(len(retained_labels)),
            "removed_objects": int(object_count - len(retained_labels)),
            "applied": True,
            "reason": (
                "multiotsu_three_physical_area_classes_keep_largest"
                if retain_policy == "large_only"
                else "multiotsu_three_physical_area_classes_remove_small_only"
            ),
            "retain_policy": retain_policy,
            "thresholds_log10_km2": [float(x) for x in thresholds],
            "thresholds_km2": [float(10.0**x) for x in thresholds],
        }
    )
    return output, class_map, score_map, rows, info


def integrated_anomaly_multiotsu_filter(
    candidate_mask: np.ndarray,
    local_excess: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Filter whole objects by area-integrated positive local anomalies.

    For each candidate object, sum cell_area * max(scaled_PC1 - T_local, 0).
    Apply three-class Otsu to the logarithms of these object scores and retain
    the largest class. No pixels are resegmented within retained objects.
    """
    labels, object_count = periodic_connected_components(candidate_mask)
    class_map = np.zeros(candidate_mask.shape, dtype=np.int8)
    score_map = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    info: Dict[str, Any] = {
        "method": "physical_area_weighted_local_anomaly_multiotsu",
        "candidate_objects": int(object_count),
        "retained_objects": int(object_count),
        "removed_objects": 0,
        "applied": False,
        "reason": "not_evaluated",
        "thresholds_log10_integrated_anomaly": None,
    }
    regions = list(regionprops(labels))
    if object_count < 3:
        info["reason"] = "too_few_candidate_objects_for_multiotsu"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            [],
            info,
        )

    cell_area = grid_cell_area_km2(latitude, longitude)
    excess = np.maximum(np.asarray(local_excess, dtype=np.float64), 0.0)
    physical_areas = np.asarray(
        [float(np.sum(cell_area[r.coords[:, 0], r.coords[:, 1]])) for r in regions],
        dtype=np.float64,
    )
    integrated_anomalies = np.asarray(
        [
            float(
                np.sum(
                    cell_area[r.coords[:, 0], r.coords[:, 1]]
                    * excess[r.coords[:, 0], r.coords[:, 1]]
                )
            )
            for r in regions
        ],
        dtype=np.float64,
    )
    log_scores = np.log10(np.maximum(integrated_anomalies, EPS))
    if np.unique(log_scores).size < 3:
        info["reason"] = "too_few_unique_integrated_anomaly_scores"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            [],
            info,
        )

    try:
        thresholds = np.asarray(
            threshold_multiotsu(log_scores, classes=3), dtype=np.float64
        )
    except ValueError:
        info["reason"] = "multiotsu_could_not_split_integrated_anomaly_scores"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            [],
            info,
        )

    score_classes = np.digitize(log_scores, thresholds, right=False)
    retained_labels = {
        int(region.label)
        for region, score_class in zip(regions, score_classes)
        if int(score_class) == 2
    }
    if not retained_labels or len(retained_labels) == object_count:
        info["reason"] = "multiotsu_did_not_create_nonempty_largest_score_class"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            [],
            info,
        )

    output = np.zeros_like(candidate_mask, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for region, physical_area, integrated_anomaly, log_score, score_class in zip(
        regions, physical_areas, integrated_anomalies, log_scores, score_classes
    ):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        class_map[yy, xx] = int(score_class) + 1
        score_map[yy, xx] = float(log_score)
        retained = int(region.label) in retained_labels
        if retained:
            output[yy, xx] = 1.0
        rows.append(
            {
                "candidate_object_id": int(region.label),
                "pixel_count": int(region.area),
                "physical_area_km2": float(physical_area),
                "integrated_local_anomaly_km2": float(integrated_anomaly),
                "log10_integrated_local_anomaly": float(log_score),
                "mean_local_excess": float(np.mean(excess[yy, xx])),
                "max_local_excess": float(np.max(excess[yy, xx])),
                "anomaly_class": int(score_class + 1),
                "anomaly_class_name": ("small", "medium", "large")[int(score_class)],
                "multiotsu_thresholds_log10_integrated_anomaly": ";".join(
                    f"{x:.8g}" for x in thresholds
                ),
                "retained_by_integrated_anomaly_multiotsu": int(retained),
            }
        )
    info.update(
        {
            "retained_objects": int(len(retained_labels)),
            "removed_objects": int(object_count - len(retained_labels)),
            "applied": True,
            "reason": "multiotsu_three_integrated_anomaly_classes_keep_largest",
            "thresholds_log10_integrated_anomaly": [float(x) for x in thresholds],
            "thresholds_integrated_anomaly_km2": [float(10.0**x) for x in thresholds],
        }
    )
    return output, class_map, score_map, rows, info


def global_local_rank_multiotsu_filter(
    candidate_mask: np.ndarray,
    local_excess: np.ndarray,
    ivt: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Filter whole objects using local anomalies and global IVT support.

    For each object, calculate its area-integrated positive local anomaly and
    its area-weighted mean global empirical IVT percentile. Convert both
    attributes to midranks across objects at the current time, and use their
    minimum as a joint score. Three-class Otsu retains the highest score class.

    This score is not a Bayesian posterior. Selection favours candidates with
    both forms of support and does not establish the absence of weak real ARs.
    """
    labels, object_count = periodic_connected_components(candidate_mask)
    class_map = np.zeros(candidate_mask.shape, dtype=np.int8)
    score_map = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    global_support_map = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    local_rank_map = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    global_ivt_percentile = empirical_percentile_map(ivt)
    info: Dict[str, Any] = {
        "method": "whole_object_global_local_rank_multiotsu",
        "candidate_objects": int(object_count),
        "retained_objects": int(object_count),
        "removed_objects": 0,
        "applied": False,
        "reason": "not_evaluated",
        "thresholds_joint_rank": None,
        "pixel_resegmentation_after_first_otsu": False,
    }
    regions = list(regionprops(labels))
    if object_count < 3:
        info["reason"] = "too_few_candidate_objects_for_multiotsu"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            global_support_map,
            local_rank_map,
            global_ivt_percentile,
            [],
            info,
        )

    cell_area = grid_cell_area_km2(latitude, longitude)
    excess = np.maximum(np.asarray(local_excess, dtype=np.float64), 0.0)
    physical_areas = np.asarray(
        [float(np.sum(cell_area[r.coords[:, 0], r.coords[:, 1]])) for r in regions],
        dtype=np.float64,
    )
    local_masses = np.asarray(
        [
            float(
                np.sum(
                    cell_area[r.coords[:, 0], r.coords[:, 1]]
                    * excess[r.coords[:, 0], r.coords[:, 1]]
                )
            )
            for r in regions
        ],
        dtype=np.float64,
    )
    global_supports = np.asarray(
        [
            float(
                np.sum(
                    cell_area[r.coords[:, 0], r.coords[:, 1]]
                    * global_ivt_percentile[r.coords[:, 0], r.coords[:, 1]]
                )
                / max(physical_area, EPS)
            )
            for r, physical_area in zip(regions, physical_areas)
        ],
        dtype=np.float64,
    )
    local_ranks = empirical_midrank(np.log10(np.maximum(local_masses, EPS)))
    global_ranks = empirical_midrank(global_supports)
    joint_scores = np.minimum(local_ranks, global_ranks)

    if np.unique(joint_scores).size < 3:
        info["reason"] = "too_few_unique_global_local_object_scores_for_multiotsu"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            global_support_map,
            local_rank_map,
            global_ivt_percentile,
            [],
            info,
        )

    try:
        thresholds = np.asarray(
            threshold_multiotsu(joint_scores, classes=3), dtype=np.float64
        )
    except ValueError:
        info["reason"] = "multiotsu_could_not_split_global_local_object_scores"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            global_support_map,
            local_rank_map,
            global_ivt_percentile,
            [],
            info,
        )

    score_classes = np.digitize(joint_scores, thresholds, right=False)
    retained_labels = {
        int(region.label)
        for region, score_class in zip(regions, score_classes)
        if int(score_class) == 2
    }
    if not retained_labels or len(retained_labels) == object_count:
        info["reason"] = "multiotsu_did_not_create_nonempty_largest_joint_score_class"
        return (
            np.asarray(candidate_mask, dtype=np.float32),
            class_map,
            score_map,
            global_support_map,
            local_rank_map,
            global_ivt_percentile,
            [],
            info,
        )

    output = np.zeros_like(candidate_mask, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for (
        region,
        physical_area,
        local_mass,
        global_support,
        local_rank,
        global_rank,
        joint_score,
        score_class,
    ) in zip(
        regions,
        physical_areas,
        local_masses,
        global_supports,
        local_ranks,
        global_ranks,
        joint_scores,
        score_classes,
    ):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        class_map[yy, xx] = int(score_class) + 1
        score_map[yy, xx] = float(joint_score)
        global_support_map[yy, xx] = float(global_support)
        local_rank_map[yy, xx] = float(local_rank)
        retained = int(region.label) in retained_labels
        if retained:
            output[yy, xx] = 1.0
        rows.append(
            {
                "candidate_object_id": int(region.label),
                "pixel_count": int(region.area),
                "physical_area_km2": float(physical_area),
                "integrated_local_anomaly_km2": float(local_mass),
                "local_anomaly_rank": float(local_rank),
                "global_ivt_percentile_mean": float(global_support),
                "global_ivt_support_rank": float(global_rank),
                "joint_global_local_rank_score": float(joint_score),
                "joint_score_class": int(score_class + 1),
                "joint_score_class_name": ("small", "medium", "large")[
                    int(score_class)
                ],
                "multiotsu_thresholds_joint_rank": ";".join(
                    f"{x:.8g}" for x in thresholds
                ),
                "retained_by_global_local_rank_multiotsu": int(retained),
            }
        )
    info.update(
        {
            "retained_objects": int(len(retained_labels)),
            "removed_objects": int(object_count - len(retained_labels)),
            "applied": True,
            "reason": "three_object_classes_from_global_and_local_evidence_keep_largest",
            "thresholds_joint_rank": [float(x) for x in thresholds],
            "joint_score_formula": "min(rank(integrated_local_anomaly), rank(global_area_weighted_IVT_percentile))",
        }
    )
    return (
        output,
        class_map,
        score_map,
        global_support_map,
        local_rank_map,
        global_ivt_percentile,
        rows,
        info,
    )


def tropical_filter(
    mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    minimum_latitude: float,
    maximum_latitude: float,
    removal_fraction: float,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Remove whole objects whose tropical area fraction exceeds the cutoff.

    Use spherical grid-cell areas for both the tropical and total object area.
    """
    labels, _ = periodic_connected_components(mask)
    tropical_latitudes = (latitude >= minimum_latitude) & (latitude <= maximum_latitude)
    cell_area = grid_cell_area_km2(latitude, longitude)
    output = np.zeros_like(mask, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for region in regionprops(labels):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        physical_area = float(np.sum(cell_area[yy, xx]))
        tropical_count = int(tropical_latitudes[yy].sum())
        tropical_physical_area = float(
            np.sum(cell_area[yy, xx] * tropical_latitudes[yy])
        )
        fraction = float(tropical_physical_area / max(physical_area, EPS))
        removed = bool(enabled and fraction > removal_fraction)
        if not removed:
            output[yy, xx] = 1.0
        rows.append(
            {
                "pre_tropical_object_id": int(region.label),
                "pixel_count": int(region.area),
                "physical_area_km2": physical_area,
                "tropical_pixel_count": tropical_count,
                "tropical_physical_area_km2": tropical_physical_area,
                "tropical_fraction": fraction,
                "tropical_latitude_min": float(minimum_latitude),
                "tropical_latitude_max": float(maximum_latitude),
                "tropical_removal_fraction": float(removal_fraction),
                "removed_by_tropical_filter": int(removed),
                "retained_after_tropical_filter": int(not removed),
            }
        )
    return output, labels, rows


def whole_object_selection_diagnostics(
    first_object_mask: np.ndarray,
    selected_mask: np.ndarray,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Check that each initial candidate object is retained or removed whole.

    The selected pixel count must be zero or the original object's full size.
    Raise an error if a later filter has retained only part of an object.
    """
    first_labels, first_count = periodic_connected_components(first_object_mask)
    selected = np.asarray(selected_mask) > 0.5
    partial_ids: List[int] = []
    retained_ids: List[int] = []
    for region in regionprops(first_labels):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        selected_count = int(selected[yy, xx].sum())
        if selected_count == int(region.area):
            retained_ids.append(int(region.label))
        elif selected_count != 0:
            partial_ids.append(int(region.label))

    if partial_ids:
        raise RuntimeError(
            f"{stage} violates whole-object selection: objects {partial_ids[:12]} were partially removed. "
            "Later filters must not resegment pixels within first-Otsu objects."
        )
    return {
        "stage": stage,
        "first_otsu_objects": int(first_count),
        "whole_objects_retained": int(len(retained_ids)),
        "whole_objects_removed": int(first_count - len(retained_ids)),
        "partially_modified_object_count": 0,
        "integrity_check_passed": True,
    }


# Input data, object diagnostics and plotting
def load_single_time(data_root: Path, target_time: str) -> Dict[str, Any]:
    target_datetime = datetime.fromisoformat(target_time)
    year, month = target_datetime.year, target_datetime.month
    input_path = data_root / str(year) / f"IVT_{year}_{month:02d}.nc"
    if not input_path.exists():
        raise FileNotFoundError(f"Monthly IVT file not found: {input_path}")

    target_minute = np.datetime64(target_datetime, "m")
    with xr.open_dataset(input_path, engine="netcdf4") as dataset:
        rename: Dict[str, str] = {}
        if "lat" in dataset.dims:
            rename["lat"] = "latitude"
        if "lon" in dataset.dims:
            rename["lon"] = "longitude"
        if rename:
            dataset = dataset.rename(rename)
        required = set(INPUT_VARIABLES) | {"time", "latitude", "longitude"}
        missing = required - set(dataset.variables)
        if missing:
            raise KeyError(
                "Missing variables or coordinates: " + ", ".join(sorted(missing))
            )

        times = np.asarray(dataset["time"].values)
        matches = np.flatnonzero(times.astype("datetime64[m]") == target_minute)
        if matches.size != 1:
            available = ", ".join(np.datetime_as_string(x, unit="m") for x in times[:8])
            raise ValueError(
                f"No unique match for target time {target_time}; first available times: {available}"
            )
        time_index = int(matches[0])
        latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
        longitude = np.asarray(dataset["longitude"].values, dtype=np.float64)
        fields: Dict[str, np.ndarray] = {}
        for variable in INPUT_VARIABLES:
            data = (
                dataset[variable]
                .isel(time=time_index)
                .transpose("latitude", "longitude")
            )
            values = np.asarray(data.values, dtype=np.float32)
            if values.shape != (latitude.size, longitude.size):
                raise ValueError(f"{variable} has an unexpected shape: {values.shape}")
            fields[variable] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "input_path": str(input_path),
        "matched_time": times[time_index],
        "latitude": latitude,
        "longitude": longitude,
        "ivt": fields["IVT"],
        "uivt": fields["uIVT"],
        "vivt": fields["vIVT"],
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def object_rows(
    labels: np.ndarray,
    ivt: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> List[Dict[str, Any]]:
    """Summarize final objects and their IVT and geographic properties.

    Report periodic seam crossing and minimum circular longitude span because
    ordinary longitude minima and maxima can misrepresent seam-crossing objects.
    These properties are diagnostic and do not affect object selection.
    """
    cell_area = grid_cell_area_km2(latitude, longitude)
    rows: List[Dict[str, Any]] = []
    for region in regionprops(labels):
        yy, xx = region.coords[:, 0], region.coords[:, 1]
        unique_columns = np.unique(xx)
        wrapped_longitudes = np.sort(
            np.mod(np.asarray(longitude[unique_columns], dtype=np.float64), 360.0)
        )
        if wrapped_longitudes.size <= 1:
            periodic_span_degrees = 0.0
        else:
            cyclic_gaps = np.diff(
                np.concatenate((wrapped_longitudes, wrapped_longitudes[:1] + 360.0))
            )
            periodic_span_degrees = float(360.0 - np.max(cyclic_gaps))
        rows.append(
            {
                "object_id": int(region.label),
                "pixel_count": int(region.area),
                "physical_area_km2": float(np.sum(cell_area[yy, xx])),
                "mean_ivt": float(np.mean(ivt[yy, xx])),
                "max_ivt": float(np.max(ivt[yy, xx])),
                "min_latitude": float(np.min(latitude[yy])),
                "max_latitude": float(np.max(latitude[yy])),
                "min_longitude": float(np.min(longitude[xx])),
                "max_longitude": float(np.max(longitude[xx])),
                "crosses_periodic_longitude_boundary": int(
                    bool(
                        np.any(unique_columns == 0)
                        and np.any(unique_columns == longitude.size - 1)
                    )
                ),
                "periodic_longitude_span_degrees": periodic_span_degrees,
            }
        )
    return rows


def _subplot_kwargs() -> Dict[str, Any]:
    if HAS_CARTOPY:
        return {"projection": ccrs.PlateCarree(central_longitude=180)}
    return {}


def periodic_plot_arrays(
    longitude: np.ndarray,
    field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Append a cyclic first column for plotting without changing saved data.

    The extra column closes the longitude seam without interpolation. It is used
    only for rendering, not for segmentation, object labelling or NetCDF output.
    """
    lon = np.asarray(longitude, dtype=np.float64)
    values = np.asarray(field)
    if values.ndim != 2 or values.shape[1] != lon.size or lon.size < 2:
        return lon, values

    cyclic_longitude = float(lon[0] + 360.0)
    # Do not append a second cyclic endpoint.
    if np.isclose(lon[-1], cyclic_longitude, atol=1.0e-8, rtol=0.0):
        return lon, values
    return (
        np.concatenate((lon, np.asarray([cyclic_longitude], dtype=np.float64))),
        np.concatenate((values, values[:, :1]), axis=1),
    )


def draw_field(
    axis: Any,
    longitude: np.ndarray,
    latitude: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: Any,
    *,
    norm: Optional[Any] = None,
    show_colorbar: bool = True,
) -> Any:
    kwargs: Dict[str, Any] = {"shading": "auto", "cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm
    plot_longitude, plot_field = periodic_plot_arrays(longitude, field)
    if HAS_CARTOPY:
        mesh = axis.pcolormesh(
            plot_longitude,
            latitude,
            plot_field,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )
        axis.coastlines(linewidth=0.45)
        axis.set_global()
    else:
        mesh = axis.pcolormesh(plot_longitude, latitude, plot_field, **kwargs)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    axis.set_title(title, fontsize=10)
    if show_colorbar:
        axis.figure.colorbar(mesh, ax=axis, shrink=0.72)
    return mesh


def object_colormap(object_count: int) -> Tuple[ListedColormap, BoundaryNorm]:
    if object_count <= 0:
        colors = np.asarray([[0.0, 0.0, 0.0, 1.0]])
    else:
        hue = np.linspace(0.0, 1.0, object_count, endpoint=False)
        hsv = np.column_stack(
            (hue, np.full(object_count, 0.72), np.full(object_count, 0.92))
        )
        rgb = hsv_to_rgb(hsv)
        colors = np.vstack(
            (
                np.asarray([[0.0, 0.0, 0.0, 1.0]]),
                np.column_stack((rgb, np.ones(object_count))),
            )
        )
    cmap = ListedColormap(colors, name="AR_objects")
    norm = BoundaryNorm(np.arange(-0.5, object_count + 1.5, 1.0), cmap.N)
    return cmap, norm


def draw_object_ids(
    axis: Any,
    longitude: np.ndarray,
    latitude: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    count = int(np.max(labels)) if labels.size else 0
    cmap, norm = object_colormap(count)
    draw_field(
        axis,
        longitude,
        latitude,
        labels,
        f"{title} ({count} objects)",
        cmap,
        norm=norm,
        show_colorbar=False,
    )
    axis.text(
        0.01,
        0.02,
        f"{count} periodic connected objects",
        transform=axis.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
    )


def save_figure(
    path: Path,
    *,
    longitude: np.ndarray,
    latitude: np.ndarray,
    ivt: np.ndarray,
    manifold: np.ndarray,
    first_threshold: np.ndarray,
    first_threshold_title: str,
    first_core: np.ndarray,
    object_attribute_map: np.ndarray,
    object_attribute_title: str,
    object_score_map: np.ndarray,
    object_score_title: str,
    object_classes: np.ndarray,
    object_classes_title: str,
    after_primary_filter: np.ndarray,
    pre_tropical_labels: np.ndarray,
    after_tropical_filter: np.ndarray,
    area_postfilter_score_map: np.ndarray,
    area_postfilter_score_title: str,
    area_postfilter_classes: np.ndarray,
    area_postfilter_classes_title: str,
    final_mask: np.ndarray,
    final_labels: np.ndarray,
    ivt_inside_final: np.ndarray,
    summary_text: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(28, 19), subplot_kw=_subplot_kwargs())
    axes = axes.ravel()
    draw_field(axes[0], longitude, latitude, ivt, "IVT", "viridis")
    draw_field(axes[1], longitude, latitude, manifold, "abs(PC1)", "magma")
    draw_field(
        axes[2], longitude, latitude, first_threshold, first_threshold_title, "magma"
    )
    draw_field(
        axes[3],
        longitude,
        latitude,
        first_core,
        "PCA + first local Otsu objects",
        "gray",
    )
    draw_field(
        axes[4],
        longitude,
        latitude,
        object_attribute_map,
        object_attribute_title,
        "plasma",
    )
    draw_field(
        axes[5], longitude, latitude, object_score_map, object_score_title, "plasma"
    )

    area_cmap = ListedColormap(["black", "#d73027", "#fdae61", "#1a9850"])
    area_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), area_cmap.N)
    class_mesh = draw_field(
        axes[6],
        longitude,
        latitude,
        object_classes,
        object_classes_title,
        area_cmap,
        norm=area_norm,
        show_colorbar=False,
    )
    class_colorbar = fig.colorbar(
        class_mesh, ax=axes[6], shrink=0.72, ticks=[0, 1, 2, 3]
    )
    class_colorbar.ax.set_yticklabels(["background", "small", "medium", "large"])
    draw_field(
        axes[7],
        longitude,
        latitude,
        after_primary_filter,
        "After primary whole-object filter",
        "gray",
    )
    draw_object_ids(
        axes[8], longitude, latitude, pre_tropical_labels, "Before tropical filter"
    )
    draw_field(
        axes[9],
        longitude,
        latitude,
        after_tropical_filter,
        "After tropical filter; before area filter",
        "gray",
    )
    draw_field(
        axes[10],
        longitude,
        latitude,
        area_postfilter_score_map,
        area_postfilter_score_title,
        "plasma",
    )
    area_mesh = draw_field(
        axes[11],
        longitude,
        latitude,
        area_postfilter_classes,
        area_postfilter_classes_title,
        area_cmap,
        norm=area_norm,
        show_colorbar=False,
    )
    area_colorbar = fig.colorbar(
        area_mesh, ax=axes[11], shrink=0.72, ticks=[0, 1, 2, 3]
    )
    area_colorbar.ax.set_yticklabels(["background", "small", "medium", "large"])
    draw_field(
        axes[12], longitude, latitude, final_mask, "Final AR-candidate mask", "gray"
    )
    draw_object_ids(
        axes[13], longitude, latitude, final_labels, "Final AR-candidate objects"
    )
    draw_field(
        axes[14],
        longitude,
        latitude,
        ivt_inside_final,
        "IVT inside final candidates",
        "Blues",
    )
    axes[15].axis("off")
    axes[15].text(
        0.03,
        0.97,
        summary_text,
        transform=axes[15].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        wrap=True,
        bbox={"facecolor": "#f4f4f4", "edgecolor": "0.55", "pad": 10},
    )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# Detection and output for each requested analysis time
def output_tag(time_value: np.datetime64) -> str:
    return np.datetime_as_string(time_value, unit="m").replace("-", "").replace(":", "")


def detect_one_time(
    model: nn.Module,
    device: torch.device,
    target_time: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    item = load_single_time(Path(args.data_root), target_time)
    ivt, uivt, vivt = item["ivt"], item["uivt"], item["vivt"]
    latitude, longitude = item["latitude"], item["longitude"]
    matched_time = np.datetime64(item["matched_time"], "ns")

    input_tensor = torch.from_numpy(np.stack((ivt, uivt, vivt), axis=0)[None]).to(
        device
    )
    with torch.inference_mode():
        _, latent = model(input_tensor)
    latent_array = latent[0].detach().cpu().numpy()
    del latent, input_tensor

    manifold, explained_variance_ratio = absolute_first_principal_component(
        latent_array, chunk_size=args.pca_chunk_size
    )
    if args.otsu_mode == "local":
        first_threshold, first_threshold_map, core_mask, scaled_pc1 = local_otsu_mask(
            manifold, args.local_otsu_radius
        )
        first_threshold_title = "First local Otsu threshold (scaled)"
        first_method = "local_rank_otsu"
    else:
        first_threshold, first_threshold_map, core_mask, scaled_pc1 = global_otsu_mask(
            manifold
        )
        first_threshold_title = "First global Otsu threshold (abs PC1)"
        first_method = "global_otsu"

    # All subsequent filters operate on complete first-Otsu objects.
    # Retain or remove each object without re-thresholding its pixels.
    local_excess = local_excess_map(scaled_pc1, first_threshold_map, core_mask)
    candidate_mask = np.asarray(core_mask, dtype=np.float32)

    empty_object_metric = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    object_global_support_map = empty_object_metric.copy()
    object_local_rank_map = empty_object_metric.copy()
    global_ivt_percentile = empty_object_metric.copy()

    if args.object_filter == "integrated-anomaly-multiotsu":
        (
            after_object_filter,
            object_classes,
            object_score_map,
            filter_rows,
            object_info,
        ) = integrated_anomaly_multiotsu_filter(
            candidate_mask, local_excess, latitude, longitude
        )
        object_score_title = "log10 integrated local anomaly per object"
        object_classes_title = (
            "Object integrated-anomaly multi-Otsu: small / medium / large"
        )
        object_attribute_map = local_excess
        object_attribute_title = "Local excess inside full objects"
    elif args.object_filter == "area-multiotsu":
        (
            after_object_filter,
            object_classes,
            object_score_map,
            filter_rows,
            object_info,
        ) = area_multiotsu_filter(candidate_mask, latitude, longitude)
        object_score_title = "log10 physical area per object (km²)"
        object_classes_title = "Object physical-area multi-Otsu: small / medium / large"
        object_attribute_map = local_excess
        object_attribute_title = "Local excess inside full objects"
    elif args.object_filter == "global-local-rank-multiotsu":
        (
            after_object_filter,
            object_classes,
            object_score_map,
            object_global_support_map,
            object_local_rank_map,
            global_ivt_percentile,
            filter_rows,
            object_info,
        ) = global_local_rank_multiotsu_filter(
            candidate_mask, local_excess, ivt, latitude, longitude
        )
        object_score_title = "Joint rank = min(local anomaly, global IVT)"
        object_classes_title = (
            "Object global-local rank multi-Otsu: small / medium / large"
        )
        object_attribute_map = global_ivt_percentile
        object_attribute_title = "Global empirical IVT percentile"
    else:
        after_object_filter = candidate_mask
        candidate_labels, candidate_count = periodic_connected_components(
            candidate_mask
        )
        object_classes = np.where(candidate_labels > 0, 3, 0).astype(np.int8)
        object_score_map = np.where(candidate_labels > 0, local_excess, np.nan).astype(
            np.float32
        )
        filter_rows = []
        object_info = {
            "method": "none",
            "candidate_objects": int(candidate_count),
            "retained_objects": int(candidate_count),
            "removed_objects": 0,
            "applied": False,
            "reason": "object_filter_none_keep_full_first_otsu_objects",
        }
        object_score_title = "Local excess (object filter disabled)"
        object_classes_title = "All first-Otsu objects retained"
        object_attribute_map = local_excess
        object_attribute_title = "Local excess inside full objects"

    integrity_after_primary_filter = whole_object_selection_diagnostics(
        candidate_mask,
        after_object_filter,
        stage="after_primary_whole_object_filter",
    )
    after_tropical_filter, pre_tropical_labels, tropical_rows = tropical_filter(
        after_object_filter,
        latitude,
        longitude,
        minimum_latitude=args.tropical_latitude_min,
        maximum_latitude=args.tropical_latitude_max,
        removal_fraction=args.tropical_removal_fraction,
        enabled=not args.disable_tropical_filter,
    )
    integrity_after_tropical_filter = whole_object_selection_diagnostics(
        candidate_mask,
        after_tropical_filter,
        stage="after_tropical_filter",
    )

    if args.area_postfilter == "multiotsu-remove-small":
        (
            final_mask,
            area_postfilter_classes,
            area_postfilter_score_map,
            area_postfilter_rows,
            area_postfilter_info,
        ) = area_multiotsu_filter(
            after_tropical_filter,
            latitude,
            longitude,
            retain_policy="remove_small",
        )
        area_postfilter_score_title = "log10 physical area after tropical filter (km²)"
        area_postfilter_classes_title = (
            "Post-tropical area multi-Otsu: remove small class"
        )
    elif args.area_postfilter == "multiotsu-keep-large":
        (
            final_mask,
            area_postfilter_classes,
            area_postfilter_score_map,
            area_postfilter_rows,
            area_postfilter_info,
        ) = area_multiotsu_filter(
            after_tropical_filter,
            latitude,
            longitude,
            retain_policy="large_only",
        )
        area_postfilter_score_title = "log10 physical area after tropical filter (km²)"
        area_postfilter_classes_title = (
            "Post-tropical area multi-Otsu: keep large class"
        )
    else:
        final_mask = np.asarray(after_tropical_filter, dtype=np.float32)
        tropical_labels, tropical_count = periodic_connected_components(
            after_tropical_filter
        )
        area_postfilter_classes = np.where(tropical_labels > 0, 3, 0).astype(np.int8)
        area_postfilter_score_map = np.full(
            candidate_mask.shape, np.nan, dtype=np.float32
        )
        area_postfilter_rows = []
        area_postfilter_info = {
            "method": "none",
            "candidate_objects": int(tropical_count),
            "retained_objects": int(tropical_count),
            "removed_objects": 0,
            "applied": False,
            "reason": "area_postfilter_none_keep_all_post_tropical_objects",
        }
        area_postfilter_score_title = "Physical-area postfilter disabled"
        area_postfilter_classes_title = "All post-tropical objects retained"

    integrity_after_area_postfilter = whole_object_selection_diagnostics(
        candidate_mask,
        final_mask,
        stage="after_area_postfilter",
    )
    final_labels, final_object_count = periodic_connected_components(final_mask)
    ivt_inside_final = (ivt * final_mask).astype(np.float32)

    tag = output_tag(matched_time)
    filter_tag = (
        str(args.object_filter)
        if args.area_postfilter == "none"
        else f"{args.object_filter}_{args.area_postfilter}"
    )
    run_stem = f"ar_{tag}_{filter_tag}"
    output_root = Path(args.output_root)
    data_dir = output_root / "data"
    figure_dir = output_root / "figures"
    diagnostics_dir = output_root / "diagnostics"
    objects_dir = output_root / "objects"
    for directory in (data_dir, figure_dir, diagnostics_dir, objects_dir):
        directory.mkdir(parents=True, exist_ok=True)

    final_rows = object_rows(final_labels, ivt, latitude, longitude)
    # Include filter names in output filenames to distinguish filter settings.
    final_object_csv = objects_dir / f"{run_stem}_final_objects.csv"
    filter_csv = diagnostics_dir / f"{run_stem}_object_filter.csv"
    tropical_csv = diagnostics_dir / f"{run_stem}_tropical_filter.csv"
    area_postfilter_csv = diagnostics_dir / f"{run_stem}_area_postfilter.csv"
    write_csv(final_object_csv, final_rows)
    write_csv(filter_csv, filter_rows)
    write_csv(tropical_csv, tropical_rows)
    write_csv(area_postfilter_csv, area_postfilter_rows)

    result = xr.Dataset(
        data_vars={
            "IVT": (("latitude", "longitude"), ivt.astype(np.float32)),
            "uIVT": (("latitude", "longitude"), uivt.astype(np.float32)),
            "vIVT": (("latitude", "longitude"), vivt.astype(np.float32)),
            "abs_PC1": (("latitude", "longitude"), manifold.astype(np.float32)),
            "scaled_abs_PC1": (
                ("latitude", "longitude"),
                scaled_pc1.astype(np.float32),
            ),
            "first_otsu_threshold": (
                ("latitude", "longitude"),
                first_threshold_map.astype(np.float32),
            ),
            "first_otsu_core": (("latitude", "longitude"), core_mask.astype(np.int8)),
            "local_excess_inside_first_otsu_object": (
                ("latitude", "longitude"),
                local_excess.astype(np.float32),
            ),
            "grid_cell_area_km2": (
                ("latitude", "longitude"),
                grid_cell_area_km2(latitude, longitude).astype(np.float32),
            ),
            "object_filter_score": (
                ("latitude", "longitude"),
                object_score_map.astype(np.float32),
            ),
            "object_filter_class": (
                ("latitude", "longitude"),
                object_classes.astype(np.int8),
            ),
            "global_ivt_empirical_percentile": (
                ("latitude", "longitude"),
                global_ivt_percentile.astype(np.float32),
            ),
            "object_global_ivt_support": (
                ("latitude", "longitude"),
                object_global_support_map.astype(np.float32),
            ),
            "object_local_anomaly_rank": (
                ("latitude", "longitude"),
                object_local_rank_map.astype(np.float32),
            ),
            "after_whole_object_filter": (
                ("latitude", "longitude"),
                after_object_filter.astype(np.int8),
            ),
            "after_tropical_filter_before_area": (
                ("latitude", "longitude"),
                after_tropical_filter.astype(np.int8),
            ),
            "object_area_postfilter_score_log10_km2": (
                ("latitude", "longitude"),
                area_postfilter_score_map.astype(np.float32),
            ),
            "object_area_postfilter_class": (
                ("latitude", "longitude"),
                area_postfilter_classes.astype(np.int8),
            ),
            "after_automatic_area_filter": (
                ("latitude", "longitude"),
                final_mask.astype(np.int8),
            ),
            "pre_tropical_object_id": (
                ("latitude", "longitude"),
                pre_tropical_labels.astype(np.int32),
            ),
            "final_ar_candidate_mask": (
                ("latitude", "longitude"),
                final_mask.astype(np.int8),
            ),
            "final_ar_candidate_object_id": (
                ("latitude", "longitude"),
                final_labels.astype(np.int32),
            ),
            "IVT_inside_final_AR_candidate": (
                ("latitude", "longitude"),
                ivt_inside_final,
            ),
        },
        coords={"latitude": latitude, "longitude": longitude},
        attrs={
            "matched_time": str(matched_time),
            "model": f"{args.model_path.name}; 3-64-64-64-3 CNN",
            "recognition": (
                "CNN latent -> PCA -> abs(PC1) -> first global/local Otsu -> "
                "periodic connected candidate objects -> primary whole-object filter "
                "-> tropical domain screen -> optional whole-object physical-area "
                "postfilter"
            ),
            "first_otsu_mode": str(args.otsu_mode),
            "local_otsu_radius_pixels": (
                int(args.local_otsu_radius) if args.otsu_mode == "local" else -1
            ),
            "local_otsu_disk_diameter_pixels": (
                int(2 * args.local_otsu_radius + 1) if args.otsu_mode == "local" else -1
            ),
            "pixel_resegmentation_after_first_otsu": "no",
            "object_operation_rule": "Every first-Otsu periodic connected object is retained or deleted as a whole; its pixels are never re-thresholded.",
            "whole_object_integrity_checked": "yes",
            "bayesian_posterior_used": "no",
            "shape_filter_used": "no",
            "object_filter": str(args.object_filter),
            "object_filter_score_description": str(object_score_title),
            "global_ivt_support_used": (
                "yes" if args.object_filter == "global-local-rank-multiotsu" else "no"
            ),
            "area_postfilter": str(args.area_postfilter),
            "area_postfilter_score_description": str(area_postfilter_score_title),
            "periodic_longitude_plotting": "cyclic first-column closure for rendering only; source data unchanged",
            "tropical_filter_enabled": int(not args.disable_tropical_filter),
            "tropical_latitude_min": float(args.tropical_latitude_min),
            "tropical_latitude_max": float(args.tropical_latitude_max),
        },
    )
    netcdf_path = data_dir / f"{run_stem}.nc"
    result.to_netcdf(netcdf_path)
    np.save(data_dir / f"{run_stem}_final_mask.npy", final_mask.astype(np.int8))

    tropical_removed = int(
        sum(int(row["removed_by_tropical_filter"]) for row in tropical_rows)
    )

    segmentation_suffix = (
        f"local_otsu_r{args.local_otsu_radius}"
        if args.otsu_mode == "local"
        else "global_otsu"
    )
    figure_path = figure_dir / f"{run_stem}_pca_{segmentation_suffix}.png"
    save_figure(
        figure_path,
        longitude=longitude,
        latitude=latitude,
        ivt=ivt,
        manifold=manifold,
        first_threshold=first_threshold_map,
        first_threshold_title=first_threshold_title,
        first_core=core_mask,
        object_attribute_map=object_attribute_map,
        object_attribute_title=object_attribute_title,
        object_score_map=object_score_map,
        object_score_title=object_score_title,
        object_classes=object_classes,
        object_classes_title=object_classes_title,
        after_primary_filter=after_object_filter,
        pre_tropical_labels=pre_tropical_labels,
        after_tropical_filter=after_tropical_filter,
        area_postfilter_score_map=area_postfilter_score_map,
        area_postfilter_score_title=area_postfilter_score_title,
        area_postfilter_classes=area_postfilter_classes,
        area_postfilter_classes_title=area_postfilter_classes_title,
        final_mask=final_mask,
        final_labels=final_labels,
        ivt_inside_final=ivt_inside_final,
        summary_text=(
            "Whole-object sequence\n\n"
            f"1. First {args.otsu_mode} Otsu objects: "
            f"{int(periodic_connected_components(candidate_mask)[1])}\n"
            f"2. Primary {args.object_filter}: "
            f"keep {object_info['retained_objects']}/"
            f"{object_info['candidate_objects']}\n"
            f"3. Tropical screen: removed {tropical_removed}/"
            f"{int(np.max(pre_tropical_labels))}\n"
            f"4. Area postfilter ({args.area_postfilter}): "
            f"keep {area_postfilter_info['retained_objects']}/"
            f"{area_postfilter_info['candidate_objects']}\n\n"
            "Integrity: every operation retains or removes a complete "
            "first-Otsu periodic connected object.\n"
            "No pixel-level re-thresholding after the first Otsu."
        ),
        title=(
            f"Objective AR candidates | PCA -> {args.otsu_mode} Otsu"
            + (
                f" (radius={args.local_otsu_radius})"
                if args.otsu_mode == "local"
                else ""
            )
            + f" -> whole-object {args.object_filter} -> tropical -> {args.area_postfilter} | "
            + str(matched_time)
        ),
    )

    diagnostics: Dict[str, Any] = {
        "requested_time": target_time,
        "matched_time": str(matched_time),
        "input_file": item["input_path"],
        "latent_shape": list(latent_array.shape),
        "pca_explained_variance_ratio": float(explained_variance_ratio),
        "first_pixel_segmentation": {
            "method": first_method,
            "radius_pixels": (
                int(args.local_otsu_radius) if args.otsu_mode == "local" else None
            ),
            "disk_diameter_pixels": (
                int(2 * args.local_otsu_radius + 1)
                if args.otsu_mode == "local"
                else None
            ),
            "representative_threshold": float(first_threshold),
            "core_pixels": int(core_mask.sum()),
        },
        "candidate_objects": {
            "source": "first global/local Otsu core",
            "periodic_connectivity": (
                "8-neighbour; longitude first/last columns joined, including "
                "diagonal seam neighbours"
            ),
            "candidate_pixels": int(candidate_mask.sum()),
            "candidate_objects": int(periodic_connected_components(candidate_mask)[1]),
            "pixel_resegmentation_after_first_otsu": False,
            "local_excess_definition": "max(scaled_abs_PC1 - local_otsu_threshold, 0), used only as an object attribute",
            "global_ivt_support_definition": (
                "area-weighted mean global empirical IVT percentile per object; "
                "used only by global-local-rank-multiotsu"
            ),
        },
        "object_filter": object_info,
        "area_postfilter": area_postfilter_info,
        "whole_object_integrity": {
            "after_primary_object_filter": integrity_after_primary_filter,
            "after_tropical_filter": integrity_after_tropical_filter,
            "after_area_postfilter": integrity_after_area_postfilter,
        },
        "tropical_filter": {
            "enabled": bool(not args.disable_tropical_filter),
            "latitude_min": float(args.tropical_latitude_min),
            "latitude_max": float(args.tropical_latitude_max),
            "removal_fraction": float(args.tropical_removal_fraction),
            "pre_tropical_objects": int(np.max(pre_tropical_labels)),
            "removed_objects": tropical_removed,
        },
        "final_pixels": int(final_mask.sum()),
        "final_objects": int(final_object_count),
        "outputs": {
            "netcdf": str(netcdf_path),
            "figure": str(figure_path),
            "final_object_csv": str(final_object_csv),
            "object_filter_csv": str(filter_csv),
            "tropical_csv": str(tropical_csv),
            "area_postfilter_csv": str(area_postfilter_csv),
        },
    }
    json_path = diagnostics_dir / f"{run_stem}.json"
    json_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  NetCDF     : {netcdf_path}")
    print(f"  Figure     : {figure_path}")
    print(f"  Diagnostics: {json_path}")
    print(
        f"time={matched_time} | PC1={explained_variance_ratio:.6f} | "
        f"core={int(core_mask.sum())} | full_object_candidates={int(candidate_mask.sum())} | "
        f"object_filter={args.object_filter}:{object_info['removed_objects']}/{object_info['candidate_objects']} | "
        f"tropical_removed={tropical_removed}/{int(np.max(pre_tropical_labels))} | "
        f"area_postfilter={args.area_postfilter}:{area_postfilter_info['removed_objects']}/{area_postfilter_info['candidate_objects']} | "
        f"partial_objects=0 | "
        f"final_objects={final_object_count}",
        flush=True,
    )
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify AR candidates using CNN features, PCA, Otsu segmentation and whole-object filters."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--time",
        action="append",
        default=None,
        help="Process this time; repeat for multiple times, e.g. --time 2016-06-15T12:00:00.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--otsu-mode",
        choices=("global", "local"),
        default="local",
        help="First pixel segmentation of PCA features: local (default) or global for comparison.",
    )
    parser.add_argument(
        "--local-otsu-radius",
        type=int,
        default=48,
        help="Local Otsu disk radius in cells; 48 (default) gives a 97 x 97 bounding window.",
    )
    parser.add_argument(
        "--object-filter",
        choices=(
            "integrated-anomaly-multiotsu",
            "global-local-rank-multiotsu",
            "area-multiotsu",
            "none",
        ),
        default="integrated-anomaly-multiotsu",
        help=(
            "integrated-anomaly-multiotsu: split area-integrated local anomalies into "
            "three classes and keep the largest (default); global-local-rank-multiotsu: "
            "require joint local-anomaly and global-IVT support without "
            "resegmenting pixels; area-multiotsu: classify objects by physical area; "
            "none: retain all first-Otsu objects."
        ),
    )
    parser.add_argument(
        "--area-postfilter",
        choices=("none", "multiotsu-remove-small", "multiotsu-keep-large"),
        default="none",
        help=(
            "Apply three-class Otsu to spherical object areas after the primary and "
            "tropical filters. multiotsu-remove-small removes only the smallest class; "
            "multiotsu-keep-large retains only the largest class; none disables this "
            "step. No fixed area or pixel-count cutoff is imposed."
        ),
    )
    parser.add_argument("--disable-tropical-filter", action="store_true")
    parser.add_argument("--tropical-latitude-min", type=float, default=-15.0)
    parser.add_argument("--tropical-latitude-max", type=float, default=15.0)
    parser.add_argument("--tropical-removal-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_otsu_radius < 1:
        raise ValueError("--local-otsu-radius must be at least 1.")
    if args.pca_chunk_size < 2:
        raise ValueError("--pca-chunk-size must be at least 2.")
    if args.tropical_latitude_min >= args.tropical_latitude_max:
        raise ValueError(
            "The lower tropical latitude bound must be below the upper bound."
        )
    if not 0.0 <= args.tropical_removal_fraction <= 1.0:
        raise ValueError("--tropical-removal-fraction must lie in [0, 1].")

    seed_everything(args.seed)
    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; add --cpu to run on the CPU.")
        device = torch.device(f"cuda:{args.gpu}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model = SSSARStructureModel().to(device)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()

    times = tuple(args.time) if args.time else DEFAULT_TIMES
    print("=" * 92)
    print("AR candidates | Single-time PCA, local Otsu and whole-object filtering")
    print("=" * 92)
    print(f"Device      : {device}")
    print(f"Model       : {args.model_path}")
    print(f"Output root : {args.output_root}")
    print(f"Times       : {', '.join(times)}")
    if args.otsu_mode == "local":
        print(f"First Otsu  : local rank Otsu, radius={args.local_otsu_radius} px")
    else:
        print("First Otsu  : global Otsu on abs(PC1) [comparison only]")
    print("Second pixel Otsu : none; first-Otsu objects are never re-segmented")
    if args.object_filter == "integrated-anomaly-multiotsu":
        print(
            "Object      : full-object integrated local-anomaly multi-Otsu (keep large class)"
        )
    elif args.object_filter == "global-local-rank-multiotsu":
        print(
            "Object      : full-object global-local rank multi-Otsu "
            "(local anomaly AND global IVT support; keep large class)"
        )
    elif args.object_filter == "area-multiotsu":
        print("Object      : full-object physical-area multi-Otsu (keep large class)")
    else:
        print("Object      : off; preserve all full first-Otsu objects")
    if args.area_postfilter == "multiotsu-remove-small":
        print(
            "Area post   : post-tropical physical-area multi-Otsu (remove small class only)"
        )
    elif args.area_postfilter == "multiotsu-keep-large":
        print(
            "Area post   : post-tropical physical-area multi-Otsu (keep large class only)"
        )
    else:
        print("Area post   : off")
    print(
        "Postfilter  : shape=off, "
        f"tropical={'off' if args.disable_tropical_filter else 'on'} "
        f"({args.tropical_latitude_min:g} to {args.tropical_latitude_max:g})"
    )

    summaries = []
    for index, target_time in enumerate(times, start=1):
        print(f"\n[{index}/{len(times)}] Processing {target_time}")
        summaries.append(detect_one_time(model, device, target_time, args))

    summary_tag = (
        str(args.object_filter)
        if args.area_postfilter == "none"
        else f"{args.object_filter}_{args.area_postfilter}"
    )
    summary_path = Path(args.output_root) / f"run_summary_{summary_tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nProcessing complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
