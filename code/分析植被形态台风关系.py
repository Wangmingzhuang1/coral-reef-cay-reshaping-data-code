"""开展非建模的植被比例、形态变化与台风暴露描述和关联分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, spearmanr, wilcoxon
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from 分析范围 import (
    ANALYSIS_CONTRACT_VERSION,
    analysis_contract_digest,
    exclude_reefs,
    module_contract,
)
from 分析沙洲面积轨迹 import MULTIMETRIC_ORDER, configure_style, panel_label
from 对齐流场波浪与沙洲变化 import bh_fdr


VEGETATION_CONTRACT = module_contract("vegetation_state")
CORE_MIN_INTERVAL_DAYS, CORE_MAX_INTERVAL_DAYS = VEGETATION_CONTRACT["interval_days"]
WINDOW_SENSITIVITY = [
    tuple(days) for days in VEGETATION_CONTRACT["sensitivity_interval_days"]
]
VEGETATION_BINS = [-np.inf, 0.03, 0.15, 0.45, np.inf]
VEGETATION_LABELS = ["none", "sparse", "partial", "dominant"]
VEGETATION_THRESHOLD_PERTURBATIONS = [-0.02, 0.0, 0.02]
VEGETATION_STATE_ORDER = {"none": 0, "sparse": 1, "partial": 2, "dominant": 3}
OUTCOMES = [
    "erosion_fraction_per_year",
    "deposition_fraction_per_year",
    "gross_mobility_fraction_per_year",
    "centroid_shift_m_per_year",
    "centroid_shift_normalized_per_year",
]


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    numeric_columns = [
        "time_interval_days",
        "area_t_m2",
        "erosion_area_m2",
        "deposition_area_m2",
        "gross_boundary_change_m2",
        "centroid_shift_m",
        "centroid_shift_m_per_year",
        "vegetation_fraction_t",
        "typhoon_strong_count",
        "typhoon_r34_count",
        "typhoon_min_distance_km",
        "typhoon_max_nearest_wind_m_s",
        "typhoon_max_storm_wind_m_s",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["years"] = data["time_interval_days"] / 365.2425
    data["erosion_fraction_per_year"] = data["erosion_area_m2"] / data["area_t_m2"] / data["years"]
    data["deposition_fraction_per_year"] = data["deposition_area_m2"] / data["area_t_m2"] / data["years"]
    data["gross_mobility_fraction_per_year"] = (
        data["gross_boundary_change_m2"] / data["area_t_m2"] / data["years"]
    )
    data["centroid_shift_normalized_per_year"] = (
        data["centroid_shift_m"] / np.sqrt(data["area_t_m2"]) / data["years"]
    )
    data["vegetation_band"] = pd.cut(
        data["vegetation_fraction_t"],
        bins=VEGETATION_BINS,
        labels=VEGETATION_LABELS,
        right=False,
    ).astype("string")
    data["strong_typhoon_exposure"] = np.where(
        data["typhoon_strong_count"].fillna(0) > 0,
        "strong_typhoon",
        "no_strong_typhoon",
    )
    valid = data[
        data["boundary_change_status"].eq("ok")
        & data["area_t_m2"].gt(0)
        & data["vegetation_fraction_t"].notna()
        & data["years"].gt(0)
    ].copy()
    core = valid[
        valid["time_interval_days"].between(CORE_MIN_INTERVAL_DAYS, CORE_MAX_INTERVAL_DAYS)
    ].copy()
    return valid, core


def describe(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["sensor", "vegetation_band", "strong_typhoon_exposure"]
    for keys, group in core.groupby(group_columns, observed=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_intervals"] = int(len(group))
        row["n_cays"] = int(group["sand_cay_id"].nunique())
        row["median_interval_days"] = float(group["time_interval_days"].median())
        for outcome in OUTCOMES:
            values = group[outcome].dropna()
            row[f"{outcome}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{outcome}_q25"] = float(values.quantile(0.25)) if len(values) else np.nan
            row[f"{outcome}_q75"] = float(values.quantile(0.75)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns)


def rank_associations(core: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor, sensor_data in core.groupby("sensor"):
        datasets = {
            "interval_exploratory": sensor_data,
            "cay_median": sensor_data.groupby("sand_cay_id", as_index=False).median(numeric_only=True),
        }
        for analysis_unit, frame in datasets.items():
            for outcome in OUTCOMES:
                selected = frame[["vegetation_fraction_t", outcome]].dropna()
                if len(selected) < 5 or selected.nunique().min() < 2:
                    rho = p_value = np.nan
                else:
                    rho, p_value = spearmanr(
                        selected["vegetation_fraction_t"], selected[outcome]
                    )
                rows.append(
                    {
                        "sensor": sensor,
                        "analysis_unit": analysis_unit,
                        "outcome": outcome,
                        "n": int(len(selected)),
                        "spearman_rho": rho,
                        "two_sided_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def window_sensitivity(valid: pd.DataFrame) -> pd.DataFrame:
    """预定义时间窗敏感性：不改变主核心样本，只记录方向和FDR结果。"""
    rows: list[dict[str, object]] = []
    for low, high in WINDOW_SENSITIVITY:
        core = valid.loc[valid["time_interval_days"].between(low, high)].copy()
        associations = rank_associations(core)
        if associations.empty:
            continue
        associations["fdr_q_value"] = associations.groupby("sensor")[
            "two_sided_p_value"
        ].transform(bh_fdr)
        for row in associations.itertuples(index=False):
            rows.append(
                {
                    "window_days": f"{low}-{high}",
                    "sensor": row.sensor,
                    "analysis_unit": row.analysis_unit,
                    "outcome": row.outcome,
                    "n": int(row.n),
                    "spearman_rho": row.spearman_rho,
                    "two_sided_p_value": row.two_sided_p_value,
                    "fdr_q_value": row.fdr_q_value,
                }
            )
    return pd.DataFrame(rows)


def mask_perturbation_sensitivity(core: pd.DataFrame, perturbation: pd.DataFrame) -> pd.DataFrame:
    """用±1像元掩膜扰动重算植被—稳定性秩相关，不改变主分析样本。"""
    if perturbation.empty:
        return pd.DataFrame()
    perturbed = perturbation.loc[
        perturbation["status"].eq("ok") & perturbation["perturbation_pixels"].ne(0)
    ].copy()
    if perturbed.empty:
        return pd.DataFrame()
    merged = core.merge(
        perturbed[
            [
                "transition_id",
                "perturbation_pixels",
                "area_t_m2",
                "gross_boundary_change_m2",
            ]
        ],
        on="transition_id",
        how="inner",
        suffixes=("", "_perturbed"),
    )
    merged["gross_mobility_fraction_per_year"] = (
        merged["gross_boundary_change_m2_perturbed"]
        / merged["area_t_m2_perturbed"]
        / merged["years"]
    )
    rows: list[dict[str, object]] = []
    for pixels, group in merged.groupby("perturbation_pixels"):
        for sensor, sensor_data in group.groupby("sensor"):
            cay_median = sensor_data.groupby("sand_cay_id", as_index=False).median(
                numeric_only=True
            )
            selected = cay_median[
                ["vegetation_fraction_t", "gross_mobility_fraction_per_year"]
            ].dropna()
            if len(selected) < 5 or selected.nunique().min() < 2:
                continue
            rho, p_value = spearmanr(
                selected["vegetation_fraction_t"],
                selected["gross_mobility_fraction_per_year"],
            )
            rows.append(
                {
                    "perturbation_pixels": int(pixels),
                    "sensor": sensor,
                    "analysis_unit": "cay_median",
                    "outcome": "gross_mobility_fraction_per_year",
                    "n": int(len(selected)),
                    "spearman_rho": float(rho),
                    "two_sided_p_value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby("perturbation_pixels")[
            "two_sided_p_value"
        ].transform(bh_fdr)
    return result


def color_proxy_validation(color_csv: Path) -> pd.DataFrame:
    """用人工审计标签评估连续颜色植被比例的测量一致性；标签不进入模型。"""
    color = pd.read_csv(color_csv)
    valid = color.loc[
        color["status"].eq("ok")
        & color["vegetation_fraction"].notna()
        & color["manual_vegetation_state"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for sensor, group in valid.groupby("sensor"):
        manual = group["manual_vegetation_state"].astype("string")
        automatic = group["automatic_vegetation_state"].astype("string")
        fraction = pd.to_numeric(group["vegetation_fraction"], errors="coerce")
        agreement = manual.eq(automatic)
        manual_order = manual.map(VEGETATION_STATE_ORDER)
        automatic_order = automatic.map(VEGETATION_STATE_ORDER)
        ordinal_agreement = manual_order.eq(automatic_order)
        adjacent_agreement = (manual_order - automatic_order).abs() <= 1
        near_threshold = np.zeros(len(group), dtype=bool)
        for threshold in VEGETATION_BINS[1:-1]:
            near_threshold |= (fraction - threshold).abs() <= 0.05
        for delta in VEGETATION_THRESHOLD_PERTURBATIONS:
            perturbed_fraction = (fraction + delta).clip(0, 1)
            perturbed_band = pd.cut(
                perturbed_fraction,
                bins=VEGETATION_BINS,
                labels=VEGETATION_LABELS,
                right=False,
            ).astype("string")
            rows.append(
                {
                    "sensor": sensor,
                    "check": "threshold_perturbation",
                    "threshold_delta": delta,
                    "n_audit_records": int(len(group)),
                    "n_manual_automatic_agreement": int(agreement.sum()),
                    "manual_automatic_agreement": float(agreement.mean()),
                    "n_ordinal_agreement": int(ordinal_agreement.sum()),
                    "ordinal_agreement": float(ordinal_agreement.mean()),
                    "n_adjacent_or_exact_agreement": int(adjacent_agreement.sum()),
                    "adjacent_or_exact_agreement": float(adjacent_agreement.mean()),
                    "n_perturbed_matches_manual": int(perturbed_band.eq(manual).sum()),
                    "perturbed_agreement": float(perturbed_band.eq(manual).mean()),
                    "n_near_threshold": int(near_threshold.sum()),
                    "near_threshold_fraction": float(near_threshold.mean()),
                    "median_vegetation_fraction": float(fraction.median()),
                }
            )
        band_means = group.groupby(manual)["vegetation_fraction"].agg(
            ["count", "median", "min", "max"]
        )
        for label, row in band_means.iterrows():
            rows.append(
                {
                    "sensor": sensor,
                    "check": "manual_state_fraction_distribution",
                    "threshold_delta": np.nan,
                    "manual_state": str(label),
                    "n_audit_records": int(row["count"]),
                    "median_vegetation_fraction": float(row["median"]),
                    "min_vegetation_fraction": float(row["min"]),
                    "max_vegetation_fraction": float(row["max"]),
                }
            )
    return pd.DataFrame(rows)


SECTOR_COUNT = 12
SECTOR_MIN_MARGIN_PIXELS = 5
SECTOR_VEGETATED_THRESHOLD = 0.15


def pixel_vegetation_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Reproduce the project colour pipeline at pixel level (same rules/thresholds)."""
    selected = mask > 0
    if not selected.any():
        return None
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    raw_pixels = rgb[selected]
    raw_value = raw_pixels.max(axis=1)
    white_cutoff = float(np.quantile(raw_value, 0.90))
    white_candidates = raw_pixels[raw_value >= white_cutoff]
    if len(white_candidates) == 0:
        return None
    white_reference = np.median(white_candidates, axis=0)
    neutral_level = float(np.mean(white_reference))
    gains = np.clip(
        neutral_level / np.maximum(white_reference, 1e-6), 0.50, 2.00
    )
    pixels = np.clip(raw_pixels * gains, 0.0, 1.0)
    hsv_pixels = cv2.cvtColor(
        np.round(pixels.reshape(-1, 1, 3) * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    ).reshape(-1, 3)
    hues = hsv_pixels[:, 0].astype(np.float32) * 2.0
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    value = pixels.max(axis=1)
    channel_min = pixels.min(axis=1)
    saturation = (value - channel_min) / np.maximum(value, 1e-6)
    green_ratio = green / np.maximum(red + green + blue, 1e-6)
    green_excess = 2.0 * green - red - blue
    vegetation = (
        (hues >= 45.0)
        & (hues <= 165.0)
        & (saturation >= 0.12)
        & (value >= 0.08)
        & (green_excess >= 0.025)
        & (green_ratio >= 0.345)
    )
    veg_2d = np.zeros(mask.shape, dtype=bool)
    veg_2d[selected] = vegetation
    return veg_2d


def _read_mask(path_text: object) -> np.ndarray | None:
    path = Path(str(path_text))
    if not path.is_file():
        return None
    buffer = np.fromfile(path, dtype=np.uint8)
    if buffer.size == 0:
        return None
    mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return mask > 0 if mask is not None else None


def vegetation_sector_contrast(
    transitions: pd.DataFrame,
    color: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M4: within-cay, same-image contrast of vegetated vs bare boundary sectors."""
    sentinel = transitions.loc[transitions["sensor"].eq("sentinel2")].copy()
    if sentinel.empty:
        return pd.DataFrame(), pd.DataFrame()
    obs = observations.loc[observations["sensor"].eq("sentinel2")].copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    image_lookup = obs.set_index(["sand_cay_id", "date"])["image_id"]
    paths = color.set_index("image_id")[["image_path", "mask_path"]]
    rows: list[dict[str, object]] = []
    for transition in sentinel.itertuples(index=False):
        time_t = pd.to_datetime(transition.time_t)
        time_t1 = pd.to_datetime(transition.time_t1)
        pre_key = (transition.sand_cay_id, time_t)
        post_key = (transition.sand_cay_id, time_t1)
        if pre_key not in image_lookup.index or post_key not in image_lookup.index:
            continue
        pre_value = image_lookup.loc[pre_key]
        post_value = image_lookup.loc[post_key]
        pre_image_id = (
            pre_value.iloc[0] if isinstance(pre_value, pd.Series) else pre_value
        )
        post_image_id = (
            post_value.iloc[0] if isinstance(post_value, pd.Series) else post_value
        )
        if pre_image_id not in paths.index or post_image_id not in paths.index:
            continue
        pre_image_path = Path(str(paths.loc[pre_image_id, "image_path"]))
        pre_mask = _read_mask(paths.loc[pre_image_id, "mask_path"])
        post_mask = _read_mask(paths.loc[post_image_id, "mask_path"])
        if pre_mask is None or post_mask is None or pre_mask.shape != post_mask.shape:
            continue
        if not pre_image_path.is_file():
            continue
        image_bgr = cv2.imdecode(
            np.fromfile(pre_image_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image_bgr is None or image_bgr.shape[:2] != pre_mask.shape:
            continue
        veg_2d = pixel_vegetation_mask(image_bgr, pre_mask)
        if veg_2d is None:
            continue
        mask_binary = pre_mask.astype(np.uint8)
        area_pixels = int(pre_mask.sum())
        radius = max(2, int(round(0.15 * np.sqrt(area_pixels / np.pi))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        interior = cv2.erode(mask_binary, kernel) > 0
        margin = pre_mask & ~interior
        if int(margin.sum()) < SECTOR_MIN_MARGIN_PIXELS:
            continue
        ys, xs = np.nonzero(pre_mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        erosion = np.logical_and(pre_mask, ~post_mask)
        deposition = np.logical_and(~pre_mask, post_mask)
        change = erosion | deposition

        def sector_map(mask2d: np.ndarray) -> np.ndarray:
            yy, xx = np.nonzero(mask2d)
            if len(xx) == 0:
                return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
            bearings = np.degrees(np.arctan2(xx - cx, -(yy - cy))) % 360.0
            sectors = (bearings // (360.0 / SECTOR_COUNT)).astype(int)
            return sectors, (yy, xx)

        margin_sectors, margin_coords = sector_map(margin)
        change_sectors, _ = sector_map(change)
        erosion_sectors, _ = sector_map(erosion)
        deposition_sectors, _ = sector_map(deposition)
        veg_values = veg_2d[margin_coords]
        sector_rows = []
        for sector in range(SECTOR_COUNT):
            margin_count = int((margin_sectors == sector).sum())
            if margin_count < SECTOR_MIN_MARGIN_PIXELS:
                continue
            change_count = int((change_sectors == sector).sum())
            erosion_count = int((erosion_sectors == sector).sum())
            deposition_count = int((deposition_sectors == sector).sum())
            veg_fraction = float(veg_values[margin_sectors == sector].mean())
            sector_rows.append(
                {
                    "sector": sector,
                    "margin_pixels": margin_count,
                    "mobility": change_count / margin_count,
                    "retreat": (erosion_count - deposition_count) / margin_count,
                    "vegetation_fraction": veg_fraction,
                    "vegetated": bool(veg_fraction >= SECTOR_VEGETATED_THRESHOLD),
                }
            )
        sector_frame = pd.DataFrame(sector_rows)
        if sector_frame.empty:
            continue
        veg_sectors = sector_frame.loc[sector_frame["vegetated"]]
        bare_sectors = sector_frame.loc[~sector_frame["vegetated"]]
        if veg_sectors.empty or bare_sectors.empty:
            continue
        rows.append(
            {
                "transition_id": transition.transition_id,
                "sand_cay_id": transition.sand_cay_id,
                "reef_id": transition.reef_id,
                "time_interval_days": transition.time_interval_days,
                "pulse": bool(
                    (transition.typhoon_strong_count or 0) > 0
                    or (transition.typhoon_r34_count or 0) > 0
                ),
                "n_sectors_vegetated": int(len(veg_sectors)),
                "n_sectors_bare": int(len(bare_sectors)),
                "mobility_vegetated": float(veg_sectors["mobility"].mean()),
                "mobility_bare": float(bare_sectors["mobility"].mean()),
                "mobility_difference": float(
                    veg_sectors["mobility"].mean() - bare_sectors["mobility"].mean()
                ),
                "retreat_vegetated": float(veg_sectors["retreat"].mean()),
                "retreat_bare": float(bare_sectors["retreat"].mean()),
                "retreat_difference": float(
                    veg_sectors["retreat"].mean() - bare_sectors["retreat"].mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    stats_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260909)

    def summarize(subset: pd.DataFrame, stratum: str) -> dict[str, object]:
        differences = subset["mobility_difference"].to_numpy(dtype=float)
        retreats = subset["retreat_difference"].to_numpy(dtype=float)
        if len(differences) < 8:
            return {
                "stratum": stratum,
                "n_transitions": int(len(subset)),
                "status": "insufficient_sample",
            }
        cays = subset["sand_cay_id"].unique()
        bootstrap_means = []
        for _ in range(2000):
            sampled = rng.choice(cays, size=len(cays), replace=True)
            parts = [
                subset.loc[subset["sand_cay_id"].eq(cay), "mobility_difference"].to_numpy(
                    dtype=float
                )
                for cay in sampled
            ]
            pooled = np.concatenate(parts)
            bootstrap_means.append(float(pooled.mean()))
        try:
            wilcoxon_p = float(
                wilcoxon(differences, alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = np.nan
        return {
            "stratum": stratum,
            "n_transitions": int(len(subset)),
            "n_cays": int(len(cays)),
            "status": "ok",
            "mean_mobility_difference": float(differences.mean()),
            "median_mobility_difference": float(np.median(differences)),
            "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
            "wilcoxon_p_value": wilcoxon_p,
            "mean_retreat_difference": float(retreats.mean()),
            "supported_negative": bool(
                differences.mean() < 0
                and np.quantile(bootstrap_means, 0.975) < 0
                and wilcoxon_p < 0.05
            ),
        }

    stats_rows.append(summarize(frame, "all"))
    for pulse, subset in frame.groupby("pulse"):
        stats_rows.append(summarize(subset, "pulse" if pulse else "background"))
    for low, high in [(90, 540), (120, 730)]:
        subset = frame.loc[frame["time_interval_days"].between(low, high)]
        stats_rows.append(summarize(subset, f"window_{low}_{high}"))
    return frame, pd.DataFrame(stats_rows)


def _sector_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def sector_causal_tests(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    color: pd.DataFrame,
) -> pd.DataFrame:
    """M7/M9：扇区尺度前向（植被→后续可动）与反向（可动→后续植被）检验。"""
    sentinel = transitions.loc[
        transitions["sensor"].eq("sentinel2")
        & transitions["boundary_change_status"].eq("ok")
        & transitions["time_interval_days"].between(90, 730)
    ].copy()
    if sentinel.empty:
        return pd.DataFrame()
    obs = observations.loc[observations["sensor"].eq("sentinel2")].copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    image_lookup = obs.set_index(["sand_cay_id", "date"])
    paths = color.set_index("image_id")[["image_path", "mask_path"]]
    cache: dict[str, dict[str, object]] = {}

    def sector_state(image_id: str) -> dict[str, object] | None:
        if image_id in cache:
            return cache[image_id]
        if image_id not in paths.index:
            return None
        mask = _read_mask(paths.loc[image_id, "mask_path"])
        image_path = Path(str(paths.loc[image_id, "image_path"]))
        if mask is None or not image_path.is_file():
            return None
        image_bgr = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image_bgr is None or image_bgr.shape[:2] != mask.shape:
            return None
        veg_2d = pixel_vegetation_mask(image_bgr, mask)
        if veg_2d is None:
            return None
        mask_binary = mask.astype(np.uint8)
        area_pixels = int(mask.sum())
        radius = max(2, int(round(0.15 * np.sqrt(area_pixels / np.pi))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        interior = cv2.erode(mask_binary, kernel) > 0
        margin = mask & ~interior
        ys, xs = np.nonzero(margin)
        if len(xs) == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
        bearings = np.degrees(np.arctan2(xs - cx, -(ys - cy))) % 360.0
        sectors = (bearings // (360.0 / SECTOR_COUNT)).astype(int)
        veg_values = veg_2d[margin]
        sector_veg = {}
        sector_margin = {}
        for sector in range(SECTOR_COUNT):
            sel = sectors == sector
            count = int(sel.sum())
            if count < SECTOR_MIN_MARGIN_PIXELS:
                continue
            sector_veg[sector] = float(veg_values[sel].mean())
            sector_margin[sector] = count
        state = {
            "sector_veg": sector_veg,
            "sector_margin": sector_margin,
            "centroid": (cx, cy),
            "mask": mask,
        }
        cache[image_id] = state
        return state

    def sector_mobility(pre_id: str, post_id: str) -> dict[int, float] | None:
        pre = sector_state(pre_id)
        post = sector_state(post_id)
        if pre is None or post is None:
            return None
        pre_mask = pre["mask"]
        post_mask = post["mask"]
        if pre_mask.shape != post_mask.shape:
            return None
        change = np.logical_xor(pre_mask, post_mask)
        ys, xs = np.nonzero(pre_mask)
        if len(xs) == 0:
            return None
        cx, cy = pre["centroid"]
        bearings = np.degrees(np.arctan2(xs - cx, -(ys - cy))) % 360.0
        change_ys, change_xs = np.nonzero(change)
        change_bearings = (
            np.degrees(np.arctan2(change_xs - cx, -(change_ys - cy))) % 360.0
        )
        change_sectors = (change_bearings // (360.0 / SECTOR_COUNT)).astype(int)
        mobility = {}
        for sector, margin_count in pre["sector_margin"].items():
            change_count = int((change_sectors == sector).sum())
            mobility[sector] = change_count / margin_count
        return mobility

    image_ids = {}
    for transition in sentinel.itertuples(index=False):
        for tag, when in [("pre", transition.time_t), ("post", transition.time_t1)]:
            key = (transition.sand_cay_id, pd.to_datetime(when))
            if key in image_lookup.index:
                record = image_lookup.loc[key]
                if isinstance(record, pd.DataFrame):
                    record = record.iloc[0]
                image_ids[(transition.transition_id, tag)] = record["image_id"]
    chain = (
        sentinel[["transition_id", "sand_cay_id", "time_t", "time_t1"]]
        .sort_values(["sand_cay_id", "time_t"])
        .reset_index(drop=True)
    )
    chain["time_t"] = pd.to_datetime(chain["time_t"])
    chain["time_t1"] = pd.to_datetime(chain["time_t1"])
    next_map = {}
    prev_map = {}
    strict_link_rows: list[dict[str, object]] = []
    for cay, group in chain.groupby("sand_cay_id"):
        records = group.to_dict("records")
        for position, record in enumerate(records):
            if position > 0:
                previous = records[position - 1]
                adjacent = previous["time_t1"] == record["time_t"]
                gap_days = int((record["time_t"] - previous["time_t1"]).days)
                strict_link_rows.append(
                    {
                        "sand_cay_id": cay,
                        "direction": "prev",
                        "transition_id": record["transition_id"],
                        "linked_transition_id": previous["transition_id"],
                        "strictly_adjacent": bool(adjacent),
                        "gap_days": gap_days,
                    }
                )
                if adjacent:
                    prev_map[record["transition_id"]] = previous["transition_id"]
            if position < len(records) - 1:
                following = records[position + 1]
                adjacent = record["time_t1"] == following["time_t"]
                gap_days = int((following["time_t"] - record["time_t1"]).days)
                strict_link_rows.append(
                    {
                        "sand_cay_id": cay,
                        "direction": "next",
                        "transition_id": record["transition_id"],
                        "linked_transition_id": following["transition_id"],
                        "strictly_adjacent": bool(adjacent),
                        "gap_days": gap_days,
                    }
                )
                if adjacent:
                    next_map[record["transition_id"]] = following["transition_id"]
    strict_links = pd.DataFrame(strict_link_rows)
    forward_rows: list[dict[str, object]] = []
    reverse_rows: list[dict[str, object]] = []
    for transition in sentinel.itertuples(index=False):
        pre_id = image_ids.get((transition.transition_id, "pre"))
        post_id = image_ids.get((transition.transition_id, "post"))
        if not pre_id or not post_id:
            continue
        mobility = sector_mobility(pre_id, post_id)
        pre_state = sector_state(pre_id)
        post_state = sector_state(post_id)
        if mobility is None or pre_state is None or post_state is None:
            continue
        prev_id = prev_map.get(transition.transition_id)
        prev_mobility = None
        if prev_id:
            prev_pre = image_ids.get((prev_id, "pre"))
            prev_post = image_ids.get((prev_id, "post"))
            if prev_pre and prev_post:
                prev_mobility = sector_mobility(prev_pre, prev_post)
        for sector, mob in mobility.items():
            if sector not in pre_state["sector_veg"]:
                continue
            forward_rows.append(
                {
                    "transition_id": transition.transition_id,
                    "sand_cay_id": transition.sand_cay_id,
                    "sector": sector,
                    "y_mobility": mob,
                    "x_vegetation": pre_state["sector_veg"][sector],
                    "control_prev_mobility": (
                        prev_mobility.get(sector, np.nan) if prev_mobility else np.nan
                    ),
                }
            )
        next_id = next_map.get(transition.transition_id)
        if not next_id:
            continue
        next_post = image_ids.get((next_id, "post"))
        if not next_post:
            continue
        next_state = sector_state(next_post)
        if next_state is None:
            continue
        for sector, mob in mobility.items():
            if sector not in post_state["sector_veg"] or sector not in next_state["sector_veg"]:
                continue
            reverse_rows.append(
                {
                    "transition_id": transition.transition_id,
                    "sand_cay_id": transition.sand_cay_id,
                    "sector": sector,
                    "y_vegetation_change": next_state["sector_veg"][sector]
                    - post_state["sector_veg"][sector],
                    "x_mobility": mob,
                    "control_vegetation": post_state["sector_veg"][sector],
                }
            )
    forward = pd.DataFrame(forward_rows)
    reverse = pd.DataFrame(reverse_rows)
    results: list[dict[str, object]] = []

    def fit_scheme(
        frame: pd.DataFrame,
        y_column: str,
        x_column: str,
        control_column: str,
        scheme: str,
        test: str,
    ) -> None:
        if frame.empty:
            return
        work = frame.dropna(subset=[y_column, x_column]).copy()
        if control_column in work.columns:
            work = work.dropna(subset=[control_column])
        if len(work) < 30 or work["sand_cay_id"].nunique() < 10:
            results.append(
                {
                    "test": test,
                    "scheme": scheme,
                    "status": "insufficient_sample",
                    "n_sectors": int(len(work)),
                    "n_transitions": int(work["transition_id"].nunique()),
                    "n_cays": int(work["sand_cay_id"].nunique()),
                }
            )
            return
        group_column = "sand_cay_id" if scheme == "within_cay" else "transition_id"
        grouped = work.groupby(group_column)
        y = work[y_column] - grouped[y_column].transform("mean")
        x = work[x_column] - grouped[x_column].transform("mean")
        columns = [x]
        names = [x_column]
        if control_column in work.columns:
            control = work[control_column] - grouped[control_column].transform("mean")
            columns.append(control)
            names.append(control_column)
        design = pd.concat(columns, axis=1)
        design.columns = names
        informative = design.abs().sum(axis=1).gt(0) & y.notna()
        fitted = sm.OLS(y.loc[informative], design.loc[informative]).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": work.loc[informative, "sand_cay_id"],
                "use_correction": True,
            },
        )
        ci_low, ci_high = fitted.conf_int().loc[x_column].astype(float)
        results.append(
            {
                "test": test,
                "scheme": scheme,
                "status": "ok",
                "n_sectors": int(informative.sum()),
                "n_transitions": int(work.loc[informative, "transition_id"].nunique()),
                "n_cays": int(work.loc[informative, "sand_cay_id"].nunique()),
                "coefficient": float(fitted.params[x_column]),
                "cluster_robust_se": float(fitted.bse[x_column]),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_value": float(fitted.pvalues[x_column]),
            }
        )

    fit_scheme(
        forward, "y_mobility", "x_vegetation", "control_prev_mobility", "within_cay", "M7_forward"
    )
    fit_scheme(
        forward,
        "y_mobility",
        "x_vegetation",
        "control_prev_mobility",
        "within_transition",
        "M7_forward",
    )
    fit_scheme(
        reverse,
        "y_vegetation_change",
        "x_mobility",
        "control_vegetation",
        "within_cay",
        "M9_reverse",
    )
    fit_scheme(
        reverse,
        "y_vegetation_change",
        "x_mobility",
        "control_vegetation",
        "within_transition",
        "M9_reverse",
    )
    if not strict_links.empty:
        results.append(
            {
                "test": "chain_audit",
                "scheme": "strict_adjacency",
                "status": "ok",
                "n_sectors": np.nan,
                "n_transitions": int(strict_links["transition_id"].nunique()),
                "n_cays": int(strict_links["sand_cay_id"].nunique()),
                "coefficient": float(strict_links["strictly_adjacent"].mean()),
                "cluster_robust_se": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "p_value": np.nan,
                "note": (
                    "coefficient is strict adjacent-link fraction; M7/M9 use only "
                    "links whose predecessor end date equals successor start date"
                ),
            }
        )
    result = pd.DataFrame(results)
    if not result.empty and "p_value" in result.columns:
        valid = result["p_value"].notna()
        if valid.any():
            result.loc[valid, "fdr_q_value"] = multipletests(
                result.loc[valid, "p_value"], method="fdr_bh"
            )[1]
    return result


def event_candidates(core: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transition_id",
        "sensor",
        "reef_id",
        "sand_cay_id",
        "time_t",
        "time_t1",
        "time_interval_days",
        "vegetation_fraction_t",
        "vegetation_band",
        "erosion_fraction_per_year",
        "deposition_fraction_per_year",
        "gross_mobility_fraction_per_year",
        "centroid_shift_m_per_year",
        "typhoon_strong_count",
        "typhoon_r34_count",
        "typhoon_min_distance_km",
        "typhoon_max_nearest_wind_m_s",
        "typhoon_max_storm_wind_m_s",
    ]
    candidates = core[core["typhoon_strong_count"].gt(0)][columns].copy()
    return candidates.sort_values(
        ["sensor", "gross_mobility_fraction_per_year"], ascending=[True, False]
    )


STABILITY_OUTCOMES = [
    "centroid_net_normalized",
    "centroid_path_normalized",
    "perimeter_max_adjacent_ratio",
    "orientation_max_step_deg",
    "abs_relative_change",
    "abs_elongation_relative_change",
]
SENSOR_LABELS = {"sentinel2": "Sentinel-2", "google_earth": "Google Earth"}
SENSOR_COLORS = {"sentinel2": "#386CB0", "google_earth": "#D95F02"}
CLASS_TICKS = {
    "Persistent reworking": "Persist. rework.",
    "Complex": "Complex",
    "Net contraction": "Net contr.",
    "Net growth": "Net growth",
}


def load_trajectory_frame(output_dir: Path) -> pd.DataFrame:
    """主轨迹表与轨迹点植被比例合并为沙洲级稳定性分析框。"""
    primary = pd.read_csv(
        output_dir / "沙洲面积主轨迹分类.csv", encoding="utf-8-sig"
    )
    points = pd.read_csv(
        output_dir / "figures" / "source_data" / "figure3_area_trajectory_points.csv",
        encoding="utf-8-sig",
    )
    vegetation = (
        points.groupby("track_id")["vegetation_fraction"]
        .median()
        .rename("vegetation_fraction_median")
    )
    frame = primary.merge(vegetation, on="track_id", how="left")
    frame["abs_relative_change"] = frame["relative_change"].abs()
    frame["abs_elongation_relative_change"] = frame[
        "elongation_relative_change"
    ].abs()
    frame["bare_cay"] = frame["vegetation_fraction_median"].lt(VEGETATION_BINS[1])
    return frame


def trajectory_rank_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor, data in frame.groupby("sensor"):
        for outcome in STABILITY_OUTCOMES:
            selected = data[["vegetation_fraction_median", outcome]].dropna()
            if len(selected) < 5 or selected.nunique().min() < 2:
                rho = p_value = np.nan
            else:
                rho, p_value = spearmanr(
                    selected["vegetation_fraction_median"], selected[outcome]
                )
            rows.append(
                {
                    "sensor": sensor,
                    "analysis_unit": "cay_trajectory",
                    "outcome": outcome,
                    "n": int(len(selected)),
                    "spearman_rho": rho,
                    "two_sided_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = result.groupby("sensor")["two_sided_p_value"].transform(
        bh_fdr
    )
    return result


def stability_class_summary(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (sensor, cls), group in frame.groupby(["sensor", "multimetric_class"]):
        veg = group["vegetation_fraction_median"].dropna()
        rows.append(
            {
                "sensor": sensor,
                "multimetric_class": cls,
                "n_cays": int(len(group)),
                "vegetation_median": float(veg.median()) if len(veg) else np.nan,
                "vegetation_q25": float(veg.quantile(0.25)) if len(veg) else np.nan,
                "vegetation_q75": float(veg.quantile(0.75)) if len(veg) else np.nan,
                "centroid_net_normalized_median": float(
                    group["centroid_net_normalized"].median()
                ),
                "perimeter_max_adjacent_ratio_median": float(
                    group["perimeter_max_adjacent_ratio"].median()
                ),
            }
        )
    class_table = pd.DataFrame(rows).sort_values(["sensor", "multimetric_class"])
    bare_rows: list[dict[str, object]] = []
    for sensor, data in frame.groupby("sensor"):
        valid = data.dropna(subset=["vegetation_fraction_median"])
        persistent = valid["multimetric_class"].eq("Persistent reworking")
        bare = valid["bare_cay"]
        table = pd.crosstab(bare, persistent)
        if table.shape == (2, 2):
            _, p_value = fisher_exact(table.to_numpy())
        else:
            p_value = np.nan
        bare_n = int(bare.sum())
        vegetated_n = int((~bare).sum())
        bare_rows.append(
            {
                "sensor": sensor,
                "bare_n": bare_n,
                "bare_persistent_n": int((bare & persistent).sum()),
                "vegetated_n": vegetated_n,
                "vegetated_persistent_n": int((~bare & persistent).sum()),
                "persistent_reworking_fraction_bare": (
                    float((bare & persistent).sum() / bare_n) if bare_n else np.nan
                ),
                "persistent_reworking_fraction_vegetated": (
                    float((~bare & persistent).sum() / vegetated_n)
                    if vegetated_n
                    else np.nan
                ),
                "fisher_p_value": p_value,
            }
        )
    return class_table, pd.DataFrame(bare_rows)


MODULATION_PAIRS = (
    ("wave_hs_p90", "gross_mobility_fraction_per_year"),
    ("wave_hs_p90", "centroid_shift_normalized_per_year"),
    ("current_cross_mean", "centroid_cross_shift_normalized_per_year"),
    ("current_along_mean", "centroid_along_shift_normalized_per_year"),
)


def forcing_vegetation_modulation(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """背景区间内检验植被是否调节强迫—可动性耦合（S2 主分析）。"""
    decomposition = pd.read_csv(
        output_dir / "流场波浪沿轴横轴分解.csv", encoding="utf-8-sig"
    )
    decomposition["forcing_regime"] = np.where(
        decomposition[["typhoon_strong_count", "typhoon_r34_count"]]
        .fillna(0)
        .sum(axis=1)
        > 0,
        "typhoon_pulse",
        "background",
    )
    frame = load_trajectory_frame(output_dir)
    vegetation = frame[
        ["sand_cay_id", "sensor", "vegetation_fraction_median", "bare_cay"]
    ]
    sub = decomposition[
        decomposition["sensor"].eq("sentinel2")
        & decomposition["forcing_regime"].eq("background")
    ].merge(vegetation, on=["sand_cay_id", "sensor"], how="left")
    sub = sub[sub["bare_cay"].notna()].copy()
    rows: list[dict[str, object]] = []
    for bare_value, group in sub.groupby("bare_cay"):
        label = "bare" if bare_value else "vegetated"
        for driver, outcome in MODULATION_PAIRS:
            selected = group[[driver, outcome]].dropna()
            if len(selected) < 5 or selected.nunique().min() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(selected[driver], selected[outcome])
            rows.append(
                {
                    "vegetation_group": label,
                    "driver": driver,
                    "outcome": outcome,
                    "n": int(len(selected)),
                    "spearman_rho": rho,
                    "two_sided_p_value": p_value,
                }
            )
    associations = pd.DataFrame(rows)
    sub["wave_quartile"] = pd.qcut(
        sub["wave_hs_p90"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    stratified = (
        sub.groupby(["wave_quartile", "bare_cay"], observed=True)[
            "gross_mobility_fraction_per_year"
        ]
        .agg(["median", "count"])
        .reset_index()
        .rename(
            columns={
                "bare_cay": "is_bare",
                "median": "median_gross_mobility",
                "count": "n_intervals",
            }
        )
    )
    return associations, stratified, sub


STABILIZATION_THRESHOLD = 0.15
STATE_ORDER = {"none": 0, "sparse": 1, "partial": 2, "dominant": 3}
TRANSITION_SURFACE_STATE_MIN = 0.15
STATE4_ORDER = {
    "bright_bare": 0,
    "spectral_transition": 1,
    "low_green": 2,
    "moderate_green": 3,
    "high_green": 4,
}
TRANSITION_NAMES = (
    "bright_to_spectral_transition",
    "spectral_transition_to_low_green",
    "direct_bright_to_low_green",
    "low_green_to_high_green",
)


def vegetation_stabilization_ladder(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """检验植被状态层级稳定化假说：梯度、沙洲内前后对照、演替方向、岛状候选。"""
    decomposition = pd.read_csv(
        output_dir / "流场波浪沿轴横轴分解.csv", encoding="utf-8-sig"
    )
    decomposition["forcing_regime"] = np.where(
        decomposition[["typhoon_strong_count", "typhoon_r34_count"]]
        .fillna(0)
        .sum(axis=1)
        > 0,
        "typhoon_pulse",
        "background",
    )
    s2 = decomposition[
        decomposition["sensor"].eq("sentinel2")
        & decomposition["forcing_regime"].eq("background")
    ].copy()
    s2["vegetation_band"] = pd.cut(
        s2["vegetation_fraction_t"],
        bins=VEGETATION_BINS,
        labels=VEGETATION_LABELS,
        right=False,
    ).astype("string")
    gradient_rows: list[dict[str, object]] = []
    for band, group in s2.groupby("vegetation_band", observed=True):
        gradient_rows.append(
            {
                "vegetation_band": str(band),
                "state_order": STATE_ORDER[str(band)],
                "n_intervals": int(len(group)),
                "n_cays": int(group["sand_cay_id"].nunique()),
                "median_gross_mobility": float(
                    group["gross_mobility_fraction_per_year"].median()
                ),
                "median_centroid_shift": float(
                    group["centroid_shift_normalized_per_year"].median()
                ),
            }
        )
    gradient = pd.DataFrame(gradient_rows).sort_values("state_order")
    cay_median = (
        s2.assign(state_order=s2["vegetation_band"].map(STATE_ORDER))
        .groupby("sand_cay_id")
        .agg(
            state=("state_order", "median"),
            gross=("gross_mobility_fraction_per_year", "median"),
            centroid=("centroid_shift_normalized_per_year", "median"),
        )
        .dropna()
    )
    rho_gross, p_gross = spearmanr(cay_median["state"], cay_median["gross"])
    rho_centroid, p_centroid = spearmanr(cay_median["state"], cay_median["centroid"])
    gradient.attrs["trend"] = (rho_gross, p_gross, rho_centroid, p_centroid)

    observations = pd.read_csv(
        output_dir / "沙洲观测主表.csv",
        encoding="utf-8-sig",
        usecols=[
            "reef_id",
            "sand_cay_id",
            "sensor",
            "date",
            "vegetation_fraction",
            "image_id",
        ],
    )
    observations = exclude_reefs(observations)
    observations["date"] = pd.to_datetime(observations["date"])
    observations = observations.merge(
        pd.read_csv(
            output_dir / "沙洲颜色组成.csv",
            encoding="utf-8-sig",
            usecols=["image_id", "core_margin_structure"],
        ),
        on="image_id",
        how="left",
    )
    paired_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    island_rows: list[dict[str, object]] = []
    for sensor in ["sentinel2", "google_earth"]:
        obs = observations[observations["sensor"].eq(sensor)]
        for cay, group in obs.groupby("sand_cay_id"):
            group = group.sort_values("date")
            bands = pd.cut(
                group["vegetation_fraction"],
                bins=VEGETATION_BINS,
                labels=VEGETATION_LABELS,
                right=False,
            )
            orders = bands.map(STATE_ORDER).to_numpy(dtype=float)
            steps = np.diff(orders[~np.isnan(orders)])
            transition_rows.append(
                {
                    "sensor": sensor,
                    "sand_cay_id": cay,
                    "n_up_steps": int((steps > 0).sum()),
                    "n_down_steps": int((steps < 0).sum()),
                }
            )
            established = group[
                group["vegetation_fraction"].ge(STABILIZATION_THRESHOLD)
            ]
            if established.empty:
                continue
            t_star = established["date"].iloc[0]
            later = group[group["date"].ge(t_star)]
            if len(later) < 2:
                continue
            late_median = float(later["vegetation_fraction"].median())
            if late_median < STABILIZATION_THRESHOLD:
                continue
            intervals = decomposition[
                decomposition["sensor"].eq(sensor)
                & decomposition["sand_cay_id"].eq(cay)
            ]
            before = intervals[
                pd.to_datetime(intervals["time_t1"]).le(t_star)
            ]["gross_mobility_fraction_per_year"].dropna()
            after = intervals[
                pd.to_datetime(intervals["time_t"]).ge(t_star)
            ]["gross_mobility_fraction_per_year"].dropna()
            if len(before) >= 2 and len(after) >= 2:
                paired_rows.append(
                    {
                        "sensor": sensor,
                        "sand_cay_id": cay,
                        "establishment_date": t_star.date().isoformat(),
                        "n_before": int(len(before)),
                        "n_after": int(len(after)),
                        "median_before": float(before.median()),
                        "median_after": float(after.median()),
                        "log_ratio_after_before": float(
                            np.log((after.median() + 1e-6) / (before.median() + 1e-6))
                        ),
                    }
                )
            dominant_share = float(
                later["vegetation_fraction"].ge(0.45).mean()
            )
            structure_share = float(
                later["core_margin_structure"].fillna(False).mean()
            )
            span_years = float(
                (later["date"].iloc[-1] - later["date"].iloc[0]).days / 365.2425
            )
            after_all = intervals[
                pd.to_datetime(intervals["time_t"]).ge(t_star)
            ]["gross_mobility_fraction_per_year"].dropna()
            if (
                dominant_share >= 0.5
                and structure_share >= 0.5
                and span_years >= 3.0
                and len(after_all) >= 2
                and float(after_all.median()) <= 1.0
            ):
                island_rows.append(
                    {
                        "sensor": sensor,
                        "sand_cay_id": cay,
                        "establishment_date": t_star.date().isoformat(),
                        "dominant_share_after": dominant_share,
                        "structure_share_after": structure_share,
                        "span_years_after": span_years,
                        "median_mobility_after": float(after_all.median()),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    transitions = pd.DataFrame(transition_rows)
    islands = pd.DataFrame(island_rows)
    observations["year"] = observations["date"].dt.year
    annual = (
        observations.groupby(["sensor", "sand_cay_id", "year"])["vegetation_fraction"]
        .agg(annual_median="median", n_obs="size")
        .reset_index()
    )
    annual["within_year_range"] = np.nan
    for (sensor, cay, year), idx in annual.groupby(
        ["sensor", "sand_cay_id", "year"]
    ).groups.items():
        values = observations.loc[
            observations["sensor"].eq(sensor)
            & observations["sand_cay_id"].eq(cay)
            & observations["year"].eq(year),
            "vegetation_fraction",
        ].dropna()
        if len(values) >= 2:
            annual.loc[idx, "within_year_range"] = float(
                values.max() - values.min()
            )
    multi = annual[annual["n_obs"].ge(2)]
    seasonal = (
        multi.groupby(["sensor", "sand_cay_id"])
        .agg(
            median_within_year_range=("within_year_range", "median"),
            n_years_multi_obs=("within_year_range", "size"),
        )
        .reset_index()
    )
    midpoint_year = (
        pd.to_datetime(s2["time_t"]) + pd.to_timedelta(s2["time_interval_days"] / 2, unit="D")
    ).dt.year
    deseason = s2.copy()
    deseason["midpoint_year"] = midpoint_year
    deseason = deseason.merge(
        annual[annual["sensor"].eq("sentinel2")][
            ["sand_cay_id", "year", "annual_median"]
        ].rename(columns={"year": "midpoint_year"}),
        on=["sand_cay_id", "midpoint_year"],
        how="left",
    )
    cay_overall = (
        deseason.groupby("sand_cay_id")["vegetation_fraction_t"].median()
    )
    deseason["annual_median"] = deseason["annual_median"].fillna(
        deseason["sand_cay_id"].map(cay_overall)
    )
    deseason["vegetation_band_deseasoned"] = pd.cut(
        deseason["annual_median"],
        bins=VEGETATION_BINS,
        labels=VEGETATION_LABELS,
        right=False,
    ).astype("string")
    deseason["state_order"] = deseason["vegetation_band_deseasoned"].map(
        STATE_ORDER
    )
    deseason_rows: list[dict[str, object]] = []
    for band, group in deseason.groupby("vegetation_band_deseasoned", observed=True):
        deseason_rows.append(
            {
                "vegetation_band": str(band),
                "state_order": STATE_ORDER[str(band)],
                "n_intervals": int(len(group)),
                "n_cays": int(group["sand_cay_id"].nunique()),
                "median_gross_mobility": float(
                    group["gross_mobility_fraction_per_year"].median()
                ),
                "median_centroid_shift": float(
                    group["centroid_shift_normalized_per_year"].median()
                ),
            }
        )
    deseasoned = pd.DataFrame(deseason_rows).sort_values("state_order")
    cay_level = (
        deseason.groupby("sand_cay_id")
        .agg(
            state=("state_order", "max"),
            annual_level=("annual_median", "median"),
            gross=("gross_mobility_fraction_per_year", "median"),
            centroid=("centroid_shift_normalized_per_year", "median"),
        )
        .dropna()
    )
    rho_d, p_d = spearmanr(cay_level["annual_level"], cay_level["gross"])
    rho_dc, p_dc = spearmanr(cay_level["annual_level"], cay_level["centroid"])
    deseasoned.attrs["trend"] = (rho_d, p_d, rho_dc, p_dc)
    return gradient, paired, transitions, islands, seasonal, deseasoned


def stage_duration_estimates(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以年度中位光谱表面状态构建可审计的状态 run 与转移界限。

    本函数只计算观测到的状态跨度、上下界、删失类型和路径计数，不把
    `end_year - start_year + 1` 当作真实阶段持续时间，也不假设单向演替。
    """
    observations = pd.read_csv(
        output_dir / "沙洲观测主表.csv",
        encoding="utf-8-sig",
        usecols=[
            "reef_id",
            "sand_cay_id",
            "sensor",
            "date",
            "vegetation_fraction",
            "image_id",
        ],
    )
    observations = exclude_reefs(observations)
    observations["date"] = pd.to_datetime(observations["date"])
    observations = observations[
        observations["sensor"].isin(["sentinel2", "google_earth"])
    ]
    observations["year"] = observations["date"].dt.year
    observations = observations.merge(
        pd.read_csv(
            output_dir / "沙洲颜色组成.csv",
            encoding="utf-8-sig",
            usecols=["image_id", "dark_olive_anomaly_fraction"],
        ),
        on="image_id",
        how="left",
    )
    annual = (
        observations.groupby(["sensor", "sand_cay_id", "year"])
        .agg(
            annual_median=("vegetation_fraction", "median"),
            annual_transition_surface=("dark_olive_anomaly_fraction", "median"),
            n_images=("image_id", "size"),
        )
        .reset_index()
    )
    annual["annual_transition_surface"] = annual[
        "annual_transition_surface"
    ].fillna(0.0)
    annual["surface_state"] = np.select(
        [
            annual["annual_median"].ge(0.45),
            annual["annual_median"].ge(0.15),
            annual["annual_median"].ge(0.03),
            annual["annual_transition_surface"].ge(TRANSITION_SURFACE_STATE_MIN),
        ],
        [
            "high_green",
            "moderate_green",
            "low_green",
            "spectral_transition",
        ],
        default="bright_bare",
    )
    run_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for (sensor, cay), group in annual.groupby(["sensor", "sand_cay_id"]):
        group = group.sort_values("year")
        if len(group) < 2:
            continue
        years = group["year"].to_numpy()
        states = group["surface_state"].to_numpy()
        runs: list[dict[str, object]] = []
        start = 0
        for i in range(1, len(years)):
            if states[i] != states[start] or years[i] != years[i - 1] + 1:
                runs.append(
                    {"state": states[start], "y0": years[start], "y1": years[i - 1]}
                )
                start = i
        runs.append({"state": states[start], "y0": years[start], "y1": years[-1]})
        for rank, run in enumerate(runs):
            observed_span = float(run["y1"] - run["y0"] + 1)
            left_censored = rank == 0
            right_censored = rank == len(runs) - 1
            run_rows.append(
                {
                    "sensor": sensor,
                    "sand_cay_id": cay,
                    "surface_state": str(run["state"]),
                    "state_order": STATE4_ORDER[str(run["state"])],
                    "start_year": int(run["y0"]),
                    "end_year": int(run["y1"]),
                    "observed_span_years": observed_span,
                    "left_censored": bool(left_censored),
                    "right_censored": bool(right_censored),
                    "completed_run": bool(not left_censored and not right_censored),
                }
            )
        first_state = str(runs[0]["state"])
        transition_runs = [
            r for r in runs if str(r["state"]) == "spectral_transition"
        ]
        green_runs = [
            r
            for r in runs
            if str(r["state"]) in ("low_green", "moderate_green", "high_green")
        ]
        low_green_runs = [
            r
            for r in runs
            if str(r["state"]) in ("low_green", "moderate_green")
        ]
        high_green_runs = [r for r in runs if str(r["state"]) == "high_green"]

        def first_passage(
            source_runs: list[dict[str, object]],
            target_states: tuple[str, ...],
            require_target_after_source: bool,
        ) -> tuple[float, float, str]:
            if not source_runs:
                return np.nan, np.nan, "source_state_not_observed"
            source = source_runs[0]
            candidates = [
                r
                for r in runs
                if str(r["state"]) in target_states
                and (
                    not require_target_after_source
                    or r["y0"] >= source["y0"]
                )
            ]
            if not candidates:
                return np.nan, np.nan, "target_state_not_observed"
            target = candidates[0]
            lower = float(target["y0"] - source["y1"])
            upper = float(target["y0"] - source["y0"])
            if lower < 0:
                lower = 0.0
            status = "observed_interval_bound"
            if source is runs[0]:
                status = "left_censored_source"
            if target is runs[-1]:
                status = "right_censored_target"
            return lower, upper, status

        records = {
            "bright_to_spectral_transition": first_passage(
                [r for r in runs if str(r["state"]) == "bright_bare"],
                ("spectral_transition",),
                True,
            ),
            "spectral_transition_to_low_green": first_passage(
                transition_runs,
                ("low_green", "moderate_green", "high_green"),
                True,
            ),
            "direct_bright_to_low_green": first_passage(
                [r for r in runs if str(r["state"]) == "bright_bare"],
                ("low_green", "moderate_green", "high_green"),
                True,
            ),
            "low_green_to_high_green": first_passage(
                low_green_runs,
                ("high_green",),
                True,
            ),
        }
        if first_state != "bright_bare":
            path_type = "start_not_bright_bare"
        elif transition_runs and any(
            r["y0"] >= transition_runs[0]["y0"] for r in green_runs
        ):
            path_type = "via_spectral_transition"
        elif green_runs:
            path_type = "direct_to_green"
        elif transition_runs:
            path_type = "spectral_transition_only"
        else:
            path_type = "bright_bare_persistent"
        row = {
            "sensor": sensor,
            "sand_cay_id": cay,
            "first_state": first_state,
            "path_type": path_type,
            "n_runs": len(runs),
            "n_up_steps": int(
                sum(
                    STATE4_ORDER[str(runs[i + 1]["state"])]
                    > STATE4_ORDER[str(runs[i]["state"])]
                    for i in range(len(runs) - 1)
                )
            ),
            "n_down_steps": int(
                sum(
                    STATE4_ORDER[str(runs[i + 1]["state"])]
                    < STATE4_ORDER[str(runs[i]["state"])]
                    for i in range(len(runs) - 1)
                )
            ),
        }
        for name, (lower, upper, status) in records.items():
            row[f"{name}_lower_years"] = lower
            row[f"{name}_upper_years"] = upper
            row[f"{name}_censoring_status"] = status
        transition_rows.append(row)
    runs_frame = pd.DataFrame(run_rows)
    transitions_frame = pd.DataFrame(transition_rows)
    return runs_frame, transitions_frame


def make_vegetation_stability_figure(
    frame: pd.DataFrame,
    associations: pd.DataFrame,
    class_table: pd.DataFrame,
    bare_table: pd.DataFrame,
    modulation: pd.DataFrame,
    modulation_associations: pd.DataFrame,
    figure_dir: Path,
    source_dir: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7))
    ax_a, ax_b, ax_c, ax_d = axes

    for sensor, data in frame.groupby("sensor"):
        sub = data[["vegetation_fraction_median", "centroid_net_normalized"]].dropna()
        ax_a.scatter(
            sub["vegetation_fraction_median"],
            sub["centroid_net_normalized"],
            s=14,
            color=SENSOR_COLORS[sensor],
            label=SENSOR_LABELS[sensor],
            alpha=0.75,
            linewidth=0,
        )
        row = associations[
            associations["sensor"].eq(sensor)
            & associations["outcome"].eq("centroid_net_normalized")
        ]
        if len(row):
            stats = row.iloc[0]
            ax_a.text(
                0.03,
                0.96 - 0.10 * list(SENSOR_LABELS).index(sensor),
                f"{SENSOR_LABELS[sensor]}: rho={stats['spearman_rho']:.2f}, "
                f"q={stats['fdr_q_value']:.3f}",
                transform=ax_a.transAxes,
                va="top",
                fontsize=6.5,
                color=SENSOR_COLORS[sensor],
            )
    ax_a.set_xlabel("Cay median vegetation fraction")
    ax_a.set_ylabel("Centroid net displacement (equiv. radii)")
    ax_a.set_title("Vegetation vs. translation", loc="left")
    panel_label(ax_a, "a")

    order = [c for c in MULTIMETRIC_ORDER if c in set(frame["multimetric_class"])]
    groups = [
        frame.loc[frame["multimetric_class"].eq(c), "vegetation_fraction_median"]
        .dropna()
        .to_numpy()
        for c in order
    ]
    positions = np.arange(len(order))
    ax_b.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#333333"},
    )
    for position, cls in zip(positions, order):
        sub = frame.loc[frame["multimetric_class"].eq(cls)]
        for sensor, sensor_data in sub.groupby("sensor"):
            values = sensor_data["vegetation_fraction_median"].dropna()
            jitter = -0.16 + 0.32 * list(SENSOR_LABELS).index(sensor)
            ax_b.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=9,
                color=SENSOR_COLORS[sensor],
                alpha=0.7,
                linewidth=0,
            )
    ax_b.set_xticks(positions, [CLASS_TICKS.get(c, c) for c in order])
    ax_b.set_ylabel("Cay median vegetation fraction")
    ax_b.set_title("Vegetation by stability class", loc="left")
    panel_label(ax_b, "b")

    width = 0.36
    x = np.arange(len(bare_table))
    labels = [SENSOR_LABELS[s] for s in bare_table["sensor"]]
    ax_c.bar(
        x - width / 2,
        bare_table["persistent_reworking_fraction_bare"],
        width=width,
        color="#BDBDBD",
        label="Bare (veg < 0.03)",
    )
    ax_c.bar(
        x + width / 2,
        bare_table["persistent_reworking_fraction_vegetated"],
        width=width,
        color="#1B9E77",
        label="Vegetated",
    )
    for xi, row in zip(x, bare_table.itertuples()):
        ax_c.text(
            xi - width / 2,
            row.persistent_reworking_fraction_bare + 0.02,
            f"{row.bare_persistent_n}/{row.bare_n}",
            ha="center",
            fontsize=6.5,
        )
        ax_c.text(
            xi + width / 2,
            row.persistent_reworking_fraction_vegetated + 0.02,
            f"{row.vegetated_persistent_n}/{row.vegetated_n}",
            ha="center",
            fontsize=6.5,
        )
    ax_c.set_xticks(x, labels)
    ax_c.set_ylim(0, 1.12)
    ax_c.set_ylabel("Persistent-reworking fraction")
    ax_c.set_title("Stability of bare vs. vegetated cays", loc="left")
    ax_c.legend(frameon=False, loc="upper right", fontsize=6.5)
    panel_label(ax_c, "c")

    for bare_value, color, label in (
        (True, "#BDBDBD", "Bare"),
        (False, "#1B9E77", "Vegetated"),
    ):
        group = modulation[modulation["bare_cay"].eq(bare_value)]
        selected = group[["wave_hs_p90", "gross_mobility_fraction_per_year"]].dropna()
        ax_d.scatter(
            selected["wave_hs_p90"],
            selected["gross_mobility_fraction_per_year"],
            s=9,
            color=color,
            alpha=0.6,
            linewidth=0,
            label=label,
        )
        stats = modulation_associations[
            modulation_associations["vegetation_group"].eq(label.lower())
            & modulation_associations["driver"].eq("wave_hs_p90")
            & modulation_associations["outcome"].eq("gross_mobility_fraction_per_year")
        ]
        if len(stats):
            row = stats.iloc[0]
            ax_d.text(
                0.03,
                0.96 - 0.10 * (0 if bare_value else 1),
                f"{label}: rho={row['spearman_rho']:.2f}, n={int(row['n'])}",
                transform=ax_d.transAxes,
                va="top",
                fontsize=6.5,
                color=color,
            )
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Interval 90th-pct wave height (m)")
    ax_d.set_ylabel("Gross reworking per year (log)")
    ax_d.set_title("Background forcing x vegetation", loc="left")
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.86, wspace=0.38)
    stem = Path(__file__).resolve().parent / '归档_非主线图件与代码' / 'figures' / "figure_5_vegetation_stability"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)
    frame[
        [
            "sand_cay_id",
            "sensor",
            "multimetric_class",
            "bare_cay",
            "vegetation_fraction_median",
            "centroid_net_normalized",
            "centroid_path_normalized",
            "perimeter_max_adjacent_ratio",
            "orientation_max_step_deg",
        ]
    ].to_csv(
        source_dir / "figure5_vegetation_stability_cays.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_table.to_csv(
        source_dir / "figure5_vegetation_stability_classes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    modulation[
        [
            "sand_cay_id",
            "bare_cay",
            "vegetation_fraction_median",
            "wave_hs_p90",
            "gross_mobility_fraction_per_year",
            "wave_quartile",
        ]
    ].to_csv(
        source_dir / "figure5_vegetation_modulation.csv",
        index=False,
        encoding="utf-8-sig",
    )


def make_stage_timeline_figure(
    transition_times: pd.DataFrame,
    islands: pd.DataFrame,
    figure_dir: Path,
    source_dir: Path,
) -> None:
    """绘制表面状态路径、转移界限与绿色核心候选持续时间。"""
    configure_style()
    transition_specs = [
        (
            "bright_to_spectral_transition",
            "Bright bare -> spectral transition",
            "via_spectral_transition",
        ),
        (
            "spectral_transition_to_low_green",
            "Spectral transition -> low green",
            "via_spectral_transition",
        ),
        (
            "direct_bright_to_low_green",
            "Bright bare -> low green",
            "direct_to_green",
        ),
        (
            "low_green_to_high_green",
            "Low/moderate green -> high green",
            "all_observed_paths",
        ),
    ]
    transition_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for sensor in ["sentinel2", "google_earth"]:
        sensor_data = transition_times.loc[transition_times["sensor"].eq(sensor)]
        for order, (name, label, path_type) in enumerate(transition_specs):
            observed = sensor_data.loc[
                sensor_data[f"{name}_lower_years"].notna()
                & sensor_data[f"{name}_upper_years"].notna(),
                [
                    "sensor",
                    "sand_cay_id",
                    "path_type",
                    f"{name}_lower_years",
                    f"{name}_upper_years",
                    f"{name}_censoring_status",
                ],
            ].copy()
            observed = observed.rename(
                columns={
                    f"{name}_lower_years": "lower_bound_years",
                    f"{name}_upper_years": "upper_bound_years",
                    f"{name}_censoring_status": "censoring_status",
                }
            )
            observed["transition"] = label
            observed["transition_order"] = order
            observed["plot_years"] = observed["upper_bound_years"]
            transition_rows.extend(observed.to_dict("records"))
            lower = observed["lower_bound_years"]
            upper = observed["upper_bound_years"]
            summary_rows.append(
                {
                    "sensor": sensor,
                    "transition": label,
                    "path_type": path_type,
                    "n": int(len(observed)),
                    "median_lower_years": float(lower.median())
                    if len(observed)
                    else np.nan,
                    "median_upper_years": float(upper.median())
                    if len(observed)
                    else np.nan,
                    "q25_upper_years": float(upper.quantile(0.25))
                    if len(observed)
                    else np.nan,
                    "q75_upper_years": float(upper.quantile(0.75))
                    if len(observed)
                    else np.nan,
                    "censoring_or_resolution": (
                        "observed first-passage interval bounds; not exact transition times"
                    ),
                }
            )
    transition_points = pd.DataFrame(transition_rows)
    transition_summary = pd.DataFrame(summary_rows)
    island_source = islands.copy()
    island_source["censoring"] = "right-censored at last observation"

    source_dir.mkdir(parents=True, exist_ok=True)
    transition_points.to_csv(
        source_dir / "figure5_stage_transition_times.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transition_summary.to_csv(
        source_dir / "figure5_stage_transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    island_source.to_csv(
        source_dir / "figure5_island_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.78, 2.05],
        width_ratios=[1.62, 1.0],
        hspace=0.55,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    stage_labels = [
        "Bright bare",
        "Spectral transition",
        "Low / moderate green",
        "High green",
        "Persistent green-core\ncandidate",
    ]
    stage_colors = ["#D9C9A7", "#A98C63", "#A8D5BA", "#3E9B76", "#164B6D"]
    x_positions = np.array([0.0, 1.45, 3.05, 4.62, 6.42])
    widths = [1.0, 0.92, 1.18, 0.96, 1.34]
    for x, width, label, color in zip(
        x_positions, widths, stage_labels, stage_colors
    ):
        ax_a.add_patch(
            plt.Rectangle(
                (x - width / 2, 0.34),
                width,
                0.42,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax_a.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=6.8 if "\n" in label else 7.2,
            color="white" if color in ("#3E9B76", "#164B6D") else "#252525",
        )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax_a.annotate(
            "",
            xy=(right - 0.58, 0.55),
            xytext=(left + 0.55, 0.55),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
        )
    ax_a.annotate(
        "Direct path",
        xy=(x_positions[2] - 0.55, 0.31),
        xytext=(x_positions[0] + 0.15, 0.08),
        arrowprops={
            "arrowstyle": "->",
            "color": "#7570B3",
            "lw": 0.9,
            "connectionstyle": "arc3,rad=-0.18",
        },
        ha="center",
        fontsize=6.7,
        color="#7570B3",
    )
    ax_a.text(
        x_positions[-1],
        0.16,
        "Green core + bright margin; persistence is right-censored",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#555555",
    )
    ax_a.set_xlim(-0.72, 7.22)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.set_title("Observed spectral-surface pathways", loc="left", pad=3)
    panel_label(ax_a, "a")

    sensor_styles = {
        "sentinel2": {
            "label": "Sentinel-2",
            "color": "#164B6D",
            "offset": -0.17,
        },
        "google_earth": {
            "label": "Google Earth",
            "color": "#D95F02",
            "offset": 0.17,
        },
    }
    base_positions = np.arange(4)
    rng = np.random.default_rng(20260907)
    for sensor, style in sensor_styles.items():
        for order, (_, label, _) in enumerate(transition_specs):
            selected = transition_points.loc[
                transition_points["sensor"].eq(sensor)
                & transition_points["transition"].eq(label)
            ]
            position = base_positions[order] + style["offset"]
            if len(selected):
                values = selected["plot_years"].to_numpy()
                box = ax_b.boxplot(
                    [values],
                    positions=[position],
                    widths=0.27,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#252525", "linewidth": 1.0},
                    whiskerprops={"color": style["color"], "linewidth": 0.8},
                    capprops={"color": style["color"], "linewidth": 0.8},
                    boxprops={"edgecolor": style["color"], "linewidth": 0.9},
                )
                box["boxes"][0].set_facecolor(style["color"])
                box["boxes"][0].set_alpha(0.28 if sensor == "google_earth" else 0.55)
                for row in selected.itertuples(index=False):
                    jitter = rng.uniform(-0.055, 0.055)
                    ax_b.plot(
                        [position + jitter, position + jitter],
                        [row.lower_bound_years, row.upper_bound_years],
                        color=style["color"],
                        linewidth=0.8,
                        alpha=0.55,
                        solid_capstyle="butt",
                        zorder=2,
                    )
                jitter = rng.uniform(-0.055, 0.055, len(values))
                ax_b.scatter(
                    position + jitter,
                    values,
                    s=13,
                    facecolor=style["color"]
                    if sensor == "sentinel2"
                    else "white",
                    edgecolor=style["color"],
                    linewidth=0.65,
                    alpha=0.82,
                    zorder=3,
                )
                ax_b.text(
                    position,
                    min(25.6, float(np.nanmax(values)) + 1.2),
                    f"n={len(values)}",
                    ha="center",
                    va="bottom",
                    fontsize=6.1,
                    color=style["color"],
                )
            else:
                ax_b.text(
                    position,
                    1.0,
                    "n=0",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.1,
                    color=style["color"],
                )
    ax_b.set_xticks(
        base_positions,
        [
            "Bright bare →\nspectral transition",
            "Spectral transition →\nlow green",
            "Bright bare →\nlow green",
            "Low/moderate green →\nhigh green",
        ],
    )
    ax_b.set_ylim(0, 27.5)
    ax_b.set_ylabel("Observed transition bound (years)")
    ax_b.set_title("First-passage bounds, not exact durations", loc="left")
    ax_b.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=style["color"]
                if sensor == "sentinel2"
                else "white",
                markeredgecolor=style["color"],
                markersize=5,
                label=style["label"],
            )
            for sensor, style in sensor_styles.items()
        ],
        frameon=False,
        loc="upper left",
    )
    clean_axes(ax_b)
    panel_label(ax_b, "b")

    island_plot = island_source.sort_values(
        ["span_years_after", "sensor"], ascending=[True, True]
    ).reset_index(drop=True)
    y_island = np.arange(len(island_plot))
    for sensor, style in sensor_styles.items():
        selected = island_plot.loc[island_plot["sensor"].eq(sensor)]
        if selected.empty:
            continue
        ax_c.barh(
            y_island[selected.index],
            selected["span_years_after"],
            color=style["color"],
            alpha=0.65,
            label=style["label"],
        )
    for yi, row in zip(y_island, island_plot.itertuples()):
        ax_c.text(
            row.span_years_after + 0.35,
            yi,
            f"{row.sand_cay_id.replace('_cay_', ' ')} ({'S2' if row.sensor == 'sentinel2' else 'GE'})",
            va="center",
            fontsize=5.8,
        )
    ax_c.set_yticks([])
    ax_c.set_xlim(0, max(24, island_plot["span_years_after"].max() + 8.0))
    ax_c.set_xlabel("Observed persistence after candidate start (years)")
    ax_c.set_title("Green-core candidates (right-censored)", loc="left")
    ax_c.legend(frameon=False, loc="lower right", fontsize=6.2)
    clean_axes(ax_c)
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.92)
    fig.text(
        0.08,
        0.018,
        (
            "Vertical lines are observed lower and upper first-passage bounds; "
            "arrowheads mark persistence continuing at the last observation."
        ),
        ha="left",
        fontsize=6.4,
        color="#555555",
    )
    stem = Path(__file__).resolve().parent / '归档_非主线图件与代码' / 'figures' / "figure_5_stage_timeline"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    research_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=research_root / "outputs" / "沙洲变化区间.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    transitions = exclude_reefs(pd.read_csv(args.transition_csv))
    valid, core = prepare(transitions)
    color = pd.read_csv(args.output_dir / "沙洲颜色组成.csv", low_memory=False)
    observations = pd.read_csv(
        args.output_dir / "沙洲观测主表.csv", low_memory=False
    )
    sector_contrast, sector_stats = vegetation_sector_contrast(
        valid, color, observations
    )
    causal_tests = sector_causal_tests(transitions, observations, color)
    descriptive = describe(core)
    associations = rank_associations(core)
    windows = window_sensitivity(valid)
    perturbation_path = args.output_dir / "掩膜边界扰动稳健性.csv"
    perturbation = (
        pd.read_csv(perturbation_path, encoding="utf-8-sig")
        if perturbation_path.is_file()
        else pd.DataFrame()
    )
    perturbation_sensitivity = mask_perturbation_sensitivity(core, perturbation)
    color_validation = color_proxy_validation(
        args.output_dir / "沙洲颜色组成.csv"
    )
    candidates = event_candidates(core)
    trajectory_frame = load_trajectory_frame(args.output_dir)
    stability_associations = trajectory_rank_associations(trajectory_frame)
    class_table, bare_table = stability_class_summary(trajectory_frame)
    modulation_associations, stratified, modulation_frame = (
        forcing_vegetation_modulation(args.output_dir)
    )
    gradient, paired, transitions, islands, seasonal, deseasoned = (
        vegetation_stabilization_ladder(args.output_dir)
    )
    runs_frame, transition_times = stage_duration_estimates(args.output_dir)
    runs_frame["is_last_run"] = False
    runs_frame.loc[
        runs_frame.groupby(["sensor", "sand_cay_id"])["end_year"].idxmax(),
        "is_last_run",
    ] = True
    residence = (
        runs_frame.groupby(["sensor", "surface_state"])
        .agg(
            n_runs=("observed_span_years", "size"),
            n_completed=("completed_run", "sum"),
            n_left_censored=("left_censored", "sum"),
            n_right_censored=("right_censored", "sum"),
            median_observed_span_years_completed=(
                "observed_span_years",
                lambda s: float(s[runs_frame.loc[s.index, "completed_run"]].median())
                if runs_frame.loc[s.index, "completed_run"].any()
                else np.nan,
            ),
            q25_observed_span_years_completed=(
                "observed_span_years",
                lambda s: float(
                    s[runs_frame.loc[s.index, "completed_run"]].quantile(0.25)
                )
                if runs_frame.loc[s.index, "completed_run"].any()
                else np.nan,
            ),
            q75_observed_span_years_completed=(
                "observed_span_years",
                lambda s: float(
                    s[runs_frame.loc[s.index, "completed_run"]].quantile(0.75)
                )
                if runs_frame.loc[s.index, "completed_run"].any()
                else np.nan,
            ),
            median_observed_span_years_ongoing=(
                "observed_span_years",
                lambda s: float(
                    s[runs_frame.loc[s.index, "is_last_run"]].median()
                )
                if runs_frame.loc[s.index, "is_last_run"].any()
                else np.nan,
            ),
            n_ongoing=(
                "is_last_run",
                lambda s: int(runs_frame.loc[s.index, "is_last_run"].sum()),
            ),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid_path = args.output_dir / "形态植被台风有效区间.csv"
    core_path = args.output_dir / "形态植被台风核心区间.csv"
    descriptive_path = args.output_dir / "植被台风分层描述统计.csv"
    associations_path = args.output_dir / "植被形态秩相关.csv"
    window_path = args.output_dir / "植被时间窗敏感性.csv"
    sector_path = args.output_dir / "植被扇区对照.csv"
    sector_stats_path = args.output_dir / "植被扇区对照统计.csv"
    causal_path = args.output_dir / "植被扇区因果滞后检验.csv"
    perturbation_sensitivity_path = (
        args.output_dir / "植被掩膜扰动敏感性.csv"
    )
    color_validation_path = args.output_dir / "植被颜色代理验证.csv"
    candidates_path = args.output_dir / "强台风事件候选.csv"
    summary_path = args.output_dir / "植被形态台风分析核查.json"
    stability_assoc_path = args.output_dir / "植被稳定性轨迹秩相关.csv"
    stability_class_path = args.output_dir / "植被稳定性分类汇总.csv"
    stability_bare_path = args.output_dir / "植被稳定性裸沙对照.csv"
    modulation_path = args.output_dir / "植被调节强迫响应.csv"
    stratified_path = args.output_dir / "植被强迫分层可动性.csv"
    ladder_path = args.output_dir / "植被状态稳定性梯度.csv"
    paired_ladder_path = args.output_dir / "植被建立沙洲内前后对照.csv"
    transition_path = args.output_dir / "植被状态转移计数.csv"
    island_path = args.output_dir / "岛状稳定候选.csv"
    seasonal_path = args.output_dir / "植被年内季节振幅.csv"
    deseason_path = args.output_dir / "植被去季节化稳定性梯度.csv"
    residence_path = args.output_dir / "沙洲阶段驻留时长汇总.csv"
    stage_runs_path = args.output_dir / "沙洲阶段驻留run明细.csv"
    transition_times_path = args.output_dir / "沙洲阶段转移时间.csv"
    valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
    core.to_csv(core_path, index=False, encoding="utf-8-sig")
    descriptive.to_csv(descriptive_path, index=False, encoding="utf-8-sig")
    associations.to_csv(associations_path, index=False, encoding="utf-8-sig")
    windows.to_csv(window_path, index=False, encoding="utf-8-sig")
    sector_contrast.to_csv(sector_path, index=False, encoding="utf-8-sig")
    sector_stats.to_csv(sector_stats_path, index=False, encoding="utf-8-sig")
    causal_tests.to_csv(causal_path, index=False, encoding="utf-8-sig")
    perturbation_sensitivity.to_csv(
        perturbation_sensitivity_path, index=False, encoding="utf-8-sig"
    )
    color_validation.to_csv(
        color_validation_path, index=False, encoding="utf-8-sig"
    )
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    stability_associations.to_csv(
        stability_assoc_path, index=False, encoding="utf-8-sig"
    )
    class_table.to_csv(stability_class_path, index=False, encoding="utf-8-sig")
    bare_table.to_csv(stability_bare_path, index=False, encoding="utf-8-sig")
    modulation_associations.to_csv(modulation_path, index=False, encoding="utf-8-sig")
    stratified.to_csv(stratified_path, index=False, encoding="utf-8-sig")
    gradient.to_csv(ladder_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_ladder_path, index=False, encoding="utf-8-sig")
    transitions.to_csv(transition_path, index=False, encoding="utf-8-sig")
    islands.to_csv(island_path, index=False, encoding="utf-8-sig")
    seasonal.to_csv(seasonal_path, index=False, encoding="utf-8-sig")
    deseasoned.to_csv(deseason_path, index=False, encoding="utf-8-sig")
    residence.to_csv(residence_path, index=False, encoding="utf-8-sig")
    runs_frame.to_csv(stage_runs_path, index=False, encoding="utf-8-sig")
    transition_times.to_csv(transition_times_path, index=False, encoding="utf-8-sig")
    figure_dir = args.output_dir / "figures"
    source_dir = figure_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    make_vegetation_stability_figure(
        trajectory_frame,
        stability_associations,
        class_table,
        bare_table,
        modulation_frame,
        modulation_associations,
        figure_dir,
        source_dir,
    )
    make_stage_timeline_figure(
        transition_times,
        islands,
        figure_dir,
        source_dir,
    )

    summary = {
        "analysis_type": "descriptive and rank-association only; no predictive model fitted",
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "valid_intervals": int(len(valid)),
        "core_intervals": int(len(core)),
        "core_rule": {
            "minimum_interval_days": CORE_MIN_INTERVAL_DAYS,
            "maximum_interval_days": CORE_MAX_INTERVAL_DAYS,
            "sensitivity_windows": [list(days) for days in WINDOW_SENSITIVITY],
            "reason": "减少过短观测间隔的配准/像元噪声，以及过长间隔中无法分辨事件时序的累计变化。",
        },
        "mask_perturbation_sensitivity": {
            "status": "not_attempted" if perturbation.empty else "completed",
            "perturbation_pixels": sorted(
                perturbation_sensitivity["perturbation_pixels"].unique().tolist()
            )
            if not perturbation_sensitivity.empty
            else [],
        },
        "M4_vegetation_sector_contrast": {
            "transitions": int(len(sector_contrast)),
            "strata": sector_stats.loc[
                sector_stats["status"].eq("ok"), ["stratum", "mean_mobility_difference", "bootstrap_ci_high", "wilcoxon_p_value", "supported_negative"]
            ].to_dict("records")
            if not sector_stats.empty
            else [],
        },
        "M7_M9_sector_causal_lag": causal_tests.to_dict("records")
        if not causal_tests.empty
        else [],
        "color_proxy_validation": {
            "status": "completed" if not color_validation.empty else "no_audit_records",
            "rows": int(len(color_validation)),
            "threshold_perturbations": VEGETATION_THRESHOLD_PERTURBATIONS,
            "manual_label_policy": "audit only; never an analysis input",
        },
        "core_by_sensor": {str(k): int(v) for k, v in core["sensor"].value_counts().items()},
        "strong_typhoon_candidates": int(len(candidates)),
        "association_warning": "重复时序和未控制潮位、传感器、礁盘地貌等混杂因素，因此相关系数仅描述关联，不代表因果。",
        "trajectory_stability": {
            "unit": "primary cay trajectory (same cay x sensor x reference frame)",
            "bare_threshold_vegetation_fraction": VEGETATION_BINS[1],
            "outcome_columns": STABILITY_OUTCOMES,
            "class_counts": {
                str(key): int(value)
                for key, value in trajectory_frame["multimetric_class"]
                .value_counts()
                .items()
            },
            "warning": (
                "Vegetation fraction is a remote-sensing color proxy; associations "
                "with stability remain observational and tide/sensor/reef confounded."
            ),
            "modulation": {
                "unit": "Sentinel-2 background intervals (no typhoon exposure)",
                "groups": "bare (cay median vegetation < 0.03) vs vegetated",
                "pairs": [
                    {"driver": d, "outcome": o} for d, o in MODULATION_PAIRS
                ],
                "warning": (
                    "Modulation is descriptive stratification; reef exposure and cay "
                    "size co-vary with vegetation and remain uncontrolled."
                ),
            },
            "stabilization_ladder": {
                "hypothesis": (
                    "Vegetation states form a progressive stabilization ladder "
                    "(cay to island), setting stability level rather than modulating "
                    "forcing-response slopes."
                ),
                "threshold_vegetation_fraction": STABILIZATION_THRESHOLD,
                "gradient_trend_gross_rho": float(gradient.attrs["trend"][0]),
                "gradient_trend_gross_p": float(gradient.attrs["trend"][1]),
                "gradient_trend_centroid_rho": float(gradient.attrs["trend"][2]),
                "gradient_trend_centroid_p": float(gradient.attrs["trend"][3]),
                "paired_within_cay": (
                    {
                        "n_s2_cays": int(
                            (paired["sensor"].eq("sentinel2")).sum()
                        ),
                        "n_s2_decreases": int(
                            paired.loc[
                                paired["sensor"].eq("sentinel2"),
                                "log_ratio_after_before",
                            ]
                            .lt(0)
                            .sum()
                        ),
                        "wilcoxon_p_s2": (
                            float(
                                wilcoxon(
                                    paired.loc[
                                        paired["sensor"].eq("sentinel2"),
                                        "log_ratio_after_before",
                                    ]
                                ).pvalue
                            )
                            if (paired["sensor"].eq("sentinel2")).sum() >= 8
                            else None
                        ),
                    }
                    if len(paired)
                    else {"n_s2_cays": 0}
                ),
                "transition_sums": {
                    str(sensor): {
                        "up": int(row["n_up_steps"]),
                        "down": int(row["n_down_steps"]),
                    }
                    for sensor, row in transitions.groupby("sensor")[
                        ["n_up_steps", "n_down_steps"]
                    ]
                    .sum()
                    .iterrows()
                },
                "island_like_candidates": int(len(islands)),
                "seasonal_amplitude": {
                    str(sensor): {
                        "n_cays": int(row["n_cays"]),
                        "median_within_year_range": float(
                            row["median_within_year_range"]
                        ),
                    }
                    for sensor, row in seasonal.groupby("sensor")
                    .agg(
                        n_cays=("sand_cay_id", "nunique"),
                        median_within_year_range=(
                            "median_within_year_range",
                            "median",
                        ),
                    )
                    .iterrows()
                },
                "deseasoned_gradient_trend_gross_rho": float(
                    deseasoned.attrs["trend"][0]
                ),
                "deseasoned_gradient_trend_gross_p": float(
                    deseasoned.attrs["trend"][1]
                ),
                "deseasoned_gradient_trend_centroid_rho": float(
                    deseasoned.attrs["trend"][2]
                ),
                "deseasoned_gradient_trend_centroid_p": float(
                    deseasoned.attrs["trend"][3]
                ),
                "vegetation_source": (
                    "All vegetation variables are continuous color-classification "
                    "fractions from cay masks (per-cay white balance); manual labels "
                    "are used only for audit/conflict flags, never as analysis inputs."
                ),
                "stage_durations": {
                    "method": (
                        "Annual-median color-derived surface states collapsed to "
                        "bright bare / spectral transition / low-moderate green / "
                        "high green; runs report observed spans and first-passage "
                        "lower/upper bounds, not exact transition times. First/last "
                        "runs are left/right censored; missing years break runs."
                    ),
                    "transition_lower_years_median": {
                        name: {
                            str(sensor): float(
                                sub.loc[
                                    sub[f"{name}_lower_years"].notna(),
                                    f"{name}_lower_years",
                                ].median()
                            )
                            if sub[f"{name}_lower_years"].notna().any()
                            else None
                            for sensor, sub in transition_times.groupby("sensor")
                        }
                        for name in TRANSITION_NAMES
                    },
                    "transition_upper_years_median": {
                        name: {
                            str(sensor): float(
                                sub.loc[
                                    sub[f"{name}_upper_years"].notna(),
                                    f"{name}_upper_years",
                                ].median()
                            )
                            if sub[f"{name}_upper_years"].notna().any()
                            else None
                            for sensor, sub in transition_times.groupby("sensor")
                        }
                        for name in TRANSITION_NAMES
                    },
                    "transition_observed_n": {
                        name: {
                            str(sensor): int(
                                sub[f"{name}_lower_years"].notna().sum()
                            )
                            for sensor, sub in transition_times.groupby("sensor")
                        }
                        for name in TRANSITION_NAMES
                    },
                    "path_type_counts": {
                        str(sensor): {
                            str(key): int(value)
                            for key, value in sub["path_type"].value_counts().items()
                        }
                        for sensor, sub in transition_times.groupby("sensor")
                    },
                    "island_persistence_years_median": (
                        float(islands["span_years_after"].median())
                        if len(islands)
                        else None
                    ),
                    "caveats": (
                        "Observation cadence (S2 ~95 d, GE ~350 d) sets the lower "
                        "resolution; reported values are interval bounds, not exact "
                        "durations; states are color-proxy and reversible."
                    ),
                },
                "warning": (
                    "State transitions may follow prior stability (reverse causality); "
                    "vegetation remains a color proxy."
                ),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in [
        valid_path,
        core_path,
        descriptive_path,
        associations_path,
        window_path,
        sector_path,
        sector_stats_path,
        causal_path,
        perturbation_sensitivity_path,
        color_validation_path,
        candidates_path,
        stability_assoc_path,
        stability_class_path,
        stability_bare_path,
        modulation_path,
        stratified_path,
        ladder_path,
        paired_ladder_path,
        transition_path,
        island_path,
        seasonal_path,
        deseason_path,
        residence_path,
        stage_runs_path,
        transition_times_path,
        summary_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
