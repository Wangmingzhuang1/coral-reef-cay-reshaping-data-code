"""开展非建模的植被比例、形态变化与台风暴露描述和关联分析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from 提取沙洲颜色组成 import vegetation_pixels
from 构建沙洲观测与变化表 import read_mask as _read_mask, select_analysis_observations
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
    if not np.asarray(mask).any():
        return None
    return vegetation_pixels(image_bgr, mask)


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










def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-csv", type=Path, default=root / "outputs" / "沙洲变化区间.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    intervals = exclude_reefs(pd.read_csv(args.transition_csv))
    valid, core = prepare(intervals)
    color = pd.read_csv(args.output_dir / "沙洲颜色组成.csv", low_memory=False)
    observations = select_analysis_observations(pd.read_csv(args.output_dir / "沙洲观测主表.csv", low_memory=False))
    contrast, sector_stats = vegetation_sector_contrast(valid, color, observations)
    lag_tests = sector_causal_tests(intervals, observations, color)
    tracks = load_trajectory_frame(args.output_dir)
    track_correlations = trajectory_rank_associations(tracks)
    classes, bare = stability_class_summary(tracks)
    perturbation = pd.read_csv(args.output_dir / "掩膜边界扰动稳健性.csv")
    tables = {
        "形态植被台风有效区间.csv": valid,
        "形态植被台风核心区间.csv": core,
        "植被台风分层描述统计.csv": describe(core),
        "植被形态秩相关.csv": rank_associations(core),
        "植被时间窗敏感性.csv": window_sensitivity(valid),
        "植被扇区对照.csv": contrast,
        "植被扇区对照统计.csv": sector_stats,
        "植被扇区因果滞后检验.csv": lag_tests,
        "植被掩膜扰动敏感性.csv": mask_perturbation_sensitivity(core, perturbation),
        "植被颜色代理验证.csv": color_proxy_validation(args.output_dir / "沙洲颜色组成.csv"),
        "强台风事件候选.csv": event_candidates(core),
        "植被稳定性轨迹秩相关.csv": track_correlations,
        "植被稳定性分类汇总.csv": classes,
        "植被稳定性裸沙对照.csv": bare,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")
    source = args.output_dir / "figures" / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    columns = ["sand_cay_id", "sensor", "multimetric_class", "bare_cay", "vegetation_fraction_median", *STABILITY_OUTCOMES]
    tracks[columns].to_csv(source / "figure5_vegetation_stability_cays.csv", index=False, encoding="utf-8-sig")
    summary = {
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "primary_window_days": [CORE_MIN_INTERVAL_DAYS, CORE_MAX_INTERVAL_DAYS],
        "n_valid_intervals": int(len(valid)), "n_core_intervals": int(len(core)),
        "n_core_cays": int(core.sand_cay_id.nunique()),
        "sector_contrasts": sector_stats.to_dict("records"),
        "adjacent_sector_associations": lag_tests.to_dict("records"),
        "trajectory_correlations": track_correlations.to_dict("records"),
        "interpretation": "覆盖水平、同影像岸段对照与严格相邻观测关联；不将其作为植被干预效应。",
    }
    (args.output_dir / "植被形态台风分析核查.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"core_intervals": len(core), "sector_pairs": len(contrast), "trajectory_cays": len(tracks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
