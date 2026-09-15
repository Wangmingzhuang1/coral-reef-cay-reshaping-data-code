"""构建台风事件前后沙洲影像对，并进行事件尺度的定量描述。

主分析窗口为 T-60 至 T-5 天和 T+5 至 T+60 天。前后影像必须属于
同一沙洲、同一传感器和同一固定参考框架。分析结果是事件关联，不是因果效应。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from 分析范围 import (
    ANALYSIS_CONTRACT_VERSION,
    analysis_contract_digest,
    exclude_reefs,
    load_excluded_reefs,
    module_contract,
)


TYPHOON_CONTRACT = module_contract("typhoon_event")
MAIN_WINDOW_DAYS = abs(TYPHOON_CONTRACT["event_window_days"][0])
EVENT_BUFFER_DAYS = abs(TYPHOON_CONTRACT["event_window_days"][1])
SENSITIVITY_WINDOWS = [MAIN_WINDOW_DAYS] + [
    abs(window[3]) for window in TYPHOON_CONTRACT["sensitivity_window_days"]
]
EXTENDED_WINDOW_DAYS = 90
EXTENDED_RADIUS_KM = 250
MINIMUM_STORMS_FOR_INFERENCE = 5
LOCAL_WIND_THRESHOLD_M_S = 17.5
# 研究总体包括裸沙与含植被沙洲；人工阶段仅用于数据质量核查。
STABLE_STAGES = None
ACCEPTED_QUALITY = {"A", "B"}
OUTCOMES = [
    "erosion_fraction",
    "deposition_fraction",
    "gross_mobility_fraction",
    "net_area_fraction",
    "centroid_shift_normalized",
    "redistribution_balance_index",
]
RANDOM_SEED = 20260903


def read_mask(path_text: object) -> np.ndarray | None:
    path = Path(str(path_text))
    if not path.is_file():
        return None
    try:
        buffer = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return mask > 0 if mask is not None else None


def finite_number(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else np.nan


def prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Prepare the Sentinel-2-only observations used for event inference.

    Google Earth remains available for the long-term descriptive products, but
    event hypothesis tests must not pool its irregular acquisitions with
    Sentinel-2 observations.
    """
    data = observations.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["quality_rank"] = data["quality_grade"].map({"A": 3, "B": 2, "C": 1}).fillna(0)
    data["pixel_count"] = pd.to_numeric(data["pixel_count"], errors="coerce")
    data = data[
        data["sensor"].eq("sentinel2")
        & data["quality_grade"].isin(ACCEPTED_QUALITY)
        & data["development_stage"].notna()
        & data["date"].notna()
    ].copy()
    keys = ["sensor", "sand_cay_id", "reference_frame_id", "date"]
    return (
        data.sort_values(
            keys + ["quality_rank", "pixel_count"],
            ascending=[True, True, True, True, False, False],
        )
        .drop_duplicates(keys)
        .reset_index(drop=True)
    )


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    data["nearest_time_utc"] = pd.to_datetime(data["nearest_time_utc"], errors="coerce")
    for column in [
        "nearest_dist_km",
        "wind_at_nearest_ms",
        "storm_max_wind_ms",
        "inside_r34_quadrant_at_nearest",
        "event_relevant_quadrant",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["inside_r34"] = data["inside_r34_quadrant_at_nearest"].fillna(0).astype(int)
    data["event_relevant"] = data["event_relevant_quadrant"].fillna(0).eq(1)
    data["event_candidate_250km"] = data["nearest_dist_km"].le(250)
    return data[data["nearest_time_utc"].notna()].sort_values(
        ["sand_cay_id", "nearest_time_utc"]
    )


def response_from_pair(
    pre: pd.Series,
    post: pd.Series,
) -> dict[str, object]:
    pixel_size = float(
        np.nanmean([finite_number(pre.get("pixel_size_m")), finite_number(post.get("pixel_size_m"))])
    )
    area_pre = finite_number(pre.get("sand_cay_area_m2"))
    area_post = finite_number(post.get("sand_cay_area_m2"))
    mask_pre = read_mask(pre.get("mask_path", ""))
    mask_post = read_mask(post.get("mask_path", ""))
    if (
        mask_pre is None
        or mask_post is None
        or mask_pre.shape != mask_post.shape
        or not np.isfinite(pixel_size)
        or not np.isfinite(area_pre)
        or area_pre <= 0
    ):
        return {"response_status": "mask_scale_or_area_unavailable"}

    erosion_area = float(np.logical_and(mask_pre, ~mask_post).sum() * pixel_size**2)
    deposition_area = float(np.logical_and(~mask_pre, mask_post).sum() * pixel_size**2)
    dx = (
        finite_number(post.get("sand_cay_centroid_x"))
        - finite_number(pre.get("sand_cay_centroid_x"))
    ) * pixel_size
    dy = (
        finite_number(post.get("sand_cay_centroid_y"))
        - finite_number(pre.get("sand_cay_centroid_y"))
    ) * pixel_size
    centroid_shift = float(np.hypot(dx, dy))
    equivalent_radius = float(np.sqrt(area_pre / np.pi))
    gross_change = erosion_area + deposition_area
    redistribution_balance = (
        2 * min(erosion_area, deposition_area) / gross_change
        if gross_change > 0
        else np.nan
    )
    extreme_response = (
        gross_change / area_pre > 1
        or centroid_shift / equivalent_radius > 1
        or abs((area_post - area_pre) / area_pre) > 0.75
    )
    return {
        "response_status": "ok",
        "response_qc_flag": (
            "extreme_requires_visual_review" if extreme_response else "within_screening_range"
        ),
        "area_pre_m2": area_pre,
        "area_post_m2": area_post,
        "erosion_area_m2": erosion_area,
        "deposition_area_m2": deposition_area,
        "gross_boundary_change_m2": gross_change,
        "erosion_fraction": erosion_area / area_pre,
        "deposition_fraction": deposition_area / area_pre,
        "gross_mobility_fraction": gross_change / area_pre,
        "net_area_fraction": (area_post - area_pre) / area_pre,
        "centroid_shift_m": centroid_shift,
        "centroid_shift_normalized": centroid_shift / equivalent_radius,
        "redistribution_balance_index": redistribution_balance,
        "vegetation_fraction_pre": finite_number(pre.get("vegetation_fraction")),
        "vegetation_fraction_post": finite_number(post.get("vegetation_fraction")),
    }


def build_event_pairs(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int,
    event_flag: str = "event_relevant",
) -> pd.DataFrame:
    relevant = events[events[event_flag]].copy()
    rows: list[dict[str, object]] = []
    for sand_cay_id, cay_events in relevant.groupby("sand_cay_id"):
        cay_observations = observations[observations["sand_cay_id"].eq(sand_cay_id)]
        if cay_observations.empty:
            continue
        for event in cay_events.itertuples(index=False):
            event_time = event.nearest_time_utc
            for (sensor, reference_frame_id), group in cay_observations.groupby(
                ["sensor", "reference_frame_id"], dropna=False
            ):
                pre_candidates = group[
                    group["date"].between(
                        event_time - pd.Timedelta(days=window_days),
                        event_time - pd.Timedelta(days=EVENT_BUFFER_DAYS),
                    )
                ]
                post_candidates = group[
                    group["date"].between(
                        event_time + pd.Timedelta(days=EVENT_BUFFER_DAYS),
                        event_time + pd.Timedelta(days=window_days),
                    )
                ]
                if pre_candidates.empty or post_candidates.empty:
                    continue
                pre = pre_candidates.loc[pre_candidates["date"].idxmax()]
                post = post_candidates.loc[post_candidates["date"].idxmin()]
                competing = relevant[
                    relevant["sand_cay_id"].eq(sand_cay_id)
                    & relevant["nearest_time_utc"].between(
                        pre["date"], post["date"], inclusive="neither"
                    )
                    & ~relevant["sid"].eq(event.sid)
                ]
                if not competing.empty:
                    continue
                row = {
                    "window_days": window_days,
                    "event_pair_id": (
                        f"{event.sid}__{sand_cay_id}__{sensor}__"
                        f"{pre['date'].date()}__{post['date'].date()}"
                    ),
                    "sid": event.sid,
                    "storm_name": event.name,
                    "reef_id": event.reef_id,
                    "sand_cay_id": sand_cay_id,
                    "sensor": sensor,
                    "reference_frame_id": reference_frame_id,
                    "event_time_utc": event_time,
                    "pre_date": pre["date"],
                    "post_date": post["date"],
                    "pre_lag_days": int((event_time.normalize() - pre["date"]).days),
                    "post_lag_days": int((post["date"] - event_time.normalize()).days),
                    "pair_span_days": int((post["date"] - pre["date"]).days),
                    "nearest_dist_km": event.nearest_dist_km,
                    "wind_at_nearest_ms": event.wind_at_nearest_ms,
                    "storm_max_wind_ms": event.storm_max_wind_ms,
                    "inside_r34": event.inside_r34,
                    "storm_center_bearing_from_cay_deg": getattr(
                        event, "storm_center_bearing_from_cay_deg", np.nan
                    ),
                    "storm_motion_bearing_deg": getattr(
                        event, "storm_motion_bearing_deg", np.nan
                    ),
                    "local_duration_within_250km_h": getattr(
                        event, "local_duration_within_250km_h", np.nan
                    ),
                    "local_duration_within_100km_h": getattr(
                        event, "local_duration_within_100km_h", np.nan
                    ),
                    "local_r34_duration_approx_h": getattr(
                        event, "local_r34_duration_approx_h", np.nan
                    ),
                    "center_wind_ge17_5_within_250km_h": getattr(
                        event, "center_wind_ge17_5_within_250km_h", np.nan
                    ),
                    "quality_grade_pre": pre["quality_grade"],
                    "quality_grade_post": post["quality_grade"],
                    "development_stage_pre": pre["development_stage"],
                    "development_stage_post": post["development_stage"],
                    "pre_image_id": pre["image_id"],
                    "post_image_id": post["image_id"],
                    "pre_mask_path": pre["mask_path"],
                    "post_mask_path": post["mask_path"],
                }
                row.update(response_from_pair(pre, post))
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["event_time_utc", "sid", "sand_cay_id", "sensor"])


def pairing_funnel(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int,
) -> dict[str, int]:
    relevant = events[events["event_relevant"]]
    counts = {
        "direct_event_cay_records": int(len(relevant)),
        "direct_storms": int(relevant["sid"].nunique()),
        "direct_cays": int(relevant["sand_cay_id"].nunique()),
        "event_cay_records_with_any_observation": 0,
        "event_cay_records_inside_observation_span": 0,
        "event_cay_records_with_pre_image": 0,
        "event_cay_records_with_post_image": 0,
        "event_cay_records_with_same_frame_pair": 0,
        "isolated_same_frame_pairs": 0,
    }
    for event in relevant.itertuples(index=False):
        cay_observations = observations[
            observations["sand_cay_id"].eq(event.sand_cay_id)
        ]
        if cay_observations.empty:
            continue
        counts["event_cay_records_with_any_observation"] += 1
        if (
            cay_observations["date"].min() < event.nearest_time_utc
            and cay_observations["date"].max() > event.nearest_time_utc
        ):
            counts["event_cay_records_inside_observation_span"] += 1
        has_pre = False
        has_post = False
        has_pair = False
        has_isolated_pair = False
        for _, group in cay_observations.groupby(
            ["sensor", "reference_frame_id"], dropna=False
        ):
            pre = group[
                group["date"].between(
                    event.nearest_time_utc - pd.Timedelta(days=window_days),
                    event.nearest_time_utc - pd.Timedelta(days=EVENT_BUFFER_DAYS),
                )
            ]
            post = group[
                group["date"].between(
                    event.nearest_time_utc + pd.Timedelta(days=EVENT_BUFFER_DAYS),
                    event.nearest_time_utc + pd.Timedelta(days=window_days),
                )
            ]
            has_pre |= not pre.empty
            has_post |= not post.empty
            if pre.empty or post.empty:
                continue
            has_pair = True
            pre_date = pre["date"].max()
            post_date = post["date"].min()
            competing = relevant[
                relevant["sand_cay_id"].eq(event.sand_cay_id)
                & relevant["nearest_time_utc"].between(
                    pre_date, post_date, inclusive="neither"
                )
                & ~relevant["sid"].eq(event.sid)
            ]
            has_isolated_pair |= competing.empty
        counts["event_cay_records_with_pre_image"] += int(has_pre)
        counts["event_cay_records_with_post_image"] += int(has_post)
        counts["event_cay_records_with_same_frame_pair"] += int(has_pair)
        counts["isolated_same_frame_pairs"] += int(has_isolated_pair)
    return counts


def add_stage_to_transitions(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    stage = observations[
        ["sensor", "sand_cay_id", "reference_frame_id", "date", "development_stage"]
    ].copy()
    result = transitions.copy()
    result["time_t"] = pd.to_datetime(result["time_t"], errors="coerce")
    result["time_t1"] = pd.to_datetime(result["time_t1"], errors="coerce")
    result = result.merge(
        stage.rename(columns={"date": "time_t", "development_stage": "stage_t"}),
        on=["sensor", "sand_cay_id", "reference_frame_id", "time_t"],
        how="left",
        validate="many_to_one",
    )
    return result.merge(
        stage.rename(columns={"date": "time_t1", "development_stage": "stage_t1"}),
        on=["sensor", "sand_cay_id", "reference_frame_id", "time_t1"],
        how="left",
        validate="many_to_one",
    )


def eligible_controls(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    relevant_events: pd.DataFrame,
) -> pd.DataFrame:
    data = add_stage_to_transitions(transitions, observations)
    numeric = [
        "time_interval_days",
        "area_t_m2",
        "erosion_area_m2",
        "deposition_area_m2",
        "gross_boundary_change_m2",
        "centroid_shift_m",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        data["boundary_change_status"].eq("ok")
        & data["quality_grade_t"].isin(ACCEPTED_QUALITY)
        & data["quality_grade_t1"].isin(ACCEPTED_QUALITY)
        & data["stage_t"].notna()
        & data["stage_t1"].notna()
        & data["area_t_m2"].gt(0)
    ].copy()
    has_relevant_event = []
    for transition in data.itertuples(index=False):
        selected = relevant_events[
            relevant_events["sand_cay_id"].eq(transition.sand_cay_id)
            & relevant_events["nearest_time_utc"].between(
                transition.time_t, transition.time_t1, inclusive="right"
            )
        ]
        has_relevant_event.append(not selected.empty)
    data["has_relevant_event"] = has_relevant_event
    data = data[~data["has_relevant_event"]].copy()
    data["erosion_fraction"] = data["erosion_area_m2"] / data["area_t_m2"]
    data["deposition_fraction"] = data["deposition_area_m2"] / data["area_t_m2"]
    data["gross_mobility_fraction"] = data["gross_boundary_change_m2"] / data["area_t_m2"]
    data["net_area_fraction"] = data["area_change_m2"] / data["area_t_m2"]
    data["centroid_shift_normalized"] = data["centroid_shift_m"] / np.sqrt(
        data["area_t_m2"] / np.pi
    )
    gross = data["erosion_area_m2"] + data["deposition_area_m2"]
    data["redistribution_balance_index"] = np.where(
        gross.gt(0),
        2 * np.minimum(data["erosion_area_m2"], data["deposition_area_m2"]) / gross,
        np.nan,
    )
    data["control_midpoint"] = data["time_t"] + (data["time_t1"] - data["time_t"]) / 2
    return data


def match_controls(
    event_pairs: pd.DataFrame,
    controls: pd.DataFrame,
    tide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if event_pairs.empty:
        return event_pairs
    used_controls: set[str] = set()
    rows: list[dict[str, object]] = []
    for event in event_pairs.sort_values("event_time_utc").itertuples(index=False):
        candidates = controls[
            controls["sand_cay_id"].eq(event.sand_cay_id)
            & controls["sensor"].eq(event.sensor)
            & controls["reference_frame_id"].eq(event.reference_frame_id)
            & controls["time_interval_days"].between(
                max(10, event.pair_span_days * 0.5),
                event.pair_span_days * 2.0,
            )
            & ~controls["transition_id"].isin(used_controls)
        ].copy()
        if candidates.empty:
            continue
        candidates["duration_score"] = np.abs(
            np.log(candidates["time_interval_days"] / event.pair_span_days)
        )
        candidates["time_score"] = (
            candidates["control_midpoint"] - event.event_time_utc
        ).abs().dt.days / 365.2425
        candidates["tide_score"] = np.nan
        candidates["match_score"] = candidates["duration_score"] + 0.05 * candidates["time_score"]
        if tide is not None and "tide_mean_m" in candidates.columns:
            candidates["tide_score"] = np.abs(
                candidates["tide_mean_m"] - event.tide_mean_m
            )
            candidates["match_score"] = (
                candidates["duration_score"]
                + 0.05 * candidates["time_score"]
                + 0.25 * candidates["tide_score"].fillna(
                    candidates["tide_score"].max() if candidates["tide_score"].notna().any() else 0
                )
            )
        control = candidates.sort_values(["match_score", "time_score"]).iloc[0]
        used_controls.add(str(control["transition_id"]))
        row = {
            "event_pair_id": event.event_pair_id,
            "sid": event.sid,
            "storm_name": event.storm_name,
            "sand_cay_id": event.sand_cay_id,
            "sensor": event.sensor,
            "event_time_utc": event.event_time_utc,
            "event_response_qc_flag": event.response_qc_flag,
            "event_pair_span_days": event.pair_span_days,
            "control_transition_id": control["transition_id"],
            "control_start": control["time_t"],
            "control_end": control["time_t1"],
            "control_span_days": control["time_interval_days"],
            "match_score": control["match_score"],
        }
        for outcome in OUTCOMES:
            event_value = finite_number(getattr(event, outcome))
            control_value = finite_number(control[outcome])
            row[f"event_{outcome}"] = event_value
            row[f"control_{outcome}"] = control_value
            row[f"difference_{outcome}"] = event_value - control_value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_windows(all_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_days, group in all_pairs.groupby("window_days"):
        valid = group[group["response_status"].eq("ok")]
        row: dict[str, object] = {
            "window_days": int(window_days),
            "n_pairs": int(len(group)),
            "n_valid_pairs": int(len(valid)),
            "n_storms": int(valid["sid"].nunique()),
            "n_cays": int(valid["sand_cay_id"].nunique()),
            "n_reefs": int(valid["reef_id"].nunique()),
            "n_google_earth": int(valid["sensor"].eq("google_earth").sum()),
            "n_sentinel2": int(valid["sensor"].eq("sentinel2").sum()),
        }
        for outcome in OUTCOMES:
            row[f"{outcome}_median"] = float(valid[outcome].median())
            row[f"{outcome}_q25"] = float(valid[outcome].quantile(0.25))
            row[f"{outcome}_q75"] = float(valid[outcome].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("window_days")


def summarize_events(main_pairs: pd.DataFrame) -> pd.DataFrame:
    if main_pairs.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "event_time_utc": ("event_time_utc", "first"),
        "storm_name": ("storm_name", "first"),
        "n_event_cay_pairs": ("event_pair_id", "size"),
        "n_cays": ("sand_cay_id", "nunique"),
        "n_reefs": ("reef_id", "nunique"),
        "nearest_distance_km": ("nearest_dist_km", "min"),
        "maximum_local_wind_m_s": ("wind_at_nearest_ms", "max"),
        "r34_pair_count": ("inside_r34", "sum"),
    }
    for outcome in OUTCOMES:
        aggregations[f"{outcome}_median"] = (outcome, "median")
    result = main_pairs.groupby("sid", as_index=False).agg(**aggregations)
    return result.sort_values(
        ["gross_mobility_fraction_median", "n_cays"],
        ascending=[False, False],
    )


def summarize_matched(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    specifications = {
        "all_events": matched,
        "exclude_extreme_pending_review": matched[
            ~matched["event_response_qc_flag"].eq("extreme_requires_visual_review")
        ],
    }
    for specification, frame in specifications.items():
        storm_level = frame.groupby("sid", as_index=False)[
            [f"difference_{outcome}" for outcome in OUTCOMES]
        ].median()
        for outcome in OUTCOMES:
            values = storm_level[f"difference_{outcome}"].dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            bootstrap = np.median(
                values[rng.integers(0, len(values), size=(5000, len(values)))],
                axis=1,
            )
            try:
                p_value = float(wilcoxon(values, alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "specification": specification,
                    "outcome": outcome,
                    "n_matched_event_pairs": int(
                        frame[f"difference_{outcome}"].notna().sum()
                    ),
                    "n_independent_storms": int(len(values)),
                    "storm_median_difference": float(np.median(values)),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "wilcoxon_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    for specification, index in result.groupby("specification").groups.items():
        valid = result.loc[index, "wilcoxon_p_value"].notna()
        selected = result.loc[index].index[valid]
        if len(selected):
            result.loc[selected, "fdr_q_value"] = multipletests(
                result.loc[selected, "wilcoxon_p_value"], method="fdr_bh"
            )[1]
    result["inference_status"] = np.where(
        result["n_independent_storms"] >= MINIMUM_STORMS_FOR_INFERENCE,
        "descriptive_or_inferential_allowed",
        "descriptive_only_insufficient_independent_storms",
    )
    result["interpretation"] = np.where(
        result["fdr_q_value"].ge(0.05),
        "current CI does not identify a general excess event effect; absence of evidence is not evidence of absence",
        "FDR-supported difference from matched background controls",
    )
    return result


def extended_dose_response(extended_pairs: pd.DataFrame) -> pd.DataFrame:
    valid = extended_pairs.loc[
        extended_pairs["response_status"].eq("ok")
        & extended_pairs["nearest_dist_km"].notna()
        & extended_pairs["gross_mobility_fraction"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    storm_groups = valid.groupby("sid")
    for dose_name, dose_column in [
        ("nearest_distance_km", "nearest_dist_km"),
        ("local_duration_within_250km_h", "local_duration_within_250km_h"),
        ("wind_at_nearest_ms", "wind_at_nearest_ms"),
    ]:
        if dose_column not in valid.columns:
            continue
        storm_medians = (
            storm_groups[[dose_column, "gross_mobility_fraction"]]
            .median()
            .dropna()
        )
        if len(storm_medians) < MINIMUM_STORMS_FOR_INFERENCE:
            rows.append(
                {
                    "dose": dose_name,
                    "status": "insufficient_independent_storms",
                    "n_storms": int(len(storm_medians)),
                    "n_pairs": int(valid[dose_column].notna().sum()),
                }
            )
            continue
        rho, p_value = spearmanr(
            storm_medians[dose_column], storm_medians["gross_mobility_fraction"]
        )
        rows.append(
            {
                "dose": dose_name,
                "status": "exploratory_only",
                "n_storms": int(len(storm_medians)),
                "n_pairs": int(valid[dose_column].notna().sum()),
                "storm_median_dose": float(storm_medians[dose_column].median()),
                "storm_median_response": float(
                    storm_medians["gross_mobility_fraction"].median()
                ),
                "spearman_rho": float(rho),
                "two_sided_p_value": float(p_value),
                "interpretation": (
                    "Exploratory dose association clustered by storm; no causal dose effect"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and "two_sided_p_value" in result.columns:
        valid_p = result["two_sided_p_value"].notna()
        if valid_p.any():
            result.loc[valid_p, "fdr_q_value"] = multipletests(
                result.loc[valid_p, "two_sided_p_value"], method="fdr_bh"
            )[1]
    return result


BOOTSTRAP_ITERATIONS = 2000
RAYLEIGH_CONCENTRATION_ALPHA = 0.05
MECHANISM_MEAN_ANGLE_TOLERANCE_DEG = 45.0


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _bearing_deg(east: float, north: float) -> float:
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _wrap_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _fold_axial_deg(angle: float) -> float:
    return float((angle + 90.0) % 180.0 - 90.0)


def _axial_summary(angles_deg: np.ndarray) -> tuple[float, float, float, int]:
    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = int(len(angles))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    doubled = np.radians(np.asarray([_fold_axial_deg(a) for a in angles]) * 2.0)
    c = float(np.cos(doubled).mean())
    s = float(np.sin(doubled).mean())
    resultant = float(np.hypot(c, s))
    mean_axial = float(np.degrees(np.arctan2(s, c)) / 2.0)
    z = n * resultant**2
    p_value = float(np.exp(-z))
    return mean_axial, resultant, p_value, n


def _circular_summary(angles_deg: np.ndarray) -> tuple[float, float, float, int]:
    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]
    n = int(len(angles))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rad = np.radians(angles)
    c = float(np.cos(rad).mean())
    s = float(np.sin(rad).mean())
    resultant = float(np.hypot(c, s))
    mean_angle = float(np.degrees(np.arctan2(s, c)) % 360.0)
    z = n * resultant**2
    p_value = float(np.exp(-z))
    return mean_angle, resultant, p_value, n


def _storm_clustered_circular_bootstrap(
    frame: pd.DataFrame,
    angle_column: str,
    rng: np.random.Generator,
    axial: bool = False,
) -> dict[str, float]:
    storms = frame["sid"].unique()
    if len(storms) < 3:
        return {
            "bootstrap_mean_angle_low": np.nan,
            "bootstrap_mean_angle_high": np.nan,
            "bootstrap_resultant_low": np.nan,
            "bootstrap_resultant_high": np.nan,
        }
    means = []
    resultants = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(storms, size=len(storms), replace=True)
        parts = []
        for sid in sampled:
            subset = frame.loc[frame["sid"].eq(sid), angle_column].dropna()
            if not subset.empty:
                parts.append(subset.to_numpy(dtype=float))
        if not parts:
            continue
        mean_angle, resultant, _, n = _circular_summary(np.concatenate(parts))
        if axial:
            mean_angle, resultant, _, n = _axial_summary(np.concatenate(parts))
        if n >= 3:
            means.append(mean_angle)
            resultants.append(resultant)
    if not means:
        return {
            "bootstrap_mean_angle_low": np.nan,
            "bootstrap_mean_angle_high": np.nan,
            "bootstrap_resultant_low": np.nan,
            "bootstrap_resultant_high": np.nan,
        }
    means_arr = np.asarray(means, dtype=float)
    resultants_arr = np.asarray(resultants, dtype=float)
    return {
        "bootstrap_mean_angle_low": float(np.quantile(means_arr, 0.025)),
        "bootstrap_mean_angle_high": float(np.quantile(means_arr, 0.975)),
        "bootstrap_resultant_low": float(np.quantile(resultants_arr, 0.025)),
        "bootstrap_resultant_high": float(np.quantile(resultants_arr, 0.975)),
    }


def event_spatial_fingerprint(
    pairs: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M1：侵蚀/堆积质心方位是否带有风暴几何决定的方向指纹。"""
    axis_lookup = (
        observations.drop_duplicates(["image_id"])
        .set_index("image_id")["sand_cay_major_axis_angle"]
    )
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        if getattr(pair, "response_status", "") != "ok":
            continue
        pre_mask = read_mask(getattr(pair, "pre_mask_path", ""))
        post_mask = read_mask(getattr(pair, "post_mask_path", ""))
        if pre_mask is None or post_mask is None or pre_mask.shape != post_mask.shape:
            continue
        erosion = np.logical_and(pre_mask, ~post_mask)
        deposition = np.logical_and(~pre_mask, post_mask)
        origin = _mask_centroid(pre_mask)
        erosion_centroid = _mask_centroid(erosion)
        deposition_centroid = _mask_centroid(deposition)
        if origin is None or erosion_centroid is None or deposition_centroid is None:
            continue
        erosion_bearing = _bearing_deg(
            erosion_centroid[0] - origin[0],
            -(erosion_centroid[1] - origin[1]),
        )
        deposition_bearing = _bearing_deg(
            deposition_centroid[0] - origin[0],
            -(deposition_centroid[1] - origin[1]),
        )
        windward = float(getattr(pair, "storm_center_bearing_from_cay_deg", np.nan))
        motion = float(getattr(pair, "storm_motion_bearing_deg", np.nan))
        axis_angle = float(
            axis_lookup.get(getattr(pair, "pre_image_id", ""), np.nan)
        )
        rows.append(
            {
                "event_pair_id": pair.event_pair_id,
                "sid": pair.sid,
                "sand_cay_id": pair.sand_cay_id,
                "window_days": pair.window_days,
                "sample": "strict" if pair.window_days == MAIN_WINDOW_DAYS else "extended",
                "erosion_bearing_deg": erosion_bearing,
                "deposition_bearing_deg": deposition_bearing,
                "windward_bearing_deg": windward,
                "leeward_bearing_deg": _wrap_deg(windward + 180.0) % 360.0,
                "storm_motion_bearing_deg": motion,
                "cay_axis_bearing_deg": (
                    _bearing_deg(np.sin(np.radians(axis_angle)), -np.cos(np.radians(axis_angle)))
                    if np.isfinite(axis_angle)
                    else np.nan
                ),
                "erosion_axis_residual_deg": (
                    _fold_axial_deg(erosion_bearing - _bearing_deg(np.sin(np.radians(axis_angle)), -np.cos(np.radians(axis_angle))))
                    if np.isfinite(axis_angle)
                    else np.nan
                ),
                "deposition_axis_residual_deg": (
                    _fold_axial_deg(deposition_bearing - _bearing_deg(np.sin(np.radians(axis_angle)), -np.cos(np.radians(axis_angle))))
                    if np.isfinite(axis_angle)
                    else np.nan
                ),
                "erosion_residual_deg": _wrap_deg(erosion_bearing - windward),
                "deposition_residual_deg": _wrap_deg(
                    deposition_bearing - (windward + 180.0)
                ),
                "transport_residual_deg": _wrap_deg(
                    _wrap_deg(deposition_bearing - erosion_bearing) - motion
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    rng = np.random.default_rng(RANDOM_SEED)
    stats_rows: list[dict[str, object]] = []
    for sample, group in frame.groupby("sample"):
        for angle_column, prediction in [
            ("erosion_residual_deg", "erosion on storm-facing side"),
            ("deposition_residual_deg", "deposition on lee side"),
            ("transport_residual_deg", "transport aligned with storm motion"),
            ("erosion_axis_residual_deg", "erosion aligned with cay major axis"),
            ("deposition_axis_residual_deg", "deposition aligned with cay major axis"),
        ]:
            axial = angle_column.endswith("axis_residual_deg")
            if axial:
                mean_angle, resultant, p_value, n = _axial_summary(
                    group[angle_column].to_numpy(dtype=float)
                )
                centered = mean_angle
                tolerance = 30.0
            else:
                mean_angle, resultant, p_value, n = _circular_summary(
                    group[angle_column].to_numpy(dtype=float)
                )
                centered = _wrap_deg(mean_angle)
                tolerance = MECHANISM_MEAN_ANGLE_TOLERANCE_DEG
            bootstrap = _storm_clustered_circular_bootstrap(
                group, angle_column, rng, axial=axial
            )
            stats_rows.append(
                {
                    "sample": sample,
                    "angle_column": angle_column,
                    "prediction": prediction,
                    "n_pairs": n,
                    "n_storms": int(group["sid"].nunique()),
                    "mean_residual_deg": mean_angle,
                    "centered_mean_residual_deg": centered,
                    "resultant_length": resultant,
                    "rayleigh_p_value": p_value,
                    **bootstrap,
                    "concentrated": bool(
                        p_value < RAYLEIGH_CONCENTRATION_ALPHA
                        and abs(centered) < tolerance
                    ),
                }
            )
    stats = pd.DataFrame(stats_rows)
    return frame, stats


def directional_dose_response(
    pairs: pd.DataFrame,
    observations: pd.DataFrame,
    reef_daily_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M2：事件响应是否由轴向对齐能量（矢量剂量）而非标量能量解释。"""
    valid = pairs.loc[pairs["response_status"].eq("ok")].copy()
    if valid.empty or not reef_daily_csv.is_file():
        return pd.DataFrame(), pd.DataFrame()
    daily = pd.read_csv(
        reef_daily_csv,
        usecols=["reef_id", "source", "time", "uo", "vo", "VHM0", "VSDX", "VSDY", "u10", "v10"],
        low_memory=False,
    )
    raw_time = daily["time"].copy()
    daily["time"] = pd.to_datetime(raw_time, errors="coerce", format="mixed")
    if daily["time"].isna().any():
        examples = raw_time.loc[daily["time"].isna()].astype(str).head(5).tolist()
        raise ValueError(
            "环境日表含无法解析的 time；示例：" + ", ".join(examples)
        )
    axis_lookup = (
        observations.drop_duplicates(["image_id"])
        .set_index("image_id")["sand_cay_major_axis_angle"]
    )
    rows: list[dict[str, object]] = []
    for pair in valid.itertuples(index=False):
        pre_date = pd.to_datetime(pair.pre_date)
        post_date = pd.to_datetime(pair.post_date)
        window = daily.loc[
            daily["reef_id"].eq(pair.reef_id)
            & daily["time"].gt(pre_date)
            & daily["time"].le(post_date)
        ]
        wave = window.loc[window["source"].eq("waverys")]
        wind = window.loc[window["source"].eq("era5")]
        expected_days = max(int((post_date - pre_date).days), 1)
        axis_angle = float(axis_lookup.get(pair.pre_image_id, np.nan))
        if not np.isfinite(axis_angle):
            continue
        phi = np.radians(axis_angle)
        along_unit = np.array([np.sin(phi), -np.cos(phi)])
        cross_unit = np.array([np.cos(phi), np.sin(phi)])
        wave_vector = np.array(
            [float(wave["VSDX"].mean()), float(wave["VSDY"].mean())]
        ) if len(wave) else np.array([np.nan, np.nan])
        wind_vector = np.array(
            [float(wind["u10"].mean()), float(wind["v10"].mean())]
        ) if len(wind) else np.array([np.nan, np.nan])
        pre_mask = read_mask(pair.pre_mask_path)
        post_mask = read_mask(pair.post_mask_path)
        along_shift = cross_shift = np.nan
        if pre_mask is not None and post_mask is not None and pre_mask.shape == post_mask.shape:
            pre_centroid = _mask_centroid(pre_mask)
            post_centroid = _mask_centroid(post_mask)
            if pre_centroid and post_centroid:
                east = post_centroid[0] - pre_centroid[0]
                north = -(post_centroid[1] - pre_centroid[1])
                along_shift = float(np.dot([east, north], along_unit))
                cross_shift = float(np.dot([east, north], cross_unit))
        wave_along = float(np.dot(wave_vector, along_unit))
        wave_cross = float(np.dot(wave_vector, cross_unit))
        wind_along = float(np.dot(wind_vector, along_unit))
        wind_cross = float(np.dot(wind_vector, cross_unit))
        wave_magnitude = float(np.hypot(*wave_vector))
        years = pair.pair_span_days / 365.2425
        rows.append(
            {
                "event_pair_id": pair.event_pair_id,
                "sid": pair.sid,
                "sand_cay_id": pair.sand_cay_id,
                "reef_id": pair.reef_id,
                "window_days": pair.window_days,
                "sample": "strict" if pair.window_days == MAIN_WINDOW_DAYS else "extended",
                "pair_span_days": pair.pair_span_days,
                "wave_days_observed": int(len(wave)),
                "wave_complete": bool(len(wave) >= 0.9 * expected_days),
                "wind_days_observed": int(len(wind)),
                "wind_complete": bool(len(wind) >= 0.9 * expected_days),
                "wave_hs_p90": float(wave["VHM0"].quantile(0.9)) if len(wave) else np.nan,
                "wave_along": wave_along,
                "wave_cross": wave_cross,
                "wave_aligned_energy": abs(wave_along),
                "wave_magnitude": wave_magnitude,
                "wave_misalign_cos": (
                    abs(wave_along) / wave_magnitude if wave_magnitude > 0 else np.nan
                ),
                "wind_along": wind_along,
                "wind_cross": wind_cross,
                "gross_mobility_fraction": pair.gross_mobility_fraction,
                "along_shift_normalized_per_year": (
                    along_shift / np.sqrt(pair.area_pre_m2 / np.pi) / years
                    if np.isfinite(along_shift)
                    else np.nan
                ),
                "cross_shift_normalized_per_year": (
                    cross_shift / np.sqrt(pair.area_pre_m2 / np.pi) / years
                    if np.isfinite(cross_shift)
                    else np.nan
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    frame = frame.loc[frame["wave_complete"]].copy()
    if frame.empty:
        return frame, pd.DataFrame()
    for column in [
        "wave_hs_p90",
        "wave_along",
        "wave_cross",
        "wave_aligned_energy",
        "wave_misalign_cos",
        "wind_along",
        "wind_cross",
        "gross_mobility_fraction",
        "along_shift_normalized_per_year",
        "cross_shift_normalized_per_year",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["log_duration"] = np.log(frame["pair_span_days"])
    positive = frame.loc[frame["gross_mobility_fraction"].gt(0), "gross_mobility_fraction"]
    floor = float(positive.quantile(0.01) / 2) if len(positive) else 1e-6
    frame["log_gross_mobility"] = np.log(frame["gross_mobility_fraction"].clip(lower=0) + floor)

    def z(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            return values * 0
        return (values - float(values.mean())) / sd

    frame["wave_hs_p90_z"] = z(frame["wave_hs_p90"])
    frame["wave_along_abs_z"] = z(frame["wave_along"].abs())
    frame["wave_cross_abs_z"] = z(frame["wave_cross"].abs())
    frame["wave_aligned_energy_z"] = z(frame["wave_aligned_energy"])
    frame["wave_misalign_cos_z"] = z(frame["wave_misalign_cos"])
    frame["wave_along_z"] = z(frame["wave_along"])
    frame["wave_cross_z"] = z(frame["wave_cross"])
    frame["log_duration_z"] = z(frame["log_duration"])

    model_specs = {
        "scalar_dose": ("log_gross_mobility", ["wave_hs_p90_z", "log_duration_z"]),
        "vector_dose": (
            "log_gross_mobility",
            ["wave_along_abs_z", "wave_cross_abs_z", "log_duration_z"],
        ),
        "aligned_dose": (
            "log_gross_mobility",
            ["wave_aligned_energy_z", "wave_misalign_cos_z", "log_duration_z"],
        ),
        "directional_along": (
            "along_shift_normalized_per_year",
            ["wave_along_z", "log_duration_z"],
        ),
        "directional_cross": (
            "cross_shift_normalized_per_year",
            ["wave_cross_z", "log_duration_z"],
        ),
    }
    model_rows: list[dict[str, object]] = []
    for name, (outcome, terms) in model_specs.items():
        subset = frame.loc[frame[[outcome] + terms].notna().all(axis=1)].copy()
        if len(subset) < 20 or subset["sid"].nunique() < 5:
            model_rows.append(
                {
                    "model": name,
                    "outcome": outcome,
                    "status": "insufficient_sample",
                    "n_pairs": int(len(subset)),
                    "n_storms": int(subset["sid"].nunique()),
                }
            )
            continue
        x = sm.add_constant(subset[terms], has_constant="add")
        fitted = sm.OLS(subset[outcome], x).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["sid"], "use_correction": True},
        )
        for term in terms:
            ci_low, ci_high = fitted.conf_int().loc[term].astype(float)
            model_rows.append(
                {
                    "model": name,
                    "outcome": outcome,
                    "term": term,
                    "status": "ok",
                    "n_pairs": int(len(subset)),
                    "n_storms": int(subset["sid"].nunique()),
                    "coefficient": float(fitted.params[term]),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "p_value": float(fitted.pvalues[term]),
                    "model_aic": float(fitted.aic),
                    "model_bic": float(fitted.bic),
                    "adj_r_squared": float(fitted.rsquared_adj),
                }
            )
    models = pd.DataFrame(model_rows)
    return frame, models


def resolution_audit(
    pairs: pd.DataFrame,
    observations: pd.DataFrame,
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """审计掩膜派生面积/质心/主轴的使用正确性，并量化分辨率对方位的影响。"""
    obs = observations.drop_duplicates(["image_id"]).set_index("image_id")
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        if getattr(pair, "response_status", "") != "ok":
            continue
        for tag, image_id, mask_path in [
            ("pre", pair.pre_image_id, pair.pre_mask_path),
            ("post", pair.post_image_id, pair.post_mask_path),
        ]:
            mask = read_mask(mask_path)
            if mask is None or image_id not in obs.index:
                continue
            feature = obs.loc[image_id]
            pixel_size = float(pd.to_numeric(pd.Series([feature["pixel_size_m"]]), errors="coerce").iloc[0])
            mask_area = float(mask.sum()) * pixel_size**2
            feature_area = float(pd.to_numeric(pd.Series([feature["sand_cay_area_m2"]]), errors="coerce").iloc[0])
            centroid = _mask_centroid(mask)
            if centroid is None:
                continue
            feat_cx = float(pd.to_numeric(pd.Series([feature["sand_cay_centroid_x"]]), errors="coerce").iloc[0])
            feat_cy = float(pd.to_numeric(pd.Series([feature["sand_cay_centroid_y"]]), errors="coerce").iloc[0])
            centroid_dist_m = float(
                np.hypot(
                    (feat_cx - centroid[0]) * pixel_size,
                    (feat_cy - centroid[1]) * pixel_size,
                )
            )
            rows.append(
                {
                    "event_pair_id": pair.event_pair_id,
                    "sample": "strict" if pair.window_days == MAIN_WINDOW_DAYS else "extended",
                    "sensor": pair.sensor,
                    "endpoint": tag,
                    "pixel_size_m": pixel_size,
                    "mask_area_m2": mask_area,
                    "feature_area_m2": feature_area,
                    "area_relative_difference": (
                        abs(mask_area - feature_area) / feature_area if feature_area > 0 else np.nan
                    ),
                    "centroid_offset_m": centroid_dist_m,
                }
            )
    frame = pd.DataFrame(rows)
    jitter_rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        if getattr(pair, "response_status", "") != "ok":
            continue
        pre_mask = read_mask(pair.pre_mask_path)
        post_mask = read_mask(pair.post_mask_path)
        if pre_mask is None or post_mask is None or pre_mask.shape != post_mask.shape:
            continue
        pixel_size = float(
            pd.to_numeric(pd.Series([obs.loc[pair.pre_image_id, "pixel_size_m"]]), errors="coerce").iloc[0]
        )
        origin = _mask_centroid(pre_mask)
        erosion = np.logical_and(pre_mask, ~post_mask)
        deposition = np.logical_and(~pre_mask, post_mask)
        for name, change_mask in [("erosion", erosion), ("deposition", deposition)]:
            centroid = _mask_centroid(change_mask)
            if origin is None or centroid is None:
                continue
            distance_m = float(
                np.hypot(centroid[0] - origin[0], centroid[1] - origin[1]) * pixel_size
            )
            jitter_rows.append(
                {
                    "event_pair_id": pair.event_pair_id,
                    "sample": "strict" if pair.window_days == MAIN_WINDOW_DAYS else "extended",
                    "sensor": pair.sensor,
                    "change": name,
                    "pixel_size_m": pixel_size,
                    "centroid_distance_m": distance_m,
                    "one_pixel_angular_jitter_deg": float(
                        np.degrees(np.arctan2(pixel_size, max(distance_m, 1e-6)))
                    ),
                }
            )
    jitter = pd.DataFrame(jitter_rows)
    stats_rows: list[dict[str, object]] = []
    if not frame.empty:
        for (sample, sensor), group in frame.groupby(["sample", "sensor"]):
            stats_rows.append(
                {
                    "scope": f"pair_{sample}_{sensor}",
                    "n": int(len(group)),
                    "median_area_relative_difference": float(
                        group["area_relative_difference"].median()
                    ),
                    "median_centroid_offset_m": float(group["centroid_offset_m"].median()),
                    "max_centroid_offset_m": float(group["centroid_offset_m"].max()),
                }
            )
    if not jitter.empty:
        for (sample, sensor), group in jitter.groupby(["sample", "sensor"]):
            stats_rows.append(
                {
                    "scope": f"jitter_{sample}_{sensor}",
                    "n": int(len(group)),
                    "median_one_pixel_angular_jitter_deg": float(
                        group["one_pixel_angular_jitter_deg"].median()
                    ),
                    "max_one_pixel_angular_jitter_deg": float(
                        group["one_pixel_angular_jitter_deg"].max()
                    ),
                }
            )
    ge = transitions.loc[
        transitions["sensor"].eq("google_earth")
        & transitions["boundary_change_status"].eq("ok")
        & transitions["time_interval_days"].between(90, 730)
    ].copy()
    ge_rows: list[dict[str, object]] = []
    date_lookup = (
        observations.loc[observations["sensor"].eq("google_earth")]
        .copy()
    )
    date_lookup["date"] = pd.to_datetime(date_lookup["date"], errors="coerce")
    lookup = date_lookup.set_index(["sand_cay_id", "date"])
    for transition in ge.itertuples(index=False):
        keys = [
            (transition.sand_cay_id, pd.to_datetime(transition.time_t)),
            (transition.sand_cay_id, pd.to_datetime(transition.time_t1)),
        ]
        masks = []
        meta = []
        for key in keys:
            if key not in lookup.index:
                break
            record = lookup.loc[key]
            if isinstance(record, pd.DataFrame):
                record = record.iloc[0]
            mask = read_mask(record["mask_path"])
            if mask is None:
                break
            masks.append(mask)
            meta.append(record)
        if len(masks) != 2 or masks[0].shape != masks[1].shape:
            continue
        pre_mask, post_mask = masks
        pixel_size = float(
            pd.to_numeric(pd.Series([meta[0]["pixel_size_m"]]), errors="coerce").iloc[0]
        )
        origin = _mask_centroid(pre_mask)
        erosion = np.logical_and(pre_mask, ~post_mask)
        deposition = np.logical_and(~pre_mask, post_mask)
        erosion_centroid = _mask_centroid(erosion)
        deposition_centroid = _mask_centroid(deposition)
        axis_angle = float(
            pd.to_numeric(pd.Series([meta[0]["sand_cay_major_axis_angle"]]), errors="coerce").iloc[0]
        )
        if origin is None or erosion_centroid is None or deposition_centroid is None:
            continue
        if not np.isfinite(axis_angle):
            continue
        axis_bearing = _bearing_deg(np.sin(np.radians(axis_angle)), -np.cos(np.radians(axis_angle)))
        for name, centroid in [("erosion", erosion_centroid), ("deposition", deposition_centroid)]:
            bearing = _bearing_deg(centroid[0] - origin[0], -(centroid[1] - origin[1]))
            ge_rows.append(
                {
                    "transition_id": transition.transition_id,
                    "change": name,
                    "pixel_size_m": pixel_size,
                    "axis_residual_deg": _fold_axial_deg(bearing - axis_bearing),
                }
            )
    ge_frame = pd.DataFrame(ge_rows)
    if not ge_frame.empty:
        for change, group in ge_frame.groupby("change"):
            mean_axial, resultant, p_value, n = _axial_summary(
                group["axis_residual_deg"].to_numpy(dtype=float)
            )
            stats_rows.append(
                {
                    "scope": f"google_earth_axis_{change}",
                    "n": n,
                    "mean_axial_residual_deg": mean_axial,
                    "resultant_length": resultant,
                    "rayleigh_p_value": p_value,
                    "median_pixel_size_m": float(group["pixel_size_m"].median()),
                }
            )
    return frame, pd.DataFrame(stats_rows)


def main() -> None:
    research_root = Path(__file__).resolve().parent
    dataset_root = research_root / "data" / "source_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-csv",
        type=Path,
        default=research_root / "outputs" / "沙洲观测主表.csv",
    )
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=research_root / "outputs" / "沙洲变化区间.csv",
    )
    parser.add_argument(
        "--event-csv",
        type=Path,
        default=research_root / "outputs" / "台风事件正式暴露.csv",
    )
    parser.add_argument(
        "--event-feature-csv",
        type=Path,
        default=research_root / "outputs" / "台风事件暴露特征.csv",
    )
    parser.add_argument(
        "--reef-daily-csv",
        type=Path,
        default=research_root
        / "data"
        / "environment"
        / "harmonized"
        / "daily"
        / "reef_daily_features.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    observations = prepare_observations(
        exclude_reefs(pd.read_csv(args.observation_csv, low_memory=False))
    )
    observations_raw = exclude_reefs(
        pd.read_csv(args.observation_csv, low_memory=False)
    )
    events_raw = exclude_reefs(pd.read_csv(args.event_csv, low_memory=False))
    if args.event_feature_csv.is_file():
        features = pd.read_csv(args.event_feature_csv, low_memory=False)
        feature_columns = [
            "reef_id",
            "sand_cay_id",
            "sid",
            "storm_center_bearing_from_cay_deg",
            "storm_motion_bearing_deg",
            "local_duration_within_250km_h",
            "local_duration_within_100km_h",
            "local_r34_duration_approx_h",
            "center_wind_ge17_5_within_250km_h",
        ]
        events_raw = events_raw.merge(
            features[feature_columns],
            on=["reef_id", "sand_cay_id", "sid"],
            how="left",
            validate="one_to_one",
        )
    events = prepare_events(events_raw)
    transitions = exclude_reefs(pd.read_csv(args.transition_csv, low_memory=False))
    transitions_raw = transitions.copy()
    transitions = transitions.loc[transitions["sensor"].eq("sentinel2")].copy()
    tide_path = args.output_dir / "潮位敏感性区间.csv"
    tide = None
    if tide_path.is_file() and tide_path.stat().st_size > 0:
        try:
            tide = pd.read_csv(tide_path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            tide = None
    if tide is not None and tide.empty:
        tide = None

    pair_frames = [
        build_event_pairs(observations, events, window_days)
        for window_days in SENSITIVITY_WINDOWS
    ]
    all_pairs = pd.concat(pair_frames, ignore_index=True)
    extended_pairs = build_event_pairs(
        observations,
        events,
        window_days=EXTENDED_WINDOW_DAYS,
        event_flag="event_candidate_250km",
    )
    main_pairs = all_pairs[
        all_pairs["window_days"].eq(MAIN_WINDOW_DAYS)
        & all_pairs["response_status"].eq("ok")
    ].copy()
    controls = eligible_controls(
        transitions,
        observations,
        events[events["event_relevant"]],
    )
    if tide is not None and {"transition_id", "tide_mean_m"}.issubset(tide.columns):
        controls = controls.merge(
            tide[["transition_id", "tide_mean_m", "tide_delta_m"]],
            on="transition_id",
            how="left",
        )
        main_pairs = main_pairs.merge(
            tide.rename(columns={"tide_mean_m": "event_tide_mean_m"})[
                ["event_pair_id", "event_tide_mean_m", "tide_delta_m"]
            ]
            if "event_pair_id" in tide.columns
            else tide[["transition_id", "tide_mean_m"]].rename(
                columns={"tide_mean_m": "event_tide_mean_m"}
            ),
            how="left",
        )
    matched = match_controls(main_pairs, controls, tide)
    extended_dose = extended_dose_response(extended_pairs)
    mechanism_pairs = pd.concat([main_pairs, extended_pairs], ignore_index=True)
    fingerprint, fingerprint_stats = event_spatial_fingerprint(
        mechanism_pairs, observations
    )
    dose_rows, dose_models = directional_dose_response(
        mechanism_pairs, observations, args.reef_daily_csv
    )
    resolution_rows, resolution_stats = resolution_audit(
        mechanism_pairs, observations_raw, transitions_raw
    )
    window_summary = summarize_windows(all_pairs)
    event_summary = summarize_events(main_pairs)
    matched_summary = summarize_matched(matched)
    funnel = pairing_funnel(observations, events, MAIN_WINDOW_DAYS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pair_path = args.output_dir / "台风事件前后影像配对.csv"
    main_pair_path = args.output_dir / "台风事件前后主分析样本.csv"
    window_path = args.output_dir / "台风事件窗口敏感性.csv"
    event_summary_path = args.output_dir / "台风事件响应汇总.csv"
    extended_path = args.output_dir / "台风事件扩展候选样本.csv"
    matched_path = args.output_dir / "台风事件匹配对照.csv"
    matched_summary_path = args.output_dir / "台风事件匹配对照统计.csv"
    extended_dose_path = args.output_dir / "台风扩展剂量探索.csv"
    fingerprint_path = args.output_dir / "台风事件空间指纹.csv"
    fingerprint_stats_path = args.output_dir / "台风事件空间指纹统计.csv"
    dose_path = args.output_dir / "台风方向剂量响应.csv"
    dose_model_path = args.output_dir / "台风方向剂量响应模型.csv"
    resolution_path = args.output_dir / "台风事件空间指纹分辨率审计.csv"
    resolution_stats_path = args.output_dir / "台风事件空间指纹分辨率审计统计.csv"
    check_path = args.output_dir / "台风事件前后分析核查.json"
    all_pairs.to_csv(all_pair_path, index=False, encoding="utf-8-sig")
    main_pairs.to_csv(main_pair_path, index=False, encoding="utf-8-sig")
    window_summary.to_csv(window_path, index=False, encoding="utf-8-sig")
    event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")
    extended_pairs.to_csv(extended_path, index=False, encoding="utf-8-sig")
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
    matched_summary.to_csv(matched_summary_path, index=False, encoding="utf-8-sig")
    extended_dose.to_csv(extended_dose_path, index=False, encoding="utf-8-sig")
    fingerprint.to_csv(fingerprint_path, index=False, encoding="utf-8-sig")
    fingerprint_stats.to_csv(fingerprint_stats_path, index=False, encoding="utf-8-sig")
    dose_rows.to_csv(dose_path, index=False, encoding="utf-8-sig")
    dose_models.to_csv(dose_model_path, index=False, encoding="utf-8-sig")
    resolution_rows.to_csv(resolution_path, index=False, encoding="utf-8-sig")
    resolution_stats.to_csv(resolution_stats_path, index=False, encoding="utf-8-sig")

    summary = {
        "analysis_unit": "typhoon event - sand cay - Sentinel-2 - fixed reference frame",
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "excluded_reefs": sorted(load_excluded_reefs()),
        "event_definition": (
            "inside the cay-bearing IBTrACS R34 quadrant, or nearest distance <=100 km "
            "with local nearest wind >=17.5 m/s"
        ),
        "pairing_funnel": funnel,
        "main_pre_window": [-MAIN_WINDOW_DAYS, -EVENT_BUFFER_DAYS],
        "main_post_window": [EVENT_BUFFER_DAYS, MAIN_WINDOW_DAYS],
        "competing_event_rule": "exclude pairs containing another relevant event between images",
        "main_valid_pairs": int(len(main_pairs)),
        "main_storms": int(main_pairs["sid"].nunique()),
        "main_cays": int(main_pairs["sand_cay_id"].nunique()),
        "main_reefs": int(main_pairs["reef_id"].nunique()),
        "main_sensor_counts": {
            str(key): int(value) for key, value in main_pairs["sensor"].value_counts().items()
        },
        "main_extreme_response_pairs_pending_review": int(
            main_pairs["response_qc_flag"].eq("extreme_requires_visual_review").sum()
        ),
        "matched_control_pairs": int(len(matched)),
        "matched_independent_storms": int(matched["sid"].nunique()) if not matched.empty else 0,
        "extended_analysis_rule": {
            "window_days": EXTENDED_WINDOW_DAYS,
            "radius_km": EXTENDED_RADIUS_KM,
            "status": "exploratory_dose_only",
            "minimum_independent_storms_for_inference": MINIMUM_STORMS_FOR_INFERENCE,
        },
        "extended_valid_pairs": int(
            extended_pairs["response_status"].eq("ok").sum()
        ),
        "extended_storms": int(
            extended_pairs.loc[extended_pairs["response_status"].eq("ok"), "sid"].nunique()
        ),
        "extended_cays": int(
            extended_pairs.loc[
                extended_pairs["response_status"].eq("ok"), "sand_cay_id"
            ].nunique()
        ),
        "extended_reefs": int(
            extended_pairs.loc[
                extended_pairs["response_status"].eq("ok"), "reef_id"
            ].nunique()
        ),
        "mechanism_tests": {
            "M1_spatial_fingerprint": {
                "pairs": int(len(fingerprint)),
                "strict_pairs": int(fingerprint["sample"].eq("strict").sum())
                if not fingerprint.empty
                else 0,
                "extended_pairs": int(fingerprint["sample"].eq("extended").sum())
                if not fingerprint.empty
                else 0,
                "concentrated_tests": int(fingerprint_stats["concentrated"].sum())
                if not fingerprint_stats.empty
                else 0,
                "total_tests": int(len(fingerprint_stats))
                if not fingerprint_stats.empty
                else 0,
            },
            "M2_directional_dose_response": {
                "pairs": int(len(dose_rows)),
                "models": int(dose_models["model"].nunique())
                if not dose_models.empty
                else 0,
            },
            "M1_resolution_audit": {
                "rows": int(len(resolution_rows)),
                "stats": resolution_stats.to_dict("records")
                if not resolution_stats.empty
                else [],
            },
        },
        "tide_matching_status": "not_attempted" if tide is None else "table_present",
        "inference_warning": (
            "Event pairs and matched controls are observational and sparse. A non-significant matched contrast means the current CI cannot identify a general excess event effect; it does not demonstrate that typhoons have no effect. Tide, cloud/water color, unmeasured wave/current forcing, and storm-direction alignment remain alternative explanations."
        ),
    }
    check_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for path in [
        all_pair_path,
        main_pair_path,
        window_path,
        event_summary_path,
        extended_path,
        matched_path,
        matched_summary_path,
        extended_dose_path,
        fingerprint_path,
        fingerprint_stats_path,
        dose_path,
        dose_model_path,
        resolution_path,
        resolution_stats_path,
        check_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
