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
from 构建沙洲观测与变化表 import read_mask
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
        candidates["match_score"] = candidates["duration_score"] + 0.05 * candidates["time_score"]
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




BOOTSTRAP_ITERATIONS = 2000
RAYLEIGH_CONCENTRATION_ALPHA = 0.05
MECHANISM_MEAN_ANGLE_TOLERANCE_DEG = 45.0






















def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-csv", type=Path, default=root / "outputs" / "沙洲观测主表.csv")
    parser.add_argument("--transition-csv", type=Path, default=root / "outputs" / "沙洲变化区间.csv")
    parser.add_argument("--event-csv", type=Path, default=root / "outputs" / "台风事件正式暴露.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    observations = prepare_observations(exclude_reefs(pd.read_csv(args.observation_csv, low_memory=False)))
    events = prepare_events(exclude_reefs(pd.read_csv(args.event_csv, low_memory=False)))
    intervals = exclude_reefs(pd.read_csv(args.transition_csv, low_memory=False))
    intervals = intervals.loc[intervals.sensor.eq("sentinel2")].copy()
    all_pairs = pd.concat([build_event_pairs(observations, events, days) for days in SENSITIVITY_WINDOWS], ignore_index=True)
    main_pairs = all_pairs.loc[all_pairs.window_days.eq(MAIN_WINDOW_DAYS) & all_pairs.response_status.eq("ok")].copy()
    controls = eligible_controls(intervals, observations, events.loc[events.event_relevant])
    matched = match_controls(main_pairs, controls)
    funnel = pairing_funnel(observations, events, MAIN_WINDOW_DAYS)
    tables = {
        "台风事件前后影像配对.csv": all_pairs,
        "台风事件前后主分析样本.csv": main_pairs,
        "台风事件窗口敏感性.csv": summarize_windows(all_pairs),
        "台风事件响应汇总.csv": summarize_events(main_pairs),
        "台风事件匹配对照.csv": matched,
        "台风事件匹配对照统计.csv": summarize_matched(matched),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")
    summary = {
        "analysis_unit": "typhoon event - sand cay - Sentinel-2 - fixed reference frame",
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "excluded_reefs": sorted(load_excluded_reefs()),
        "event_definition": "R34 toward cay, or distance <=100 km and nearest wind >=17.5 m/s",
        "pairing_funnel": funnel,
        "main_pre_window": [-MAIN_WINDOW_DAYS, -EVENT_BUFFER_DAYS],
        "main_post_window": [EVENT_BUFFER_DAYS, MAIN_WINDOW_DAYS],
        "main_valid_pairs": int(len(main_pairs)), "main_storms": int(main_pairs.sid.nunique()),
        "main_cays": int(main_pairs.sand_cay_id.nunique()), "main_reefs": int(main_pairs.reef_id.nunique()),
        "matched_control_pairs": int(len(matched)),
        "matched_independent_storms": int(matched.sid.nunique()) if not matched.empty else 0,
        "competing_event_rule": "exclude another relevant event between paired observations",
    }
    (args.output_dir / "台风事件前后分析核查.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"strict_pairs": len(main_pairs), "storms": main_pairs.sid.nunique(), "matched_pairs": len(matched)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
