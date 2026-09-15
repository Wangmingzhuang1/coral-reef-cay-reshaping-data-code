"""为沙洲形态研究生成可追溯的台风事件前后定性核验队列。

该脚本不拟合模型，也不判定台风造成了变化；它只把已有台风暴露、
形态变化和观测质量信息合并为人工核验的优先顺序。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from 分析范围 import exclude_reefs


MIN_INTERVAL_DAYS = 30
MAX_INTERVAL_DAYS = 365
ACCEPTED_QUALITY = {"A", "B"}


def percentile(values: pd.Series) -> pd.Series:
    """以同一传感器为参照计算排序百分位，缺失值不参与排序。"""
    return values.rank(pct=True, method="average")


def prepare(transitions: pd.DataFrame) -> pd.DataFrame:
    data = transitions.copy()
    numerical = [
        "time_interval_days",
        "area_t_m2",
        "gross_boundary_change_m2",
        "centroid_shift_m",
        "typhoon_strong_count",
        "typhoon_r34_count",
        "typhoon_min_distance_km",
        "typhoon_max_nearest_wind_m_s",
        "typhoon_max_storm_wind_m_s",
    ]
    for column in numerical:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["years"] = data["time_interval_days"] / 365.2425
    data["gross_mobility_fraction_per_year"] = (
        data["gross_boundary_change_m2"] / data["area_t_m2"] / data["years"]
    )
    data["centroid_shift_normalized_per_year"] = (
        data["centroid_shift_m"] / np.sqrt(data["area_t_m2"]) / data["years"]
    )
    data["is_event_exposed"] = data["typhoon_strong_count"].gt(0) | data["typhoon_r34_count"].gt(0)
    data["quality_eligible"] = data["quality_grade_t"].isin(ACCEPTED_QUALITY) & data[
        "quality_grade_t1"
    ].isin(ACCEPTED_QUALITY)
    data["interval_eligible"] = data["time_interval_days"].between(
        MIN_INTERVAL_DAYS, MAX_INTERVAL_DAYS, inclusive="both"
    )
    data["boundary_eligible"] = data["boundary_change_status"].eq("ok") & data["area_t_m2"].gt(0)
    return data


def classify_priority(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    for column in [
        "gross_mobility_fraction_per_year",
        "centroid_shift_normalized_per_year",
        "typhoon_max_nearest_wind_m_s",
        "typhoon_max_storm_wind_m_s",
    ]:
        data[f"{column}_percentile"] = data.groupby("sensor")[column].transform(percentile)
    data["typhoon_min_distance_km_percentile"] = data.groupby("sensor")[
        "typhoon_min_distance_km"
    ].transform(lambda values: 1 - percentile(values))
    data["exposure_score"] = (
        0.35 * data["typhoon_r34_count"].clip(upper=1)
        + 0.20 * data["typhoon_strong_count"].clip(upper=2).div(2)
        + 0.15 * data["typhoon_max_nearest_wind_m_s_percentile"].fillna(0)
        + 0.10 * data["typhoon_max_storm_wind_m_s_percentile"].fillna(0)
        + 0.20 * data["typhoon_min_distance_km_percentile"].fillna(0)
    )
    data["change_score"] = (
        0.75 * data["gross_mobility_fraction_per_year_percentile"].fillna(0)
        + 0.25 * data["centroid_shift_normalized_per_year_percentile"].fillna(0)
    )
    data["review_priority_score"] = 0.60 * data["exposure_score"] + 0.40 * data["change_score"]
    data["priority_tier"] = pd.cut(
        data["review_priority_score"],
        bins=[-np.inf, 0.45, 0.65, np.inf],
        labels=["secondary", "high", "critical"],
        right=False,
    ).astype("string")
    data["review_reason"] = np.select(
        [
            data["typhoon_r34_count"].gt(0) & data["priority_tier"].eq("critical"),
            data["typhoon_r34_count"].gt(0),
            data["priority_tier"].eq("critical"),
        ],
        [
            "R34 exposure with high morphological response",
            "R34 exposure",
            "Strong-typhoon exposure with high morphological response",
        ],
        default="Strong-typhoon exposure",
    )
    return data.sort_values(
        ["priority_tier", "review_priority_score", "sensor", "time_t"],
        ascending=[True, False, True, True],
        key=lambda values: values.map({"critical": 0, "high": 1, "secondary": 2})
        if values.name == "priority_tier"
        else values,
    )


def main() -> None:
    research_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition-csv", type=Path, default=research_root / "outputs" / "沙洲变化区间.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    prepared = prepare(exclude_reefs(pd.read_csv(args.transition_csv)))
    candidates = prepared[
        prepared["is_event_exposed"]
        & prepared["quality_eligible"]
        & prepared["interval_eligible"]
        & prepared["boundary_eligible"]
    ].copy()
    queue = classify_priority(candidates)
    queue["qualitative_review_status"] = "pending"
    queue["observed_change_description"] = ""
    queue["image_alignment_check"] = "pending"
    queue["mask_semantic_check"] = "pending"
    queue["alternative_explanations"] = ""
    queue["reviewer_notes"] = ""

    output_columns = [
        "transition_id", "priority_tier", "review_priority_score", "review_reason", "sensor",
        "reef_id", "sand_cay_id", "reference_frame_id", "time_t", "time_t1", "time_interval_days",
        "quality_grade_t", "quality_grade_t1", "area_t_m2", "area_change_m2", "erosion_area_m2",
        "deposition_area_m2", "gross_boundary_change_m2", "gross_mobility_fraction_per_year",
        "centroid_shift_m", "centroid_shift_normalized_per_year", "typhoon_event_count",
        "typhoon_strong_count", "typhoon_r34_count", "typhoon_min_distance_km",
        "typhoon_max_nearest_wind_m_s", "typhoon_max_storm_wind_m_s", "vegetation_fraction_t",
        "vegetation_fraction_t1", "vegetation_fraction_change", "qualitative_review_status",
        "observed_change_description", "image_alignment_check", "mask_semantic_check",
        "alternative_explanations", "reviewer_notes",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "台风事件定性核验队列.csv"
    summary_path = args.output_dir / "台风事件定性核验核查.json"
    queue[output_columns].to_csv(queue_path, index=False, encoding="utf-8-sig")
    summary = {
        "purpose": "Prioritize visual verification of storm-exposed intervals; not a causal attribution or a fitted model.",
        "selection_rule": {
            "event_exposure": "at least one strong-typhoon record or R34 exposure in the interval",
            "time_interval_days": [MIN_INTERVAL_DAYS, MAX_INTERVAL_DAYS],
            "quality_grade": sorted(ACCEPTED_QUALITY),
            "boundary_change_status": "ok",
        },
        "input_transitions": int(len(prepared)),
        "event_exposed_intervals": int(prepared["is_event_exposed"].sum()),
        "eligible_review_intervals": int(len(queue)),
        "priority_counts": {str(k): int(v) for k, v in queue["priority_tier"].value_counts().items()},
        "sensor_counts": {str(k): int(v) for k, v in queue["sensor"].value_counts().items()},
        "interpretation_warning": "The queue ranks evidence for human inspection. Storm exposure and morphological change in one interval do not establish causality, because interval timing, tide, registration, reef setting, and other forcing remain potential alternatives.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(queue_path)
    print(summary_path)


if __name__ == "__main__":
    main()
