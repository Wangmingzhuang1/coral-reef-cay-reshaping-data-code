"""把人工植被状态与颜色比例合并为可审计的软标签和复核队列。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE_TO_CODE = {"none": 0, "sparse": 1, "partial": 2, "dominant": 3}
CODE_TO_STATE = {value: key for key, value in STATE_TO_CODE.items()}
STATE_BOUNDARIES = np.array([0.03, 0.15, 0.45], dtype=float)
FOUR_TO_THREE = {
    "none": "low_cover",
    "sparse": "low_cover",
    "partial": "partial_cover",
    "dominant": "dominant_cover",
}


def color_state(fraction: float) -> str:
    if fraction >= STATE_BOUNDARIES[2]:
        return "dominant"
    if fraction >= STATE_BOUNDARIES[1]:
        return "partial"
    if fraction >= STATE_BOUNDARIES[0]:
        return "sparse"
    return "none"


def color_confidence(fraction: float) -> tuple[float, str]:
    margin = float(np.min(np.abs(STATE_BOUNDARIES - fraction)))
    if margin >= 0.05:
        return margin, "high"
    if margin >= 0.02:
        return margin, "medium"
    return margin, "low"


def three_level_state(four_level_state: str) -> str:
    return FOUR_TO_THREE.get(four_level_state, "")


def temporal_flags(data: pd.DataFrame) -> pd.Series:
    flags = pd.Series("", index=data.index, dtype="object")
    group_columns = ["sensor", "sand_cay_id"]
    for _, group in data.sort_values("date").groupby(group_columns, dropna=False):
        indexed = list(group.index)
        for previous_idx, current_idx, following_idx in zip(indexed, indexed[1:], indexed[2:]):
            previous = data.loc[previous_idx]
            current = data.loc[current_idx]
            following = data.loc[following_idx]
            if not all(
                state in STATE_TO_CODE
                for state in [
                    previous["manual_vegetation_state"],
                    current["manual_vegetation_state"],
                    following["manual_vegetation_state"],
                ]
            ):
                continue
            neighbor_code = STATE_TO_CODE[previous["manual_vegetation_state"]]
            current_code = STATE_TO_CODE[current["manual_vegetation_state"]]
            if (
                previous["manual_vegetation_state"] == following["manual_vegetation_state"]
                and abs(current_code - neighbor_code) >= 2
                and abs(STATE_TO_CODE[current["color_vegetation_state"]] - neighbor_code) <= 1
            ):
                flags.at[current_idx] = "manual_temporal_outlier"
    return flags


def add_review_fields(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["color_vegetation_state"] = data["vegetation_fraction"].map(color_state)
    confidence = data["vegetation_fraction"].map(color_confidence)
    data["color_state_margin"] = confidence.map(lambda item: item[0])
    data["color_confidence"] = confidence.map(lambda item: item[1])
    data["manual_state_code"] = data["manual_vegetation_state"].map(STATE_TO_CODE)
    data["color_state_code"] = data["color_vegetation_state"].map(STATE_TO_CODE)
    data["manual_color_distance"] = (
        data["manual_state_code"] - data["color_state_code"]
    ).abs()

    data["label_status"] = "missing_manual_label"
    valid_manual = data["manual_state_code"].notna()
    data.loc[valid_manual & (data["manual_color_distance"] == 0), "label_status"] = "confirmed"
    data.loc[valid_manual & (data["manual_color_distance"] == 1), "label_status"] = "compatible"
    data.loc[valid_manual & (data["manual_color_distance"] >= 2), "label_status"] = "conflict"
    data["temporal_flag"] = temporal_flags(data)

    # 颜色比例显示 none 与 sparse 强烈重叠，因此把两者合并为较稳定的低覆盖类别。
    data["manual_cover_state_3"] = data["manual_vegetation_state"].map(three_level_state)
    data["color_cover_state_3"] = data["color_vegetation_state"].map(three_level_state)
    data["cover_state_status_3"] = "missing_manual_label"
    has_manual_cover = data["manual_cover_state_3"].ne("")
    data.loc[
        has_manual_cover
        & data["manual_cover_state_3"].eq(data["color_cover_state_3"]),
        "cover_state_status_3",
    ] = "confirmed"
    data.loc[
        has_manual_cover
        & data["manual_cover_state_3"].ne(data["color_cover_state_3"]),
        "cover_state_status_3",
    ] = "conflict"
    data.loc[
        has_manual_cover & data["label_status"].eq("compatible"),
        "cover_state_status_3",
    ] = "compatible"
    data["analysis_cover_state_3"] = data["manual_cover_state_3"]
    data.loc[
        ~data["cover_state_status_3"].isin(["confirmed", "compatible"]),
        "analysis_cover_state_3",
    ] = ""

    data["analysis_vegetation_state"] = data["manual_vegetation_state"]
    data.loc[data["label_status"].isin(["conflict", "missing_manual_label"]), "analysis_vegetation_state"] = ""
    data["suggested_vegetation_state"] = data["manual_vegetation_state"]
    data.loc[data["label_status"].isin(["conflict", "missing_manual_label"]), "suggested_vegetation_state"] = data.loc[
        data["label_status"].isin(["conflict", "missing_manual_label"]), "color_vegetation_state"
    ]

    data["review_priority"] = "none"
    data.loc[
        (data["label_status"] == "compatible") & (data["color_confidence"] == "low"),
        "review_priority",
    ] = "low"
    data.loc[data["label_status"] == "conflict", "review_priority"] = "high"
    data.loc[
        (data["label_status"] == "conflict") & (data["color_confidence"] == "high"),
        "review_priority",
    ] = "critical"
    data.loc[data["temporal_flag"] != "", "review_priority"] = "critical"
    return data


def main() -> None:
    research_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-csv",
        type=Path,
        default=research_root / "outputs" / "沙洲观测主表.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=research_root / "outputs")
    args = parser.parse_args()

    data = pd.read_csv(args.observation_csv)
    data = data[data["vegetation_fraction"].notna()].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = add_review_fields(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "植被状态审校主表.csv"
    review_path = args.output_dir / "植被状态优先复核.csv"
    summary_path = args.output_dir / "植被状态审校核查.json"
    data.to_csv(main_path, index=False, encoding="utf-8-sig")
    priority_order = {"critical": 0, "high": 1, "low": 2}
    review = data[data["review_priority"] != "none"].copy()
    review["priority_order"] = review["review_priority"].map(priority_order)
    review = review.sort_values(
        ["priority_order", "color_state_margin", "date"],
        ascending=[True, False, True],
    ).drop(columns="priority_order")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_color_valid_observations": int(len(data)),
        "label_status_counts": {str(k): int(v) for k, v in data["label_status"].value_counts().items()},
        "review_priority_counts": {str(k): int(v) for k, v in data["review_priority"].value_counts().items()},
        "temporal_outlier_count": int((data["temporal_flag"] != "").sum()),
        "n_retained_for_categorical_analysis": int(data["analysis_vegetation_state"].ne("").sum()),
        "n_excluded_from_categorical_analysis": int(data["analysis_vegetation_state"].eq("").sum()),
        "three_level_cover_status_counts": {
            str(k): int(v) for k, v in data["cover_state_status_3"].value_counts().items()
        },
        "n_retained_for_three_level_analysis": int(data["analysis_cover_state_3"].ne("").sum()),
        "three_level_definition": {
            "low_cover": "人工 none 或 sparse；对应低覆盖状态",
            "partial_cover": "人工 partial；对应局部覆盖状态",
            "dominant_cover": "人工 dominant；对应优势覆盖状态",
        },
        "continuous_variable": "vegetation_fraction 保留全部有效颜色观测，不受四级标签冲突影响。",
        "rule": "人工标签与颜色状态相差不超过一级时保留；相差两级及以上进入复核且不参与分类分组。",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"审校主表: {main_path}")
    print(f"优先复核: {review_path}")
    print(f"核查结果: {summary_path}")


if __name__ == "__main__":
    main()
