"""NH 未标注 S2 场景的可检测性 triage（只读筛查，不出掩膜）。

复用上游场景注册表 + 现有 GE 掩膜面积先验 + 场景云量，产出按优先级排序的
标注队列清单与预计收益，供标注工作台直接复用。不做任何下载/标注/修改。
"""

from __future__ import annotations

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT

UP_METADATA = (
    Path(r"{USER_HOME}/Desktop\砾洲数据集\sand_cay_temporal_dataset_builder")
    / "data"
    / "metadata"
    / "sentinel2_annotation_metadata.csv"
)
CLOUD_GOOD = 20.0
CLOUD_OK = 40.0
AREA_HIGH = 3000.0
AREA_MID = 1500.0


def main() -> None:
    metadata = pd.read_csv(UP_METADATA)
    metadata = metadata[metadata["reef_id"].astype("string").str.startswith("NH")].copy()
    metadata["cloud_cover_scene_percent"] = pd.to_numeric(
        metadata["cloud_cover_scene_percent"], errors="coerce"
    )
    audit = pd.read_csv(
        ROOT / "outputs" / "GE_Sentinel标注核查明细.csv", low_memory=False
    )
    annotated = set(
        audit.loc[audit["sensor"].eq("sentinel2"), "image_id"].astype("string")
    )
    metadata["annotated"] = metadata["image_id"].astype("string").isin(annotated)
    pending = metadata[~metadata["annotated"]].copy()

    obs = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    obs["sand_cay_area_m2"] = pd.to_numeric(obs["sand_cay_area_m2"], errors="coerce")
    ge_area = (
        obs.loc[obs["sensor"].eq("google_earth")]
        .groupby("sand_cay_id")["sand_cay_area_m2"]
        .median()
    )
    s2_cays = set(
        obs.loc[obs["sensor"].eq("sentinel2"), "sand_cay_id"].unique()
    )
    pending["ge_median_area_m2"] = pending["sand_cay_id"].map(ge_area)
    pending["cay_has_s2_mask"] = pending["sand_cay_id"].isin(s2_cays)

    def cloud_score(value: float) -> int:
        if np.isfinite(value) and value <= CLOUD_GOOD:
            return 2
        if np.isfinite(value) and value <= CLOUD_OK:
            return 1
        return 0

    def area_score(value: float) -> int:
        if np.isfinite(value) and value >= AREA_HIGH:
            return 2
        if np.isfinite(value) and value >= AREA_MID:
            return 1
        return 0

    pending["cloud_score"] = pending["cloud_cover_scene_percent"].map(cloud_score)
    pending["area_score"] = pending["ge_median_area_m2"].map(area_score)
    pending["priority_score"] = (
        pending["cloud_score"]
        + pending["area_score"]
        + pending["cay_has_s2_mask"].astype(int)
    )
    pending["recommended_action"] = np.select(
        [
            pending["priority_score"].ge(4),
            pending["priority_score"].ge(2),
        ],
        ["annotate_now", "pixel_check_then_decide"],
        default="skip_low_detectability",
    )
    pending = pending.sort_values(
        ["priority_score", "cloud_cover_scene_percent"], ascending=[False, True]
    )
    columns = [
        "image_id",
        "reef_id",
        "sand_cay_id",
        "date",
        "cloud_cover_scene_percent",
        "ge_median_area_m2",
        "cay_has_s2_mask",
        "cloud_score",
        "area_score",
        "priority_score",
        "recommended_action",
    ]
    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    pending[columns].to_csv(
        out / "NH未标注场景triage清单.csv", index=False, encoding="utf-8-sig"
    )

    per_cay = (
        pending.groupby(["reef_id", "sand_cay_id"])
        .agg(
            n_pending=("image_id", "size"),
            n_annotate_now=("recommended_action", lambda s: (s == "annotate_now").sum()),
            n_pixel_check=("recommended_action", lambda s: (s == "pixel_check_then_decide").sum()),
            ge_median_area_m2=("ge_median_area_m2", "median"),
        )
        .reset_index()
    )
    per_cay.to_csv(out / "NH_triage按沙洲汇总.csv", index=False, encoding="utf-8-sig")
    audit_json = {
        "nh_metadata_scenes": int(len(metadata)),
        "nh_annotated_scenes": int(metadata["annotated"].sum()),
        "nh_pending_scenes": int(len(pending)),
        "action_counts": pending["recommended_action"].value_counts().to_dict(),
        "cays_with_pending": int(pending["sand_cay_id"].nunique()),
        "cays_reaching_annotate_now_ge4_scenes": int(
            per_cay.loc[per_cay["n_annotate_now"].ge(4), "sand_cay_id"].nunique()
        ),
        "thresholds": {
            "cloud_good_pct": CLOUD_GOOD,
            "cloud_ok_pct": CLOUD_OK,
            "area_high_m2": AREA_HIGH,
            "area_mid_m2": AREA_MID,
        },
        "reuse_note": "清单按 priority_score 降序排列，可直接作为标注工作台队列；pixel_check_then_decide 需先做 10 m 目视/像素预筛。",
    }
    (out / "NH_triage核查.json").write_text(
        json.dumps(audit_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
