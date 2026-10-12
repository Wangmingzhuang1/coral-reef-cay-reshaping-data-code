"""同季与反季配对的质心分离度；与轮廓分析共享同源固定坐标和观测选择。"""
from __future__ import annotations

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import numpy as np
import pandas as pd
from 构建沙洲观测与变化表 import select_analysis_observations

ROOT = PROJECT_ROOT


def summarize_pairs(pairwise: pd.DataFrame, observations: pd.DataFrame, n_bootstrap: int = 2000):
    obs = select_analysis_observations(observations)
    obs = obs.loc[obs.sensor.eq("sentinel2")].copy()
    obs["requiv"] = np.sqrt(obs.sand_cay_area_m2 / np.pi)
    columns = ["image_id", "sand_cay_id", "reference_frame_id", "sand_cay_centroid_x", "sand_cay_centroid_y", "pixel_size_m", "requiv"]
    left = obs[columns].rename(columns={c: c + "_l" for c in columns})
    right = obs[columns].rename(columns={c: c + "_r" for c in columns})
    frame = pairwise.merge(left, left_on="left_image_id", right_on="image_id_l", validate="many_to_one")
    frame = frame.merge(right, left_on="right_image_id", right_on="image_id_r", validate="many_to_one")
    if not frame.sand_cay_id_l.eq(frame.sand_cay_id_r).all() or not frame.reference_frame_id_l.eq(frame.reference_frame_id_r).all():
        raise ValueError("质心配对跨沙洲或跨固定参考框。")
    if not np.allclose(frame.pixel_size_m_l, frame.pixel_size_m_r):
        raise ValueError("同一固定参考框的像素尺寸不一致。")
    frame["dn"] = np.hypot(frame.sand_cay_centroid_x_l-frame.sand_cay_centroid_x_r,
                            frame.sand_cay_centroid_y_l-frame.sand_cay_centroid_y_r) * frame.pixel_size_m_l / ((frame.requiv_l+frame.requiv_r)/2)
    rules = {
        "same_season_cross_year": frame.gap_days.between(300,430) & ~frame.same_calendar_year & frame.same_quarter,
        "opposite_season_within_year": frame.same_calendar_year & frame.gap_days.between(90,270) & frame.phase_distance.ge(1),
        "opposite_season_cross_year": ~frame.same_calendar_year & frame.gap_days.between(480,620) & frame.phase_distance.ge(1),
    }
    rng = np.random.default_rng(7)
    summaries = []
    buckets = {name: frame.loc[rule].copy() for name, rule in rules.items()}
    for name, part in buckets.items():
        if part.empty:
            summaries.append({"bucket": name, "n": 0})
            continue
        groups = [g.dn.to_numpy() for _, g in part.groupby("sand_cay_id_l", sort=False)]
        boot = [np.median(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])) for _ in range(n_bootstrap)]
        low, high = np.quantile(boot,[.025,.975])
        summaries.append({"bucket":name,"n":len(part),"n_cays":len(groups),
                          "median_normalized_centroid_separation":round(float(part.dn.median()),3),
                          "ci95_low":round(float(low),3),"ci95_high":round(float(high),3),
                          "p90":round(float(part.dn.quantile(.9)),3),"median_iou":round(float(part.iou.median()),3)})
    return pd.DataFrame(summaries), buckets["same_season_cross_year"]


def main():
    pairwise=pd.read_csv(ROOT/"outputs/Sentinel2季节形态配对.csv")
    observations=pd.read_csv(ROOT/"outputs/沙洲观测主表.csv")
    summary, pairs=summarize_pairs(pairwise,observations)
    summary.to_csv(ROOT/"outputs/季节相位质心分离统计.csv",index=False,encoding="utf-8-sig")
    pairs[["sand_cay_id_l","left_image_id","right_image_id","gap_days","phase_distance","dn","iou"]].to_csv(ROOT/"outputs/季节相位质心分离.csv",index=False,encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
