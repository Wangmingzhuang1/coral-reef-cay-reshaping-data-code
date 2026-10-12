"""Google Earth 层的 ENSO 滞后敏感性（路线 B：传感器分层，不与 S2 混合）。

GE 影像是机会性获取（云少才截图），区间中点携带气候状态信息，因此本脚本的
滞后关联只作描述性敏感性/符号复现，绝不与 S2 主结果合并估计，也不作因果解释。
估计合同与 S2 区域脚本一致：沙洲内去均值 + 沙洲聚类 SE + 预设滞后族内 FDR。
"""

from __future__ import annotations

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

enso_mod = importlib.import_module("分析ENSO与沙洲面积")
scope_mod = importlib.import_module("分析范围")
from 区域气候滞后模型 import (  # noqa: E402
    LAGS_MONTHS,
    describe_region_outcome,
    fit_region_lag_models,
    plot_region_lag_models,
    region_code,
)

CORE_DAYS = (90, 365)


def prepare_ge_intervals() -> pd.DataFrame:
    nino = enso_mod.load_nino34(
        ROOT / "data" / "environment" / "indices" / "detrend.nino34.ascii.txt"
    )
    frame = pd.read_csv(ROOT / "outputs" / "沙洲变化区间.csv", low_memory=False)
    frame = scope_mod.exclude_reefs(frame)
    for column in (
        "time_interval_days",
        "area_t_m2",
        "area_t1_m2",
        "gross_boundary_change_m2",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["time_t"] = pd.to_datetime(frame["time_t"], errors="coerce")
    frame["time_t1"] = pd.to_datetime(frame["time_t1"], errors="coerce")
    frame = frame.loc[
        frame["sensor"].eq("google_earth")
        & frame["quality_grade_t"].isin(["A", "B"])
        & frame["quality_grade_t1"].isin(["A", "B"])
        & frame["time_interval_days"].between(*CORE_DAYS)
        & frame["area_t_m2"].gt(0)
        & frame["area_t1_m2"].gt(0)
        & frame[["time_t", "time_t1"]].notna().all(axis=1)
    ].copy()
    years = frame["time_interval_days"] / 365.2425
    frame["annualized_log_area_change"] = (
        np.log(frame["area_t1_m2"]) - np.log(frame["area_t_m2"])
    ) / years
    frame["gross_mobility_fraction_per_year"] = (
        frame["gross_boundary_change_m2"] / frame["area_t_m2"] / years
    )
    positive = frame.loc[
        frame["gross_mobility_fraction_per_year"].gt(0),
        "gross_mobility_fraction_per_year",
    ]
    floor = float(positive.quantile(0.01) / 2) if len(positive) else 1e-6
    frame["log_gross_mobility"] = np.log(
        frame["gross_mobility_fraction_per_year"].clip(lower=0) + floor
    )
    midpoint = frame["time_t"] + (frame["time_t1"] - frame["time_t"]) / 2
    frame["midpoint_year"] = midpoint.dt.year + (midpoint.dt.dayofyear - 1) / 365.2425
    frame["midpoint_month"] = midpoint.dt.to_period("M").astype(str)
    frame["midpoint_calendar_year"] = midpoint.dt.year.astype(int)
    frame["season_sin"] = np.sin(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["season_cos"] = np.cos(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["log_interval"] = np.log(frame["time_interval_days"])
    lookup = dict(zip(nino["month_period"], nino["nino34_anomaly_c"]))
    for lag in LAGS_MONTHS:
        frame[f"nino34_lag_{lag}m"] = [
            enso_mod.interval_weighted_index(start, end, lag, lookup)
            for start, end in zip(frame["time_t"], frame["time_t1"])
        ]
    frame["region"] = region_code(frame["reef_id"])
    return frame


def main() -> None:
    sample = prepare_ge_intervals()
    outcomes = {
        "annualized_log_area_change": sample["annualized_log_area_change"].notna(),
        "log_gross_mobility": sample["log_gross_mobility"].notna(),
    }
    models = fit_region_lag_models(sample, outcomes, "nino34_lag_{lag}m")
    descriptive_rows = []
    for region, group in sample.groupby("region", observed=True):
        for outcome in outcomes:
            row = describe_region_outcome(group, outcome)
            row["region"] = region
            descriptive_rows.append(row)
    descriptive = pd.DataFrame(descriptive_rows)
    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "GE层ENSO滞后敏感性.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(out / "GE层ENSO描述统计.csv", index=False, encoding="utf-8-sig")
    plot_region_lag_models(
        models,
        (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r5_ge_stratum_enso_lags",
        "Nino 3.4 anomaly (deg C)",
    )
    supported = models.loc[models["status"].eq("ok") & models["fdr_q_value"].lt(0.05)]
    audit = {
        "stratum": "Google Earth only; never pooled with Sentinel-2",
        "selection_bias_caveat": "GE acquisition is opportunistic (cloud-free screenshots); interval midpoints carry climate-state information, so lag associations are descriptive sign-replication only",
        "region_sample_counts": sample.groupby("region", observed=True)
        .agg(n_intervals=("transition_id", "size"), n_cays=("sand_cay_id", "nunique"))
        .reset_index()
        .to_dict("records"),
        "fdr_supported_within_stratum": supported.to_dict("records"),
        "inference": "Within-cay observational lag association inside the GE stratum; cay-clustered SE; FDR within region-outcome family; not causal and not pooled.",
    }
    (out / "GE层ENSO滞后核查.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
