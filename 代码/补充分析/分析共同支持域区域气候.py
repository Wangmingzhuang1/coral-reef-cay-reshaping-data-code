"""共同支持域（common support）敏感性：控制 10 m 分辨率的尺寸截断后重跑区域对比。

动机：S2 样本是分辨率筛过的（单像素沙洲不可见），若各区域截断位置不同，
区域差异会混入截断差异。本脚本以 GE 掩膜面积先验定义沙洲尺寸，主阈值
3000 m2（10 m 下约 30 像素）、敏感性阈值 1500 m2，只保留共同支持域内沙洲，
重跑区域重塑率对比与区域 ENSO 滞后模型，与无限制版本对照。只读现有数据。
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
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

scope_mod = importlib.import_module("分析范围")
from 区域气候滞后模型 import (  # noqa: E402
    describe_region_outcome,
    fit_region_interaction,
    fit_region_lag_models,
    region_code,
)

THRESHOLDS_M2 = (3000.0, 1500.0)
CORE_DAYS = (90, 365)
OUTCOMES = ("gross_mobility_fraction_per_year",)


def cay_size_prior() -> pd.Series:
    obs = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    obs["sand_cay_area_m2"] = pd.to_numeric(obs["sand_cay_area_m2"], errors="coerce")
    ge = obs.loc[obs["sensor"].eq("google_earth")]
    return ge.groupby("sand_cay_id")["sand_cay_area_m2"].median()


def size_structure(prior: pd.Series) -> pd.DataFrame:
    obs = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    obs["region"] = region_code(obs["reef_id"])
    obs["sand_cay_area_m2"] = pd.to_numeric(obs["sand_cay_area_m2"], errors="coerce")
    rows = []
    for region, group in obs.groupby("region", observed=True):
        ge_cays = group.loc[group["sensor"].eq("google_earth"), "sand_cay_id"].unique()
        s2_cays = group.loc[group["sensor"].eq("sentinel2"), "sand_cay_id"].unique()
        ge_sizes = prior.reindex(ge_cays).dropna()
        s2_sizes = prior.reindex(s2_cays).dropna()
        rows.append(
            {
                "region": region,
                "n_ge_cays": int(len(ge_cays)),
                "n_s2_cays": int(len(s2_cays)),
                "ge_median_area_m2": float(ge_sizes.median()) if len(ge_sizes) else np.nan,
                "ge_p25_area_m2": float(ge_sizes.quantile(0.25)) if len(ge_sizes) else np.nan,
                "s2_min_prior_area_m2": float(s2_sizes.min()) if len(s2_sizes) else np.nan,
                "s2_median_prior_area_m2": float(s2_sizes.median()) if len(s2_sizes) else np.nan,
                "ge_fraction_ge_3000m2": float(ge_sizes.ge(3000.0).mean()) if len(ge_sizes) else np.nan,
                "ge_fraction_ge_1500m2": float(ge_sizes.ge(1500.0).mean()) if len(ge_sizes) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_intervals(prior: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "outputs" / "沙洲变化区间.csv", low_memory=False)
    frame = scope_mod.exclude_reefs(frame)
    for column in ("time_interval_days", "area_t_m2", "gross_boundary_change_m2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["sensor"].eq("sentinel2")
        & frame["quality_grade_t"].isin(["A", "B"])
        & frame["quality_grade_t1"].isin(["A", "B"])
        & frame["time_interval_days"].between(*CORE_DAYS)
        & frame["area_t_m2"].gt(0)
    ].copy()
    years = frame["time_interval_days"] / 365.2425
    frame["gross_mobility_fraction_per_year"] = (
        frame["gross_boundary_change_m2"] / frame["area_t_m2"] / years
    )
    frame["region"] = region_code(frame["reef_id"])
    frame["cay_prior_area_m2"] = frame["sand_cay_id"].map(prior)
    return frame


def regional_mobility(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        groups = [
            group[outcome].dropna().to_numpy()
            for _, group in frame.groupby("region", observed=True)
        ]
        groups = [g for g in groups if len(g) >= 5]
        n = int(sum(len(g) for g in groups))
        k = len(groups)
        if k < 2:
            rows.append({"outcome": outcome, "status": "insufficient_groups"})
            continue
        stat, p = kruskal(*groups)
        rows.append(
            {
                "outcome": outcome,
                "status": "ok",
                "n_intervals": n,
                "n_groups": k,
                "kruskal_h": float(stat),
                "two_sided_p_value": float(p),
                "epsilon_squared": float((stat - k + 1) / (n - k)) if n > k else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    ok = result["status"].eq("ok")
    if ok.any():
        result.loc[ok, "fdr_q_value"] = multipletests(
            result.loc[ok, "two_sided_p_value"], method="fdr_bh"
        )[1]
    return result


def main() -> None:
    prior = cay_size_prior()
    structure = size_structure(prior)
    intervals = load_intervals(prior)
    enso = pd.read_csv(
        ROOT / "outputs" / "ENSO滞后分析样本.csv", encoding="utf-8-sig", low_memory=False
    )
    enso["region"] = region_code(enso["reef_id"])
    enso["cay_prior_area_m2"] = enso["sand_cay_id"].map(prior)

    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    structure.to_csv(out / "共同支持域尺寸结构.csv", index=False, encoding="utf-8-sig")

    audit: dict = {"size_structure": structure.to_dict("records"), "thresholds_m2": list(THRESHOLDS_M2)}
    mobility_rows = []
    enso_rows = []
    interaction_rows = []
    for threshold in THRESHOLDS_M2:
        sub_int = intervals[intervals["cay_prior_area_m2"].ge(threshold)]
        mobility = regional_mobility(sub_int)
        mobility["threshold_m2"] = threshold
        mobility_rows.append(mobility)
        sub_enso = enso[enso["cay_prior_area_m2"].ge(threshold)]
        outcomes = {
            "annualized_log_area_change": sub_enso["annualized_log_area_change"].notna(),
            "log_gross_mobility": sub_enso["log_gross_mobility"].notna(),
        }
        models = fit_region_lag_models(sub_enso, outcomes, "nino34_lag_{lag}m")
        models["threshold_m2"] = threshold
        enso_rows.append(models)
        for outcome, valid in outcomes.items():
            for lag in (0, 3, 6, 24):
                row = fit_region_interaction(
                    sub_enso.loc[valid], outcome, f"nino34_lag_{lag}m"
                )
                row["threshold_m2"] = threshold
                row["lag_months"] = lag
                interaction_rows.append(row)
        audit[f"restricted_sample_threshold_{int(threshold)}"] = {
            "intervals": int(len(sub_int)),
            "cays": int(sub_int["sand_cay_id"].nunique()),
            "enso_intervals": int(len(sub_enso)),
            "enso_cays": int(sub_enso["sand_cay_id"].nunique()),
            "region_cays": sub_int.groupby("region", observed=True)["sand_cay_id"]
            .nunique()
            .to_dict(),
        }
    mobility_all = pd.concat(mobility_rows, ignore_index=True)
    enso_all = pd.concat(enso_rows, ignore_index=True)
    interaction_all = pd.DataFrame(interaction_rows)
    mobility_all.to_csv(out / "共同支持域区域检验.csv", index=False, encoding="utf-8-sig")
    enso_all.to_csv(out / "共同支持域区域ENSO滞后.csv", index=False, encoding="utf-8-sig")
    interaction_all.to_csv(out / "共同支持域区域ENSO交互.csv", index=False, encoding="utf-8-sig")
    supported = enso_all.loc[enso_all["status"].eq("ok") & enso_all["fdr_q_value"].lt(0.05)]
    audit["fdr_supported_restricted_lag_tests"] = supported.to_dict("records")
    audit["interpretation"] = (
        "若共同支持域内区域对比与无限制版本同向且量级相近，则区域差异不是尺寸截断伪影；"
        "NH 仍因可检测沙洲不足留在描述层。"
    )
    (out / "共同支持域核查.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
