"""跨气候水动力背景下方向耦合的预注册式补充分析。

目标不是把区域名称当作气候机制，也不重新检验 ENSO。这里以独立于形态响应的
1996--2016 年逐日 WAVERYS 波候构建两条先验背景轴：
1) chronic wave exposure（平均波高与 p90 波高）；
2) wave directional concentration（波矢量的多年合向量长度）。

主检验：这两条背景是否调节已在主模型中定义的局地波矢量--位移方向耦合。
所有模型严格从主模型的同一 Sentinel-2 样本和同一协变量合同出发；该脚本不写入
主文 outputs，也不替代既有“区域”或 ENSO 分析。

运行：
  python 跨气候水动力背景分析.py --stage audit
  python 跨气候水动力背景分析.py --stage model
"""

from __future__ import annotations

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
OUTPUT = (PROJECT_ROOT / "outputs/气候检验")
DAILY = ROOT / "data" / "environment" / "harmonized" / "daily" / "reef_daily_features.csv"
POINTS = ROOT / "data" / "environment" / "harmonized" / "daily" / "point_selection_audit.csv"
ANALYSIS = ROOT / "outputs" / "已成洲沙洲定量分析样本.csv"
DECOMPOSITION = ROOT / "outputs" / "流场波浪沿轴横轴分解.csv"
PRIMARY_MODELS = ROOT / "outputs" / "强迫植被沙洲内模型.csv"

PRE_S2_START = pd.Timestamp("1996-01-01")
PRE_S2_END = pd.Timestamp("2016-12-31")
FULL_PERIOD_START = pd.Timestamp("1996-01-01")
FULL_PERIOD_END = pd.Timestamp("2025-12-31")
MIN_DAILY_COVERAGE = 0.95
MIN_REEF_COVERAGE_SHARE = 0.90
MIN_WAVE_GRID_CELLS = 30
MIN_REEFS_PER_SPATIAL_STRATUM = 10
RANDOM_SEED = 20260916

BASE_TERMS = [
    "current_along_mean_z",
    "current_cross_mean_z",
    "wave_vector_along_mean_z",
    "wave_vector_cross_mean_z",
    "wave_hs_p90_z",
    "pulse",
    "vegetation_fraction_t_z",
    "hs_veg_interaction",
    "crosscur_veg_interaction",
    "log_interval",
    "midpoint_year_z",
    "log_area_z",
]
OUTCOMES = {
    "along_shift": "along_shift_z",
    "cross_shift": "cross_shift_z",
}
PRIMARY_TERM_BY_OUTCOME = {
    "along_shift": "wave_vector_along_mean_z",
    "cross_shift": "wave_vector_cross_mean_z",
}


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / scale


def vector_concentration(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return np.nan
    speed = np.hypot(x[valid], y[valid])
    denominator = float(speed.mean())
    if denominator <= 0:
        return np.nan
    return float(np.hypot(x[valid].mean(), y[valid].mean()) / denominator)


def load_primary_frame() -> pd.DataFrame:
    """完全复用主模型输入、标准化与沙洲内去均值的前置合同。"""
    analysis = pd.read_csv(ANALYSIS, encoding="utf-8-sig", low_memory=False)
    columns = [
        "transition_id",
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "wave_hs_p90",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
    ]
    decomposition = pd.read_csv(DECOMPOSITION, usecols=columns, low_memory=False)
    frame = analysis.loc[analysis["sensor"].eq("sentinel2")].merge(
        decomposition, on="transition_id", how="inner", validate="one_to_one"
    )
    frame = frame.loc[
        frame[["current_along_mean", "wave_hs_p90", "vegetation_fraction_t"]]
        .notna()
        .all(axis=1)
    ].copy()
    frame["pulse"] = (
        frame[["typhoon_strong_count", "typhoon_r34_count"]].fillna(0).sum(axis=1) > 0
    ).astype(float)
    for column in (
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "wave_hs_p90",
        "vegetation_fraction_t",
    ):
        frame[f"{column}_z"] = zscore(frame[column])
    frame["hs_veg_interaction"] = frame["wave_hs_p90_z"] * frame["vegetation_fraction_t_z"]
    frame["crosscur_veg_interaction"] = (
        frame["current_cross_mean_z"] * frame["vegetation_fraction_t_z"]
    )
    frame["log_interval"] = zscore(np.log(frame["time_interval_days"]))
    frame["midpoint_year_z"] = zscore(frame["midpoint_year"])
    frame["log_area_z"] = zscore(np.log(frame["area_t_m2"]))
    positive = frame.loc[
        frame["gross_mobility_fraction_per_year"].gt(0), "gross_mobility_fraction_per_year"
    ]
    frame["log_gross_mobility"] = np.log(
        frame["gross_mobility_fraction_per_year"].clip(lower=0) + float(positive.quantile(0.01) / 2)
    )
    frame["along_shift_z"] = zscore(frame["centroid_along_shift_normalized_per_year"])
    frame["cross_shift_z"] = zscore(frame["centroid_cross_shift_normalized_per_year"])
    return frame


def load_wave_grid_map(reef_ids: set[str]) -> pd.DataFrame:
    points = pd.read_csv(POINTS)
    points = points.loc[
        points["source"].eq("waverys") & points["reef_id"].isin(reef_ids)
    ].copy()
    points["wave_grid_id"] = (
        points["sampled_lon"].round(6).astype(str)
        + "_"
        + points["sampled_lat"].round(6).astype(str)
    )
    return points[["reef_id", "wave_grid_id", "sampled_lon", "sampled_lat"]].drop_duplicates()


def load_wave_descriptors(
    reef_ids: set[str], climate_start: pd.Timestamp, climate_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从本机日值表提取固定 30 年波候描述符；不读取形态响应。"""
    needed = ["reef_id", "source", "time", "VHM0", "VSDX", "VSDY"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(DAILY, usecols=needed, chunksize=250_000):
        chunk = chunk.loc[
            chunk["source"].eq("waverys") & chunk["reef_id"].isin(reef_ids)
        ].copy()
        if chunk.empty:
            continue
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.loc[chunk["time"].between(climate_start, climate_end)]
        pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(), pd.DataFrame()
    daily = pd.concat(pieces, ignore_index=True)
    expected_days = len(pd.date_range(climate_start, climate_end, freq="D"))
    coverage_rows: list[dict[str, object]] = []
    descriptor_rows: list[dict[str, object]] = []
    for reef_id, group in daily.groupby("reef_id", observed=True):
        valid = group.dropna(subset=["VHM0", "VSDX", "VSDY"]).copy()
        coverage = len(valid) / expected_days
        coverage_rows.append(
            {
                "reef_id": reef_id,
                "expected_days": expected_days,
                "complete_wave_days": int(len(valid)),
                "wave_coverage_fraction": float(coverage),
                "start": group["time"].min().date().isoformat() if len(group) else None,
                "end": group["time"].max().date().isoformat() if len(group) else None,
            }
        )
        if valid.empty:
            continue
        monthly = valid.assign(month=valid["time"].dt.month).groupby("month", observed=True)["VHM0"].mean()
        descriptor_rows.append(
            {
                "reef_id": reef_id,
                "wave_hs_mean": float(valid["VHM0"].mean()),
                "wave_hs_p90": float(valid["VHM0"].quantile(0.90)),
                "wave_hs_seasonality_cv": float(monthly.std(ddof=0) / monthly.mean())
                if monthly.mean() > 0
                else np.nan,
                "wave_directional_concentration": vector_concentration(valid["VSDX"], valid["VSDY"]),
            }
        )
    return pd.DataFrame(descriptor_rows), pd.DataFrame(coverage_rows)


def build_background(descriptors: pd.DataFrame, grid_map: pd.DataFrame) -> pd.DataFrame:
    background = descriptors.merge(grid_map, on="reef_id", how="left", validate="one_to_one")
    background["wave_exposure_z"] = (
        zscore(background["wave_hs_mean"]) + zscore(background["wave_hs_p90"])
    ) / 2
    background["wave_directionality_z"] = zscore(background["wave_directional_concentration"])
    return background


def make_audit(
    frame: pd.DataFrame,
    background: pd.DataFrame,
    coverage: pd.DataFrame,
    climate_start: pd.Timestamp,
    climate_end: pd.Timestamp,
    background_window: str,
) -> dict[str, object]:
    primary_reefs = set(frame["reef_id"].unique())
    merged = frame[["reef_id", "sand_cay_id"]].drop_duplicates().merge(
        background, on="reef_id", how="left"
    )
    coverage_for_primary = coverage.loc[coverage["reef_id"].isin(primary_reefs)].copy()
    complete_reefs = set(
        coverage_for_primary.loc[
            coverage_for_primary["wave_coverage_fraction"].ge(MIN_DAILY_COVERAGE), "reef_id"
        ]
    )
    descriptors_ok = merged[["wave_exposure_z", "wave_directionality_z", "wave_grid_id"]].notna().all(axis=1)
    n_grid = int(background.loc[background["reef_id"].isin(primary_reefs), "wave_grid_id"].nunique())
    gates = {
        "matched_primary_sample": {
            "n_intervals_before_within_demean": int(len(frame)),
            "n_cays_before_within_demean": int(frame["sand_cay_id"].nunique()),
            "n_reefs": int(len(primary_reefs)),
        },
        "daily_coverage": {
            "window": [climate_start.date().isoformat(), climate_end.date().isoformat()],
            "minimum_per_reef": MIN_DAILY_COVERAGE,
            "reefs_passing": int(len(complete_reefs)),
            "reefs_required": int(np.ceil(MIN_REEF_COVERAGE_SHARE * len(primary_reefs))),
            "minimum_reef_share": MIN_REEF_COVERAGE_SHARE,
        },
        "independent_wave_grid_cells": {
            "n_unique_sampled_cells": n_grid,
            "minimum_required": MIN_WAVE_GRID_CELLS,
        },
        "descriptor_completeness": {
            "cays_with_complete_background": int(descriptors_ok.sum()),
            "cays_in_primary_sample": int(len(merged)),
        },
    }
    permitted = (
        len(complete_reefs) >= int(np.ceil(MIN_REEF_COVERAGE_SHARE * len(primary_reefs)))
        and n_grid >= MIN_WAVE_GRID_CELLS
        and bool(descriptors_ok.all())
    )
    return {
        "analysis_name": "cross_hydroclimatic_background_directional_coupling",
        "status": "model_permitted" if permitted else "audit_gate_not_met",
        "hypothesis": "Multi-decadal wave background may modify, but does not define, within-cay local wave-direction coupling.",
        "scope": "Exploratory supplementary analysis only; observational interaction, not a causal climate-mode attribution.",
        "background_definition": {
            "reference_window": [climate_start.date().isoformat(), climate_end.date().isoformat()],
            "window_role": "primary pre-Sentinel-2 background" if background_window == "pre_s2" else "overlapping full-period sensitivity only",
            "primary_modifier": "wave_exposure_z = mean(z(long-term mean VHM0), z(long-term p90 VHM0))",
            "sensitivity_modifier": "wave_directionality_z = z(multi-decadal resultant wave-vector concentration)",
            "source": "WAVERYS daily sampled grid cell per reef",
        },
        "predefined_model": {
            "sample": "exact Sentinel-2 source contract of the primary directional model",
            "outcomes": ["along_shift", "cross_shift"],
            "primary_interactions": [
                "wave_vector_along_mean_z × wave_exposure_z for along_shift",
                "wave_vector_cross_mean_z × wave_exposure_z for cross_shift",
            ],
            "sensitivity": "Replace wave_exposure_z with wave_directionality_z; do not combine both modifiers in one model.",
            "covariates": BASE_TERMS,
            "estimation": "within-cay fixed effects; cay-clustered SE primary, reef and WAVERYS-cell clustered SE sensitivity; BH-FDR within each two-test family",
        },
        "gates": gates,
        "model_permitted": bool(permitted),
        "reasons_if_not_permitted": [] if permitted else [
            "Do not estimate climate-background interactions until all audit gates pass."
        ],
    }


def demean_and_fit(frame: pd.DataFrame, outcome: str, modifier: str, cluster: str) -> tuple[pd.DataFrame, dict[str, object]]:
    outcome_column = OUTCOMES[outcome]
    force_term = PRIMARY_TERM_BY_OUTCOME[outcome]
    interaction = f"{force_term}_x_{modifier}"
    work = frame.copy()
    work[interaction] = work[force_term] * work[modifier]
    terms = [*BASE_TERMS, interaction]
    informative = work[[outcome_column, *terms, cluster]].notna().all(axis=1)
    work = work.loc[informative].copy()
    groups = work.groupby("sand_cay_id", observed=True)
    y = work[outcome_column] - groups[outcome_column].transform("mean")
    x = work[terms] - groups[terms].transform("mean")
    keep = x.abs().sum(axis=1).gt(0)
    work, y, x = work.loc[keep], y.loc[keep], x.loc[keep]
    design = x.to_numpy(dtype=float)
    vif = {term: float(variance_inflation_factor(design, i)) for i, term in enumerate(terms)}
    fitted = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": work[cluster], "use_correction": True})
    rows = []
    for term in terms:
        low, high = fitted.conf_int().loc[term].astype(float)
        rows.append(
            {
                "outcome": outcome,
                "modifier": modifier,
                "cluster_level": cluster,
                "term": term,
                "n_intervals": int(len(work)),
                "n_cays": int(work["sand_cay_id"].nunique()),
                "n_reefs": int(work["reef_id"].nunique()),
                "n_wave_grid_cells": int(work["wave_grid_id"].nunique()),
                "coefficient": float(fitted.params[term]),
                "standard_error": float(fitted.bse[term]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "two_sided_p_value": float(fitted.pvalues[term]),
                "vif": vif[term],
                "vif_unstable": bool(vif[term] >= 5),
            }
        )
    diagnostics = {
        "outcome": outcome,
        "modifier": modifier,
        "cluster_level": cluster,
        "n_intervals": int(len(work)),
        "n_cays": int(work["sand_cay_id"].nunique()),
        "n_reefs": int(work["reef_id"].nunique()),
        "n_wave_grid_cells": int(work["wave_grid_id"].nunique()),
        "max_vif": float(max(vif.values())),
        "n_vif_ge_5": int(sum(value >= 5 for value in vif.values())),
        "r_squared": float(fitted.rsquared),
        "adj_r_squared": float(fitted.rsquared_adj),
        "residual_skew": float(fitted.resid.skew()),
        "residual_kurtosis": float(fitted.resid.kurt()),
    }
    return pd.DataFrame(rows), diagnostics


def reproduce_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """在新脚本中复现图 3 主规格，防止背景交互分析换了样本或口径。"""
    reference = pd.read_csv(PRIMARY_MODELS, encoding="utf-8-sig")
    reference = reference.loc[
        reference["specification"].eq("within_cay_fixed_effects")
        & reference["outcome"].isin(OUTCOMES)
        & reference["term"].isin(
            ["current_along_mean_z", "wave_vector_along_mean_z", "current_cross_mean_z", "wave_vector_cross_mean_z"]
        )
    ]
    rows = []
    for outcome, outcome_column in OUTCOMES.items():
        work = frame.loc[frame[[outcome_column, *BASE_TERMS]].notna().all(axis=1)].copy()
        group = work.groupby("sand_cay_id", observed=True)
        y = work[outcome_column] - group[outcome_column].transform("mean")
        x = work[BASE_TERMS] - group[BASE_TERMS].transform("mean")
        keep = x.abs().sum(axis=1).gt(0)
        work, y, x = work.loc[keep], y.loc[keep], x.loc[keep]
        fitted = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": work["sand_cay_id"], "use_correction": True})
        for term in reference.loc[reference["outcome"].eq(outcome), "term"]:
            expected = reference.loc[
                reference["outcome"].eq(outcome) & reference["term"].eq(term), "coefficient"
            ].iloc[0]
            actual = float(fitted.params[term])
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "reference_coefficient": float(expected),
                    "reproduced_coefficient": actual,
                    "absolute_difference": float(abs(actual - expected)),
                    "n_intervals": int(len(work)),
                    "n_cays": int(work["sand_cay_id"].nunique()),
                    "n_reefs": int(work["reef_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def run_model(frame: pd.DataFrame, background: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_frame = frame.merge(
        background[["reef_id", "wave_grid_id", "wave_exposure_z", "wave_directionality_z"]],
        on="reef_id", how="inner", validate="many_to_one"
    )
    all_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for modifier, family in (
        ("wave_exposure_z", "primary_exposure"),
        ("wave_directionality_z", "sensitivity_directionality"),
    ):
        for cluster in ("sand_cay_id", "reef_id", "wave_grid_id"):
            for outcome in OUTCOMES:
                result, diag = demean_and_fit(model_frame, outcome, modifier, cluster)
                result["family"] = family
                all_rows.append(result)
                diagnostics.append(diag | {"family": family})
    models = pd.concat(all_rows, ignore_index=True)
    models["fdr_q_value"] = np.nan
    primary_interactions = models.loc[
        models["term"].str.contains("_x_", regex=False)
        & models["cluster_level"].eq("sand_cay_id")
    ]
    for family, indices in primary_interactions.groupby("family").groups.items():
        models.loc[indices, "fdr_q_value"] = multipletests(
            models.loc[indices, "two_sided_p_value"], method="fdr_bh"
        )[1]
    reproduction = reproduce_primary(model_frame)
    return models, pd.DataFrame(diagnostics), reproduction


def leave_one_wave_grid_out(frame: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """逐一剔除独立 WAVERYS 采样格点，检验交互符号是否由单个背景格点主导。"""
    model_frame = frame.merge(
        background[["reef_id", "wave_grid_id", "wave_exposure_z", "wave_directionality_z"]],
        on="reef_id", how="inner", validate="many_to_one"
    )
    rows: list[dict[str, object]] = []
    for modifier in ("wave_exposure_z", "wave_directionality_z"):
        for omitted in sorted(model_frame["wave_grid_id"].dropna().unique()):
            subset = model_frame.loc[model_frame["wave_grid_id"].ne(omitted)].copy()
            for outcome in OUTCOMES:
                result, _ = demean_and_fit(subset, outcome, modifier, "wave_grid_id")
                interaction = f"{PRIMARY_TERM_BY_OUTCOME[outcome]}_x_{modifier}"
                row = result.loc[result["term"].eq(interaction)].iloc[0].to_dict()
                row["omitted_wave_grid_id"] = omitted
                rows.append(row)
    result = pd.DataFrame(rows)
    summary = (
        result.groupby(["modifier", "outcome"], observed=True)
        .agg(
            n_leave_one_grid_runs=("omitted_wave_grid_id", "size"),
            same_sign_fraction=("coefficient", lambda values: float(np.mean(np.sign(values) == np.sign(values.iloc[0])))),
            coefficient_min=("coefficient", "min"),
            coefficient_max=("coefficient", "max"),
        )
        .reset_index()
    )
    return result.merge(summary, on=["modifier", "outcome"], how="left", validate="many_to_one")


def within_region_directionality_check(
    frame: pd.DataFrame, background: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """检验波向集中度交互是否仅由大区域间差异造成。

    只保留至少 10 个礁盘的区域，并在区域内部以礁盘为单位重新标准化。
    此为否证性空间混杂检验，不改变波能暴露主检验或波向集中度敏感性的地位。
    """
    region_by_reef = frame[["reef_id", "region"]].drop_duplicates()
    reef_background = region_by_reef.merge(
        background[["reef_id", "wave_grid_id", "wave_directionality_z"]],
        on="reef_id", how="inner", validate="one_to_one"
    )
    reef_counts = reef_background.groupby("region", observed=True)["reef_id"].transform("size")
    reef_background = reef_background.loc[
        reef_counts.ge(MIN_REEFS_PER_SPATIAL_STRATUM)
    ].copy()
    modifier = "wave_directionality_within_large_region_z"
    reef_background[modifier] = reef_background.groupby("region", observed=True)[
        "wave_directionality_z"
    ].transform(zscore)
    model_frame = frame.merge(
        reef_background[["reef_id", "wave_grid_id", modifier]],
        on="reef_id", how="inner", validate="many_to_one"
    )
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for cluster in ("sand_cay_id", "reef_id", "wave_grid_id"):
        for outcome in OUTCOMES:
            result, diagnostic = demean_and_fit(model_frame, outcome, modifier, cluster)
            interaction = f"{PRIMARY_TERM_BY_OUTCOME[outcome]}_x_{modifier}"
            rows.append(result.loc[result["term"].eq(interaction)].copy())
            diagnostics.append(
                diagnostic
                | {
                    "check": "within_large_region_directionality",
                    "minimum_reefs_per_region": MIN_REEFS_PER_SPATIAL_STRATUM,
                    "regions_retained": ",".join(sorted(reef_background["region"].unique())),
                }
            )
    results = pd.concat(rows, ignore_index=True)
    results["check"] = "within_large_region_directionality"
    results["fdr_q_value"] = np.nan
    primary = results["cluster_level"].eq("sand_cay_id")
    results.loc[primary, "fdr_q_value"] = multipletests(
        results.loc[primary, "two_sided_p_value"], method="fdr_bh"
    )[1]
    return results, pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "model"), default="audit")
    parser.add_argument("--background-window", choices=("pre_s2", "full_period"), default="pre_s2")
    args = parser.parse_args()
    if args.background_window == "pre_s2":
        climate_start, climate_end, suffix = PRE_S2_START, PRE_S2_END, ""
    else:
        climate_start, climate_end, suffix = FULL_PERIOD_START, FULL_PERIOD_END, "_全时段敏感性"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_primary_frame()
    reef_ids = set(frame["reef_id"].unique())
    descriptors, coverage = load_wave_descriptors(reef_ids, climate_start, climate_end)
    background = build_background(descriptors, load_wave_grid_map(reef_ids))
    audit = make_audit(frame, background, coverage, climate_start, climate_end, args.background_window)
    background.to_csv(OUTPUT / f"跨气候水动力背景{suffix}_礁盘描述符.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUTPUT / f"跨气候水动力背景{suffix}_日值覆盖.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / f"跨气候水动力背景{suffix}_可行性审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.stage == "model":
        if not audit["model_permitted"]:
            raise SystemExit("Model stage stopped: audit gate not met.")
        models, diagnostics, reproduction = run_model(frame, background)
        models.to_csv(OUTPUT / f"跨气候水动力背景{suffix}_方向耦合模型.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(OUTPUT / f"跨气候水动力背景{suffix}_模型诊断.csv", index=False, encoding="utf-8-sig")
        reproduction.to_csv(OUTPUT / f"跨气候水动力背景{suffix}_主模型复现.csv", index=False, encoding="utf-8-sig")
        leave_one_wave_grid_out(frame, background).to_csv(
            OUTPUT / f"跨气候水动力背景{suffix}_格点逐一剔除.csv", index=False, encoding="utf-8-sig"
        )
        within_region, within_region_diagnostics = within_region_directionality_check(frame, background)
        within_region.to_csv(
            OUTPUT / f"跨气候水动力背景{suffix}_区域内波向集中度检验.csv",
            index=False,
            encoding="utf-8-sig",
        )
        within_region_diagnostics.to_csv(
            OUTPUT / f"跨气候水动力背景{suffix}_区域内波向集中度诊断.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not reproduction["absolute_difference"].lt(1e-10).all():
            raise SystemExit("Model stage stopped: primary-model reproduction mismatch.")


if __name__ == "__main__":
    main()
