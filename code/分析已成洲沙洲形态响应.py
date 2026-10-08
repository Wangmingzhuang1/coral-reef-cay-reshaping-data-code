"""已成洲沙洲的形态响应与台风暴露定量基线。

分析单位为同一沙洲的相邻观测区间。脚本不训练预测模型，也不把关联解释为因果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from 分析范围 import (
    ANALYSIS_CONTRACT_VERSION,
    analysis_contract_digest,
    exclude_reefs,
    load_excluded_reefs,
    module_contract,
)


DIRECTIONAL_CONTRACT = module_contract("directional_background")
CORE_MIN_DAYS, CORE_MAX_DAYS = DIRECTIONAL_CONTRACT["interval_days"]
WINDOW_SENSITIVITY = [
    tuple(days) for days in DIRECTIONAL_CONTRACT["sensitivity_interval_days"]
]
VIF_UNSTABLE_THRESHOLD = 5.0
# 研究总体包括裸沙与含植被沙洲；人工阶段仅用于数据质量核查，不再把分析限定为"稳定"表面。
STABLE_STAGES = None
OUTCOMES = {
    "log_area_change_per_year": "Annualized log area change",
    "erosion_fraction_per_year": "Annualized erosion fraction",
    "deposition_fraction_per_year": "Annualized deposition fraction",
    "gross_mobility_fraction_per_year": "Annualized gross boundary mobility",
    "centroid_shift_normalized_per_year": "Annualized normalized centroid displacement",
}
RANDOM_SEED = 20260903
TYPHOON_INTERVAL_MIN_EXPOSED = 30
TYPHOON_INTERVAL_MIN_CONTRAST_CAYS = 10


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def add_stage(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["sensor", "sand_cay_id", "date"]
    stage = (
        observations.sort_values(keys + ["quality_grade"])
        .drop_duplicates(keys)
        [keys + ["development_stage"]]
    )
    result = transitions.merge(
        stage.rename(columns={"date": "time_t", "development_stage": "stage_t"}),
        on=["sensor", "sand_cay_id", "time_t"],
        how="left",
        validate="many_to_one",
    )
    return result.merge(
        stage.rename(columns={"date": "time_t1", "development_stage": "stage_t1"}),
        on=["sensor", "sand_cay_id", "time_t1"],
        how="left",
        validate="many_to_one",
    )


def prepare(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    window_days: tuple[int, int] = (CORE_MIN_DAYS, CORE_MAX_DAYS),
) -> tuple[pd.DataFrame, dict[str, int]]:
    data = add_stage(transitions.copy(), observations)
    numeric = [
        "time_interval_days",
        "area_t_m2",
        "area_t1_m2",
        "erosion_area_m2",
        "deposition_area_m2",
        "gross_boundary_change_m2",
        "centroid_shift_m",
        "vegetation_fraction_t",
        "typhoon_strong_count",
        "typhoon_r34_count",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["time_t"] = pd.to_datetime(data["time_t"], errors="coerce")
    data["time_t1"] = pd.to_datetime(data["time_t1"], errors="coerce")
    data["midpoint_year"] = (
        data["time_t"] + (data["time_t1"] - data["time_t"]) / 2
    ).dt.year
    data["years"] = data["time_interval_days"] / 365.2425
    data["equivalent_radius_m"] = np.sqrt(data["area_t_m2"] / np.pi)
    data["log_area_change_per_year"] = (
        np.log(data["area_t1_m2"] / data["area_t_m2"]) / data["years"]
    )
    data["erosion_fraction_per_year"] = (
        data["erosion_area_m2"] / data["area_t_m2"] / data["years"]
    )
    data["deposition_fraction_per_year"] = (
        data["deposition_area_m2"] / data["area_t_m2"] / data["years"]
    )
    data["gross_mobility_fraction_per_year"] = (
        data["gross_boundary_change_m2"] / data["area_t_m2"] / data["years"]
    )
    data["centroid_shift_normalized_per_year"] = (
        data["centroid_shift_m"] / data["equivalent_radius_m"] / data["years"]
    )
    data["strong_typhoon_exposure"] = data["typhoon_strong_count"].fillna(0).gt(0).astype(int)
    data["r34_exposure"] = data["typhoon_r34_count"].fillna(0).gt(0).astype(int)
    data["typhoon_available"] = to_bool(data["typhoon_data_available"])
    data["region"] = data["reef_id"].astype("string").str.split("_").str[0]
    data["stable_both_ends"] = data["stage_t"].notna() & data["stage_t1"].notna()
    data["quality_good_both"] = data["quality_grade_t"].isin(["A", "B"]) & data[
        "quality_grade_t1"
    ].isin(["A", "B"])

    checks = {
        "input_intervals": int(len(data)),
        "boundary_valid": int(data["boundary_change_status"].eq("ok").sum()),
        "core_time_window": int(
            data["time_interval_days"].between(*window_days).sum()
        ),
        "stable_both_ends": int(data["stable_both_ends"].sum()),
        "quality_good_both": int(data["quality_good_both"].sum()),
    }
    analysis = data[
        data["boundary_change_status"].eq("ok")
        & data["time_interval_days"].between(*window_days)
        & data["area_t_m2"].gt(0)
        & data["years"].gt(0)
        & data["stable_both_ends"]
        & data["quality_good_both"]
        & data["typhoon_available"]
        & data["sensor"].eq(DIRECTIONAL_CONTRACT["primary_sensor"])
    ].copy()
    analysis.replace([np.inf, -np.inf], np.nan, inplace=True)
    sequential = data.copy()
    steps = {
        "boundary_valid": sequential["boundary_change_status"].eq("ok"),
        "window_valid": sequential["time_interval_days"].between(*window_days),
        "positive_initial_area": sequential["area_t_m2"].gt(0),
        "positive_interval_years": sequential["years"].gt(0),
        "stable_both_ends": sequential["stable_both_ends"],
        "quality_good_both": sequential["quality_good_both"],
        "typhoon_available": sequential["typhoon_available"],
        "primary_sensor": sequential["sensor"].eq(
            DIRECTIONAL_CONTRACT["primary_sensor"]
        ),
    }
    funnel_steps = {}
    previous_count = int(len(data))
    for name, mask in steps.items():
        sequential = sequential.loc[mask.reindex(sequential.index, fill_value=False)]
        current_count = int(len(sequential))
        funnel_steps[name] = {
            "remaining_intervals": current_count,
            "excluded_at_this_step": previous_count - current_count,
            "remaining_cays": int(sequential["sand_cay_id"].nunique()),
            "remaining_reefs": int(sequential["reef_id"].nunique()),
        }
        previous_count = current_count
    funnel = {
        "window_days": list(window_days),
        "input_intervals": int(len(data)),
        "sequential_steps": funnel_steps,
        "final_analysis": int(len(analysis)),
        "final_by_sensor": {
            str(sensor): int(count)
            for sensor, count in analysis["sensor"].value_counts().items()
        },
    }
    checks["sample_funnel"] = funnel
    checks["analysis_intervals"] = int(len(analysis))
    checks["analysis_cays"] = int(analysis["sand_cay_id"].nunique())
    checks["analysis_reefs"] = int(analysis["reef_id"].nunique())
    checks["strong_typhoon_intervals"] = int(analysis["strong_typhoon_exposure"].sum())
    checks["r34_intervals"] = int(analysis["r34_exposure"].sum())
    return analysis, checks


def describe(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_specs = [
        ("all", ["strong_typhoon_exposure"]),
        ("by_sensor", ["sensor", "strong_typhoon_exposure"]),
    ]
    for scope, columns in group_specs:
        for keys, group in analysis.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, object] = {
                "scope": scope,
                **dict(zip(columns, keys, strict=True)),
                "n_intervals": int(len(group)),
                "n_cays": int(group["sand_cay_id"].nunique()),
                "n_reefs": int(group["reef_id"].nunique()),
            }
            for outcome in OUTCOMES:
                values = group[outcome].dropna()
                row[f"{outcome}_median"] = float(values.median())
                row[f"{outcome}_q25"] = float(values.quantile(0.25))
                row[f"{outcome}_q75"] = float(values.quantile(0.75))
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def paired_cay_comparison(analysis: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for exposure in ["strong_typhoon_exposure", "r34_exposure"]:
        for outcome in OUTCOMES:
            aggregated = (
                analysis.groupby(["sand_cay_id", exposure], as_index=False)[outcome]
                .median()
                .pivot(index="sand_cay_id", columns=exposure, values=outcome)
                .dropna()
            )
            if 0 not in aggregated.columns or 1 not in aggregated.columns:
                continue
            difference = (aggregated[1] - aggregated[0]).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_median_ci(difference, rng)
            try:
                p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "exposure": exposure,
                    "outcome": outcome,
                    "n_exposed_intervals": int(analysis[exposure].sum()),
                    "n_cays_with_both_states": int(len(aggregated)),
                    "unexposed_cay_median": float(aggregated[0].median()),
                    "exposed_cay_median": float(aggregated[1].median()),
                    "median_within_cay_difference": float(np.median(difference)),
                    "difference_ci95_low": ci_low,
                    "difference_ci95_high": ci_high,
                    "wilcoxon_p_value": p_value,
                    "inference_status": (
                        "inferential_eligible"
                        if (
                            int(analysis[exposure].sum())
                            >= TYPHOON_INTERVAL_MIN_EXPOSED
                            and len(aggregated) >= TYPHOON_INTERVAL_MIN_CONTRAST_CAYS
                        )
                        else "exploratory_below_exposure_gate"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    for exposure, index in result.groupby("exposure").groups.items():
        valid = result.loc[index, "wilcoxon_p_value"].notna()
        selected = result.loc[index].index[valid]
        if len(selected):
            result.loc[selected, "fdr_q_value"] = multipletests(
                result.loc[selected, "wilcoxon_p_value"], method="fdr_bh"
            )[1]
    return result


def zscore(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return series * 0
    return (series - series.mean()) / standard_deviation


def years_from_days(days: pd.Series) -> pd.Series:
    return pd.to_numeric(days, errors="coerce") / 365.2425


def mask_perturbation_sensitivity(
    analysis: pd.DataFrame,
    perturbation: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> pd.DataFrame:
    """用±1像元掩膜扰动重算方向分解，并检验主模型方向是否保持。"""
    if perturbation.empty or decomposition.empty:
        return pd.DataFrame()
    decomposition_columns = [
        "transition_id",
        "sensor",
        "reef_id",
        "sand_cay_id",
        "time_interval_days",
        "area_t_m2",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
        "gross_mobility_fraction_per_year",
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "wave_hs_p90",
        "sand_cay_major_axis_angle",
    ]
    available = [c for c in decomposition_columns if c in decomposition.columns]
    base = decomposition[available].copy()
    perturbed = perturbation.loc[
        perturbation["status"].eq("ok")
        & perturbation["perturbation_pixels"].ne(0)
    ].copy()
    rows: list[dict[str, object]] = []
    for perturbation_pixels, group in perturbed.groupby("perturbation_pixels"):
        frame = base.merge(
            group[
                [
                    "transition_id",
                    "area_t_m2",
                    "centroid_shift_m",
                    "centroid_east_shift_m",
                    "centroid_north_shift_m",
                    "gross_boundary_change_m2",
                    "major_axis_angle_t",
                ]
            ],
            on="transition_id",
            how="inner",
            suffixes=("_base", "_perturbed"),
        )
        frame["gross_mobility_fraction_per_year"] = (
            frame["gross_boundary_change_m2"] / frame["area_t_m2_perturbed"] / years_from_days(frame["time_interval_days"])
        )
        frame["centroid_shift_normalized_per_year"] = (
            frame["centroid_shift_m"]
            / np.sqrt(frame["area_t_m2_perturbed"] / np.pi)
            / years_from_days(frame["time_interval_days"])
        )
        # Recompute along/cross projections from the perturbed centroid and axis angle.
        phi = np.radians(frame["major_axis_angle_t"].to_numpy(dtype=float))
        east = frame["centroid_east_shift_m"].to_numpy(dtype=float)
        north = frame["centroid_north_shift_m"].to_numpy(dtype=float)
        years = years_from_days(frame["time_interval_days"])
        scale = np.sqrt(frame["area_t_m2_perturbed"].to_numpy(dtype=float))
        frame["centroid_along_shift_normalized_per_year"] = (
            (east * np.sin(phi) + north * -np.cos(phi)) / scale / years
        )
        frame["centroid_cross_shift_normalized_per_year"] = (
            (east * np.cos(phi) + north * np.sin(phi)) / scale / years
        )
        frame["area_t_m2"] = frame["area_t_m2_perturbed"]
        frame = frame.drop(columns=["area_t_m2_base", "area_t_m2_perturbed"])
        models, diagnostics, _ = unified_forcing_vegetation_model(
            analysis,
            frame,
            fit_random_intercepts=False,
        )
        selected = models.loc[
            models["specification"].eq("within_cay_fixed_effects")
            & models["term"].isin(
                [
                    "current_along_mean_z",
                    "wave_vector_along_mean_z",
                    "current_cross_mean_z",
                    "wave_vector_cross_mean_z",
                    "vegetation_fraction_t_z",
                    "pulse",
                ]
            )
        ]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "perturbation_pixels": int(perturbation_pixels),
                    "outcome": row.outcome,
                    "term": row.term,
                    "n_intervals": int(row.n_intervals),
                    "n_cays": int(row.n_cays),
                    "n_reefs": int(row.n_reefs),
                    "coefficient": float(row.coefficient),
                    "ci95_low": float(row.ci95_low),
                    "ci95_high": float(row.ci95_high),
                    "fdr_q_value": float(row.fdr_q_value),
                    "vif_unstable": bool(row.vif_unstable),
                }
            )
    return pd.DataFrame(rows)


def within_cay_regression(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = {
        "primary": ["log_area", "log_interval", "midpoint_year_z", "sentinel2"],
        "vegetation_adjusted": [
            "log_area",
            "log_interval",
            "midpoint_year_z",
            "sentinel2",
            "vegetation_fraction_z",
        ],
    }
    for exposure in ["strong_typhoon_exposure", "r34_exposure"]:
        for outcome in OUTCOMES:
            for specification, covariates in specifications.items():
                columns = [
                    "sand_cay_id",
                    exposure,
                    outcome,
                    "area_t_m2",
                    "time_interval_days",
                    "midpoint_year",
                    "sensor",
                ]
                if specification == "vegetation_adjusted":
                    columns.append("vegetation_fraction_t")
                frame = analysis[columns].dropna().copy()
                if outcome == "log_area_change_per_year":
                    floor = np.nan
                    frame["outcome_model"] = frame[outcome]
                    response_scale = "annualized_log_area_ratio"
                else:
                    positive = frame.loc[frame[outcome].gt(0), outcome]
                    if positive.empty:
                        continue
                    floor = float(positive.quantile(0.01) / 2)
                    frame["outcome_model"] = np.log(frame[outcome].clip(lower=0) + floor)
                    response_scale = "log_positive_annualized_response"
                frame["log_area"] = zscore(np.log(frame["area_t_m2"]))
                frame["log_interval"] = zscore(np.log(frame["time_interval_days"]))
                frame["midpoint_year_z"] = zscore(frame["midpoint_year"])
                frame["sentinel2"] = frame["sensor"].eq("sentinel2").astype(float)
                if specification == "vegetation_adjusted":
                    frame["vegetation_fraction_z"] = zscore(frame["vegetation_fraction_t"])

                model_columns = [exposure] + covariates
                grouped = frame.groupby("sand_cay_id")
                y = frame["outcome_model"] - grouped["outcome_model"].transform("mean")
                x = frame[model_columns] - grouped[model_columns].transform("mean")
                informative = x.abs().sum(axis=1).gt(0)
                frame = frame.loc[informative]
                y = y.loc[informative]
                x = x.loc[informative]
                if frame["sand_cay_id"].nunique() < 10 or x[exposure].abs().sum() == 0:
                    continue

                fitted = sm.OLS(y, x).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": frame["sand_cay_id"], "use_correction": True},
                )
                beta = float(fitted.params[exposure])
                se = float(fitted.bse[exposure])
                p_value = float(fitted.pvalues[exposure])
                ci_low, ci_high = fitted.conf_int().loc[exposure].astype(float)
                exposure_range = grouped[exposure].agg(["min", "max"])
                contrast_cays = int((exposure_range["min"] != exposure_range["max"]).sum())
                rows.append(
                    {
                        "exposure": exposure,
                        "outcome": outcome,
                        "specification": specification,
                        "n_intervals": int(len(frame)),
                        "n_cays": int(frame["sand_cay_id"].nunique()),
                        "n_cays_with_both_exposure_states": contrast_cays,
                        "n_exposed_intervals": int(frame[exposure].sum()),
                        "log_response_floor": floor,
                        "response_scale": response_scale,
                        "coefficient_log_scale": beta,
                        "cluster_robust_se": se,
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "two_sided_p_value": p_value,
                        "approximate_percent_difference": (
                            float(100 * np.expm1(beta))
                            if response_scale == "log_positive_annualized_response"
                            else np.nan
                        ),
                        "inference_status": (
                            "inferential_eligible"
                            if (
                                int(frame[exposure].sum())
                                >= TYPHOON_INTERVAL_MIN_EXPOSED
                                and contrast_cays >= TYPHOON_INTERVAL_MIN_CONTRAST_CAYS
                            )
                            else "exploratory_below_exposure_gate"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    for (_, specification), index in result.groupby(["exposure", "specification"]).groups.items():
        result.loc[index, "fdr_q_value"] = multipletests(
            result.loc[index, "two_sided_p_value"], method="fdr_bh"
        )[1]
    return result


UNIFIED_FORCING_TERMS = (
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
)

BETWEEN_WITHIN_TERMS = (
    "current_along_mean_w",
    "current_cross_mean_w",
    "wave_vector_along_mean_w",
    "wave_vector_cross_mean_w",
    "wave_hs_p90_w",
    "pulse_w",
    "veg_between_z",
    "veg_within_z",
    "hs_w_x_veg_between",
    "log_interval",
    "midpoint_year_z",
    "log_area_w",
    "area_between_z",
)


def unified_forcing_vegetation_model(
    analysis: pd.DataFrame,
    decomposition: pd.DataFrame,
    *,
    fit_random_intercepts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """沙洲内统一模型：方向分解强迫 + 台风脉冲 + 植被调节（仅 S2，预定义项）。"""
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
    frame = analysis[analysis["sensor"].eq("sentinel2")].merge(
        decomposition[columns], on="transition_id", how="inner"
    )
    frame = frame[
        frame[["current_along_mean", "wave_hs_p90", "vegetation_fraction_t"]]
        .notna()
        .all(axis=1)
    ].copy()
    frame["pulse"] = (
        frame[["typhoon_strong_count", "typhoon_r34_count"]].fillna(0).sum(axis=1)
        > 0
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
    frame["hs_veg_interaction"] = (
        frame["wave_hs_p90_z"] * frame["vegetation_fraction_t_z"]
    )
    frame["crosscur_veg_interaction"] = (
        frame["current_cross_mean_z"] * frame["vegetation_fraction_t_z"]
    )
    frame["log_interval"] = zscore(np.log(frame["time_interval_days"]))
    frame["midpoint_year_z"] = zscore(frame["midpoint_year"])
    frame["log_area_z"] = zscore(np.log(frame["area_t_m2"]))
    grouped_for_bw = frame.groupby("sand_cay_id")
    for column in (
        "current_along_mean_z",
        "current_cross_mean_z",
        "wave_vector_along_mean_z",
        "wave_vector_cross_mean_z",
        "wave_hs_p90_z",
        "pulse",
        "log_area_z",
    ):
        name = column[:-2] if column.endswith("_z") else column
        frame[f"{name}_w"] = frame[column] - grouped_for_bw[column].transform(
            "mean"
        )
    frame["veg_between_z"] = zscore(
        grouped_for_bw["vegetation_fraction_t"].transform("mean")
    )
    frame["veg_within_z"] = zscore(
        frame["vegetation_fraction_t"]
        - grouped_for_bw["vegetation_fraction_t"].transform("mean")
    )
    frame["area_between_z"] = zscore(grouped_for_bw["log_area_z"].transform("mean"))
    frame["hs_w_x_veg_between"] = frame["wave_hs_p90_w"] * frame["veg_between_z"]
    positive = frame.loc[
        frame["gross_mobility_fraction_per_year"].gt(0),
        "gross_mobility_fraction_per_year",
    ]
    floor = float(positive.quantile(0.01) / 2)
    frame["log_gross_mobility"] = np.log(
        frame["gross_mobility_fraction_per_year"].clip(lower=0) + floor
    )
    frame["along_shift_z"] = zscore(
        frame["centroid_along_shift_normalized_per_year"]
    )
    frame["cross_shift_z"] = zscore(
        frame["centroid_cross_shift_normalized_per_year"]
    )
    outcomes = {
        "log_gross_mobility": "log_gross_mobility",
        "along_shift": "along_shift_z",
        "cross_shift": "cross_shift_z",
    }
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    robustness: list[dict[str, object]] = []
    for outcome_name, outcome_column in outcomes.items():
        specifications = {
            "within_cay_fixed_effects": list(UNIFIED_FORCING_TERMS),
            "within_cay_reef_cluster_sensitivity": list(UNIFIED_FORCING_TERMS),
        }
        for specification, terms in specifications.items():
            informative = frame[terms].notna().all(axis=1) & frame[outcome_column].notna()
            sub_frame = frame.loc[informative]
            if len(sub_frame) < 30 or sub_frame["sand_cay_id"].nunique() < 15:
                diagnostics.append(
                    {
                        "outcome": outcome_name,
                        "specification": specification,
                        "status": "insufficient_sample",
                        "n_intervals": int(len(sub_frame)),
                        "n_cays": int(sub_frame["sand_cay_id"].nunique()),
                        "n_reefs": int(sub_frame["reef_id"].nunique()),
                    }
                )
                continue
            cay_groups = sub_frame.groupby("sand_cay_id")
            y = sub_frame[outcome_column] - cay_groups[outcome_column].transform("mean")
            x = sub_frame[terms] - cay_groups[terms].transform("mean")
            informative_rows = x.abs().sum(axis=1).gt(0)
            sub_frame = sub_frame.loc[informative_rows]
            y = y.loc[informative_rows]
            x = x.loc[informative_rows]
            if len(sub_frame) < 30 or sub_frame["sand_cay_id"].nunique() < 15:
                diagnostics.append(
                    {
                        "outcome": outcome_name,
                        "specification": specification,
                        "status": "insufficient_informative_rows",
                        "n_intervals": int(len(sub_frame)),
                        "n_cays": int(sub_frame["sand_cay_id"].nunique()),
                        "n_reefs": int(sub_frame["reef_id"].nunique()),
                    }
                )
                continue
            cluster_groups = (
                sub_frame["reef_id"]
                if specification == "within_cay_reef_cluster_sensitivity"
                else sub_frame["sand_cay_id"]
            )
            design = x.to_numpy(dtype=float)
            vif_values = {
                term: float(variance_inflation_factor(design, index))
                for index, term in enumerate(terms)
            }
            try:
                fitted = sm.OLS(y, x).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": cluster_groups, "use_correction": True},
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "outcome": outcome_name,
                        "specification": specification,
                        "status": f"fit_failed:{type(exc).__name__}",
                        "error": str(exc),
                        "n_intervals": int(len(sub_frame)),
                        "n_cays": int(sub_frame["sand_cay_id"].nunique()),
                        "n_reefs": int(sub_frame["reef_id"].nunique()),
                    }
                )
                continue
            residuals = pd.Series(fitted.resid, index=sub_frame.index)
            influence = fitted.get_influence()
            hat_diag = np.asarray(influence.hat_matrix_diag, dtype=float)
            denominator = np.sqrt(
                np.maximum(fitted.mse_resid * np.clip(1 - hat_diag, 1e-12, None), 1e-12)
            )
            studentized = residuals / denominator
            cook = influence.cooks_distance[0]
            diagnostics.append(
                {
                    "outcome": outcome_name,
                    "specification": specification,
                    "status": "ok",
                    "n_intervals": int(len(sub_frame)),
                    "n_cays": int(sub_frame["sand_cay_id"].nunique()),
                    "n_reefs": int(sub_frame["reef_id"].nunique()),
                    "cluster_level": (
                        "reef"
                        if specification == "within_cay_reef_cluster_sensitivity"
                        else "sand_cay"
                    ),
                    "max_vif": float(max(vif_values.values(), default=np.nan)),
                    "n_terms_vif_ge_5": int(
                        sum(v >= VIF_UNSTABLE_THRESHOLD for v in vif_values.values())
                    ),
                    "residual_skew": float(residuals.skew()),
                    "residual_kurtosis": float(residuals.kurt()),
                    "abs_studentized_gt_3": int((studentized.abs() > 3).sum()),
                    "max_cooks_d": float(np.nanmax(cook)) if len(cook) else np.nan,
                    "n_cooks_d_gt_4_over_n": int(
                        (cook > 4 / max(len(sub_frame), 1)).sum()
                    ),
                    "r_squared": float(fitted.rsquared),
                    "adj_r_squared": float(fitted.rsquared_adj),
                }
            )
            for term in terms:
                ci_low, ci_high = fitted.conf_int().loc[term].astype(float)
                rows.append(
                    {
                        "outcome": outcome_name,
                        "specification": specification,
                        "term": term,
                        "n_intervals": int(len(sub_frame)),
                        "n_cays": int(sub_frame["sand_cay_id"].nunique()),
                        "n_reefs": int(sub_frame["reef_id"].nunique()),
                        "coefficient": float(fitted.params[term]),
                        "se_type": (
                            "reef_cluster_robust_se"
                            if specification == "within_cay_reef_cluster_sensitivity"
                            else "cay_cluster_robust_se"
                        ),
                        "standard_error": float(fitted.bse[term]),
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "two_sided_p_value": float(fitted.pvalues[term]),
                        "vif": float(vif_values.get(term, np.nan)),
                        "vif_unstable": bool(
                            vif_values.get(term, np.nan) >= VIF_UNSTABLE_THRESHOLD
                        ),
                    }
                )
        if not fit_random_intercepts:
            continue
        re_terms = list(BETWEEN_WITHIN_TERMS)
        informative_re = (
            frame[re_terms].notna().all(axis=1) & frame[outcome_column].notna()
        )
        sub_frame_re = frame.loc[informative_re]
        if len(sub_frame_re) < 30 or sub_frame_re["sand_cay_id"].nunique() < 15:
            diagnostics.append(
                {
                    "outcome": outcome_name,
                    "specification": "between_within_random_intercept",
                    "status": "insufficient_sample",
                    "n_intervals": int(len(sub_frame_re)),
                    "n_cays": int(sub_frame_re["sand_cay_id"].nunique()),
                    "n_reefs": int(sub_frame_re["reef_id"].nunique()),
                }
            )
            continue
        x_re = sm.add_constant(sub_frame_re[re_terms], has_constant="add")
        y_re = sub_frame_re[outcome_column]
        model = sm.MixedLM(
            y_re,
            x_re,
            groups=sub_frame_re["sand_cay_id"],
        )
        start_variances = [0.1, 0.01, 1.0]
        attempts: list[tuple[float, object | None, str]] = []
        mixed = None
        for start_variance in start_variances:
            start_params = MixedLMParams(model.k_fe, model.k_re, model.k_vc)
            start_params.fe_params = np.zeros(model.k_fe)
            start_params.cov_re = np.eye(model.k_re) * start_variance
            start_params.vcomp = np.ones(model.k_vc)
            try:
                candidate = model.fit(
                    reml=True,
                    method="lbfgs",
                    start_params=start_params,
                    disp=False,
                )
                attempts.append((start_variance, candidate, "ok"))
                if candidate.converged:
                    mixed = candidate
                    break
            except Exception as exc:
                attempts.append((start_variance, None, f"{type(exc).__name__}: {exc}"))
        if mixed is None:
            diagnostics.append(
                {
                    "outcome": outcome_name,
                    "specification": "between_within_random_intercept",
                    "status": "fit_failed_all_start_variances",
                    "error": "; ".join(
                        f"variance={variance:.3g}: {message}"
                        for variance, _, message in attempts
                    ),
                    "n_intervals": int(len(sub_frame_re)),
                    "n_cays": int(sub_frame_re["sand_cay_id"].nunique()),
                    "n_reefs": int(sub_frame_re["reef_id"].nunique()),
                }
            )
            continue
        random_variance = float(np.asarray(mixed.cov_re).ravel()[0])
        residual_variance = float(mixed.scale)
        icc = random_variance / (random_variance + residual_variance)
        design_re = sub_frame_re[re_terms].to_numpy(dtype=float)
        vif_re = {
            term: float(variance_inflation_factor(design_re, index))
            for index, term in enumerate(re_terms)
        }
        mixed_residuals = pd.Series(mixed.resid, index=sub_frame_re.index)
        diagnostics.append(
            {
                "outcome": outcome_name,
                "specification": "between_within_random_intercept",
                "status": "ok" if bool(mixed.converged) else "not_converged",
                "converged": bool(mixed.converged),
                "n_intervals": int(len(sub_frame_re)),
                "n_cays": int(sub_frame_re["sand_cay_id"].nunique()),
                "n_reefs": int(sub_frame_re["reef_id"].nunique()),
                "log_likelihood": float(mixed.llf),
                "reml": bool(mixed.reml),
                "random_intercept_variance": random_variance,
                "residual_variance": residual_variance,
                "icc": float(icc),
                "max_vif": float(max(vif_re.values(), default=np.nan)),
                "n_terms_vif_ge_5": int(
                    sum(v >= VIF_UNSTABLE_THRESHOLD for v in vif_re.values())
                ),
                "residual_skew": float(mixed_residuals.skew()),
                "residual_kurtosis": float(mixed_residuals.kurt()),
            }
        )
        ci_table = mixed.conf_int()
        for term in re_terms:
            ci_low, ci_high = ci_table.loc[term].astype(float)
            rows.append(
                {
                    "outcome": outcome_name,
                    "specification": "between_within_random_intercept",
                    "term": term,
                    "n_intervals": int(len(sub_frame_re)),
                    "n_cays": int(sub_frame_re["sand_cay_id"].nunique()),
                    "n_reefs": int(sub_frame_re["reef_id"].nunique()),
                    "coefficient": float(mixed.params[term]),
                    "se_type": "mixedlm_model_se",
                    "standard_error": float(mixed.bse[term]),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "two_sided_p_value": float(mixed.pvalues[term]),
                    "vif": float(vif_re.get(term, np.nan)),
                    "vif_unstable": bool(
                        vif_re.get(term, np.nan) >= VIF_UNSTABLE_THRESHOLD
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby(["outcome", "specification"])[
            "two_sided_p_value"
        ].transform(lambda p: multipletests(p, method="fdr_bh")[1])
        robustness = evaluate_specification_robustness(result)
    return result, pd.DataFrame(diagnostics), pd.DataFrame(robustness)


def directional_leave_one_cay_sensitivity(
    analysis: pd.DataFrame, decomposition: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐一剔除沙洲的方向项影响检验。

    该检验仅复用主模型的 Sentinel-2 观测、预定义协变量和沙洲固定效应；
    每次剔除一个沙洲后重新标准化和拟合，避免由少数沙洲驱动方向结论。
    """
    columns = [
        "transition_id", "current_along_mean", "current_cross_mean",
        "wave_vector_along_mean", "wave_vector_cross_mean", "wave_hs_p90",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
    ]
    base = analysis[analysis["sensor"].eq("sentinel2")].merge(
        decomposition[columns], on="transition_id", how="inner"
    )
    base = base[base[["current_along_mean", "wave_hs_p90", "vegetation_fraction_t"]].notna().all(axis=1)].copy()
    eligible_cays = base.groupby("sand_cay_id").filter(
        lambda group: len(group) > 1
    )["sand_cay_id"].unique()
    base = base.loc[base["sand_cay_id"].isin(eligible_cays)].copy()
    targets = {
        "along_shift": ["current_along_mean_z", "wave_vector_along_mean_z"],
        "cross_shift": ["current_cross_mean_z", "wave_vector_cross_mean_z"],
    }
    rows: list[dict[str, object]] = []
    for omitted_cay in sorted(base["sand_cay_id"].unique()):
        frame = base.loc[base["sand_cay_id"].ne(omitted_cay)].copy()
        frame["pulse"] = (
            frame[["typhoon_strong_count", "typhoon_r34_count"]].fillna(0).sum(axis=1) > 0
        ).astype(float)
        for column in (
            "current_along_mean", "current_cross_mean", "wave_vector_along_mean",
            "wave_vector_cross_mean", "wave_hs_p90", "vegetation_fraction_t",
        ):
            frame[f"{column}_z"] = zscore(frame[column])
        frame["hs_veg_interaction"] = frame["wave_hs_p90_z"] * frame["vegetation_fraction_t_z"]
        frame["crosscur_veg_interaction"] = frame["current_cross_mean_z"] * frame["vegetation_fraction_t_z"]
        frame["log_interval"] = zscore(np.log(frame["time_interval_days"]))
        frame["midpoint_year_z"] = zscore(frame["midpoint_year"])
        frame["log_area_z"] = zscore(np.log(frame["area_t_m2"]))
        grouped = frame.groupby("sand_cay_id")
        for column in (
            "current_along_mean_z", "current_cross_mean_z", "wave_vector_along_mean_z",
            "wave_vector_cross_mean_z", "wave_hs_p90_z", "pulse", "log_area_z",
        ):
            name = column[:-2] if column.endswith("_z") else column
            frame[f"{name}_w"] = frame[column] - grouped[column].transform("mean")
        frame["along_shift_z"] = zscore(frame["centroid_along_shift_normalized_per_year"])
        frame["cross_shift_z"] = zscore(frame["centroid_cross_shift_normalized_per_year"])
        for outcome, terms_of_interest in targets.items():
            outcome_column = f"{outcome}_z"
            cay_groups = frame.groupby("sand_cay_id")
            y = frame[outcome_column] - cay_groups[outcome_column].transform("mean")
            x = frame[list(UNIFIED_FORCING_TERMS)] - cay_groups[list(UNIFIED_FORCING_TERMS)].transform("mean")
            informative = x.abs().sum(axis=1).gt(0)
            sub = frame.loc[informative]
            try:
                fitted = sm.OLS(y.loc[informative], x.loc[informative]).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": sub["sand_cay_id"], "use_correction": True},
                )
            except Exception:
                continue
            ci = fitted.conf_int()
            for term in terms_of_interest:
                rows.append({
                    "omitted_sand_cay_id": omitted_cay,
                    "outcome": outcome,
                    "term": term,
                    "n_intervals": int(len(sub)),
                    "n_cays": int(sub["sand_cay_id"].nunique()),
                    "coefficient": float(fitted.params[term]),
                    "ci95_low": float(ci.loc[term, 0]),
                    "ci95_high": float(ci.loc[term, 1]),
                    "two_sided_p_value": float(fitted.pvalues[term]),
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = (
        detail.groupby(["outcome", "term"], as_index=False)
        .agg(
            n_deletions=("omitted_sand_cay_id", "nunique"),
            coefficient_min=("coefficient", "min"),
            coefficient_max=("coefficient", "max"),
            all_same_sign_as_full=("coefficient", lambda x: bool((np.sign(x) == np.sign(x.iloc[0])).all() and np.sign(x.iloc[0]) != 0)),
            n_ci_excluding_zero=("ci95_low", lambda x: 0),
        )
    )
    excluded = detail.assign(ci_excludes_zero=(detail["ci95_low"] * detail["ci95_high"] > 0))
    counts = excluded.groupby(["outcome", "term"], as_index=False).agg(n_ci_excluding_zero=("ci_excludes_zero", "sum"))
    summary = summary.drop(columns="n_ci_excluding_zero").merge(counts, on=["outcome", "term"], how="left")
    return detail, summary


def evaluate_specification_robustness(models: pd.DataFrame) -> list[dict[str, object]]:
    primary_terms = sorted(
        {
            ("along_shift", "current_along_mean_z"),
            ("along_shift", "wave_vector_along_mean_z"),
            ("cross_shift", "current_cross_mean_z"),
            ("cross_shift", "wave_vector_cross_mean_z"),
            ("log_gross_mobility", "vegetation_fraction_t_z"),
            ("log_gross_mobility", "pulse"),
        }
    )
    robustness: list[dict[str, object]] = []
    for outcome, term in primary_terms:
        selected = models.loc[models["outcome"].eq(outcome) & models["term"].eq(term)]
        primary = selected.loc[selected["specification"].eq("within_cay_fixed_effects")]
        if primary.empty:
            robustness.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "robustness_status": "primary_missing",
                }
            )
            continue
        primary_row = primary.iloc[0]
        signs: dict[str, int] = {}
        supported: dict[str, bool] = {}
        unstable: dict[str, bool] = {}
        for specification, group in selected.groupby("specification"):
            coefficient = float(group.iloc[0]["coefficient"])
            signs[str(specification)] = int(np.sign(coefficient))
            supported[str(specification)] = bool(group.iloc[0]["fdr_q_value"] < 0.05)
            unstable[str(specification)] = bool(group.iloc[0]["vif_unstable"])
        same_sign = (
            len(set(signs.values())) == 1
            and signs.get("within_cay_fixed_effects", 0) != 0
        )
        if float(primary_row["fdr_q_value"]) >= 0.05:
            status = "primary_not_supported"
        elif same_sign and supported.get(
            "within_cay_reef_cluster_sensitivity", False
        ):
            status = "robust"
        elif same_sign:
            status = "directionally_consistent"
        else:
            status = "not_robust"
        robustness.append(
            {
                "outcome": outcome,
                "term": term,
                "primary_coefficient": float(primary_row["coefficient"]),
                "primary_ci95_low": float(primary_row["ci95_low"]),
                "primary_ci95_high": float(primary_row["ci95_high"]),
                "primary_q_value": float(primary_row["fdr_q_value"]),
                "same_sign_across_specifications": bool(same_sign),
                "robustness_status": status,
                "specification_signs": json.dumps(signs, ensure_ascii=False),
                "specification_q_below_005": json.dumps(supported, ensure_ascii=False),
                "specification_vif_unstable": json.dumps(unstable, ensure_ascii=False),
            }
        )
    return robustness


def evaluate_primary_robustness(
    models: pd.DataFrame,
    window_sensitivity: pd.DataFrame,
    perturbation_models: pd.DataFrame,
) -> pd.DataFrame:
    """按预定义规则合并主规格、礁盘聚类、时间窗和±1像元边界扰动。"""
    primary_terms = sorted(
        {
            ("along_shift", "current_along_mean_z"),
            ("along_shift", "wave_vector_along_mean_z"),
            ("cross_shift", "current_cross_mean_z"),
            ("cross_shift", "wave_vector_cross_mean_z"),
            ("log_gross_mobility", "vegetation_fraction_t_z"),
            ("log_gross_mobility", "pulse"),
        }
    )
    rows: list[dict[str, object]] = []
    for outcome, term in primary_terms:
        primary = models.loc[
            models["specification"].eq("within_cay_fixed_effects")
            & models["outcome"].eq(outcome)
            & models["term"].eq(term)
        ]
        if primary.empty:
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "robustness_status": "primary_missing",
                }
            )
            continue
        primary_row = primary.iloc[0]
        primary_sign = float(np.sign(primary_row["coefficient"]))
        primary_supported = bool(primary_row["fdr_q_value"] < 0.05)
        reef = models.loc[
            models["specification"].eq("within_cay_reef_cluster_sensitivity")
            & models["outcome"].eq(outcome)
            & models["term"].eq(term)
        ]
        windows = window_sensitivity.loc[
            window_sensitivity["status"].eq("ok")
            & window_sensitivity["outcome"].eq(outcome)
            & window_sensitivity["term"].eq(term)
        ]
        perturbations = perturbation_models.loc[
            perturbation_models["outcome"].eq(outcome)
            & perturbation_models["term"].eq(term)
        ]
        window_ok = len(windows) > 0 and bool(
            (np.sign(windows["coefficient"]) == primary_sign).all()
            and (windows["fdr_q_value"] < 0.05).any()
        )
        perturbation_ok = len(perturbations) > 0 and bool(
            (np.sign(perturbations["coefficient"]) == primary_sign).all()
            and (perturbations["fdr_q_value"] < 0.05).any()
        )
        reef_ok = (
            not reef.empty
            and float(np.sign(reef.iloc[0]["coefficient"])) == primary_sign
            and bool(reef.iloc[0]["fdr_q_value"] < 0.05)
        )
        if not primary_supported:
            status = "primary_not_supported"
        elif reef_ok and window_ok and perturbation_ok:
            status = "robust"
        elif (
            float(np.sign(reef.iloc[0]["coefficient"])) == primary_sign
            if not reef.empty
            else False
        ) and (
            (np.sign(windows["coefficient"]) == primary_sign).all()
            if len(windows)
            else False
        ) and (
            (np.sign(perturbations["coefficient"]) == primary_sign).all()
            if len(perturbations)
            else False
        ):
            status = "directionally_consistent"
        else:
            status = "not_robust"
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "primary_coefficient": float(primary_row["coefficient"]),
                "primary_ci95_low": float(primary_row["ci95_low"]),
                "primary_ci95_high": float(primary_row["ci95_high"]),
                "primary_q_value": float(primary_row["fdr_q_value"]),
                "primary_sign": int(primary_sign),
                "reef_cluster_q_below_005": reef_ok,
                "window_alternative_supported": window_ok,
                "window_alternative_same_sign": bool(
                    len(windows)
                    and (np.sign(windows["coefficient"]) == primary_sign).all()
                ),
                "perturbation_same_sign": bool(
                    len(perturbations)
                    and (np.sign(perturbations["coefficient"]) == primary_sign).all()
                ),
                "perturbation_alternative_supported": perturbation_ok,
                "robustness_status": status,
                "n_windows": int(len(windows)),
                "n_perturbations": int(len(perturbations)),
            }
        )
    return pd.DataFrame(rows)


def wind_increment_analysis(
    analysis: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P1：检验风场相对流+浪的增量解释，并排除局地台风脉冲作敏感性。"""
    columns = [
        "transition_id",
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "wave_hs_p90",
        "wind_along_mean",
        "wind_cross_mean",
        "flow_wave_wind_complete",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
    ]
    available = [c for c in columns if c in decomposition.columns]
    frame = analysis.merge(decomposition[available], on="transition_id", how="inner")
    frame = frame.loc[
        frame["flow_wave_wind_complete"].eq(True)
        & frame[["wind_along_mean", "wind_cross_mean"]].notna().all(axis=1)
    ].copy()
    coverage_by_sensor = {
        str(sensor): int(count)
        for sensor, count in frame["sensor"].value_counts().items()
    }
    frame = frame.loc[frame["sensor"].eq("sentinel2")].copy()
    comparison_rows: list[dict[str, object]] = [
        {
            "model": "coverage",
            "term": "wind_complete_subset",
            "n_intervals": int(len(frame)),
            "n_cays": int(frame["sand_cay_id"].nunique()),
            "n_reefs": int(frame["reef_id"].nunique()),
            "coverage_by_sensor": coverage_by_sensor,
        }
    ]
    minimum_intervals = 30
    minimum_cays = 10
    if len(frame) < minimum_intervals or frame["sand_cay_id"].nunique() < minimum_cays:
        comparison_rows.append(
            {
                "model": "decision",
                "term": "exploratory_only_below_inferential_gate",
                "minimum_intervals": minimum_intervals,
                "minimum_cays": minimum_cays,
                "reason": "Coverage-restricted sample cannot support inferential wind increment.",
            }
        )
        return pd.DataFrame(), pd.DataFrame(comparison_rows)
    for column in [
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "wave_hs_p90",
        "wind_along_mean",
        "wind_cross_mean",
        "vegetation_fraction_t",
    ]:
        frame[f"{column}_z"] = zscore(frame[column])
    frame["hs_veg_interaction"] = (
        frame["wave_hs_p90_z"] * frame["vegetation_fraction_t_z"]
    )
    frame["crosscur_veg_interaction"] = (
        frame["current_cross_mean_z"] * frame["vegetation_fraction_t_z"]
    )
    frame["log_interval"] = zscore(np.log(frame["time_interval_days"]))
    frame["midpoint_year_z"] = zscore(frame["midpoint_year"])
    frame["log_area_z"] = zscore(np.log(frame["area_t_m2"]))
    frame["pulse"] = (
        frame[["typhoon_strong_count", "typhoon_r34_count"]].fillna(0).sum(axis=1) > 0
    ).astype(float)
    base_terms = [
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
    wind_terms = base_terms + ["wind_along_mean_z", "wind_cross_mean_z"]
    positive = frame.loc[
        frame["gross_mobility_fraction_per_year"].gt(0),
        "gross_mobility_fraction_per_year",
    ]
    floor = float(positive.quantile(0.01) / 2) if len(positive) else 1e-6
    frame["log_gross_mobility"] = np.log(
        frame["gross_mobility_fraction_per_year"].clip(lower=0) + floor
    )
    outcomes = {
        "log_gross_mobility": "log_gross_mobility",
        "along_shift": "centroid_along_shift_normalized_per_year",
        "cross_shift": "centroid_cross_shift_normalized_per_year",
    }
    rows: list[dict[str, object]] = []
    scopes = [
        ("all_intervals_pulse_adjusted", frame, base_terms),
        (
            "no_typhoon_pulse_intervals",
            frame.loc[frame["pulse"].eq(0)].copy(),
            [term for term in base_terms if term != "pulse"],
        ),
    ]
    for scope_name, scope_frame, scope_base_terms in scopes:
        scope_wind_terms = scope_base_terms + ["wind_along_mean_z", "wind_cross_mean_z"]
        if (
            len(scope_frame) < minimum_intervals
            or scope_frame["sand_cay_id"].nunique() < minimum_cays
        ):
            comparison_rows.append(
                {
                    "analysis_scope": scope_name,
                    "model": "decision",
                    "term": "exploratory_only_below_inferential_gate",
                    "n_intervals": int(len(scope_frame)),
                    "n_cays": int(scope_frame["sand_cay_id"].nunique()),
                    "minimum_intervals": minimum_intervals,
                    "minimum_cays": minimum_cays,
                    "reason": "Scope cannot support inferential wind increment.",
                }
            )
            continue
        for outcome_name, outcome_column in outcomes.items():
            grouped = scope_frame.groupby("sand_cay_id")
            y = scope_frame[outcome_column] - grouped[outcome_column].transform("mean")
            x_all = scope_frame[scope_wind_terms] - grouped[scope_wind_terms].transform("mean")
            informative = (
                x_all.notna().all(axis=1)
                & y.notna()
                & np.isfinite(x_all).all(axis=1)
                & np.isfinite(y)
                & x_all.abs().sum(axis=1).gt(0)
            )
            fits = {}
            for model_name, terms in [
                ("flow_wave", scope_base_terms),
                ("flow_wave_wind", scope_wind_terms),
            ]:
                x = x_all[terms]
                maximum_likelihood_fit = sm.OLS(
                    y.loc[informative], x.loc[informative]
                ).fit()
                fitted = maximum_likelihood_fit.get_robustcov_results(
                    cov_type="cluster",
                    groups=scope_frame.loc[informative, "sand_cay_id"],
                    use_correction=True,
                )
                robust_params = pd.Series(np.asarray(fitted.params), index=terms)
                robust_bse = pd.Series(np.asarray(fitted.bse), index=terms)
                robust_pvalues = pd.Series(np.asarray(fitted.pvalues), index=terms)
                confidence_intervals = pd.DataFrame(
                    np.asarray(fitted.conf_int()), index=terms
                )
                fits[model_name] = (
                    maximum_likelihood_fit,
                    fitted,
                    robust_params,
                    robust_pvalues,
                )
                for term in terms:
                    ci_low, ci_high = confidence_intervals.loc[term].astype(float)
                    rows.append(
                        {
                            "analysis_scope": scope_name,
                            "outcome": outcome_name,
                            "model": model_name,
                            "term": term,
                            "n_intervals": int(informative.sum()),
                            "n_cays": int(
                                scope_frame.loc[informative, "sand_cay_id"].nunique()
                            ),
                            "coefficient": float(robust_params[term]),
                            "cluster_robust_se": float(robust_bse[term]),
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "two_sided_p_value": float(robust_pvalues[term]),
                        }
                    )
            base_ml_fit, _, _, _ = fits["flow_wave"]
            (
                wind_ml_fit,
                wind_robust_fit,
                wind_robust_params,
                wind_robust_pvalues,
            ) = fits["flow_wave_wind"]
            try:
                ml_lr_p = float(wind_ml_fit.compare_lr_test(base_ml_fit)[1])
                partial_f_p = float(wind_ml_fit.compare_f_test(base_ml_fit)[1])
            except Exception:  # noqa: BLE001 - non-nested edge case
                ml_lr_p = np.nan
                partial_f_p = np.nan
            restriction = np.zeros((2, len(scope_wind_terms)))
            restriction[0, scope_wind_terms.index("wind_along_mean_z")] = 1.0
            restriction[1, scope_wind_terms.index("wind_cross_mean_z")] = 1.0
            try:
                joint_wind_p = float(
                    np.asarray(
                        wind_robust_fit.wald_test(restriction, scalar=True).pvalue
                    ).squeeze()
                )
            except Exception:  # noqa: BLE001 - singular covariance edge case
                joint_wind_p = np.nan
            comparison_rows.append(
                {
                    "analysis_scope": scope_name,
                    "model": "increment",
                    "term": outcome_name,
                    "n_intervals": int(informative.sum()),
                    "n_cays": int(
                        scope_frame.loc[informative, "sand_cay_id"].nunique()
                    ),
                    "aic_flow_wave": float(base_ml_fit.aic),
                    "aic_flow_wave_wind": float(wind_ml_fit.aic),
                    "bic_flow_wave": float(base_ml_fit.bic),
                    "bic_flow_wave_wind": float(wind_ml_fit.bic),
                    "adj_r2_flow_wave": float(base_ml_fit.rsquared_adj),
                    "adj_r2_flow_wave_wind": float(wind_ml_fit.rsquared_adj),
                    "ml_lr_p_value": ml_lr_p,
                    "partial_f_p_value": partial_f_p,
                    "joint_wind_cluster_p_value": joint_wind_p,
                    "wind_along_coefficient": float(
                        wind_robust_params.get("wind_along_mean_z", np.nan)
                    ),
                    "wind_cross_coefficient": float(
                        wind_robust_params.get("wind_cross_mean_z", np.nan)
                    ),
                    "wind_along_p": float(
                        wind_robust_pvalues.get("wind_along_mean_z", np.nan)
                    ),
                    "wind_cross_p": float(
                        wind_robust_pvalues.get("wind_cross_mean_z", np.nan)
                    ),
                    "comparison_note": (
                        "AIC/BIC/adjusted R2 and nested tests use ordinary maximum-"
                        "likelihood OLS on identical rows; coefficient inference and "
                        "the joint wind test use cay-clustered robust covariance."
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = result.groupby(["analysis_scope", "outcome", "model"])[
            "two_sided_p_value"
        ].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    comparison = pd.DataFrame(comparison_rows)
    comparison["joint_wind_cluster_fdr_q_value"] = np.nan
    increment = comparison["model"].eq("increment")
    if increment.any():
        comparison.loc[increment, "joint_wind_cluster_fdr_q_value"] = (
            comparison.loc[increment]
            .groupby("analysis_scope")["joint_wind_cluster_p_value"]
            .transform(lambda p: multipletests(p, method="fdr_bh")[1])
        )
    return result, comparison


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=root / "outputs" / "沙洲变化区间.csv",
    )
    parser.add_argument(
        "--observation-csv",
        type=Path,
        default=root / "outputs" / "沙洲观测主表.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()

    transitions = exclude_reefs(pd.read_csv(args.transition_csv, low_memory=False))
    observations = exclude_reefs(pd.read_csv(args.observation_csv, low_memory=False))
    analysis, checks = prepare(transitions, observations)
    descriptive = describe(analysis)
    composition = (
        analysis.groupby(["region", "sensor", "strong_typhoon_exposure"], dropna=False)
        .agg(
            n_intervals=("transition_id", "size"),
            n_cays=("sand_cay_id", "nunique"),
            n_reefs=("reef_id", "nunique"),
        )
        .reset_index()
    )
    paired = paired_cay_comparison(analysis)
    regression = within_cay_regression(analysis)
    decomposition = pd.read_csv(
        args.output_dir / "流场波浪沿轴横轴分解.csv", encoding="utf-8-sig"
    )
    unified, diagnostics, robustness = unified_forcing_vegetation_model(
        analysis, decomposition
    )
    leave_one_cay, leave_one_cay_summary = directional_leave_one_cay_sensitivity(
        analysis, decomposition
    )
    wind_terms, wind_comparison = wind_increment_analysis(analysis, decomposition)
    primary_robustness = pd.DataFrame()
    perturbation_path = args.output_dir / "掩膜边界扰动稳健性.csv"
    perturbation = (
        pd.read_csv(perturbation_path, encoding="utf-8-sig")
        if perturbation_path.is_file()
        else pd.DataFrame()
    )
    perturbation_models = mask_perturbation_sensitivity(
        analysis, perturbation, decomposition
    )
    window_rows: list[dict[str, object]] = []
    for low, high in WINDOW_SENSITIVITY:
        if (low, high) == (CORE_MIN_DAYS, CORE_MAX_DAYS):
            window_analysis = analysis.copy()
        else:
            window_analysis, _ = prepare(transitions, observations, (low, high))
        window_decomposition = decomposition.loc[
            decomposition["time_interval_days"].between(low, high)
        ].copy()
        if len(window_analysis) < 30:
            window_rows.append(
                {
                    "window_days": f"{low}-{high}",
                    "status": "insufficient_sample",
                    "n_intervals": int(len(window_analysis)),
                    "n_cays": int(window_analysis["sand_cay_id"].nunique()),
                    "n_reefs": int(window_analysis["reef_id"].nunique()),
                }
            )
            continue
        window_models, _, _ = unified_forcing_vegetation_model(
            window_analysis, window_decomposition, fit_random_intercepts=False
        )
        selected = window_models.loc[
            window_models["specification"].eq("within_cay_fixed_effects")
            & window_models["term"].isin(
                [
                    "current_along_mean_z",
                    "wave_vector_along_mean_z",
                    "current_cross_mean_z",
                    "wave_vector_cross_mean_z",
                    "vegetation_fraction_t_z",
                    "pulse",
                ]
            )
        ]
        if selected.empty:
            window_rows.append(
                {
                    "window_days": f"{low}-{high}",
                    "status": "no_primary_rows",
                    "n_intervals": int(len(window_analysis)),
                    "n_cays": int(window_analysis["sand_cay_id"].nunique()),
                    "n_reefs": int(window_analysis["reef_id"].nunique()),
                }
            )
            continue
        for row in selected.itertuples(index=False):
            window_rows.append(
                {
                    "window_days": f"{low}-{high}",
                    "status": "ok",
                    "outcome": row.outcome,
                    "term": row.term,
                    "n_intervals": int(row.n_intervals),
                    "n_cays": int(row.n_cays),
                    "n_reefs": int(row.n_reefs),
                    "coefficient": float(row.coefficient),
                    "ci95_low": float(row.ci95_low),
                    "ci95_high": float(row.ci95_high),
                    "fdr_q_value": float(row.fdr_q_value),
                    "vif_unstable": bool(row.vif_unstable),
                }
            )
    window_sensitivity = pd.DataFrame(window_rows)
    primary_robustness = evaluate_primary_robustness(
        unified, window_sensitivity, perturbation_models
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "已成洲沙洲定量分析样本.csv"
    descriptive_path = args.output_dir / "已成洲沙洲响应描述统计.csv"
    composition_path = args.output_dir / "已成洲沙洲台风样本构成.csv"
    paired_path = args.output_dir / "台风暴露沙洲内配对比较.csv"
    regression_path = args.output_dir / "台风暴露固定效应回归.csv"
    unified_path = args.output_dir / "强迫植被沙洲内模型.csv"
    wind_path = args.output_dir / "风场增量检验.csv"
    wind_comparison_path = args.output_dir / "风场增量检验模型比较.csv"
    diagnostics_path = args.output_dir / "强迫植被沙洲内模型诊断.csv"
    robustness_path = args.output_dir / "强迫植被沙洲内模型稳健性.csv"
    primary_robustness_path = args.output_dir / "方向模型主要结果稳健性.csv"
    window_path = args.output_dir / "方向模型时间窗敏感性.csv"
    perturbation_model_path = args.output_dir / "方向模型掩膜扰动敏感性.csv"
    leave_one_cay_path = args.output_dir / "方向模型逐沙洲剔除敏感性.csv"
    leave_one_cay_summary_path = args.output_dir / "方向模型逐沙洲剔除敏感性汇总.csv"
    check_path = args.output_dir / "已成洲沙洲定量分析核查.json"
    analysis.to_csv(sample_path, index=False, encoding="utf-8-sig")
    descriptive.to_csv(descriptive_path, index=False, encoding="utf-8-sig")
    composition.to_csv(composition_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    regression.to_csv(regression_path, index=False, encoding="utf-8-sig")
    unified.to_csv(unified_path, index=False, encoding="utf-8-sig")
    wind_terms.to_csv(wind_path, index=False, encoding="utf-8-sig")
    wind_comparison.to_csv(wind_comparison_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    robustness.to_csv(robustness_path, index=False, encoding="utf-8-sig")
    primary_robustness.to_csv(
        primary_robustness_path, index=False, encoding="utf-8-sig"
    )
    window_sensitivity.to_csv(window_path, index=False, encoding="utf-8-sig")
    perturbation_models.to_csv(
        perturbation_model_path, index=False, encoding="utf-8-sig"
    )
    leave_one_cay.to_csv(leave_one_cay_path, index=False, encoding="utf-8-sig")
    leave_one_cay_summary.to_csv(
        leave_one_cay_summary_path, index=False, encoding="utf-8-sig"
    )

    checks.update(
        {
            "excluded_reefs": sorted(load_excluded_reefs()),
            "analysis_unit": "same-cay adjacent observation interval",
            "core_window_days": [CORE_MIN_DAYS, CORE_MAX_DAYS],
            "stage_rule": "both endpoint manual development-stage labels are present; bare and vegetated cays are retained",
            "quality_rule": "both endpoint quality grades are A or B",
            "primary_response": "annualized gross boundary mobility normalized by initial area",
            "secondary_responses": [
                "annualized log area change",
                "annualized erosion fraction",
                "annualized deposition fraction",
                "annualized centroid displacement normalized by equivalent radius",
            ],
            "inference_boundary": (
                "Paired comparisons and within-cay fixed-effect regressions estimate associations. "
                "They do not identify causal typhoon effects."
            ),
            "typhoon_interval_exposure_gate": {
                "definition": (
                    "formal exposure = cay-bearing R34 quadrant or <=100 km with "
                    "nearest center wind >=17.5 m/s"
                ),
                "formal_exposed_intervals": int(
                    analysis["strong_typhoon_exposure"].sum()
                ),
                "formal_exposed_cays": int(
                    analysis.loc[
                        analysis["strong_typhoon_exposure"].eq(1), "sand_cay_id"
                    ].nunique()
                ),
                "cays_with_both_exposure_states": int(
                    analysis.groupby("sand_cay_id")["strong_typhoon_exposure"]
                    .nunique()
                    .eq(2)
                    .sum()
                ),
                "minimum_exposed_intervals": TYPHOON_INTERVAL_MIN_EXPOSED,
                "minimum_contrast_cays": TYPHOON_INTERVAL_MIN_CONTRAST_CAYS,
                "status": (
                    "inferential_eligible"
                    if (
                        int(analysis["strong_typhoon_exposure"].sum())
                        >= TYPHOON_INTERVAL_MIN_EXPOSED
                        and int(
                            analysis.groupby("sand_cay_id")[
                                "strong_typhoon_exposure"
                            ]
                            .nunique()
                            .eq(2)
                            .sum()
                        )
                        >= TYPHOON_INTERVAL_MIN_CONTRAST_CAYS
                    )
                    else "exploratory_below_exposure_gate"
                ),
            },
            "centroid_warning": (
                "Only centroid displacement magnitude is used. Geographic bearing remains provisional "
                "until reference-frame north orientation is verified."
            ),
        }
    )
    checks["unified_forcing_vegetation_model"] = {
        "sensor": "sentinel2_only",
        "analysis_contract_version": ANALYSIS_CONTRACT_VERSION,
        "analysis_contract_sha256": analysis_contract_digest(),
        "primary_response": "log annualized gross boundary mobility (within-cay)",
        "secondary_responses": ["along-axis shift z", "cross-axis shift z"],
        "moderator": "interval-start vegetation fraction (z-scored)",
        "predefined_interactions": [
            "wave_hs_p90 x vegetation",
            "cross current x vegetation",
        ],
        "robustness_specifications": [
            "within-cay fixed effects clustered by cay",
            "within-cay fixed effects clustered by reef",
            "predefined interval-window sensitivity",
            "±1 pixel mask-boundary perturbation sensitivity",
        ],
        "diagnostics": (
            "VIF, residual skew/kurtosis, studentized outliers, Cook's distance, "
            "random-intercept variance, ICC, convergence status and model likelihood"
        ),
        "wind_increment": {
            "rows": int(len(wind_terms)),
            "comparison": wind_comparison.to_dict("records")
            if not wind_comparison.empty
            else [],
        },
        "wind_status": (
            "exploratory_coverage_restricted_pending_era5_complete"
            if wind_terms.empty
            else "fitted_on_wind_complete_subset"
        ),
        "inference_boundary": "Within-cay associations with cluster-robust SE; not causal.",
    }
    check_path.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for path in [
        sample_path,
        descriptive_path,
        composition_path,
        paired_path,
        regression_path,
        unified_path,
        wind_path,
        wind_comparison_path,
        diagnostics_path,
        robustness_path,
        primary_robustness_path,
        window_path,
        perturbation_model_path,
        check_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
