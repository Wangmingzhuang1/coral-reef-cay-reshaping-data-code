"""按长期净面积变化、形态活动度和区域分层审计沙洲类型。

本脚本不把“端点净面积稳定”误称为“形态稳定”：先以既有、可追溯的
0.25 / 0.50 净面积阈值形成大变化、中等变化、净面积稳定三类，再以面积
波动、相邻面积突变和质心路径识别其中的严格形态稳定子集。

环境比较只使用 Sentinel-2 与完整风-浪-流核心区间；因类型是沙洲长期属性，
本结果是小样本描述和组间关联，不是因果模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
NET_STABLE_THRESHOLD = 0.25
NET_MAJOR_THRESHOLD = 0.50
LATE_CV_THRESHOLD = 0.15
MAX_STEP_RATIO_THRESHOLD = 1.50
CENTROID_PATH_THRESHOLD = 1.00
PERMUTATIONS = 9_999
RANDOM_SEED = 20260911


def region_code(reef_id: pd.Series) -> pd.Series:
    return reef_id.astype("string").str.extract(r"^([A-Z]+)", expand=False)


def net_area_type(relative_change: pd.Series) -> pd.Categorical:
    magnitude = pd.to_numeric(relative_change, errors="coerce").abs()
    return pd.Categorical(
        np.select(
            [magnitude.lt(NET_STABLE_THRESHOLD), magnitude.lt(NET_MAJOR_THRESHOLD)],
            ["Net-area stable (<25%)", "Moderate net change (25-<50%)"],
            default="Major net change (>=50%)",
        ),
        categories=[
            "Net-area stable (<25%)",
            "Moderate net change (25-<50%)",
            "Major net change (>=50%)",
        ],
        ordered=True,
    )


def add_type_columns(primary: pd.DataFrame) -> pd.DataFrame:
    frame = primary.copy()
    frame["region"] = region_code(frame["reef_id"])
    frame["net_area_type"] = net_area_type(frame["relative_change"])
    strict = (
        frame["relative_change"].abs().lt(NET_STABLE_THRESHOLD)
        & frame["late_cv"].le(LATE_CV_THRESHOLD)
        & frame["max_adjacent_area_ratio"].le(MAX_STEP_RATIO_THRESHOLD)
        & frame["centroid_path_normalized"].le(CENTROID_PATH_THRESHOLD)
    )
    frame["strict_morphological_stability"] = strict
    frame["two_axis_type"] = np.select(
        [
            strict,
            frame["net_area_type"].eq("Net-area stable (<25%)"),
            frame["net_area_type"].eq("Moderate net change (25-<50%)"),
        ],
        [
            "Morphologically stable",
            "Net-stable but reworking",
            "Moderate transformation",
        ],
        default="Major transformation",
    )
    return frame


def permutation_chi_square(
    frame: pd.DataFrame, row: str, column: str
) -> dict[str, float | int | str]:
    table = pd.crosstab(frame[row], frame[column])
    table = table.loc[table.sum(axis=1).gt(0), table.sum(axis=0).gt(0)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"status": "insufficient_cells", "n": int(len(frame))}
    observed, _, _, _ = chi2_contingency(table, correction=False)
    labels = frame[row].astype(str).to_numpy()
    outcomes = frame[column].astype(str).to_numpy()
    row_levels = table.index.astype(str).tolist()
    col_levels = table.columns.astype(str).tolist()
    row_codes = pd.Categorical(labels, categories=row_levels).codes
    column_codes = pd.Categorical(outcomes, categories=col_levels).codes
    expected = np.outer(
        np.bincount(row_codes, minlength=len(row_levels)),
        np.bincount(column_codes, minlength=len(col_levels)),
    ) / len(frame)
    rng = np.random.default_rng(RANDOM_SEED)
    simulated = np.zeros(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        shuffled = rng.permutation(column_codes)
        simulated_table = np.zeros_like(expected)
        np.add.at(simulated_table, (row_codes, shuffled), 1)
        simulated[index] = np.sum((simulated_table - expected) ** 2 / expected)
    return {
        "status": "ok",
        "n": int(len(frame)),
        "rows": int(table.shape[0]),
        "columns": int(table.shape[1]),
        "chi_square": float(observed),
        "permutation_p_value": float((np.sum(simulated >= observed) + 1) / (PERMUTATIONS + 1)),
    }


def aggregate_s2_environment(primary: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    s2_primary = primary.loc[primary["sensor"].eq("sentinel2")].copy()
    s2_core = core.loc[core["sensor"].eq("sentinel2")].copy()
    keys = ["reef_id", "sand_cay_id", "sensor", "reference_frame_id"]
    usable = s2_core.merge(
        s2_primary[
            keys
            + [
                "region",
                "net_area_type",
                "two_axis_type",
                "strict_morphological_stability",
                "multimetric_class",
            ]
        ],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    usable["strong_typhoon_exposure"] = (
        pd.to_numeric(usable["typhoon_strong_count"], errors="coerce").fillna(0).gt(0)
    ).astype(float)
    numeric = [
        "vegetation_fraction_t",
        "gross_mobility_fraction_per_year",
        "wave_hs_p90",
        "current_speed_mean",
        "wind_speed_p90",
        "strong_typhoon_exposure",
        "area_t_m2",
    ]
    for column in numeric:
        usable[column] = pd.to_numeric(usable[column], errors="coerce")
    cay = (
        usable.groupby(
            keys
            + [
                "region",
                "net_area_type",
                "two_axis_type",
                "strict_morphological_stability",
                "multimetric_class",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_intervals=("transition_id", "size"),
            median_vegetation_fraction=("vegetation_fraction_t", "median"),
            median_gross_mobility=("gross_mobility_fraction_per_year", "median"),
            median_wave_hs_p90_m=("wave_hs_p90", "median"),
            median_current_speed_m_s=("current_speed_mean", "median"),
            median_wind_speed_p90_m_s=("wind_speed_p90", "median"),
            strong_typhoon_interval_fraction=("strong_typhoon_exposure", "mean"),
            median_start_area_m2=("area_t_m2", "median"),
        )
        .reset_index()
    )
    return cay


def type_environment_tests(cays: pd.DataFrame) -> pd.DataFrame:
    drivers = [
        "median_vegetation_fraction",
        "median_gross_mobility",
        "median_wave_hs_p90_m",
        "median_current_speed_m_s",
        "median_wind_speed_p90_m_s",
        "strong_typhoon_interval_fraction",
        "median_start_area_m2",
    ]
    rows: list[dict[str, object]] = []
    for driver in drivers:
        selected = cays[["net_area_type", driver]].dropna()
        groups = [
            group[driver].to_numpy(dtype=float)
            for _, group in selected.groupby("net_area_type", observed=True)
            if len(group) >= 3
        ]
        counts = selected.groupby("net_area_type", observed=True).size()
        row: dict[str, object] = {
            "driver": driver,
            "n_cays": int(len(selected)),
            "n_groups_with_at_least_3_cays": int(len(groups)),
            "group_counts": json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False),
        }
        for category, group in selected.groupby("net_area_type", observed=True):
            row[f"median__{category}"] = float(group[driver].median())
        if len(groups) >= 2:
            statistic, p_value = kruskal(*groups)
            row.update(
                {
                    "status": "descriptive_small_sample",
                    "kruskal_h": float(statistic),
                    "two_sided_p_value": float(p_value),
                }
            )
        else:
            row.update({"status": "insufficient_group_size", "kruskal_h": np.nan, "two_sided_p_value": np.nan})
        rows.append(row)
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    valid = result["two_sided_p_value"].notna()
    if valid.any():
        result.loc[valid, "fdr_q_value"] = multipletests(
            result.loc[valid, "two_sided_p_value"], method="fdr_bh"
        )[1]
    return result


def stable_subtype_contrast(cays: pd.DataFrame) -> pd.DataFrame:
    """只在净面积稳定沙洲内比较严格形态稳定与持续重塑，避免混入净变化。"""
    drivers = [
        "median_vegetation_fraction",
        "median_gross_mobility",
        "median_wave_hs_p90_m",
        "median_current_speed_m_s",
        "median_wind_speed_p90_m_s",
        "median_start_area_m2",
    ]
    stable = cays.loc[
        cays["two_axis_type"].isin(
            ["Morphologically stable", "Net-stable but reworking"]
        )
    ].copy()
    rows: list[dict[str, object]] = []
    for driver in drivers:
        strict = stable.loc[
            stable["two_axis_type"].eq("Morphologically stable"), driver
        ].dropna()
        reworking = stable.loc[
            stable["two_axis_type"].eq("Net-stable but reworking"), driver
        ].dropna()
        row: dict[str, object] = {
            "driver": driver,
            "n_morphologically_stable": int(len(strict)),
            "n_net_stable_reworking": int(len(reworking)),
            "median_morphologically_stable": float(strict.median()) if len(strict) else np.nan,
            "median_net_stable_reworking": float(reworking.median()) if len(reworking) else np.nan,
        }
        if len(strict) >= 3 and len(reworking) >= 3:
            statistic, p_value = mannwhitneyu(strict, reworking, alternative="two-sided")
            row.update(
                {
                    "status": "descriptive_small_sample",
                    "mann_whitney_u": float(statistic),
                    "two_sided_p_value": float(p_value),
                    "cliffs_delta_strict_minus_reworking": float(
                        2 * statistic / (len(strict) * len(reworking)) - 1
                    ),
                }
            )
        else:
            row.update(
                {
                    "status": "insufficient_group_size",
                    "mann_whitney_u": np.nan,
                    "two_sided_p_value": np.nan,
                    "cliffs_delta_strict_minus_reworking": np.nan,
                }
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    valid = result["two_sided_p_value"].notna()
    if valid.any():
        result.loc[valid, "fdr_q_value"] = multipletests(
            result.loc[valid, "two_sided_p_value"], method="fdr_bh"
        )[1]
    return result


def main() -> None:
    primary_path = OUTPUT_DIR / "沙洲面积主轨迹分类.csv"
    core_path = OUTPUT_DIR / "流场波浪核心区间.csv"
    primary = pd.read_csv(primary_path, encoding="utf-8-sig")
    eligible = primary.loc[primary["eligible"].astype(str).str.lower().eq("true")].copy()
    typed = add_type_columns(eligible)
    regional = (
        typed.groupby(["sensor", "region", "net_area_type"], observed=True)
        .size()
        .rename("n_cays")
        .reset_index()
    )
    summary = (
        typed.groupby(["sensor", "net_area_type", "two_axis_type"], observed=True)
        .agg(
            n_cays=("sand_cay_id", "size"),
            median_relative_change=("relative_change", "median"),
            median_late_cv=("late_cv", "median"),
            median_centroid_path_normalized=("centroid_path_normalized", "median"),
        )
        .reset_index()
    )
    regional_tests: list[dict[str, object]] = []
    for sensor, group in typed.groupby("sensor"):
        result = permutation_chi_square(group, "region", "net_area_type")
        result["sensor"] = sensor
        regional_tests.append(result)
    core = pd.read_csv(core_path, low_memory=False)
    s2_cays = aggregate_s2_environment(typed, core)
    environment = type_environment_tests(s2_cays)
    stable_contrast = stable_subtype_contrast(s2_cays)

    typed.to_csv(OUTPUT_DIR / "沙洲净面积类型.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "沙洲类型分层汇总.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(OUTPUT_DIR / "沙洲类型区域构成.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(regional_tests).to_csv(
        OUTPUT_DIR / "沙洲类型区域检验.csv", index=False, encoding="utf-8-sig"
    )
    s2_cays.to_csv(OUTPUT_DIR / "沙洲类型S2环境汇总.csv", index=False, encoding="utf-8-sig")
    environment.to_csv(OUTPUT_DIR / "沙洲类型S2环境关联.csv", index=False, encoding="utf-8-sig")
    stable_contrast.to_csv(
        OUTPUT_DIR / "沙洲稳定子类S2对照.csv", index=False, encoding="utf-8-sig"
    )

    audit = {
        "analysis_unit": "primary same-cay same-sensor same-reference-frame trajectory",
        "net_area_type_rule": {
            "stable": "absolute endpoint relative area change < 0.25",
            "moderate": "0.25 <= absolute endpoint relative area change < 0.50",
            "major": "absolute endpoint relative area change >= 0.50",
            "basis": "0.25 follows the existing trajectory classifier; 0.50 is a transparent major-change boundary, not a geomorphic stage definition",
        },
        "strict_morphological_stability_rule": {
            "required": [
                "net-area stable",
                "late CV <= 0.15",
                "maximum adjacent area ratio <= 1.50",
                "normalized centroid path <= 1.00 initial equivalent radius",
            ],
            "interpretation": "A strict descriptive subset, not evidence that vegetation causes stability.",
        },
        "primary_cays": int(len(typed)),
        "net_area_type_counts": {
            str(key): int(value)
            for key, value in typed["net_area_type"].value_counts(sort=False).items()
        },
        "strict_morphological_stability_cays": int(typed["strict_morphological_stability"].sum()),
        "region_codes": sorted(typed["region"].dropna().unique().tolist()),
        "region_sensor_confounded": True,
        "s2_environment_cays": int(len(s2_cays)),
        "s2_environment_regions": sorted(s2_cays["region"].dropna().unique().tolist()),
        "interpretation_limit": "Regional composition is descriptive because sensor coverage differs by region. Environmental comparisons are S2-only, cay-level, small-sample associations and do not establish type drivers.",
    }
    with (OUTPUT_DIR / "沙洲类型分析核查.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
