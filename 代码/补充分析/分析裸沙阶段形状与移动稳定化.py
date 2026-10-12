"""Stage-stratified test of whether bare-sand cays stabilize within the record.

The vegetation-mobility contrast describes a level difference.  This script asks
whether the *temporal* stabilization the user hypothesised occurs in the bare-sand
stage: within the observed record, do bare cays show declining centroid mobility,
declining shape-change rates, and smaller centroid excursions in the later half of
their record?

Boundary: elapsed time is measured from the first Sentinel-2 observation of each
track.  It is record elapsed time, not cay age.  Submerged/emergence stages exist
only in sparse Google Earth imagery and are reported descriptively.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(HERE))
from 区域气候滞后模型 import apply_figure_style  # noqa: E402
from 分析沙洲形状与质心迁移 import (  # noqa: E402
    MOBILITY_OFFSET,
    circular_doy_gap,
    load_observations,
    same_season_pairs,
)

BARE_STAGES = ("stage2_transitional_emergence", "stage3_stable_exposed")
VEGETATED_STAGES = ("stage4_stable_vegetated",)
SHAPE_RATE_OFFSET = 0.005
MIN_RANGE_OBSERVATIONS = 6
MIN_RANGE_SPAN_YEARS = 5.0
GE_SEASON_GAP_DAYS = 30
GE_DURATION_DAYS = (300, 4000)


def fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["reef_id"], "use_correction": True}
    )


def build_pairs(observations: pd.DataFrame) -> pd.DataFrame:
    pairs = same_season_pairs(observations)
    stage = observations[["image_id", "development_stage"]].drop_duplicates("image_id")
    pairs = pairs.merge(
        stage.rename(columns={"development_stage": "stage_t"}),
        left_on="image_id_t",
        right_on="image_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["image_id"])
    starts = observations.groupby(["sand_cay_id", "reference_frame_id"], dropna=False)["date"].min().rename("track_start")
    pairs = pairs.merge(starts, on=["sand_cay_id", "reference_frame_id"], how="left")
    midpoint = pairs["time_t"] + (pairs["time_t1"] - pairs["time_t"]) / 2
    pairs["elapsed_years"] = (midpoint - pairs["track_start"]).dt.days / 365.2425
    years = pairs["duration_days"] / 365.2425
    pairs["mobility_norm_per_year"] = pairs["centroid_shift_normalized"] / years
    pairs["log_mobility"] = np.log(pairs["mobility_norm_per_year"] + MOBILITY_OFFSET)
    pairs["shape_rate"] = pairs["delta_log_elongation"].abs() / years
    pairs["log_shape_rate"] = np.log(pairs["shape_rate"] + SHAPE_RATE_OFFSET)
    pairs["bare"] = pairs["stage_t"].isin(BARE_STAGES).astype(int)
    pairs = pairs.loc[pairs["stage_t"].isin(BARE_STAGES + VEGETATED_STAGES)].copy()
    return pairs.sort_values(["sand_cay_id", "time_t"]).reset_index(drop=True)


def slope_rows(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    controls = "log_area_t_m2 + duration_deviation_days + season_gap_days + C(sand_cay_id)"
    pairs = pairs.copy()
    pairs["log_area_t_m2"] = np.log(pairs["area_t_m2"])
    rows: list[dict] = []
    specs = [
        ("mobility", "log_mobility"),
        ("shape_change_rate", "log_shape_rate"),
    ]
    for outcome_name, column in specs:
        pooled = fit_clustered(f"{column} ~ elapsed_years * bare + {controls}", pairs)
        interaction_term = "elapsed_years:bare"
        low, high = pooled.conf_int().loc[interaction_term].astype(float)
        rows.append({
            "analysis": f"{outcome_name}_bare_minus_vegetated_trend", "status": "confirmatory",
            "estimate": float(pooled.params[interaction_term]), "reef_cluster_se": float(pooled.bse[interaction_term]),
            "ci95_low": low, "ci95_high": high, "p_value": float(pooled.pvalues[interaction_term]),
            "n_observations": int(len(pairs)), "n_cays": int(pairs["sand_cay_id"].nunique()),
            "n_reefs": int(pairs["reef_id"].nunique()),
        })
        for label, subset in [("bare", pairs.loc[pairs["bare"].eq(1)]), ("vegetated", pairs.loc[pairs["bare"].eq(0)])]:
            if subset["sand_cay_id"].nunique() < 5:
                rows.append({"analysis": f"{outcome_name}_{label}_trend", "status": "insufficient_cays"})
                continue
            fit = fit_clustered(f"{column} ~ elapsed_years + {controls}", subset)
            low, high = fit.conf_int().loc["elapsed_years"].astype(float)
            rows.append({
                "analysis": f"{outcome_name}_{label}_trend", "status": "confirmatory",
                "estimate": float(fit.params["elapsed_years"]), "reef_cluster_se": float(fit.bse["elapsed_years"]),
                "ci95_low": low, "ci95_high": high, "p_value": float(fit.pvalues["elapsed_years"]),
                "n_observations": int(len(subset)), "n_cays": int(subset["sand_cay_id"].nunique()),
                "n_reefs": int(subset["reef_id"].nunique()),
            })
    return pairs, rows


def centroid_excursion(observations: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (reef_id, cay_id, frame_id), group in observations.groupby(["reef_id", "sand_cay_id", "reference_frame_id"], dropna=False):
        group = group.sort_values(["date", "image_id"]).drop_duplicates("date", keep="first").reset_index(drop=True)
        span_years = (group["date"].iloc[-1] - group["date"].iloc[0]).days / 365.2425
        if len(group) < MIN_RANGE_OBSERVATIONS or span_years < MIN_RANGE_SPAN_YEARS:
            continue
        median_date = group["date"].median()
        halves = []
        for label, subset in [("early", group.loc[group["date"].le(median_date)]), ("late", group.loc[group["date"].gt(median_date)])]:
            if len(subset) < 3:
                halves.append(np.nan)
                continue
            x = subset["sand_cay_centroid_x"].to_numpy(float) * subset["pixel_size_m"].to_numpy(float)
            y = subset["sand_cay_centroid_y"].to_numpy(float) * subset["pixel_size_m"].to_numpy(float)
            radius = np.sqrt(subset["sand_cay_area_m2"].to_numpy(float) / np.pi)
            halves.append(float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)) / np.median(radius)))
        if not np.all(np.isfinite(halves)):
            continue
        records.append({
            "reef_id": reef_id, "sand_cay_id": cay_id, "reference_frame_id": frame_id,
            "first_stage": group["development_stage"].iloc[0], "n_observations": len(group),
            "span_years": span_years, "early_excursion": halves[0], "late_excursion": halves[1],
            "bare_start": int(group["development_stage"].iloc[0] in BARE_STAGES),
        })
    return pd.DataFrame(records)


def excursion_rows(excursions: pd.DataFrame) -> list[dict]:
    rows = []
    for label, subset in [("bare", excursions.loc[excursions["bare_start"].eq(1)]), ("vegetated", excursions.loc[excursions["bare_start"].eq(0)])]:
        if len(subset) < 8:
            rows.append({"analysis": f"centroid_excursion_{label}_early_vs_late", "status": "insufficient_cays"})
            continue
        stat = wilcoxon(subset["late_excursion"], subset["early_excursion"])
        rows.append({
            "analysis": f"centroid_excursion_{label}_early_vs_late", "status": "confirmatory",
            "estimate": float(subset["late_excursion"].median() - subset["early_excursion"].median()),
            "reef_cluster_se": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "p_value": float(stat.pvalue), "n_observations": int(len(subset)),
            "n_cays": int(len(subset)), "n_reefs": int(subset["reef_id"].nunique()),
            "early_median": float(subset["early_excursion"].median()),
            "late_median": float(subset["late_excursion"].median()),
        })
    return rows


def google_earth_early_descriptive(table: pd.DataFrame) -> pd.DataFrame:
    work = table.loc[table["sensor"].eq("google_earth") & table["quality_grade"].isin(["A", "B"])].copy()
    early_cays = work.loc[work["development_stage"].isin(["stage1_submerged_initial", "stage2_transitional_emergence"]), "sand_cay_id"].unique()
    records = []
    for cay_id in early_cays:
        group = work.loc[work["sand_cay_id"].eq(cay_id)].sort_values(["date", "image_id"]).drop_duplicates("date", keep="first").reset_index(drop=True)
        if len(group) < 4:
            continue
        start = group["date"].iloc[0]
        for position in range(len(group) - 1):
            source, target = group.iloc[position], group.iloc[position + 1]
            duration = (target["date"] - source["date"]).days
            gap = circular_doy_gap(source["date"], target["date"])
            if not GE_DURATION_DAYS[0] <= duration <= GE_DURATION_DAYS[1] or gap > GE_SEASON_GAP_DAYS:
                continue
            shift = float(np.hypot(
                target["sand_cay_centroid_x"] - source["sand_cay_centroid_x"],
                target["sand_cay_centroid_y"] - source["sand_cay_centroid_y"],
            ) * source["pixel_size_m"])
            radius = float(np.sqrt(source["sand_cay_area_m2"] / np.pi))
            records.append({
                "sand_cay_id": cay_id, "reef_id": source["reef_id"], "time_t": source["date"], "time_t1": target["date"],
                "elapsed_years": ((source["date"] + (target["date"] - source["date"]) / 2) - start).days / 365.2425,
                "stage_t": source["development_stage"], "stage_t1": target["development_stage"],
                "mobility_norm_per_year": shift / radius / (duration / 365.2425),
            })
    return pd.DataFrame(records)


def make_figure(pairs: pd.DataFrame, models: pd.DataFrame, excursions: pd.DataFrame, target: Path) -> None:
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5), gridspec_kw={"width_ratios": [1.05, 1.05, 0.95]})
    colors = {1: "#D55E00", 0: "#0072B2"}

    ax = axes[0]
    for bare, label in [(1, "Bare"), (0, "Vegetated")]:
        subset = pairs.loc[pairs["bare"].eq(bare)]
        ax.scatter(subset["elapsed_years"], subset["mobility_norm_per_year"], s=6, color=colors[bare], alpha=0.22, linewidths=0)
        slope, intercept = np.polyfit(subset["elapsed_years"], np.log(subset["mobility_norm_per_year"] + MOBILITY_OFFSET), 1)
        grid = np.linspace(subset["elapsed_years"].min(), subset["elapsed_years"].max(), 50)
        end_value = float(np.exp(intercept + slope * grid[-1]) - MOBILITY_OFFSET)
        ax.plot(grid, np.exp(intercept + slope * grid) - MOBILITY_OFFSET, color=colors[bare], lw=1.2)
        ax.text(grid[-1] + 0.15, end_value, label, color=colors[bare], fontsize=6.2, va="center")
    ax.set_yscale("log")
    ax.set_xlim(pairs["elapsed_years"].min() - 0.3, pairs["elapsed_years"].max() + 1.15)
    ax.set_xlabel("Record elapsed time (years)")
    ax.set_ylabel("Centroid shift / equivalent radius per year")
    ax.set_title("a  Mobility within the record", loc="left", fontsize=8)

    ax = axes[1]
    display = models.loc[models["status"].eq("confirmatory") & models["reef_cluster_se"].notna()].copy()
    labels = {
        "mobility_bare_trend": "Bare mobility trend",
        "mobility_vegetated_trend": "Vegetated mobility trend",
        "mobility_bare_minus_vegetated_trend": "Bare - vegetated mobility",
        "shape_change_rate_bare_trend": "Bare shape-change trend",
        "shape_change_rate_vegetated_trend": "Vegetated shape-change trend",
        "shape_change_rate_bare_minus_vegetated_trend": "Bare - vegetated shape",
    }
    display = display.loc[display["analysis"].isin(labels)].set_index("analysis").loc[list(labels)].reset_index()
    positions = np.arange(len(display))[::-1]
    for position, row in zip(positions, display.itertuples(index=False)):
        ax.plot([row.ci95_low, row.ci95_high], [position, position], color="#333333", lw=1.1)
        ax.plot(row.estimate, position, "o", color="#D55E00" if "bare" in row.analysis and "minus" not in row.analysis else "#444444", ms=4.0)
    ax.axvline(0, color="#888888", lw=0.7, ls="--")
    ax.set_yticks(positions, [labels[name] for name in display["analysis"]])
    ax.tick_params(axis="y", labelsize=5.9)
    x_low = float(display["ci95_low"].min()); x_high = float(display["ci95_high"].max()); pad = max(0.02, (x_high - x_low) * 0.08)
    ax.set_xlim(x_low - pad, x_high + pad)
    ax.set_xlabel("Log-outcome change per record year (95% CI)")
    ax.set_title("b  Stabilization slopes", loc="left", fontsize=8)

    ax = axes[2]
    for bare in (1, 0):
        subset = excursions.loc[excursions["bare_start"].eq(bare)]
        for row in subset.itertuples(index=False):
            ax.plot([0, 1], [row.early_excursion, row.late_excursion], color=colors[bare], alpha=0.30, lw=0.7)
        if len(subset):
            ax.scatter([0, 1], [subset["early_excursion"].median(), subset["late_excursion"].median()], s=22, color=colors[bare], marker="D", zorder=4)
    ax.text(0.02, 0.97, "Orange: bare start; blue: vegetated start.\nDiamonds mark stratum medians.", transform=ax.transAxes, va="top", fontsize=5.6)
    ax.set_xticks([0, 1], ["Early half", "Late half"])
    ax.set_ylabel("Centroid excursion / equivalent radius")
    ax.set_title("c  Within-cay excursion halves", loc="left", fontsize=8)

    for suffix, dpi in [(".png", 400), (".tiff", 400), (".pdf", None), (".svg", None)]:
        kwargs = {"bbox_inches": "tight"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(target.with_suffix(suffix), **kwargs)
    plt.close(fig)


def main() -> None:
    output_dir = (PROJECT_ROOT / "outputs/气候检验")
    figure_dir = (PROJECT_ROOT / "outputs/figures/中间图")
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    observations = load_observations()
    pairs, slope_list = slope_rows(build_pairs(observations))
    excursions = centroid_excursion(observations)
    rows = slope_list + excursion_rows(excursions)
    models = pd.DataFrame(rows)
    tested = models.loc[models["status"].eq("confirmatory") & models["p_value"].notna()].copy()
    models.loc[tested.index, "fdr_q_value"] = multipletests(tested["p_value"], method="fdr_bh")[1]
    raw_table = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", encoding="utf-8-sig", low_memory=False)
    raw_table["date"] = pd.to_datetime(raw_table["date"], errors="coerce")
    for column in ["sand_cay_area_m2", "sand_cay_centroid_x", "sand_cay_centroid_y", "pixel_size_m"]:
        raw_table[column] = pd.to_numeric(raw_table[column], errors="coerce")
    raw_table = raw_table.loc[raw_table[["sand_cay_area_m2", "sand_cay_centroid_x", "sand_cay_centroid_y", "pixel_size_m"]].notna().all(axis=1) & raw_table["sand_cay_area_m2"].gt(0)]
    ge_early = google_earth_early_descriptive(raw_table)

    pairs.to_csv(output_dir / "裸沙阶段稳定化同季节样本.csv", index=False, encoding="utf-8-sig")
    excursions.to_csv(output_dir / "裸沙阶段质心活动范围早晚半.csv", index=False, encoding="utf-8-sig")
    models.to_csv(output_dir / "裸沙阶段稳定化模型.csv", index=False, encoding="utf-8-sig")
    ge_early.to_csv(output_dir / "裸沙阶段早期GE描述.csv", index=False, encoding="utf-8-sig")
    audit = {
        "question": "Within the observed record, do bare-sand cays show declining centroid mobility, declining shape-change rate, and smaller later-half centroid excursions?",
        "strata": "Interval baseline development stage: bare = stage2/stage3; vegetated = stage4.",
        "elapsed_time": "Years since first Sentinel-2 observation of the track; not cay age.",
        "seasonal_control": "Sentinel-2 pairs restricted to 350-380 d with circular day-of-year gap <=15 d.",
        "early_stage_boundary": "stage1/stage2 exist only in sparse Google Earth imagery; reported descriptively only.",
        "n_pairs": int(len(pairs)), "n_excursion_cays": int(len(excursions)), "n_ge_early_intervals": int(len(ge_early)),
    }
    (output_dir / "裸沙阶段稳定化核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(pairs, models, excursions, figure_dir / "figure_r15_bare_sand_stabilization")
    print(models.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
