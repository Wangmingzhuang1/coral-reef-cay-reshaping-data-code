"""Season-controlled shape--centroid mobility analysis for sand cays.

This analysis deliberately does not treat a reef bounding box as a reef-platform
polygon.  It asks three separable questions using only quality A/B Sentinel-2
observations in a fixed reference frame:

1. Has cay elongation changed systematically through the observed 2017--2025
   record, after accounting for within-year seasonality?
2. Does an elongated cay at time t predict its centroid displacement during the
   following, season-matched year?
3. Is centroid displacement associated with concurrent shape change?  This is
   explicitly exploratory because both quantities are derived from the same
   pair of masks.
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
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from 区域气候滞后模型 import apply_figure_style  # noqa: E402

ANNUAL_DAYS = (350, 380)
SEASON_GAP_DAYS = 15
MIN_TRAJECTORY_SPAN_YEARS = 5.0
MOBILITY_OFFSET = 0.01


def circular_doy_gap(left: pd.Timestamp, right: pd.Timestamp) -> int:
    difference = abs(int(left.dayofyear) - int(right.dayofyear))
    return min(difference, 365 - difference)


def load_observations() -> pd.DataFrame:
    from 构建沙洲观测与变化表 import select_analysis_observations
    table = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", encoding="utf-8-sig", low_memory=False)
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    numeric = [
        "sand_cay_area_m2", "sand_cay_elongation", "sand_cay_compactness",
        "sand_cay_centroid_x", "sand_cay_centroid_y", "pixel_size_m",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.loc[
        table["sensor"].eq("sentinel2")
        & table["quality_grade"].isin(["A", "B"])
        & table["date"].notna()
        & table["sand_cay_area_m2"].gt(0)
        & table["sand_cay_elongation"].gt(1)
        & table["sand_cay_compactness"].gt(0)
        & table[["sand_cay_centroid_x", "sand_cay_centroid_y", "pixel_size_m"]].notna().all(axis=1)
    ].copy()
    if table.duplicated(["sand_cay_id", "reference_frame_id", "date", "image_id"]).any():
        raise ValueError("Sentinel-2 observations contain duplicate image records")
    return select_analysis_observations(table)


def same_season_pairs(observations: pd.DataFrame) -> pd.DataFrame:
    """One deterministic approximately annual successor for every observation."""
    records: list[dict[str, object]] = []
    keys = ["reef_id", "sand_cay_id", "reference_frame_id"]
    for (reef_id, cay_id, frame_id), group in observations.groupby(keys, dropna=False):
        group = group.sort_values(["date", "image_id"]).drop_duplicates("date", keep="first").reset_index(drop=True)
        for position in range(len(group) - 1):
            source = group.iloc[position]
            candidates = group.iloc[position + 1 :].copy()
            candidates["duration_days"] = (candidates["date"] - source["date"]).dt.days
            candidates = candidates.loc[candidates["duration_days"].between(*ANNUAL_DAYS)]
            if candidates.empty:
                continue
            candidates["season_gap_days"] = candidates["date"].map(lambda value: circular_doy_gap(source["date"], value))
            candidates = candidates.loc[candidates["season_gap_days"].le(SEASON_GAP_DAYS)].copy()
            if candidates.empty:
                continue
            candidates["selection_score"] = (
                (candidates["duration_days"] - 365.2425).abs() * 100 + candidates["season_gap_days"]
            )
            target = candidates.sort_values(["selection_score", "date", "image_id"]).iloc[0]
            displacement_m = float(np.hypot(
                target["sand_cay_centroid_x"] - source["sand_cay_centroid_x"],
                target["sand_cay_centroid_y"] - source["sand_cay_centroid_y"],
            ) * source["pixel_size_m"])
            equivalent_radius_m = float(np.sqrt(source["sand_cay_area_m2"] / np.pi))
            records.append({
                "pair_id": f"{cay_id}__{source['date']:%Y%m%d}__{target['date']:%Y%m%d}",
                "reef_id": reef_id, "sand_cay_id": cay_id, "reference_frame_id": frame_id,
                "image_id_t": source["image_id"], "image_id_t1": target["image_id"],
                "time_t": source["date"], "time_t1": target["date"],
                "baseline_year": int(source["date"].year),
                "duration_days": int(target["duration_days"]), "season_gap_days": int(target["season_gap_days"]),
                "area_t_m2": float(source["sand_cay_area_m2"]),
                "elongation_t": float(source["sand_cay_elongation"]),
                "elongation_t1": float(target["sand_cay_elongation"]),
                "compactness_t": float(source["sand_cay_compactness"]),
                "compactness_t1": float(target["sand_cay_compactness"]),
                "centroid_shift_m": displacement_m,
                "equivalent_radius_m": equivalent_radius_m,
            })
    pairs = pd.DataFrame(records)
    if pairs.empty:
        raise ValueError("No same-season annual Sentinel-2 pairs were constructed")
    pairs["log_elongation_t"] = np.log(pairs["elongation_t"])
    pairs["delta_log_elongation"] = np.log(pairs["elongation_t1"]) - np.log(pairs["elongation_t"])
    pairs["delta_compactness"] = pairs["compactness_t1"] - pairs["compactness_t"]
    pairs["centroid_shift_normalized"] = pairs["centroid_shift_m"] / pairs["equivalent_radius_m"]
    pairs["log_centroid_shift_normalized"] = np.log(pairs["centroid_shift_normalized"] + MOBILITY_OFFSET)
    pairs["log_area_t"] = np.log(pairs["area_t_m2"])
    pairs["duration_deviation_days"] = pairs["duration_days"] - 365.2425
    return pairs.sort_values(["sand_cay_id", "time_t", "time_t1"]).reset_index(drop=True)


def zscore(frame: pd.DataFrame, column: str) -> str:
    name = f"z_{column}"
    scale = frame[column].std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Cannot standardize {column}")
    frame[name] = (frame[column] - frame[column].mean()) / scale
    return name


def fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["reef_id"], "use_correction": True}
    )


def model_results(observations: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pair_model = pairs.copy()
    for column in ["log_elongation_t", "log_area_t", "log_centroid_shift_normalized", "delta_log_elongation"]:
        zscore(pair_model, column)

    # Time ordering is preserved here: shape at t precedes the t--t+1 displacement.
    shape_predicts_mobility = fit_clustered(
        "z_log_centroid_shift_normalized ~ z_log_elongation_t + z_log_area_t + "
        "duration_deviation_days + season_gap_days + C(sand_cay_id) + C(baseline_year)", pair_model
    )
    # This shares both masks with the outcome and is therefore exploratory, not causal.
    concurrent_reshaping = fit_clustered(
        "z_delta_log_elongation ~ z_log_centroid_shift_normalized + z_log_elongation_t + z_log_area_t + "
        "duration_deviation_days + season_gap_days + C(sand_cay_id) + C(baseline_year)", pair_model
    )

    trend = observations.copy()
    trend["elapsed_years"] = (trend["date"] - trend["date"].min()).dt.days / 365.2425
    trend["season_sin"] = np.sin(2 * np.pi * trend["date"].dt.dayofyear / 365.2425)
    trend["season_cos"] = np.cos(2 * np.pi * trend["date"].dt.dayofyear / 365.2425)
    trend["log_elongation"] = np.log(trend["sand_cay_elongation"])
    overall_trend = fit_clustered(
        "log_elongation ~ elapsed_years + season_sin + season_cos + C(sand_cay_id)", trend
    )

    specs = [
        ("observed_shape_time_trend", overall_trend, "elapsed_years", trend, "confirmatory", "Per observed year"),
        ("baseline_shape_to_following_mobility", shape_predicts_mobility, "z_log_elongation_t", pair_model, "confirmatory", "SD mobility per SD baseline log elongation"),
        ("concurrent_mobility_to_shape_change", concurrent_reshaping, "z_log_centroid_shift_normalized", pair_model, "exploratory", "SD log-elongation change per SD mobility"),
    ]
    rows = []
    for name, fit, term, data, status, unit in specs:
        low, high = fit.conf_int().loc[term].astype(float)
        rows.append({
            "analysis": name, "status": status, "term": term, "estimate": float(fit.params[term]),
            "reef_cluster_se": float(fit.bse[term]), "ci95_low": low, "ci95_high": high,
            "p_value": float(fit.pvalues[term]), "effect_unit": unit,
            "n_observations": int(len(data)), "n_cays": int(data["sand_cay_id"].nunique()),
            "n_reefs": int(data["reef_id"].nunique()), "r_squared": float(fit.rsquared),
            "model": "OLS with cay fixed effects; reef-clustered SE" if name == "observed_shape_time_trend" else "OLS with cay and baseline-year fixed effects; reef-clustered SE",
        })
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def long_span_endpoints(observations: pd.DataFrame) -> pd.DataFrame:
    """Descriptive early/late comparison only when the season and sensor agree."""
    records = []
    for (reef_id, cay_id, frame_id), group in observations.groupby(["reef_id", "sand_cay_id", "reference_frame_id"], dropna=False):
        group = group.sort_values(["date", "image_id"]).drop_duplicates("date", keep="first").reset_index(drop=True)
        candidates = []
        for left in range(len(group) - 1):
            for right in range(left + 1, len(group)):
                duration = (group.loc[right, "date"] - group.loc[left, "date"]).days
                if duration < MIN_TRAJECTORY_SPAN_YEARS * 365.2425:
                    continue
                gap = circular_doy_gap(group.loc[left, "date"], group.loc[right, "date"])
                if gap <= SEASON_GAP_DAYS:
                    candidates.append((duration, -gap, left, right, gap))
        if not candidates:
            continue
        _, _, left, right, gap = max(candidates)
        start, end = group.loc[left], group.loc[right]
        records.append({
            "reef_id": reef_id, "sand_cay_id": cay_id, "reference_frame_id": frame_id,
            "early_date": start["date"], "late_date": end["date"], "span_years": (end["date"] - start["date"]).days / 365.2425,
            "season_gap_days": gap, "early_elongation": start["sand_cay_elongation"], "late_elongation": end["sand_cay_elongation"],
            "delta_log_elongation": np.log(end["sand_cay_elongation"]) - np.log(start["sand_cay_elongation"]),
        })
    return pd.DataFrame(records)


def make_figure(pairs: pd.DataFrame, models: pd.DataFrame, endpoints: pd.DataFrame, target: Path) -> None:
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

    ax = axes[0]
    x = pairs["log_elongation_t"]
    y = pairs["centroid_shift_normalized"]
    ax.scatter(x, y, s=7, color="#4C78A8", alpha=0.32, linewidths=0)
    slope, intercept = np.polyfit(x, np.log(y + MOBILITY_OFFSET), 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, np.exp(intercept + slope * grid) - MOBILITY_OFFSET, color="#1F3B5D", lw=1.1)
    rho, pvalue = spearmanr(x, y)
    ax.text(0.03, 0.97, f"Descriptive Spearman $\\rho$ = {rho:.2f}\n$p$ = {pvalue:.3f}", transform=ax.transAxes, va="top", fontsize=6.0)
    ax.set_yscale("log")
    ax.set_xlabel("Baseline log elongation")
    ax.set_ylabel("Next-year centroid shift / equivalent radius")
    ax.set_title("a  Shape and subsequent mobility", loc="left", fontsize=8)

    ax = axes[1]
    labels = ["Shape trend", "Shape → mobility", "Mobility → shape*"]
    display = models.set_index("analysis").loc[[
        "observed_shape_time_trend", "baseline_shape_to_following_mobility", "concurrent_mobility_to_shape_change"
    ]].reset_index()
    positions = np.arange(len(display))[::-1]
    x_low = float(display["ci95_low"].min())
    x_high = float(display["ci95_high"].max())
    x_pad = max(0.035, (x_high - x_low) * 0.08)
    for position, row in zip(positions, display.itertuples(index=False)):
        ax.plot([row.ci95_low, row.ci95_high], [position, position], color="#333333", lw=1.1)
        ax.plot(row.estimate, position, "o", color="#D55E00", ms=4.2)
    ax.axvline(0, color="#888888", lw=0.7, ls="--")
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlim(x_low - x_pad, x_high + x_pad * 2.2)
    ax.set_xlabel("Estimate (95% CI)")
    ax.set_title("b  Fixed-effect estimates", loc="left", fontsize=8)
    ax.text(0.02, -0.27, "*Exploratory: both metrics derive from the same mask pair.", transform=ax.transAxes, fontsize=5.6)

    ax = axes[2]
    if endpoints.empty:
        ax.text(0.5, 0.5, "No ≥5-year\nsame-season endpoints", ha="center", va="center")
    else:
        for row in endpoints.itertuples(index=False):
            color = "#0072B2" if row.delta_log_elongation < 0 else "#D55E00"
            ax.plot([0, 1], [np.log(row.early_elongation), np.log(row.late_elongation)], color=color, alpha=0.35, lw=0.75)
        ax.scatter(np.zeros(len(endpoints)), np.log(endpoints["early_elongation"]), s=6, color="#555555", zorder=3)
        ax.scatter(np.ones(len(endpoints)), np.log(endpoints["late_elongation"]), s=6, color="#555555", zorder=3)
        ax.set_xticks([0, 1], ["Earliest", "Latest"])
        ax.text(0.03, 0.97, f"n = {len(endpoints)} cays\nblue: less elongated", transform=ax.transAxes, va="top", fontsize=6.2)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylabel("Log elongation")
    ax.set_title("c  Same-season multi-year endpoints", loc="left", fontsize=8)

    for suffix, dpi in [(".png", 400), (".tiff", 400), (".pdf", None), (".svg", None)]:
        path = target.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if dpi:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)


def main() -> None:
    output_dir = (PROJECT_ROOT / "outputs/气候检验")
    figure_dir = (PROJECT_ROOT / "outputs/figures/中间图")
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    observations = load_observations()
    pairs = same_season_pairs(observations)
    endpoints = long_span_endpoints(observations)
    models = model_results(observations, pairs)
    quality = pd.DataFrame([{
        "source": "outputs/沙洲观测主表.csv",
        "sensor": "sentinel2", "quality_filter": "A/B", "n_observations": len(observations),
        "n_cays": observations["sand_cay_id"].nunique(), "n_reefs": observations["reef_id"].nunique(),
        "n_reference_frames": observations[["sand_cay_id", "reference_frame_id"]].drop_duplicates().shape[0],
        "pixel_size_constant_within_frame": bool(observations.groupby(["sand_cay_id", "reference_frame_id"])["pixel_size_m"].agg(lambda values: values.nunique()).le(1).all()),
        "same_season_pair_rule": "350-380 d and circular day-of-year gap <=15 d; deterministic nearest annual successor",
        "reef_platform_shape_test": "not run: available reef extent is a geographic bbox, not a validated reef-platform polygon",
    }])
    pairs.to_csv(output_dir / "形状质心迁移同季节样本.csv", index=False, encoding="utf-8-sig")
    endpoints.to_csv(output_dir / "形状质心迁移多年同季节端点.csv", index=False, encoding="utf-8-sig")
    models.to_csv(output_dir / "形状质心迁移模型.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(output_dir / "形状质心迁移质量核验.csv", index=False, encoding="utf-8-sig")
    audit = {
        "estimand": "Observed S2 shape evolution and its association with centroid displacement; no claim about cay genesis or causality.",
        "shape_metric": "log(major_axis_length / minor_axis_length)",
        "mobility_metric": "centroid displacement / baseline equivalent radius in the same fixed frame",
        "seasonal_control": quality.loc[0, "same_season_pair_rule"],
        "reef_platform_boundary": quality.loc[0, "reef_platform_shape_test"],
        "exploratory_boundary": "Concurrent mobility--shape-change association is not interpreted as causal because both derive from the same endpoints.",
    }
    (output_dir / "形状质心迁移分析核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(pairs, models, endpoints, figure_dir / "figure_r14_shape_centroid_mobility")
    print(models.to_string(index=False))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
