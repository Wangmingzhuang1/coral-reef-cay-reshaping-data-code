"""全球图 4：普适性与可迁移性检验。

a 尺寸标度律：标准化总重塑率随沙洲面积衰减，叠加 ±1 像素可检测地板；
b 植被梯度：表面状态对标准化移动性与重塑强度的单调控制；
c 跨源一致性：双源沙洲上标准化中位数的来源间对齐。

三个面板共同回答“全球意义”：区域谱系是否服从可迁移的尺寸、过程与测量规律。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from 分析全球形态重塑谱系 import (  # noqa: E402
    REGION_COLORS,
    REGION_ORDER,
    bootstrap_cay_median,
    configure_style,
)

STATE_ORDER = ["none", "sparse", "partial", "dominant"]
STATE_LABELS = {"none": "Bare", "sparse": "Sparse", "partial": "Partial", "dominant": "Dominant"}
PIXEL_FLOOR_M = 10.0


def load_intervals() -> pd.DataFrame:
    from 构建沙洲观测与变化表 import select_analysis_observations
    intervals = pd.read_csv(OUTPUTS / "全球形态重塑区间汇总.csv", low_memory=False)
    observations = select_analysis_observations(pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False))
    intervals["time_t"] = pd.to_datetime(intervals["time_t"], errors="coerce")
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    # The interval table already carries a starting-perimeter column; drop it so the
    # geometry merge does not create suffixed duplicates.
    intervals = intervals.drop(columns=[c for c in ["sand_cay_perimeter_m", "pixel_size_m"] if c in intervals.columns])
    keys = ["sand_cay_id", "reference_frame_id", "date", "sensor"]
    geometry = (
        observations.loc[observations["sensor"].isin(["google_earth", "sentinel2"]), keys + ["sand_cay_perimeter_m", "pixel_size_m"]]
        .groupby(keys, as_index=False)
        .agg(sand_cay_perimeter_m=("sand_cay_perimeter_m", "median"), pixel_size_m=("pixel_size_m", "median"))
    )
    intervals = intervals.merge(
        geometry,
        left_on=["sand_cay_id", "reference_frame_id", "time_t", "sensor"],
        right_on=keys,
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    intervals["years"] = pd.to_numeric(intervals["time_interval_days"], errors="coerce") / 365.2425
    intervals["detection_floor_fraction"] = (
        intervals["sand_cay_perimeter_m"] * intervals["pixel_size_m"] / intervals["area_t_m2"] / intervals["years"]
    )
    return intervals


def cay_table(intervals: pd.DataFrame) -> pd.DataFrame:
    cays = (
        intervals.groupby(["analysis_region", "reef_id", "sand_cay_id"], observed=True)
        .agg(
            area_m2=("area_t_m2", "median"),
            gross=("standardized_gross_boundary_reworking", "median"),
            translation=("standardized_centroid_translation", "median"),
            floor=("detection_floor_fraction", "median"),
            sources=("sensor", "nunique"),
        )
        .reset_index()
    )
    return cays.dropna(subset=["area_m2", "gross", "translation"])


def fit_scaling(cays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = cays.loc[cays["gross"].gt(0) & cays["area_m2"].gt(0)].copy()
    work["log_gross"] = np.log(work["gross"])
    work["log_area"] = np.log(work["area_m2"])
    fitted = sm.OLS(work["log_gross"], sm.add_constant(work["log_area"])).fit(
        cov_type="cluster", cov_kwds={"groups": work["reef_id"]}
    )
    rows.append({
        "stratum": "ALL", "n_cays": int(len(work)), "slope": float(fitted.params["log_area"]),
        "cluster_se": float(fitted.bse["log_area"]), "p_value": float(fitted.pvalues["log_area"]),
        "ci95_low": float(fitted.conf_int().loc["log_area", 0]), "ci95_high": float(fitted.conf_int().loc["log_area", 1]),
    })
    for region, part in work.groupby("analysis_region", observed=True):
        if len(part) < 10:
            continue
        regional = sm.OLS(part["log_gross"], sm.add_constant(part["log_area"])).fit(cov_type="HC3")
        rows.append({
            "stratum": region, "n_cays": int(len(part)), "slope": float(regional.params["log_area"]),
            "cluster_se": float(regional.bse["log_area"]), "p_value": float(regional.pvalues["log_area"]),
            "ci95_low": float(regional.conf_int().loc["log_area", 0]), "ci95_high": float(regional.conf_int().loc["log_area", 1]),
        })
    translation = sm.OLS(
        np.log(work["translation"].clip(lower=1e-6)), sm.add_constant(work["log_area"])
    ).fit(cov_type="cluster", cov_kwds={"groups": work["reef_id"]})
    rows.append({
        "stratum": "ALL_translation", "n_cays": int(len(work)), "slope": float(translation.params["log_area"]),
        "cluster_se": float(translation.bse["log_area"]), "p_value": float(translation.pvalues["log_area"]),
        "ci95_low": float(translation.conf_int().loc["log_area", 0]), "ci95_high": float(translation.conf_int().loc["log_area", 1]),
    })
    return pd.DataFrame(rows)


def fit_geometry_decomposition(intervals: pd.DataFrame) -> pd.DataFrame:
    """Separate area-normalized reworking from perimeter/area geometry.

    Characteristic boundary displacement is standardized gross reworking times
    baseline area divided by baseline perimeter (metres per 365.2425 days).
    Fit one observation per cay, using medians of interval-level quantities.
    """
    work = intervals.loc[
        intervals["area_t_m2"].gt(0) & intervals["sand_cay_perimeter_m"].gt(0)
    ].copy()
    work["perimeter_area_ratio"] = work["sand_cay_perimeter_m"] / work["area_t_m2"]
    work["boundary_displacement_m_per_year"] = (
        work["standardized_gross_boundary_reworking"]
        * work["area_t_m2"] / work["sand_cay_perimeter_m"]
    )
    cays = work.groupby(["analysis_region", "reef_id", "sand_cay_id"], observed=True).agg(
        area_m2=("area_t_m2", "median"),
        relative_reworking=("standardized_gross_boundary_reworking", "median"),
        perimeter_area_ratio=("perimeter_area_ratio", "median"),
        boundary_displacement_m_per_year=("boundary_displacement_m_per_year", "median"),
    ).reset_index()
    rows = []
    for metric in ["relative_reworking", "perimeter_area_ratio", "boundary_displacement_m_per_year"]:
        valid = cays.loc[cays[metric].gt(0) & cays["area_m2"].gt(0)].copy()
        for region, part in [("ALL", valid), *valid.groupby("analysis_region", observed=True)]:
            if region != "ALL" and len(part) < 10:
                continue
            model = sm.OLS(np.log(part[metric]), sm.add_constant(np.log(part["area_m2"]))).fit(
                cov_type="cluster" if region == "ALL" else "HC3",
                cov_kwds={"groups": part["reef_id"]} if region == "ALL" else None,
            )
            slope = model.params.iloc[1]
            ci = model.conf_int().iloc[1]
            rows.append({
                "metric": metric, "stratum": region, "n_cays": len(part),
                "slope": slope, "ci95_low": ci.iloc[0], "ci95_high": ci.iloc[1],
                "p_value": model.pvalues.iloc[1],
                "se_type": "reef-clustered" if region == "ALL" else "HC3",
            })
    return pd.DataFrame(rows)


def vegetation_gradient(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        part = intervals.loc[intervals["manual_vegetation_state_t"].eq(state)]
        if part.empty:
            continue
        cays = part.groupby("sand_cay_id", observed=True)[["standardized_centroid_translation", "standardized_gross_boundary_reworking"]].median()
        for metric in ["standardized_centroid_translation", "standardized_gross_boundary_reworking"]:
            low, high = bootstrap_cay_median(cays[metric])
            rows.append({
                "state": state, "metric": metric, "n_cays": int(len(cays)),
                "median": float(cays[metric].median()), "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def cross_source_agreement(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = intervals.groupby("sand_cay_id", observed=True)["sensor"].nunique()
    shared = support[support.eq(2)].index
    part = intervals.loc[intervals["sand_cay_id"].isin(shared)]
    medians = (
        part.groupby(["sand_cay_id", "sensor"], observed=True)[["standardized_gross_boundary_reworking", "standardized_centroid_translation"]]
        .median()
        .unstack("sensor")
    )
    rows = []
    for metric in ["standardized_gross_boundary_reworking", "standardized_centroid_translation"]:
        block = medians[metric].dropna()
        rows.append({
            "metric": metric,
            "n_shared_cays": int(len(block)),
            "spearman_ge_vs_s2": float(block["google_earth"].corr(block["sentinel2"], method="spearman")),
            "median_ge_over_s2": float((block["google_earth"] / block["sentinel2"]).median()),
        })
    stats = pd.DataFrame(rows)
    pivot = medians.reset_index()
    return stats, pivot


def draw_scaling_panel(ax_a: plt.Axes, cays: pd.DataFrame, scaling: pd.DataFrame, panel: str = "a") -> None:
    """Panel: size scaling of relative reworking above the detection floor."""
    overall = scaling.loc[scaling["stratum"].eq("ALL")].iloc[0]
    for region in REGION_ORDER:
        rows = scaling.loc[scaling["stratum"].eq(region)]
        part = cays.loc[cays["analysis_region"].eq(region)]
        if rows.empty or part.empty:
            continue
        slope = float(rows.iloc[0]["slope"])
        xa = part["area_m2"]
        ya = part["gross"].clip(lower=1e-6)
        mx = float(np.log(xa).mean())
        my = float(np.log(ya).mean())
        grid_r = np.logspace(np.log10(xa.min() * 0.9), np.log10(xa.max() * 1.1), 30)
        ax_a.plot(grid_r, np.exp(my) * (grid_r / np.exp(mx)) ** slope, color=REGION_COLORS[region], lw=0.7, alpha=0.8, zorder=2)
    for region in REGION_ORDER:
        part = cays.loc[cays["analysis_region"].eq(region)]
        if part.empty:
            continue
        ax_a.scatter(part["area_m2"], part["gross"], s=16, color=REGION_COLORS[region], alpha=0.72, edgecolors="white", linewidths=0.3, label=region)
    grid = np.logspace(np.log10(cays["area_m2"].min() * 0.8), np.log10(cays["area_m2"].max() * 1.2), 60)
    mean_log_area = float(np.log(cays["area_m2"]).mean())
    mean_log_gross = float(np.log(cays["gross"].clip(lower=1e-6)).mean())
    ax_a.plot(
        grid,
        np.exp(mean_log_gross) * (grid / np.exp(mean_log_area)) ** overall["slope"],
        color="#333333", lw=0.9, zorder=3,
    )
    floor = 2 * np.sqrt(np.pi * grid) * PIXEL_FLOOR_M / grid
    ax_a.plot(grid, floor, color="#888888", lw=0.8, ls="--", zorder=2)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Median cay area (m$^2$)")
    ax_a.set_ylabel("Gross boundary reworking (per 365 d)")
    ax_a.set_title(("" if not panel else f"{panel}  ") + "Relative reworking declines with cay size", loc="left", fontweight="bold")
    ax_a.text(0.03, 0.03, f"slope = {overall['slope']:.2f} (95% CI {overall['ci95_low']:.2f} to {overall['ci95_high']:.2f})\ndashed: ±1 px floor, 10 m, 365 d", transform=ax_a.transAxes, va="bottom", fontsize=5.5)
    ax_a.legend(loc="upper right", fontsize=5.4, handletextpad=0.3, labelspacing=0.18, title="Region", title_fontsize=5.6)

def draw_vegetation_panel(ax_b: plt.Axes, vegetation: pd.DataFrame, panel: str = "b") -> None:
    """Panel: vegetation-state mobility gradient; by source when a sensor column exists."""
    positions = np.arange(len(STATE_ORDER))
    metrics = [
        ("standardized_centroid_translation", "o", "#3C78A8", "Centroid translation"),
        ("standardized_gross_boundary_reworking", "D", "#D27745", "Gross reworking"),
    ]
    by_source = "sensor" in vegetation.columns
    sources = ["sentinel2", "google_earth"] if by_source else [None]
    offsets = {"sentinel2": -0.07, "google_earth": 0.07, None: 0.0}
    styles = {"sentinel2": "-", "google_earth": "--", None: "-"}
    short = {"sentinel2": "S2", "google_earth": "GE", None: "all"}
    for metric, marker, color, label in metrics:
        for source in sources:
            part = vegetation if source is None else vegetation.loc[vegetation["sensor"].eq(source)]
            series = part.loc[part["metric"].eq(metric)].set_index("state").reindex(STATE_ORDER)
            xx = positions + offsets[source]
            ax_b.errorbar(
                xx, series["median"], yerr=[series["median"] - series["ci95_low"], series["ci95_high"] - series["median"]],
                fmt=marker, ms=4.0, color=color, mec=color, mfc=color, capsize=1.6, lw=0.7,
                label=f"{label} ({short[source]})", zorder=3,
            )
            ax_b.plot(xx, series["median"], color=color, lw=0.7, alpha=0.7, linestyle=styles[source], zorder=2)
    ax_b.set_yscale("log")
    if by_source:
        counts = vegetation.loc[vegetation["metric"].eq("standardized_centroid_translation")].groupby("state")["n_cays"].sum()
        labels = [f"{STATE_LABELS[s]}\n(n={int(counts.loc[s])} seq)" for s in STATE_ORDER]
    else:
        labels = [f"{STATE_LABELS[s]}\n(n={int(vegetation.loc[vegetation.state.eq(s) & vegetation.metric.eq('standardized_centroid_translation'), 'n_cays'].iloc[0])})" for s in STATE_ORDER]
    ax_b.set_xticks(positions, labels)
    ax_b.set_ylabel("Standardized rate (per 365 d)")
    ax_b.set_title(("" if not panel else f"{panel}  ") + "Mobility declines with vegetation cover", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker=marker, color=color, mec=color, mfc=color, linestyle=styles[source], lw=0.8, ms=4.0, label=f"{label} ({short[source]})")
        for metric, marker, color, label in metrics
        for source in sources
    ]
    ax_b.legend(handles=handles, loc="lower left", fontsize=5.0, frameon=False, handletextpad=0.3, labelspacing=0.2, ncol=2)
    ax_b.grid(axis="y", color="#dddddd", lw=0.45)

def draw_agreement_panel(ax_c: plt.Axes, agreement: pd.DataFrame, pivot: pd.DataFrame, panel: str = "c") -> None:
    """Panel: cross-source agreement of standardized cay medians."""
    for metric, marker, color, label in [
        ("standardized_gross_boundary_reworking", "o", "#444444", "Gross reworking"),
        ("standardized_centroid_translation", "^", "#8064A2", "Centroid translation"),
    ]:
        part = pivot[["sand_cay_id", metric]].dropna()
        ax_c.scatter(part[(metric, "sentinel2")], part[(metric, "google_earth")], s=15, marker=marker, color=color, alpha=0.7, edgecolors="white", linewidths=0.3, label=label)
    limits = (0.008, 6.0)
    ax_c.plot(limits, limits, color="#888888", lw=0.8, ls="--")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(*limits)
    ax_c.set_ylim(*limits)
    ax_c.set_xlabel("Sentinel-2 cay median (per 365 d)")
    ax_c.set_ylabel("Google Earth cay median (per 365 d)")
    ax_c.set_title(("" if not panel else f"{panel}  ") + "Cross-source agreement", loc="left", fontweight="bold")
    notes = "\n".join(f"{row.metric.split('_')[1] if False else ('gross' if 'gross' in row.metric else 'translation')}: rho = {row.spearman_ge_vs_s2:.2f}, GE/S2 = {row.median_ge_over_s2:.2f}" for row in agreement.itertuples())
    ax_c.text(0.03, 0.97, f"n = {int(agreement['n_shared_cays'].iloc[0])} shared cays\n{notes}", transform=ax_c.transAxes, va="top", fontsize=5.5)
    ax_c.legend(loc="lower right", fontsize=5.5, frameon=False, handletextpad=0.3)

def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    intervals = load_intervals()
    cays = cay_table(intervals)
    scaling = fit_scaling(cays)
    geometry = fit_geometry_decomposition(intervals)
    vegetation = vegetation_gradient(intervals)
    agreement, pivot = cross_source_agreement(intervals)
    scaling.to_csv(OUTPUTS / "全球普适性标度拟合.csv", index=False, encoding="utf-8-sig")
    geometry.to_csv(OUTPUTS / "全球尺寸几何分解.csv", index=False, encoding="utf-8-sig")
    vegetation.to_csv(OUTPUTS / "全球植被梯度汇总.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(OUTPUTS / "全球跨源一致性.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(OUTPUTS / "全球跨源沙洲中位数pivot.csv", index=False, encoding="utf-8-sig")
    cays.to_csv(OUTPUTS / "全球普适性沙洲汇总.csv", index=False, encoding="utf-8-sig")
    audit = {
        "figure_claim": "Regional morphodynamic spectra obey transferable size, process, and measurement rules: relative reworking reflects perimeter-area geometry; vegetation cover corresponds to lower mobility overall, and calibrated cay medians show in-sample cross-source agreement.",
        "scaling": scaling.to_dict("records"),
        "geometry_decomposition": geometry.to_dict("records"),
        "vegetation_gradient": vegetation.to_dict("records"),
        "cross_source_agreement": agreement.to_dict("records"),
        "detection_floor": "±1 pixel boundary band using the observation pixel size, annualized over each interval; the plotted reference curve uses 10 m and 365.2425 d.",
        "statistics": "Scaling slopes use reef-clustered SE overall and HC3 within regions; vegetation intervals are 1,999-replicate cay bootstrap 95% CI; cross-source agreement uses Spearman correlation and median GE/S2 ratio on cays observed by both sources.",
    }
    (OUTPUTS / "全球普适性核查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
