"""联合 Google Earth 与 Sentinel-2 的全球区域沙洲重塑描述。

图 2：在全部可用 A/B 级区间中，将来源与观测间隔显式纳入模型，检验
净面积变化与边界重塑、侵蚀--堆积交换的区域组合。
图 3：在相同联合模型下，描述质心平移、周长调整、长短轴调整和长轴转向。

这不是把两个来源的原始像元变化直接相加：每项指标均以沙洲固定效应识别
来源偏移，并标准化至 Sentinel-2 参考尺度与 365.2425 天观测间隔。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"
RANDOM_SEED = 20260920
REFERENCE_DAYS = 365.2425
MIN_INTERVAL_DAYS = 30

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "额外分析_区域与气候响应"))
from 分析范围 import analysis_contract_digest, exclude_reefs  # noqa: E402
from 区域气候滞后模型 import attach_geography, build_geographic_mapping  # noqa: E402


# These are the five regional classes used in the manuscript Results (section 3).
REGION_ORDER = [
    "Great Barrier Reef",
    "Maldives",
    "South China Sea",
    "Marshall Islands",
    "Lakshadweep",
]
REGION_COLORS = {
    "Great Barrier Reef": "#3C78A8",
    "Maldives": "#D27745",
    "South China Sea": "#8064A2",
    "Marshall Islands": "#63966B",
    "Lakshadweep": "#A78043",
}
SUPPORT_MARKERS = {"Both sources": "o", "GE only": "^", "S2 only": "s"}

METRICS = {
    "net_area_change": {"raw": "net_area_change_fraction", "label": "Net area change", "positive": False},
    "erosion": {"raw": "erosion_fraction", "label": "Erosion", "positive": True},
    "deposition": {"raw": "deposition_fraction", "label": "Deposition", "positive": True},
    "gross_boundary_reworking": {"raw": "gross_boundary_reworking_fraction", "label": "Gross boundary reworking", "positive": True},
    "centroid_translation": {"raw": "centroid_translation_fraction", "label": "Centroid translation", "positive": True},
    "perimeter_adjustment": {"raw": "perimeter_adjustment_fraction", "label": "Perimeter adjustment", "positive": True},
    "axial_adjustment": {"raw": "axial_adjustment_fraction", "label": "Axial adjustment", "positive": True},
    "axis_reorientation": {"raw": "axis_reorientation_fraction", "label": "Axis reorientation", "positive": True},
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.2, "axes.labelsize": 7.1, "axes.titlesize": 8.0,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "legend.fontsize": 5.9,
        "axes.linewidth": 0.65, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def acute_orientation_change(angle_deg: pd.Series) -> pd.Series:
    """Represent axial orientation changes on their 0--90 degree equivalence."""
    angle = np.abs(pd.to_numeric(angle_deg, errors="coerce")) % 180.0
    return np.minimum(angle, 180.0 - angle)


def bootstrap_cay_median(values: pd.Series, reps: int = 1999) -> tuple[float, float]:
    """Resample independent sand cays, never time intervals."""
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    medians = np.median(rng.choice(array, size=(reps, len(array)), replace=True), axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def source_support(frame: pd.DataFrame) -> str:
    sources = set(frame["sensor"].dropna())
    labels = {
        frozenset({"google_earth"}): "GE only",
        frozenset({"sentinel2"}): "S2 only",
        frozenset({"google_earth", "sentinel2"}): "Both sources",
    }
    return labels[frozenset(sources)]


def load_analysis_frame() -> pd.DataFrame:
    from 构建沙洲观测与变化表 import select_analysis_observations
    intervals = exclude_reefs(pd.read_csv(ROOT / "outputs" / "沙洲变化区间.csv", low_memory=False))
    observations = select_analysis_observations(pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False))
    intervals["time_t"] = pd.to_datetime(intervals["time_t"], errors="coerce")
    intervals = intervals.loc[
        intervals["sensor"].isin(["google_earth", "sentinel2"])
        & intervals["quality_grade_t"].isin(["A", "B"])
        & intervals["quality_grade_t1"].isin(["A", "B"])
        & pd.to_numeric(intervals["area_t_m2"], errors="coerce").gt(0)
        & pd.to_numeric(intervals["time_interval_days"], errors="coerce").ge(MIN_INTERVAL_DAYS)
        & intervals["time_t"].notna()
    ].copy()
    intervals = attach_geography(intervals, build_geographic_mapping(intervals["reef_id"], observations))
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    # Four same-date duplicate observation keys exist; their perimeter is aggregated,
    # not selected arbitrarily, before it enters the interval normalisation.
    perimeters = (
        observations.loc[observations["sensor"].isin(["google_earth", "sentinel2"]), ["sand_cay_id", "reference_frame_id", "date", "sensor", "sand_cay_perimeter_m"]]
        .groupby(["sand_cay_id", "reference_frame_id", "date", "sensor"], as_index=False)
        .agg(sand_cay_perimeter_m=("sand_cay_perimeter_m", "median"))
    )
    intervals = intervals.merge(
        perimeters, left_on=["sand_cay_id", "reference_frame_id", "time_t", "sensor"],
        right_on=["sand_cay_id", "reference_frame_id", "date", "sensor"], how="left", validate="many_to_one",
    ).drop(columns="date")
    numeric = [
        "time_interval_days", "area_t_m2", "area_change_m2", "erosion_area_m2", "deposition_area_m2",
        "gross_boundary_change_m2", "centroid_shift_m", "perimeter_change_m", "sand_cay_perimeter_m",
        "major_axis_change_m", "minor_axis_change_m", "major_axis_angle_change_deg",
    ]
    for column in numeric:
        intervals[column] = pd.to_numeric(intervals[column], errors="coerce")
    intervals["equivalent_radius_m"] = np.sqrt(intervals["area_t_m2"] / np.pi)
    intervals["log_interval_days"] = np.log(intervals["time_interval_days"])
    intervals["net_area_change_fraction"] = intervals["area_change_m2"] / intervals["area_t_m2"]
    intervals["erosion_fraction"] = intervals["erosion_area_m2"] / intervals["area_t_m2"]
    intervals["deposition_fraction"] = intervals["deposition_area_m2"] / intervals["area_t_m2"]
    intervals["gross_boundary_reworking_fraction"] = intervals["gross_boundary_change_m2"] / intervals["area_t_m2"]
    intervals["centroid_translation_fraction"] = intervals["centroid_shift_m"] / intervals["equivalent_radius_m"]
    intervals["perimeter_adjustment_fraction"] = intervals["perimeter_change_m"].abs() / intervals["sand_cay_perimeter_m"]
    intervals["axial_adjustment_fraction"] = np.hypot(intervals["major_axis_change_m"], intervals["minor_axis_change_m"]) / intervals["equivalent_radius_m"]
    intervals["axis_reorientation_fraction"] = acute_orientation_change(intervals["major_axis_angle_change_deg"]) / 90.0
    return intervals


def standardize_metric(frame: pd.DataFrame, name: str, spec: dict) -> tuple[pd.Series, dict]:
    """Identify source effects within sand cays and predict a common 365-day reference."""
    raw = spec["raw"]
    valid = frame[[raw, "sensor", "log_interval_days", "sand_cay_id"]].notna().all(axis=1)
    valid &= np.isfinite(frame[raw]) & np.isfinite(frame["log_interval_days"])
    if spec["positive"]:
        valid &= frame[raw].ge(0)
    work = frame.loc[valid].copy()
    if work.empty:
        raise ValueError(f"No valid data for {name}")
    formula = "response ~ C(sensor, Treatment(reference='sentinel2')) + bs(log_interval_days, df=3, degree=3) + C(sand_cay_id)"
    if spec["positive"]:
        positive = work.loc[work[raw].gt(0), raw]
        epsilon = float(positive.min() * 1e-4) if not positive.empty else 1e-8
        work["response"] = work[raw].clip(lower=epsilon)
        fitted = smf.glm(formula, data=work, family=sm.families.Gamma(link=sm.families.links.Log())).fit(
            cov_type="cluster", cov_kwds={"groups": work["sand_cay_id"]}
        )
        counter = work.copy()
        counter["sensor"] = "sentinel2"
        counter["log_interval_days"] = np.log(REFERENCE_DAYS)
        standardized = work[raw] * fitted.predict(counter) / fitted.predict(work)
        family = "Gamma log-link"
        fit_statistic = float(fitted.deviance)
    else:
        work["response"] = np.arcsinh(work[raw])
        fitted = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["sand_cay_id"]})
        counter = work.copy()
        counter["sensor"] = "sentinel2"
        counter["log_interval_days"] = np.log(REFERENCE_DAYS)
        standardized = np.sinh(work["response"] - (fitted.predict(work) - fitted.predict(counter)))
        family = "OLS on asinh-transformed signed response"
        fit_statistic = float(fitted.ssr)
    source_term = next(term for term in fitted.params.index if "google_earth" in term)
    standardized = pd.Series(standardized.to_numpy(dtype=float), index=work.index, name=f"standardized_{name}")
    diagnostics = {
        "metric": name, "raw_metric": raw, "model_family": family,
        "n_intervals": int(len(work)), "n_cays": int(work["sand_cay_id"].nunique()),
        "n_shared_source_cays": int(work.groupby("sand_cay_id", observed=True)["sensor"].nunique().eq(2).sum()),
        "google_earth_coefficient": float(fitted.params[source_term]),
        "google_earth_cluster_se": float(fitted.bse[source_term]),
        "google_earth_p_value": float(fitted.pvalues[source_term]),
        "fit_deviance_or_residual_ss": fit_statistic,
    }
    return standardized, diagnostics


def standardize_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    diagnostics = []
    for name, spec in METRICS.items():
        standardized, record = standardize_metric(result, name, spec)
        result[standardized.name] = standardized
        diagnostics.append(record)
    models = pd.DataFrame(diagnostics)
    models["google_earth_fdr_q_value"] = multipletests(models["google_earth_p_value"], method="fdr_bh")[1]
    return result, models


def summarize_cays(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"standardized_{name}" for name in METRICS]
    cays = frame.groupby(["analysis_region", "reef_id", "sand_cay_id"], observed=True)[metrics + ["area_t_m2"]].median().reset_index()
    support = frame.groupby("sand_cay_id", observed=True).apply(source_support, include_groups=False).rename("source_support").reset_index()
    counts = frame.groupby(["sand_cay_id", "sensor"], observed=True).size().unstack(fill_value=0).rename(columns={"google_earth": "n_google_earth", "sentinel2": "n_sentinel2"}).reset_index()
    return cays.merge(support, on="sand_cay_id", how="left", validate="one_to_one").merge(counts, on="sand_cay_id", how="left", validate="one_to_one")


def summarize_regions(cays: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"standardized_{name}" for name in METRICS]
    rows = []
    for region in REGION_ORDER:
        part = cays.loc[cays["analysis_region"].eq(region)]
        if part.empty:
            continue
        row = {
            "analysis_region": region, "n_cays": int(part["sand_cay_id"].nunique()), "n_reefs": int(part["reef_id"].nunique()),
            "n_both_sources": int(part["source_support"].eq("Both sources").sum()),
            "n_ge_only": int(part["source_support"].eq("GE only").sum()), "n_s2_only": int(part["source_support"].eq("S2 only").sum()),
        }
        for metric in metrics:
            row[metric] = float(part[metric].median())
            low, high = bootstrap_cay_median(part[metric])
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(title if not label else f"{label}  {title}", loc="left", fontweight="bold", pad=5)


def value_limits(values: pd.Series, lower_floor: float = 1e-3) -> tuple[float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.loc[clean.gt(0)]
    if clean.empty:
        return lower_floor, 1.0
    low = max(lower_floor, float(clean.quantile(0.01)) * 0.7)
    high = float(clean.quantile(0.99)) * 1.35
    return low, max(high, low * 10)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def draw_signatures_net_panel(ax_a: plt.Axes, cays: pd.DataFrame, regions: pd.DataFrame, panel: str = "a") -> None:
    """Panel: sensor-adjusted net area change versus gross boundary reworking."""
    net, gross = "standardized_net_area_change", "standardized_gross_boundary_reworking"
    for region in REGION_ORDER:
        part = cays.loc[cays["analysis_region"].eq(region)]
        if part.empty:
            continue
        for support, marker in SUPPORT_MARKERS.items():
            data = part.loc[part["source_support"].eq(support)]
            if data.empty:
                continue
            sizes = 13 + 8 * np.log10(data["area_t_m2"].clip(lower=1000))
            ax_a.scatter(data[net] * 100, data[gross] * 100, s=sizes, marker=marker, color=REGION_COLORS[region], alpha=0.70, edgecolors="white", linewidths=0.35, zorder=2)
        summary = regions.loc[regions["analysis_region"].eq(region)].iloc[0]
        ax_a.scatter(summary[net] * 100, summary[gross] * 100, s=38, marker="D", color=REGION_COLORS[region], edgecolors="black", linewidths=0.45, zorder=4)
    ax_a.axvline(0, color="#5f5f5f", lw=0.65, ls="--", zorder=1)
    ax_a.set_yscale("log")
    ax_a.set_ylim(*value_limits(cays[gross] * 100, lower_floor=0.5))
    x = cays[net].dropna() * 100
    ax_a.set_xlim(float(x.quantile(0.01)) - 8, float(x.quantile(0.99)) + 8)
    ax_a.set_xlabel("Sensor-adjusted net area change (% per 365 d)")
    ax_a.set_ylabel("Gross boundary reworking (% per 365 d)")
    panel_label(ax_a, panel, "Net change versus planform reworking")
    ax_a.text(0.02, 0.96, "Point: cay median; diamond: regional median", transform=ax_a.transAxes, va="top", fontsize=5.45)
    region_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=REGION_COLORS[r], markeredgecolor="white", markersize=5.4, label=r) for r in REGION_ORDER]
    support_handles = [Line2D([0], [0], marker=m, color="#444444", linestyle="none", markerfacecolor="white", markersize=4.8, label=s) for s, m in SUPPORT_MARKERS.items()]
    legend_regions = ax_a.legend(handles=region_handles, loc="lower right", title="Region", title_fontsize=5.8, handletextpad=0.3, labelspacing=0.18)
    ax_a.add_artist(legend_regions)
    ax_a.legend(handles=support_handles, loc="upper right", title="Source support", title_fontsize=5.8, handletextpad=0.3, labelspacing=0.16)

def draw_signatures_balance_panel(ax_b: plt.Axes, regions: pd.DataFrame, panel: str = "b") -> None:
    """Panel: regional erosion-deposition balance of sensor-adjusted exchange."""
    positions = np.arange(len(REGION_ORDER))
    bars = regions.set_index("analysis_region").reindex(REGION_ORDER)
    erosion, deposition = bars["standardized_erosion"] * 100, bars["standardized_deposition"] * 100
    ax_b.barh(positions, -erosion, height=0.58, color="#4C78A8", label="Erosion")
    ax_b.barh(positions, deposition, height=0.58, color="#D27A45", label="Deposition")
    for position, row in enumerate(bars.itertuples()):
        entries = [
            (row.standardized_erosion * 100, row.standardized_erosion_ci95_low * 100, row.standardized_erosion_ci95_high * 100, "#253B55", -1),
            (row.standardized_deposition * 100, row.standardized_deposition_ci95_low * 100, row.standardized_deposition_ci95_high * 100, "#7A3C20", 1),
        ]
        for value, low, high, color, sign in entries:
            if np.isfinite(low):
                error = [[high - value], [value - low]] if sign < 0 else [[value - low], [high - value]]
                ax_b.errorbar(sign * value, position, xerr=error, fmt="none", color=color, capsize=1.7, lw=0.65)
    ax_b.axvline(0, color="#4f4f4f", lw=0.7)
    ax_b.set_yticks(positions, [f"{region}\n(n={int(row.n_cays)}; both={int(row.n_both_sources)})" for region, row in bars.iterrows()])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Sensor-adjusted planform exchange (% per 365 d)")
    panel_label(ax_b, panel, "Erosion--deposition balance")
    ax_b.legend(loc="lower right", handlelength=1.0, handletextpad=0.35)
    ax_b.text(0.02, 0.02, "Bars: cay medians\nError bars: 95% cay bootstrap CI (n ≥ 5)", transform=ax_b.transAxes, va="bottom", fontsize=5.25, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))


def draw_regional_spectra_condensed(ax: plt.Axes, regions: pd.DataFrame, panel: str = "c") -> None:
    """Panel: condensed regional spectra of the four standardized reworking modes."""
    metrics = ["centroid_translation", "perimeter_adjustment", "axial_adjustment", "axis_reorientation"]
    labels = ["Translation", "Perimeter", "Axial", "Reorientation"]
    positions = np.arange(len(metrics))
    for region in REGION_ORDER:
        rows = regions.loc[regions["analysis_region"].eq(region)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        centers, lows, highs = [], [], []
        for metric in metrics:
            column = f"standardized_{metric}"
            centers.append(float(row[column]))
            lows.append(float(row[f"{column}_ci95_low"]))
            highs.append(float(row[f"{column}_ci95_high"]))
        centers = np.asarray(centers, dtype=float)
        lows = np.asarray(lows, dtype=float)
        highs = np.asarray(highs, dtype=float)
        finite = np.isfinite(lows) & np.isfinite(highs)
        ax.errorbar(positions[finite], centers[finite], yerr=[centers[finite] - lows[finite], highs[finite] - centers[finite]], fmt="o", ms=4.0, color=REGION_COLORS[region], ecolor=REGION_COLORS[region], elinewidth=0.7, capsize=1.6, label=region, zorder=3)
        if bool((~finite).any()):
            ax.scatter(positions[~finite], centers[~finite], s=18, color=REGION_COLORS[region], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Standardized rate (per 365 d)")
    ax.grid(axis="y", color="#dddddd", lw=0.45)
    ax.text(0.02, 0.02, "Regional medians; error bars: 95% cay bootstrap CI (n ≥ 5); region colours as in panel a", transform=ax.transAxes, va="bottom", fontsize=5.2)
    panel_label(ax, panel, "Regions differ in reshaping modes")


def draw_forest_panel(ax: plt.Axes, cays: pd.DataFrame, regions: pd.DataFrame, metric: str, xlabel: str, panel: str, show_labels: bool) -> None:
    column = f"standardized_{metric}"
    rng = np.random.default_rng(RANDOM_SEED + sum(map(ord, metric)))
    for position, region in enumerate(REGION_ORDER):
        data = cays.loc[cays["analysis_region"].eq(region) & cays[column].notna()]
        for support, marker in SUPPORT_MARKERS.items():
            part = data.loc[data["source_support"].eq(support)]
            if not part.empty:
                ax.scatter(part[column], position + rng.uniform(-0.16, 0.16, len(part)), marker=marker, s=18, color=REGION_COLORS[region], alpha=0.63, edgecolors="white", linewidths=0.3, zorder=2)
        row = regions.loc[regions["analysis_region"].eq(region)].iloc[0]
        center, low, high = row[column], row[f"{column}_ci95_low"], row[f"{column}_ci95_high"]
        if np.isfinite(low):
            ax.errorbar(center, position, xerr=[[max(center - low, 0)], [max(high - center, 0)]], fmt="D", ms=4.4, color="black", mfc=REGION_COLORS[region], mec="black", mew=0.45, capsize=1.8, lw=0.65, zorder=4)
        else:
            ax.scatter(center, position, marker="D", s=27, color=REGION_COLORS[region], edgecolors="black", linewidths=0.45, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(*value_limits(cays[column], lower_floor=0.002))
    if show_labels:
        labels = [f"{region}\n(n={int(regions.loc[regions.analysis_region.eq(region), 'n_cays'].iloc[0])})" for region in REGION_ORDER]
        ax.set_yticks(np.arange(len(REGION_ORDER)), labels)
    else:
        ax.set_yticks(np.arange(len(REGION_ORDER)), [])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d0d0d0", lw=0.45, alpha=0.7)
    ax.set_xlabel(xlabel)
    panel_label(ax, panel, METRICS[metric]["label"])


def draw_figure_3(cays: pd.DataFrame, regions: pd.DataFrame) -> None:
    """Global Figure 3: all-source regional description of the requested shape metrics."""
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.93, bottom=0.19, wspace=0.065, hspace=0.31)
    panels = [
        ("centroid_translation", "Baseline radii per 365 d", "a", True),
        ("perimeter_adjustment", "Relative perimeter change per 365 d", "b", False),
        ("axial_adjustment", "Baseline radii per 365 d", "c", True),
        ("axis_reorientation", "90-degree turns per 365 d", "d", False),
    ]
    for ax, args in zip(axes.flat, panels):
        draw_forest_panel(ax, cays, regions, *args)
    handles = [Line2D([0], [0], marker=m, color="#444444", linestyle="none", markerfacecolor="white", markersize=5.2, label=s) for s, m in SUPPORT_MARKERS.items()]
    handles.append(Line2D([0], [0], marker="D", color="#444444", linestyle="none", markerfacecolor="#dddddd", markeredgecolor="black", markersize=5.3, label="Regional median"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0.045), frameon=False, handletextpad=0.35, columnspacing=1.05)
    fig.text(0.54, 0.012, "All A/B source-specific intervals standardized to Sentinel-2 reference scale and 365 d; error bars: 95% cay bootstrap CI when n ≥ 5.", ha="center", fontsize=5.1)
    save_figure(fig, FIGURES / "figure_global_regional_morphology")


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    intervals = load_analysis_frame()
    standardized, models = standardize_all(intervals)
    cays, regions = summarize_cays(standardized), None
    regions = summarize_regions(cays)
    standardized.to_csv(OUTPUTS / "全球形态重塑区间汇总.csv", index=False, encoding="utf-8-sig")
    cays.to_csv(OUTPUTS / "全球形态重塑沙洲汇总.csv", index=False, encoding="utf-8-sig")
    regions.to_csv(OUTPUTS / "全球形态重塑区域汇总.csv", index=False, encoding="utf-8-sig")
    models.to_csv(OUTPUTS / "全球形态重塑传感器调整模型.csv", index=False, encoding="utf-8-sig")
    audit = {
        "figure_2_claim": "Across the manuscript-defined Indo-Pacific regions, net planform change and boundary exchange are distinct dimensions, and both can be described using all source-specific intervals after explicit source and interval standardization.",
        "figure_3_claim": "Regional sand-cay morphodynamics span centroid translation, perimeter and axial adjustment, and axis reorientation; no single metric represents all regional reworking modes.",
        "region_definition": "Great Barrier Reef, Maldives, South China Sea, Marshall Islands, and Lakshadweep, matching the Results regional division in the manuscript.",
        "scope": "Multi-regional Indo-Pacific evidence supporting a transferable global monitoring framework; it does not assert uniform global behaviour.",
        "input_filter": "Google Earth and Sentinel-2, endpoint quality A/B, baseline area > 0, interval >= 30 d; excluded reefs follow the project analysis contract.",
        "source_fusion": "Sand-cay fixed effects plus source and cubic log-interval terms. Responses are separately standardized to Sentinel-2 and 365.2425 d: Gamma log-link for non-negative magnitudes and OLS/asinh for signed net area change. Standardized indicators describe separate dimensions; the exact area balance is computed from same-source masks before standardization.",
        "n_intervals_total": int(len(standardized)), "n_cays_total": int(cays["sand_cay_id"].nunique()), "n_reefs_total": int(cays["reef_id"].nunique()),
        "source_counts": {key: int(value) for key, value in standardized["sensor"].value_counts().items()},
        "regional_cay_counts": {str(row.analysis_region): int(row.n_cays) for row in regions.itertuples()},
        "uncertainty": "Regional intervals resample sand cays (1,999 replicates). Regions with fewer than five cays are descriptive and show no interval.",
        "duplicate_observation_rule": "Same-source/frame/date observations use the common A/B-quality and image-id selection rule; geometry comes from the selected starting observation.",
        "analysis_contract_sha256": analysis_contract_digest(),
        "analysis_workflow_reference": "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065",
    }
    (OUTPUTS / "全球形态重塑核查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_figure_3(cays, regions)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
