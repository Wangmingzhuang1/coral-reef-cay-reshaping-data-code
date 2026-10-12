"""Exploratory forcing contrast for the archived periodicity/episodicity result.

This addendum is EXPLORATORY and is not part of the verified core.  It uses only
already aligned local forcing fields (outputs/流场波浪区间特征.csv: ERA5 wind,
GLORYS current, WaveRYS waves) to ask three questions:

E1  Within-cay association of step speed with interval wave height (primary) and
    wind speed (secondary), bare stratum; vegetated reported descriptively.
E2  Seasonal phase match: monthly climatology of wave height and wind speed vs
    the monthly timing of bare step speed and top-decile steps.
E3  Typhoon-day fraction (local wind decomposition) in top-decile steps vs rest.
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

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from 区域气候滞后模型 import apply_figure_style  # noqa: E402
from 分析裸沙质心轨迹周期与突发 import build_steps, save_figure  # noqa: E402
from 分析沙洲形状与质心迁移 import load_observations  # noqa: E402


def event_forcing(daily: pd.DataFrame, steps: pd.DataFrame, quantile: float = 0.95) -> pd.DataFrame:
    """存档日尺度波高的高暴露天数；历史字段名hs_daily_max不代表原生3小时日最大值。"""
    work = daily.copy()
    work["date"] = work["time"].dt.normalize()
    work["wind_speed"] = np.hypot(pd.to_numeric(work["u10"], errors="coerce"), pd.to_numeric(work["v10"], errors="coerce"))
    work["VHM0"] = pd.to_numeric(work["VHM0"], errors="coerce")
    day = work.groupby(["reef_id", "date"], as_index=False).agg(hs_daily_max=("VHM0", "max"), wind_daily_max=("wind_speed", "max"))
    day = day.dropna(subset=["hs_daily_max"])
    thresholds = day.groupby("reef_id")["hs_daily_max"].quantile(quantile).rename("hs_p95")
    day = day.merge(thresholds, on="reef_id", how="left")
    records = []
    indexed = {reef: group.sort_values("date") for reef, group in day.groupby("reef_id")}
    for row in steps.itertuples(index=False):
        group = indexed.get(row.reef_id)
        if group is None:
            continue
        window = group.loc[group["date"].gt(pd.Timestamp(row.time_t).normalize()) & group["date"].le(pd.Timestamp(row.time_t1).normalize())]
        if window.empty:
            continue
        records.append({
            "transition_id": row.transition_id,
            "event_hs_max": float(window["hs_daily_max"].max()),
            "event_storm_days": int(window["hs_daily_max"].ge(window["hs_p95"].iloc[0]).sum()),
            "event_wind_max": float(window["wind_daily_max"].max()),
        })
    return pd.DataFrame(records)


def load_forcing_inputs():
    forcing = pd.read_csv(ROOT / "outputs" / "流场波浪区间特征.csv", low_memory=False)
    daily = pd.read_csv(ROOT / "data" / "environment" / "harmonized" / "daily" / "reef_daily_features.csv", low_memory=False)
    daily["time"] = pd.to_datetime(daily["time"], format="ISO8601")
    return forcing, daily


def build_usable(observations, steps, forcing, daily):
    merged = steps.merge(
        forcing[["transition_id", "wind_speed_mean", "wind_speed_p90", "wave_hs_mean", "wave_hs_p90", "local_typhoon_day_fraction", "waverys_complete", "era5_complete"]],
        on="transition_id",
        how="left",
        validate="one_to_one",
    )
    usable = merged.loc[merged["wave_hs_mean"].notna() & merged["wind_speed_mean"].gt(0) & merged["wave_hs_mean"].gt(0)].copy()
    usable["log_speed"] = np.log(usable["speed_norm"] + 0.01)
    usable["log_wave"] = np.log(usable["wave_hs_mean"])
    usable["log_wave_p90"] = np.log(usable["wave_hs_p90"])
    usable["log_wind"] = np.log(usable["wind_speed_mean"])
    events = event_forcing(daily, usable)
    usable = usable.merge(events, on="transition_id", how="left", validate="one_to_one")
    return usable, merged


def fit_forcing_models(usable):
    rows = []
    bare = usable.loc[usable["stratum"].eq("bare")]
    vegetated = usable.loc[usable["stratum"].eq("vegetated")]
    specs = [
        ("E1_wave_mean_bare", bare, "log_speed ~ log_wave + time_interval_days + C(sand_cay_id)", "log_wave"),
        ("E1_wave_p90_bare", bare, "log_speed ~ log_wave_p90 + time_interval_days + C(sand_cay_id)", "log_wave_p90"),
        ("E1_wind_bare", bare, "log_speed ~ log_wind + time_interval_days + C(sand_cay_id)", "log_wind"),
        ("E1_wave_mean_vegetated", vegetated, "log_speed ~ log_wave + time_interval_days + C(sand_cay_id)", "log_wave"),
    ]
    for name, subset, formula, term in specs:
        if subset["sand_cay_id"].nunique() < 5:
            rows.append({"analysis": name, "status": "insufficient_cays"})
            continue
        fit = fit_clustered(formula, subset)
        low, high = fit.conf_int().loc[term].astype(float)
        rows.append({
            "analysis": name, "status": "exploratory", "term": term,
            "estimate": float(fit.params[term]), "reef_cluster_se": float(fit.bse[term]),
            "ci95_low": low, "ci95_high": high, "p_value": float(fit.pvalues[term]),
            "n_observations": int(len(subset)), "n_cays": int(subset["sand_cay_id"].nunique()),
        })
    threshold = bare["speed_norm"].quantile(0.90)
    bare_with_fraction = bare.dropna(subset=["local_typhoon_day_fraction"])
    large = bare_with_fraction.loc[bare_with_fraction["speed_norm"].ge(threshold), "local_typhoon_day_fraction"]
    rest = bare_with_fraction.loc[bare_with_fraction["speed_norm"].lt(threshold), "local_typhoon_day_fraction"]
    if len(large) >= 8 and len(rest) >= 8:
        stat = mannwhitneyu(large, rest, alternative="two-sided")
        rows.append({
            "analysis": "E3_typhoon_day_fraction_bare", "status": "exploratory",
            "term": "local_typhoon_day_fraction", "estimate": float(large.median() - rest.median()),
            "reef_cluster_se": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "p_value": float(stat.pvalue), "n_observations": int(len(large)), "n_cays": int(bare_with_fraction["sand_cay_id"].nunique()),
        })
    with_events = usable.dropna(subset=["event_hs_max"])
    bare_events = with_events.loc[with_events["stratum"].eq("bare")]
    for name, formula, term in [
        ("E4_event_hs_max_bare", "log_speed ~ log_event_hs + time_interval_days + C(sand_cay_id)", "log_event_hs"),
        ("E4_event_storm_days_bare", "log_speed ~ event_storm_days + time_interval_days + C(sand_cay_id)", "event_storm_days"),
    ]:
        subset = bare_events.copy()
        subset["log_event_hs"] = np.log(subset["event_hs_max"])
        if subset["sand_cay_id"].nunique() < 5 or term not in subset.columns:
            rows.append({"analysis": name, "status": "insufficient_cays"})
            continue
        fit = fit_clustered(formula, subset)
        low, high = fit.conf_int().loc[term].astype(float)
        rows.append({
            "analysis": name, "status": "exploratory", "term": term,
            "estimate": float(fit.params[term]), "reef_cluster_se": float(fit.bse[term]),
            "ci95_low": low, "ci95_high": high, "p_value": float(fit.pvalues[term]),
            "n_observations": int(len(subset)), "n_cays": int(subset["sand_cay_id"].nunique()),
        })
    models = pd.DataFrame(rows)
    tested = models.loc[models["p_value"].notna()]
    models.loc[tested.index, "fdr_q_value"] = multipletests(tested["p_value"], method="fdr_bh")[1]
    return models


def build_phase(daily, merged, threshold):
    daily = daily.copy()
    phase, regional_phase = build_phase(daily, merged, threshold)
    return phase, regional_phase


def panel_storm(ax, usable, models, title):
    bare = usable.loc[usable["stratum"].eq("bare") & usable["event_storm_days"].notna()].copy()
    refit = smf.ols("log_speed ~ event_storm_days + time_interval_days + C(sand_cay_id)", data=bare).fit()
    beta = float(refit.params["event_storm_days"])
    bare["partial"] = refit.resid + beta * bare["event_storm_days"]
    ax.scatter(bare["event_storm_days"], bare["partial"], s=6, color="#D55E00", alpha=0.20, linewidths=0)
    x_mean = float(bare["event_storm_days"].mean())
    p_mean = float(bare["partial"].mean())
    grid = np.linspace(0, bare["event_storm_days"].max(), 50)
    ax.plot(grid, p_mean + beta * (grid - x_mean), color="#8C3A00", lw=1.1)
    bin_edges = [(0, 0), (1, 2), (3, 5), (6, 10 ** 6)]
    centers, medians, lows, highs = [], [], [], []
    for low_edge, high_edge in bin_edges:
        subset = bare.loc[bare["event_storm_days"].between(low_edge, high_edge)]
        if len(subset) >= 8:
            centers.append(subset["event_storm_days"].mean())
            medians.append(subset["partial"].mean())
            sem = subset["partial"].std(ddof=1) / np.sqrt(len(subset))
            lows.append(subset["partial"].mean() - 1.96 * sem)
            highs.append(subset["partial"].mean() + 1.96 * sem)
    # 分箱均值只作描述，推断使用礁盘聚类模型的区间。
    ax.scatter(centers, medians, marker="D", s=16, color="#222222", zorder=5)
    row = models.set_index("analysis").loc["E4_event_storm_days_bare"]
    ax.text(0.97, 0.97, f"Within-cay $\\beta$ = {row['estimate']:.4f} per high-wave day\n95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]\nq = {row['fdr_q_value']:.1e}; n = {int(row['n_observations'])} intervals, {int(row['n_cays'])} cays", transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    ax.axhline(0, color="#888888", lw=0.7, ls=":")
    ax.set_xlabel("High-wave days (daily Hs $\\geq$ reef p95)")
    ax.set_ylabel("Partial residual, log step speed")
    ax.set_title(title, loc="left", fontsize=8)


def panel_forest(ax, usable, models, title):
    bare = usable.loc[usable["stratum"].eq("bare") & usable["event_storm_days"].notna()].copy()
    if "log_event_hs" not in bare.columns:
        bare["log_event_hs"] = np.log(bare["event_hs_max"])
    index = models.set_index("analysis")
    order = ["E1_wave_mean_bare", "E1_wave_p90_bare", "E1_wind_bare", "E4_event_hs_max_bare", "E4_event_storm_days_bare"]
    predictors = {
        "E1_wave_mean_bare": "log_wave",
        "E1_wave_p90_bare": "log_wave_p90",
        "E1_wind_bare": "log_wind",
        "E4_event_hs_max_bare": "log_event_hs",
        "E4_event_storm_days_bare": "event_storm_days",
    }
    labels = ["Interval mean Hs", "Interval p90 Hs", "Interval mean wind", "Interval daily max Hs", "High-wave days"]
    positions = np.arange(len(order))[::-1]
    scaled = []
    for position, name in zip(positions, order):
        row = index.loc[name]
        sd = float(bare[predictors[name]].std(ddof=1))
        estimate, low, high = row["estimate"] * sd, row["ci95_low"] * sd, row["ci95_high"] * sd
        scaled.extend([low, high])
        color = "#D55E00" if name == "E4_event_storm_days_bare" else "#555555"
        ax.plot([low, high], [position, position], color="#333333", lw=1.1)
        ax.plot(estimate, position, "o", color=color, ms=4.2)
    ax.axvline(0, color="#888888", lw=0.7, ls="--")
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", labelsize=5.9)
    pad = max(0.02, (max(scaled) - min(scaled)) * 0.08)
    ax.set_xlim(min(scaled) - pad, max(scaled) + pad)
    ax.set_xlabel("Coefficient per 1 SD of predictor (95% CI)")
    ax.set_title(title, loc="left", fontsize=8)


def panel_regional(ax, regional_phase, region):
    subset = regional_phase.loc[regional_phase["region"].eq(region)].sort_values("month")
    total_top = subset["n_top_steps"].sum()
    if total_top <= 0:
        return
    fraction = subset["n_top_steps"] / total_top
    ax.bar(subset["month"], fraction, color="#D55E00", width=0.82, alpha=0.9)
    twin = ax.twinx()
    twin.plot(subset["month"], subset["wave_hs_mean"], color="#0072B2", lw=1.0)
    twin.set_ylim(0, subset["wave_hs_mean"].max() * 2.6)
    twin.set_yticks([])
    for spine in twin.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0.4, 12.6)
    ax.set_xticks([1, 4, 7, 10])
    ax.tick_params(labelsize=5.8)
    ax.set_ylabel(region, rotation=0, ha="right", va="center", fontsize=7.5, fontweight="bold", labelpad=12)


def make_figure(usable: pd.DataFrame, models: pd.DataFrame, regional_phase: pd.DataFrame, target: Path) -> None:
    apply_figure_style()
    fig = plt.figure(figsize=(7.3, 3.55))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.02, 0.98, 1.18], hspace=0.50, wspace=0.62, left=0.085, right=0.985, top=0.93, bottom=0.115)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    axes_c = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    panel_storm(ax_a, usable, models, "a  High-wave-day association with step speed")
    panel_forest(ax_b, usable, models, "b  Forcing coefficients, bare cays")
    for ax, region in zip(axes_c, ["NH", "GBR", "MEDF"]):
        panel_regional(ax, regional_phase, region)
    axes_c[0].text(0.98, 0.95, "bars: share of regional top-decile steps\nblue line: regional monthly mean Hs", transform=axes_c[0].transAxes, ha="right", va="top", fontsize=5.4)
    axes_c[2].set_xlabel("Month")
    axes_c[0].set_title("c  Regional seasonality of large steps", loc="left", fontsize=8)
    save_figure(fig, target)


def make_supplement_figure(usable: pd.DataFrame, models: pd.DataFrame, regional_phase: pd.DataFrame, target: Path) -> None:
    apply_figure_style()
    fig = plt.figure(figsize=(5.8, 3.55))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.95, 1.18], hspace=0.50, wspace=0.75, left=0.17, right=0.985, top=0.93, bottom=0.115)
    ax_a = fig.add_subplot(gs[:, 0])
    axes_b = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    panel_forest(ax_a, usable, models, "a  Forcing coefficients, bare cays")
    for ax, region in zip(axes_b, ["NH", "GBR", "MEDF"]):
        panel_regional(ax, regional_phase, region)
    axes_b[0].text(0.98, 0.95, "bars: share of regional top-decile steps\nblue line: regional monthly mean Hs", transform=axes_b[0].transAxes, ha="right", va="top", fontsize=5.4)
    axes_b[2].set_xlabel("Month")
    axes_b[0].set_title("b  Regional seasonality of large steps", loc="left", fontsize=8)
    save_figure(fig, target)


def fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["reef_id"], "use_correction": True}
    )


def main() -> None:
    output_dir = (PROJECT_ROOT / "outputs/时间结构")
    figure_dir = (PROJECT_ROOT / "outputs/figures/中间图")
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    observations = load_observations()
    steps = build_steps(observations)
    forcing, daily = load_forcing_inputs()
    usable, merged = build_usable(observations, steps, forcing, daily)

    models = fit_forcing_models(usable)
    threshold = usable.loc[usable["stratum"].eq("bare"), "speed_norm"].quantile(0.90)

    daily["wind_speed"] = np.hypot(pd.to_numeric(daily["u10"], errors="coerce"), pd.to_numeric(daily["v10"], errors="coerce"))
    daily["month"] = daily["time"].dt.month
    climatology = daily.groupby("month").agg(wave_hs_mean=("VHM0", "mean"), wind_speed_mean=("wind_speed", "mean")).reset_index()
    bare_steps = merged.loc[merged["stratum"].eq("bare")].copy()
    bare_steps["month"] = bare_steps["midpoint"].dt.month
    timing = bare_steps.groupby("month").agg(mean_speed_norm=("speed_norm", "mean"), n_steps=("speed_norm", "size")).reset_index()
    top = bare_steps.loc[bare_steps["speed_norm"].ge(threshold)].groupby("month").size().rename("n_top_steps").reset_index()
    phase = climatology.merge(timing, on="month", how="left").merge(top, on="month", how="left").fillna({"n_top_steps": 0})
    daily["region"] = daily["reef_id"].astype(str).str.extract(r"^(GBR|MEDF|OTHER|NH)")
    regional_clim = daily.dropna(subset=["region"]).groupby(["region", "month"]).agg(wave_hs_mean=("VHM0", "mean"), wind_speed_mean=("wind_speed", "mean")).reset_index()
    bare_steps["region"] = bare_steps["reef_id"].astype(str).str.extract(r"^(GBR|MEDF|OTHER|NH)")
    regional_timing = bare_steps.groupby(["region", "month"]).agg(mean_speed_norm=("speed_norm", "mean"), n_steps=("speed_norm", "size")).reset_index()
    regional_top = bare_steps.loc[bare_steps["speed_norm"].ge(threshold)].groupby(["region", "month"]).size().rename("n_top_steps").reset_index()
    regional_phase = regional_clim.merge(regional_timing, on=["region", "month"], how="left").merge(regional_top, on=["region", "month"], how="left").fillna({"n_top_steps": 0})

    models.to_csv(output_dir / "裸沙质心轨迹风浪对照模型.csv", index=False, encoding="utf-8-sig")
    phase.to_csv(output_dir / "裸沙质心轨迹风浪季节相位.csv", index=False, encoding="utf-8-sig")
    regional_phase.to_csv(output_dir / "裸沙质心轨迹风浪季节相位分区域.csv", index=False, encoding="utf-8-sig")
    audit = {
        "status": "EXPLORATORY addendum; not part of the verified core result",
        "forcing_source": "outputs/流场波浪区间特征.csv (ERA5 wind, GLORYS current, WaveRYS wave) and data/environment/harmonized/daily/reef_daily_features.csv",
        "n_steps_merged": int(len(merged)), "n_steps_with_wave_wind": int(len(usable)),
        "coverage": float(len(usable) / max(len(merged), 1)),
    }
    (output_dir / "裸沙质心轨迹风浪对照核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(usable, models, regional_phase, figure_dir / "figure_r17_storm_day_forcing")
    make_supplement_figure(usable, models, regional_phase, ROOT / "outputs" / "figures" / "figure_s4_storm_day_regional")
    print(models.to_string(index=False))
    print(phase.to_string(index=False))
    print(regional_phase.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
