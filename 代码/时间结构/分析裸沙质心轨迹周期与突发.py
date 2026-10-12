"""Periodicity and episodicity of bare-sand cay centroid trajectories.

Analysis family (FDR-BH across harmonic and timing tests):
  P1 annual harmonic, bare / vegetated stratum (within-cay, reef-clustered F test)
  P1 semiannual harmonic, bare stratum
  P2 paired-axis Lomb-Scargle screening against a date-exchangeability baseline
  A1 descriptive largest-step path share, bare / vegetated
  A2 day-of-year concentration of top-decile steps (Rayleigh), bare
  A2 typhoon wind association of top-decile steps (Mann-Whitney), bare

Positions are pixel centroids scaled by the frame-constant pixel size and demeaned
within cay x reference frame, so only within-cay wandering is analysed.
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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
from 区域气候滞后模型 import apply_figure_style  # noqa: E402
from 分析沙洲形状与质心迁移 import load_observations  # noqa: E402

BARE_STAGES = ("stage2_transitional_emergence", "stage3_stable_exposed")
VEGETATED_STAGES = ("stage4_stable_vegetated",)
STEP_DAYS = (30, 400)
LS_PERIODS = np.linspace(0.25, 4.0, 120)
LS_PERMUTATIONS = 1999
ANNUAL_BAND = (0.9, 1.1)
RNG_SEED = 20260918


def harmonics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    phase = 2 * np.pi * frame["date"].dt.dayofyear / 365.2425
    frame["sa"], frame["ca"] = np.sin(phase), np.cos(phase)
    frame["sa2"], frame["ca2"] = np.sin(2 * phase), np.cos(2 * phase)
    return frame


def position_frame(observations: pd.DataFrame) -> pd.DataFrame:
    work = observations.copy()
    work["east_m"] = work["sand_cay_centroid_x"] * work["pixel_size_m"]
    work["north_m"] = -work["sand_cay_centroid_y"] * work["pixel_size_m"]
    grouped = work.groupby(["sand_cay_id", "reference_frame_id"], dropna=False)
    work["east_dm"] = work["east_m"] - grouped["east_m"].transform("mean")
    work["north_dm"] = work["north_m"] - grouped["north_m"].transform("mean")
    work["stratum"] = np.where(work["development_stage"].isin(BARE_STAGES), "bare", np.where(work["development_stage"].isin(VEGETATED_STAGES), "vegetated", "other"))
    return harmonics(work.loc[work["stratum"].ne("other")])


def harmonic_tests(positions: pd.DataFrame) -> list[dict]:
    rows = []
    for stratum in ("bare", "vegetated"):
        subset = positions.loc[positions["stratum"].eq(stratum)]
        if subset["sand_cay_id"].nunique() < 5:
            continue
        for component in ("east_dm", "north_dm"):
            fit = smf.ols(f"{component} ~ sa + ca + sa2 + ca2 + C(sand_cay_id)", data=subset).fit(
                cov_type="cluster", cov_kwds={"groups": subset["reef_id"], "use_correction": True}
            )
            for label, terms in [("annual", ["sa", "ca"]), ("semiannual", ["sa2", "ca2"])]:
                test = fit.f_test(", ".join(f"{term} = 0" for term in terms))
                amplitude = float(np.sqrt(sum(fit.params[term] ** 2 for term in terms)))
                rows.append({
                    "analysis": f"harmonic_{label}_{component}_{stratum}", "stratum": stratum,
                    "component": component, "harmonic": label, "amplitude_m": amplitude,
                    "f_stat": float(test.fvalue), "p_value": float(test.pvalue),
                    "n_observations": int(len(subset)), "n_cays": int(subset["sand_cay_id"].nunique()),
                })
    return rows


def ls_basis(times: np.ndarray, periods: np.ndarray):
    """Precompute the Lomb-Scargle design; tau depends only on sampling times."""
    omega = 2 * np.pi / (periods * 365.2425)
    wt = omega[:, None] * times[None, :]
    tau = np.arctan2(np.sum(np.sin(2 * wt), axis=1), np.sum(np.cos(2 * wt), axis=1)) / (2 * omega)
    diff = wt - (omega * tau)[:, None]
    cos_term, sin_term = np.cos(diff), np.sin(diff)
    return cos_term, sin_term, (cos_term ** 2).sum(axis=1), (sin_term ** 2).sum(axis=1)


def ls_band_power(basis, values: np.ndarray, band: np.ndarray) -> float:
    cos_term, sin_term, denominator_cos, denominator_sin = basis
    variance = np.var(values)
    if variance <= 0:
        return 0.0
    cos_projection = cos_term @ values
    sin_projection = sin_term @ values
    power = 0.5 * (
        cos_projection ** 2 / np.maximum(denominator_cos, 1e-12)
        + sin_projection ** 2 / np.maximum(denominator_sin, 1e-12)
    ) / variance
    return float(power[band].max())


def periodogram_tests(positions: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    significant = 0
    tested = 0
    for (cay_id, frame_id), group in positions.loc[positions["stratum"].eq("bare")].groupby(["sand_cay_id", "reference_frame_id"], dropna=False):
        if len(group) < 12:
            continue
        times = (group["date"] - group["date"].iloc[0]).dt.days.to_numpy(float)
        basis = ls_basis(times, LS_PERIODS)
        band = (LS_PERIODS >= ANNUAL_BAND[0]) & (LS_PERIODS <= ANNUAL_BAND[1])
        values = group[["east_dm", "north_dm"]].to_numpy(float)
        observed = max(ls_band_power(basis, values[:, axis], band) for axis in range(2))
        null_best = []
        for _ in range(LS_PERMUTATIONS):
            permuted = values[rng.permutation(len(values))]
            null_best.append(max(ls_band_power(basis, permuted[:, axis], band) for axis in range(2)))
        p_value = (1 + np.sum(np.array(null_best) >= observed)) / (1 + len(null_best))
        tested += 1
        significant += int(p_value < 0.05)
        rows.append({"sand_cay_id": cay_id, "reference_frame_id": frame_id, "n_observations": len(group), "annual_band_power": observed, "permutation_p": p_value})
    detail = pd.DataFrame(rows)
    if tested:
        summary = [{
            "analysis": "lomb_scargle_annual_band_bare", "stratum": "bare", "component": "east+north",
            "harmonic": "annual_band", "amplitude_m": np.nan,
            "f_stat": np.nan, "p_value": np.nan,
            "n_observations": int(detail["n_observations"].sum()), "n_cays": tested,
            "significant_cays": significant,
            "evidence_role": "exploratory date-exchangeability screening; no population binomial inference",
        }]
    else:
        summary = []
    return summary, detail


def build_steps(observations: pd.DataFrame, grouping: str = "historical_stage") -> pd.DataFrame:
    """统一计算移动速率；当前主文按审校覆盖标签分组，历史阶段仅供旧分析复现。"""
    if grouping not in {"cover", "historical_stage"}:
        raise ValueError("grouping must be cover or historical_stage")
    intervals = pd.read_csv(ROOT / "outputs" / "沙洲变化区间.csv", low_memory=False)
    steps = intervals.loc[intervals["sensor"].eq("sentinel2")].copy()
    for column in ("time_interval_days", "centroid_shift_m", "area_t_m2"):
        steps[column] = pd.to_numeric(steps[column], errors="coerce")
    steps = steps.loc[steps["time_interval_days"].between(*STEP_DAYS) & steps["centroid_shift_m"].notna() & steps["area_t_m2"].gt(0)].copy()
    steps["time_t"] = pd.to_datetime(steps["time_t"])
    steps["time_t1"] = pd.to_datetime(steps["time_t1"])
    steps["midpoint"] = steps["time_t"] + (steps["time_t1"] - steps["time_t"]) / 2
    steps["years"] = steps["time_interval_days"] / 365.2425
    steps["speed_norm"] = steps["centroid_shift_m"] / np.sqrt(steps["area_t_m2"] / np.pi) / steps["years"]
    if grouping == "cover":
        keys = ["sensor", "sand_cay_id", "reference_frame_id"]
        endpoints = observations[keys + ["date"]].drop_duplicates().copy()
        endpoints["date"] = pd.to_datetime(endpoints["date"])
        for date in ["time_t", "time_t1"]:
            steps = steps.merge(endpoints.rename(columns={"date": date}), on=keys + [date],
                                how="inner", validate="many_to_one")
        labels = pd.read_csv(ROOT / "outputs" / "植被状态审校主表.csv", low_memory=False)
        from 构建沙洲观测与变化表 import select_analysis_observations
        labels = select_analysis_observations(labels)
        labels["date"] = pd.to_datetime(labels["date"])
        labels = labels.loc[labels.analysis_vegetation_state.isin(["none", "sparse", "partial", "dominant"]),
                            keys + ["date", "analysis_vegetation_state"]]
        # 标签必须与同一来源、沙洲、参考框和初始日期对应，不能用历史发育阶段代替覆盖。
        steps = steps.merge(labels.rename(columns={"date": "time_t", "analysis_vegetation_state": "cover_state_t"}),
                            on=keys + ["time_t"], how="inner", validate="many_to_one")
        steps["stratum"] = np.where(steps.cover_state_t.eq("none"), "bare", "vegetated")
        steps["grouping_definition"] = "accepted_initial_cover_none_vs_sparse_partial_dominant"
        return steps
    stage = (
        observations[["sand_cay_id", "date", "development_stage"]]
        .sort_values(["sand_cay_id", "date", "development_stage"])
        .drop_duplicates(["sand_cay_id", "date"], keep="first")
        .rename(columns={"date": "time_t"})
    )
    steps = steps.merge(stage, on=["sand_cay_id", "time_t"], how="left", validate="many_to_one")
    steps["stratum"] = np.where(steps["development_stage"].isin(BARE_STAGES), "bare", np.where(steps["development_stage"].isin(VEGETATED_STAGES), "vegetated", "other"))
    return steps.loc[steps["stratum"].ne("other")].copy()


def episodicity_tests(steps: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rows, cay_rows = [], []
    for stratum in ("bare", "vegetated"):
        subset = steps.loc[steps["stratum"].eq(stratum)]
        for (cay_id, frame_id), group in subset.groupby(["sand_cay_id", "reference_frame_id"], dropna=False):
            if len(group) < 5:
                continue
            share = group["centroid_shift_m"].max() / group["centroid_shift_m"].sum()
            cay_rows.append({"stratum": stratum, "sand_cay_id": cay_id, "reference_frame_id": frame_id, "n_steps": len(group), "largest_step_share": share})
        frame = pd.DataFrame([row for row in cay_rows if row["stratum"] == stratum])
        if len(frame) >= 8:
            rows.append({
                "analysis": f"episodicity_largest_step_{stratum}", "stratum": stratum,
                "estimate": float(frame["largest_step_share"].median()),
                "p_value": np.nan, "n_cays": int(len(frame)),
                "evidence_role": "descriptive concentration of observed displacement; no event-time inference",
            })
    return rows, pd.DataFrame(cay_rows)


def large_step_locking(steps: pd.DataFrame) -> list[dict]:
    bare = steps.loc[steps["stratum"].eq("bare")].copy()
    threshold = bare["speed_norm"].quantile(0.90)
    bare["large"] = bare["speed_norm"].ge(threshold).astype(int)
    angles = 2 * np.pi * bare["midpoint"].dt.dayofyear / 365.2425
    top = bare.loc[bare["large"].eq(1)]
    vector = np.mean(np.exp(1j * angles.loc[top.index].to_numpy()))
    rayleigh_z = len(top) * abs(vector) ** 2
    rayleigh_p = float(np.exp(-rayleigh_z))
    wind_column = "typhoon_max_nearest_wind_m_s"
    rows = [{
        "analysis": "large_step_doy_rayleigh_bare", "stratum": "bare",
        "estimate": float(abs(vector)), "smooth_null": np.nan, "p_value": rayleigh_p,
        "n_cays": int(len(top)), "mean_day_of_year": float((np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * 365.2425),
    }]
    if wind_column in bare.columns:
        with_wind = bare.dropna(subset=[wind_column])
        large = with_wind.loc[with_wind["large"].eq(1), wind_column]
        rest = with_wind.loc[with_wind["large"].eq(0), wind_column]
        if len(large) >= 8 and len(rest) >= 8:
            stat = mannwhitneyu(large, rest, alternative="two-sided")
            rows.append({
                "analysis": "large_step_typhoon_wind_bare", "stratum": "bare",
                "estimate": float(large.median() - rest.median()), "smooth_null": float(rest.median()),
                "p_value": float(stat.pvalue), "n_cays": int(len(large)),
            })
    return rows


COLORS = {"bare": "#D55E00", "vegetated": "#0072B2"}


def save_figure(fig, target: Path) -> None:
    for suffix, dpi in [(".png", 360), (".tiff", 600), (".pdf", None), (".svg", None)]:
        kwargs = {"bbox_inches": "tight"}
        if dpi:
            kwargs["dpi"] = dpi
            if suffix == ".tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(target.with_suffix(suffix), **kwargs)
    plt.close(fig)


def panel_cycle(ax, positions: pd.DataFrame, title: str) -> None:
    doy = np.linspace(1, 365.2425, 120)
    phase = 2 * np.pi * doy / 365.2425
    for stratum in ("bare", "vegetated"):
        subset = positions.loc[positions["stratum"].eq(stratum)]
        if subset["sand_cay_id"].nunique() < 5:
            continue
        points = {"east": [], "north": []}
        for component, key in [("east_dm", "east"), ("north_dm", "north")]:
            fit = smf.ols(f"{component} ~ sa + ca + sa2 + ca2 + C(sand_cay_id)", data=subset).fit()
            points[key] = fit.params["sa"] * np.sin(phase) + fit.params["ca"] * np.cos(phase) + fit.params["sa2"] * np.sin(2 * phase) + fit.params["ca2"] * np.cos(2 * phase)
        ax.plot(points["east"], points["north"], color=COLORS[stratum], lw=1.2, label=f"{stratum} fitted cycle")
        ax.scatter([points["east"][0]], [points["north"][0]], s=10, color=COLORS[stratum], marker="o")
    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.axvline(0, color="#999999", lw=0.6, ls=":")
    ax.set_xlabel("East deviation (m)")
    ax.set_ylabel("North deviation (m)")
    ax.legend(fontsize=5.8, loc="upper left")
    ax.set_title(title, loc="left", fontsize=8)


def panel_tail(ax, steps: pd.DataFrame, title: str) -> None:
    for stratum in ("bare", "vegetated"):
        values = np.sort(steps.loc[steps["stratum"].eq(stratum), "speed_norm"].to_numpy())
        cdf = 1 - np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, color=COLORS[stratum], lw=1.1, label=stratum)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized centroid speed per year")
    ax.set_ylabel("Complementary CDF")
    ax.legend(fontsize=5.8, loc="lower left")
    ax.set_title(title, loc="left", fontsize=8)


def panel_pooled_season(ax, steps: pd.DataFrame, title: str) -> None:
    bare = steps.loc[steps["stratum"].eq("bare")].copy()
    threshold = bare["speed_norm"].quantile(0.90)
    bins = np.linspace(0, 365.2425, 13)
    all_frac, _ = np.histogram(bare["midpoint"].dt.dayofyear, bins=bins)
    top_frac, _ = np.histogram(bare.loc[bare["speed_norm"].ge(threshold), "midpoint"].dt.dayofyear, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    all_frac = all_frac / max(all_frac.sum(), 1)
    top_frac = top_frac / max(top_frac.sum(), 1)
    ax.bar(centers - 7, all_frac, width=13, color="#BBBBBB", label="all bare steps")
    ax.bar(centers + 7, top_frac, width=13, color=COLORS["bare"], label="top-decile steps")
    ax.set_xlabel("Day of year of interval midpoint")
    ax.set_ylabel("Fraction of steps")
    ax.legend(fontsize=5.6, loc="upper left")
    ax.set_title(title, loc="left", fontsize=8)


def make_figure(positions: pd.DataFrame, steps: pd.DataFrame, harmonic_rows: pd.DataFrame, episodic_rows: pd.DataFrame, target: Path) -> None:
    apply_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5))
    panel_cycle(axes[0], positions, "a  Fitted seasonal centroid cycle")
    panel_tail(axes[1], steps, "b  Step-size tail")
    panel_pooled_season(axes[2], steps, "c  Seasonal timing of large steps")
    save_figure(fig, target)


def make_main_figure(positions: pd.DataFrame, steps: pd.DataFrame, observations: pd.DataFrame, target: Path) -> None:
    """Main-text Figure 6: annual cycle, tail, storm-day association, and regional timing."""
    from 分析裸沙质心轨迹风浪对照 import build_usable, fit_forcing_models, load_forcing_inputs, panel_storm, panel_regional

    apply_figure_style()
    fig = plt.figure(figsize=(7.3, 5.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, .88], hspace=.48, wspace=.48, left=.085, right=.985, top=.95, bottom=.09)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_cycle(axes[0], positions, "a  Fitted seasonal centroid cycle")
    panel_tail(axes[1], steps, "b  Step-size tail")
    forcing, daily = load_forcing_inputs()
    usable, _ = build_usable(observations, steps, forcing, daily)
    models = fit_forcing_models(usable)
    panel_storm(axes[2], usable, models, "c  High-wave-day association with step speed")
    regional_path = (PROJECT_ROOT / "outputs/时间结构") / "裸沙质心轨迹风浪季节相位分区域.csv"
    regional_phase = pd.read_csv(regional_path)
    regional_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, region in zip(regional_axes, ["NH", "GBR", "MEDF"]):
        panel_regional(ax, regional_phase, region)
    regional_axes[0].set_title("d  Regional seasonality of large steps", loc="left", fontsize=8)
    regional_axes[0].text(.98, .95, "bars: share of regional top-decile steps\nblue line: regional monthly mean Hs", transform=regional_axes[0].transAxes, ha="right", va="top", fontsize=5.4)
    regional_axes[1].set_xlabel("Month")
    save_figure(fig, target)


def main() -> None:
    output_dir = (PROJECT_ROOT / "outputs/时间结构")
    figure_dir = (PROJECT_ROOT / "outputs/figures/中间图")
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    observations = load_observations()
    positions = position_frame(observations)
    harmonic_rows = harmonic_tests(positions)
    ls_summary, ls_detail = periodogram_tests(positions)
    steps = build_steps(observations)
    episodic_rows, episodic_detail = episodicity_tests(steps)
    locking_rows = large_step_locking(steps)

    family = pd.DataFrame(harmonic_rows + ls_summary + episodic_rows + locking_rows)
    tested = family.loc[family["p_value"].notna()]
    family.loc[tested.index, "fdr_q_value"] = multipletests(tested["p_value"], method="fdr_bh")[1]

    family.to_csv(output_dir / "裸沙质心轨迹周期与突发检验.csv", index=False, encoding="utf-8-sig")
    ls_detail.to_csv(output_dir / "裸沙质心轨迹周期谱明细.csv", index=False, encoding="utf-8-sig")
    episodic_detail.to_csv(output_dir / "裸沙质心轨迹步长突发明细.csv", index=False, encoding="utf-8-sig")
    audit = {
        "question": "Do cay centroid trajectories contain annual position structure and intervals of large displacement?",
        "position_unit": "metres, demeaned within cay x reference frame",
        "step_definition": "consecutive Sentinel-2 A/B observations 30-400 d apart, from outputs/沙洲变化区间.csv",
        "null_models": "reef-clustered harmonic F tests; paired-axis maximum Lomb-Scargle date-exchangeability screening",
        "ls_permutations": LS_PERMUTATIONS,
        "ls_scope": "screening assumes exchangeable dates; temporal autocorrelation is not preserved; no binomial aggregate test",
        "largest_step_scope": "descriptive only; interval observations do not resolve the duration of geomorphic events",
        "fdr_family_size": int(len(tested)),
        "n_positions": int(len(positions)), "n_steps": int(len(steps)),
    }
    (output_dir / "裸沙质心轨迹周期与突发核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(positions, steps, family, pd.DataFrame(episodic_rows), figure_dir / "figure_r16_bare_centroid_periodicity")
    make_main_figure(positions, steps, observations, ROOT / "outputs" / "figures" / "figure_6_trajectory_dual_structure")
    print(family.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
