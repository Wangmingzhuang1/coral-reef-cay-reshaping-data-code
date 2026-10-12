"""Main-text promotion tests for the bare-sand dual-structure claims.

The archived exploratory result supports three claim fragments only descriptively.
This script supplies the missing formal contrasts, analysed as one family
(FDR-BH over the reported contrasts and regional timing tests):

T1  Seasonal amplitude stratum contrast: annual harmonic x bare interaction,
    east and north components (does bare-phase wobble exceed vegetated-phase?).
T2  Extreme-displacement stratum contrast: cay x stratum largest-step path share
    (primary, reef-clustered bootstrap of the median difference) and step-level
    top-decile fraction (secondary).
T3  Regional seasonal locking: Rayleigh test of top-decile step day-of-year within
    each region (NH, GBR, MEDF, OTHER), replacing the confounded pooled view.

Robustness of the storm-day coefficient (reported, not part of the family):
storm-day threshold p90/p975, interval window 90-365 d, leave-one-region-out.
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
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from 分析裸沙质心轨迹周期与突发 import (  # noqa: E402
    build_steps,
    episodicity_tests,
    position_frame,
)
from 分析裸沙质心轨迹风浪对照 import event_forcing  # noqa: E402
from 分析沙洲形状与质心迁移 import load_observations  # noqa: E402

REGIONS = ("NH", "GBR", "MEDF", "OTHER")
BOOTSTRAP_REPS = 1999
RNG_SEED = 20260918


def region_code(reef_id: str) -> str:
    for region in REGIONS:
        if str(reef_id).startswith(region):
            return region
    return "UNASSIGNED"


def fit_clustered(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["reef_id"], "use_correction": True}
    )


def amplitude_contrast(positions: pd.DataFrame) -> list[dict]:
    rows = []
    positions = positions.copy()
    positions["bare_ind"] = positions["stratum"].eq("bare").astype(int)
    for component in ("east_dm", "north_dm"):
        fit = fit_clustered(
            f"{component} ~ sa + ca + sa2 + ca2 + sa:bare_ind + ca:bare_ind + C(sand_cay_id)",
            positions,
        )
        test = fit.f_test("sa:bare_ind = 0, ca:bare_ind = 0")
        bare_amp = float(np.sqrt(fit.params["sa"] ** 2 + fit.params["ca"] ** 2))
        tot_sa = fit.params["sa"] + fit.params["sa:bare_ind"]
        tot_ca = fit.params["ca"] + fit.params["ca:bare_ind"]
        rows.append({
            "analysis": f"T1_annual_amplitude_contrast_{component}",
            "estimate": float(np.sqrt(tot_sa ** 2 + tot_ca ** 2) - bare_amp),
            "bare_amplitude_m": float(np.sqrt(tot_sa ** 2 + tot_ca ** 2)),
            "vegetated_amplitude_m": bare_amp,
            "f_stat": float(test.fvalue), "p_value": float(test.pvalue),
            "n_observations": int(len(positions)), "n_cays": int(positions["sand_cay_id"].nunique()),
        })
    return rows


def extreme_contrast(steps: pd.DataFrame, episodic_detail: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    reef_of_cay = steps.drop_duplicates("sand_cay_id").set_index("sand_cay_id")["reef_id"]
    detail = episodic_detail.copy()
    detail["reef_id"] = detail["sand_cay_id"].map(reef_of_cay)
    detail = detail.dropna(subset=["reef_id", "largest_step_share"])
    bare = detail.loc[detail["stratum"].eq("bare")]
    vegetated = detail.loc[detail["stratum"].eq("vegetated")]
    observed = float(bare["largest_step_share"].median() - vegetated["largest_step_share"].median())
    threshold = steps["speed_norm"].quantile(0.90)
    steps = steps.copy()
    steps["large"] = steps["speed_norm"].ge(threshold).astype(float)
    bare_steps = steps.loc[steps["stratum"].eq("bare")]
    veg_steps = steps.loc[steps["stratum"].eq("vegetated")]
    observed_fraction = float(bare_steps["large"].mean() - veg_steps["large"].mean())
    # Draw reefs jointly for both strata and retain every sampled copy.
    reefs = steps["reef_id"].unique()
    reef_detail = {reef: detail.loc[detail["reef_id"].eq(reef)] for reef in reefs}
    aggregated = steps.groupby(["reef_id", "stratum"])["large"].agg(["sum", "count"]).unstack("stratum").reindex(reefs).fillna(0)
    large_counts = aggregated["sum"].reindex(columns=["bare", "vegetated"]).to_numpy()
    step_counts = aggregated["count"].reindex(columns=["bare", "vegetated"]).to_numpy()
    diffs, fraction_diffs = [], []
    for _ in range(BOOTSTRAP_REPS):
        selected = rng.integers(0, len(reefs), len(reefs))
        totals = large_counts[selected].sum(axis=0)
        counts = step_counts[selected].sum(axis=0)
        if np.all(counts > 0):
            fraction_diffs.append(float(totals[0] / counts[0] - totals[1] / counts[1]))
        sampled = pd.concat([reef_detail[reefs[index]] for index in selected])
        medians = sampled.groupby("stratum")["largest_step_share"].median()
        if {"bare", "vegetated"}.issubset(medians.index):
            diffs.append(float(medians["bare"] - medians["vegetated"]))

    def summarize_bootstrap(values, estimate):
        values = np.asarray(values)
        low, high = np.quantile(values, [0.025, 0.975])
        # Approximate two-sided percentile-bootstrap tail probability, with finite-replicate correction.
        lower = (1 + np.sum(values <= 0)) / (len(values) + 1)
        upper = (1 + np.sum(values >= 0)) / (len(values) + 1)
        return float(low), float(high), float(min(1.0, 2 * min(lower, upper)))

    rows = []
    for analysis, estimate, values, b, v in [
        ("T2_largest_step_share_bare_minus_vegetated", observed, diffs,
         bare["largest_step_share"].median(), vegetated["largest_step_share"].median()),
        ("T2_top_decile_fraction_bare_minus_vegetated", observed_fraction, fraction_diffs,
         bare_steps["large"].mean(), veg_steps["large"].mean()),
    ]:
        low, high, p_value = summarize_bootstrap(values, estimate)
        rows.append({
            "analysis": analysis, "estimate": estimate,
            "ci95_low": low, "ci95_high": high, "p_value": p_value,
            "bare_median": float(b), "vegetated_median": float(v),
            "n_bare_cays": int(bare["sand_cay_id"].nunique()) if "largest" in analysis else int(bare_steps["sand_cay_id"].nunique()),
            "n_vegetated_cays": int(vegetated["sand_cay_id"].nunique()) if "largest" in analysis else int(veg_steps["sand_cay_id"].nunique()),
            "bootstrap_unit": "reef, jointly sampled across strata with multiplicity",
            "bootstrap_reps": len(values), "n_reefs": len(reefs),
            "threshold_scope": "fixed pooled observed p90 for the top-decile contrast",
        })
    return rows


WINDOWS = {"NH": (12, 1, 2), "GBR": (12, 1, 2, 3), "MEDF": (12, 1, 2, 3, 4), "OTHER": (12, 1, 2)}


def window_binomial(steps: pd.DataFrame) -> list[dict]:
    """T4: do top-decile steps concentrate in the regional storm window more than
    the base rate of all bare steps in that window (controls sampling season)?"""
    from scipy.stats import binomtest

    bare = steps.loc[steps["stratum"].eq("bare")].copy()
    threshold = bare["speed_norm"].quantile(0.90)
    bare["large"] = bare["speed_norm"].ge(threshold)
    bare["month"] = bare["midpoint"].dt.month
    bare["region"] = bare["reef_id"].map(region_code)
    rows = []
    for region, window in WINDOWS.items():
        subset = bare.loc[bare["region"].eq(region)]
        in_window = subset["month"].isin(window)
        n_top = int(subset["large"].sum())
        k_top = int(subset.loc[subset["large"], "month"].isin(window).sum())
        p0 = float(in_window.mean())
        if n_top < 8:
            rows.append({"analysis": f"T4_window_binomial_{region}", "status": "insufficient_steps"})
            continue
        test = binomtest(k_top, n_top, p0, alternative="greater")
        rows.append({
            "analysis": f"T4_window_binomial_{region}", "status": "ok",
            "estimate": k_top / n_top, "base_rate": p0, "p_value": float(test.pvalue),
            "n_top_steps": n_top, "k_in_window": k_top,
        })
    return rows


def regional_locking(steps: pd.DataFrame) -> list[dict]:
    bare = steps.loc[steps["stratum"].eq("bare")].copy()
    threshold = bare["speed_norm"].quantile(0.90)
    bare["large"] = bare["speed_norm"].ge(threshold)
    bare["region"] = bare["reef_id"].map(region_code)
    rows = []
    for region in REGIONS:
        subset = bare.loc[bare["region"].eq(region) & bare["large"]]
        if len(subset) < 8:
            rows.append({"analysis": f"T3_rayleigh_{region}", "status": "insufficient_steps"})
            continue
        angles = 2 * np.pi * subset["midpoint"].dt.dayofyear / 365.2425
        vector = np.mean(np.exp(1j * angles.to_numpy()))
        z_value = len(subset) * abs(vector) ** 2
        rows.append({
            "analysis": f"T3_rayleigh_{region}", "status": "ok",
            "estimate": float(abs(vector)), "p_value": float(np.exp(-z_value)),
            "mean_day_of_year": float((np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * 365.2425),
            "n_top_steps": int(len(subset)),
        })
    return rows


def storm_day_robustness(steps: pd.DataFrame, daily: pd.DataFrame) -> list[dict]:
    rows = []
    variants = [
        ("base_p95_30_400d", 0.95, (30, 400), None),
        ("threshold_p90", 0.90, (30, 400), None),
        ("threshold_p975", 0.975, (30, 400), None),
        ("window_90_365d", 0.95, (90, 365), None),
    ]
    for region in REGIONS:
        variants.append((f"leave_out_{region}", 0.95, (30, 400), region))
    for name, quantile, window, dropped in variants:
        subset = steps.loc[steps["time_interval_days"].between(*window) & steps["stratum"].eq("bare")].copy()
        if dropped:
            subset = subset.loc[subset["reef_id"].map(region_code).ne(dropped)]
        events = event_forcing(daily, subset, quantile=quantile)
        subset = subset.merge(events, on="transition_id", how="left", validate="one_to_one").dropna(subset=["event_storm_days"])
        subset["log_speed"] = np.log(subset["speed_norm"] + 0.01)
        if subset["sand_cay_id"].nunique() < 5:
            rows.append({"variant": name, "status": "insufficient_cays"})
            continue
        fit = fit_clustered("log_speed ~ event_storm_days + time_interval_days + C(sand_cay_id)", subset)
        low, high = fit.conf_int().loc["event_storm_days"].astype(float)
        rows.append({
            "variant": name, "status": "ok", "estimate": float(fit.params["event_storm_days"]),
            "ci95_low": low, "ci95_high": high, "p_value": float(fit.pvalues["event_storm_days"]),
            "n_observations": int(len(subset)), "n_cays": int(subset["sand_cay_id"].nunique()),
        })
    return rows


def main() -> None:
    output_dir = (PROJECT_ROOT / "outputs/时间结构")
    observations = load_observations()
    positions = position_frame(observations)
    steps = build_steps(observations)
    _, episodic_detail = episodicity_tests(steps)
    daily = pd.read_csv(ROOT / "data" / "environment" / "harmonized" / "daily" / "reef_daily_features.csv", low_memory=False)
    daily["time"] = pd.to_datetime(daily["time"], format="ISO8601")

    rows = amplitude_contrast(positions) + extreme_contrast(steps, episodic_detail) + regional_locking(steps) + window_binomial(steps)
    family = pd.DataFrame(rows)
    tested = family.loc[family["p_value"].notna()]
    family.loc[tested.index, "fdr_q_value"] = multipletests(tested["p_value"], method="fdr_bh")[1]
    robustness = pd.DataFrame(storm_day_robustness(steps, daily))

    family.to_csv(output_dir / "裸沙主文升级检验.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(output_dir / "裸沙主文升级稳健性.csv", index=False, encoding="utf-8-sig")
    audit = {
        "purpose": "Formal contrasts required before promoting the dual-structure claims to the main text.",
        "family": f"{len(tested)} tests, FDR-BH; robustness variants reported separately.",
        "bootstrap": "1999 joint reef-cluster draws retaining multiplicity; percentile CI; approximate two-sided percentile-bootstrap tail p values; pooled p90 held fixed",
        "language_rule": "Observational design supports association/statistical encoding, not causal determination.",
    }
    (output_dir / "裸沙主文升级检验核验.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(family.to_string(index=False))
    print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
