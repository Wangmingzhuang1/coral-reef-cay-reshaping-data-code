"""区域分层 ENSO 滞后响应：复用主文 ENSO 样本表与公共滞后估计合同。

区域模型门槛：≥30 区间且 ≥10 沙洲；不足门槛的区域只报描述统计。
区域×滞后差异用去均值模型内交互项的联合 Wald F 检验（参考区 GBR）。
结果为观测关联，不作因果解释。
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
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from 区域气候滞后模型 import (  # noqa: E402
    LAGS_MONTHS,
    describe_region_outcome,
    attach_geography,
    build_geographic_mapping,
    fit_lag_model,
    fit_region_interaction,
    fit_region_lag_models,
    plot_region_lag_models,
)


def plot_climate_figure(total, interaction_terms, dhw, stem):
    from 区域气候滞后模型 import apply_figure_style
    apply_figure_style(); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 132 / 25.4), constrained_layout=True)
    for ax, outcome, title in [(axes[0,0], "annualized_log_area_change", "a  ENSO and area change"), (axes[0,1], "log_gross_mobility", "b  ENSO and boundary mobility")]:
        part = total.loc[(total.outcome.eq(outcome)) & total.status.eq("ok")].sort_values("lag_months")
        coef = part.coefficient.to_numpy(); ax.errorbar(part.lag_months, coef, yerr=[coef-part.two_way_ci95_low.to_numpy(), part.two_way_ci95_high.to_numpy()-coef], marker="o",color="#333333",capsize=2)
        ax.axhline(0,color="#777777",lw=.65,ls="--"); ax.set_title(title,loc="left",fontweight="bold",fontsize=7.8); ax.set_xlabel("Lag (months)",fontsize=6.6); ax.set_ylabel("Within-cay coefficient (two-way 95% CI)",fontsize=6.2); ax.tick_params(labelsize=6)
    ax=axes[1,0]; part=interaction_terms.loc[interaction_terms.outcome.eq("annualized_log_area_change")].sort_values("lag_months") if not interaction_terms.empty else pd.DataFrame()
    if len(part):
        coef=part.coefficient.to_numpy(); ax.errorbar(part.lag_months,coef,yerr=[coef-part.two_way_ci95_low.to_numpy(),part.two_way_ci95_high.to_numpy()-coef],marker="o",color="#D95F02",capsize=2); ax.axhline(0,color="#777",lw=.7,ls="--")
    ax.set_title("c  GBR--Maldives ENSO interaction",loc="left",fontweight="bold",fontsize=7.8); ax.set_xlabel("Lag (months)",fontsize=6.6); ax.set_ylabel("Interaction coefficient (two-way 95% CI)",fontsize=6.2); ax.tick_params(labelsize=6)
    ax=axes[1,1]; part=dhw.loc[dhw.analysis.eq("DHW threshold event") & dhw.region.eq("GBR") & dhw.scope.eq("reef_area_fraction_ge8_0.50")].copy(); part=part.loc[part.lag_label.isin(["0-<1 year","1-<2 years","2-<3 years"])]
    x=np.arange(len(part)); coef=part.coefficient_log_area_ratio.to_numpy(); ax.errorbar(x,coef,yerr=[coef-part.ci95_low_cay_cluster.to_numpy(),part.ci95_high_cay_cluster.to_numpy()-coef],marker="o",color="#386CB0",capsize=2); ax.axhline(0,color="#777",lw=.65,ls="--"); ax.set_xticks(x); ax.set_xticklabels(["0--1 y","1--2 y","2--3 y"],fontsize=6); ax.set_title("d  GBR DHW ≥8 response",loc="left",fontweight="bold",fontsize=7.8); ax.set_ylabel("Log area ratio (cay-cluster 95% CI)",fontsize=6.2); ax.tick_params(labelsize=6); ax.text(.98,.04,"Other regions: not estimable",transform=ax.transAxes,ha="right",fontsize=5.3)
    stem.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"): fig.savefig(stem.with_suffix(suffix), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression":"tiff_lzw"}); plt.close(fig)


def main() -> None:
    sample = pd.read_csv(
        ROOT / "outputs" / "ENSO滞后分析样本.csv", encoding="utf-8-sig", low_memory=False
    )
    observations = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    mapping = build_geographic_mapping(pd.concat([sample["reef_id"], observations["reef_id"]]), observations)
    sample = attach_geography(sample, mapping)
    sample["region"] = sample["analysis_region"]
    outcomes = {
        "annualized_log_area_change": sample["annualized_log_area_change"].notna(),
        "log_gross_mobility": sample["log_gross_mobility"].notna(),
    }
    models = fit_region_lag_models(sample, outcomes, "nino34_lag_{lag}m")
    total_rows: list[dict] = []
    for outcome, valid in outcomes.items():
        for lag in LAGS_MONTHS:
            row = fit_lag_model(sample.loc[valid], outcome, f"nino34_lag_{lag}m")
            row["lag_months"] = lag
            total_rows.append(row)
    total = pd.DataFrame(total_rows)
    total["two_way_fdr_q_value"] = np.nan
    for _, index in total.loc[total.status.eq("ok")].groupby("outcome").groups.items():
        total.loc[index, "two_way_fdr_q_value"] = multipletests(total.loc[index, "two_way_p_value"], method="fdr_bh")[1]
    total["fdr_q_value"] = total["two_way_fdr_q_value"]
    interaction_rows: list[dict] = []
    interaction_term_rows: list[dict] = []
    for outcome, valid in outcomes.items():
        for lag in LAGS_MONTHS:
            row = fit_region_interaction(
                sample.loc[valid], outcome, f"nino34_lag_{lag}m"
            )
            terms = row.pop("_term_rows", [])
            row["lag_months"] = lag
            for term in terms:
                term["lag_months"] = lag
                interaction_term_rows.append(term)
            interaction_rows.append(row)
    interaction = pd.DataFrame(interaction_rows)
    interaction["fdr_q_value"] = np.nan
    ok = interaction["status"].eq("ok")
    if ok.any():
        interaction.loc[ok, "fdr_q_value"] = multipletests(
            interaction.loc[ok, "two_way_wald_p_value"], method="fdr_bh"
        )[1]
    interaction["two_way_fdr_q_value"] = interaction["fdr_q_value"]
    interaction_terms = pd.DataFrame(interaction_term_rows)
    if not interaction_terms.empty:
        interaction_terms["two_way_fdr_q_value"] = np.nan
        for _, index in interaction_terms.groupby("outcome").groups.items():
            interaction_terms.loc[index, "two_way_fdr_q_value"] = multipletests(interaction_terms.loc[index, "two_way_p_value"], method="fdr_bh")[1]
    descriptive_rows: list[dict] = []
    for region, group in sample.groupby("region", observed=True):
        for outcome in outcomes:
            row = describe_region_outcome(group, outcome)
            row["region"] = region
            descriptive_rows.append(row)
    descriptive = pd.DataFrame(descriptive_rows)

    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "区域ENSO滞后模型.csv", index=False, encoding="utf-8-sig")
    total.to_csv(out / "ENSO总体滞后模型.csv", index=False, encoding="utf-8-sig")
    interaction.to_csv(out / "区域ENSO交互检验.csv", index=False, encoding="utf-8-sig")
    interaction_terms.to_csv(out / "区域ENSO交互项.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(out / "区域ENSO描述统计.csv", index=False, encoding="utf-8-sig")
    dhw = pd.read_csv(out / "礁盘DHW_ENSO面积模型.csv", low_memory=False)
    plot_climate_figure(total, interaction_terms, dhw, (PROJECT_ROOT / "outputs/figures/中间图") / "figure_s2_climate_regional")
    plot_region_lag_models(
        models, (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r2_regional_enso_lags", "Nino 3.4 anomaly (deg C)"
    )
    supported = models.loc[models["status"].eq("ok") & models["fdr_q_value"].lt(0.05)]
    interaction_supported = interaction.loc[
        interaction["status"].eq("ok") & interaction["fdr_q_value"].lt(0.05)
    ]
    region_counts = (
        sample.groupby("region", observed=True)
        .agg(
            n_intervals=("transition_id", "size"),
            n_cays=("sand_cay_id", "nunique"),
            n_reefs=("reef_id", "nunique"),
        )
        .reset_index()
    )
    audit = {
        "index": "NOAA CPC monthly detrended Nino 3.4 SST anomaly",
        "lags_months": list(LAGS_MONTHS),
        "region_gate": ">=30 intervals and >=10 cays per region model",
        "region_sample_counts": region_counts.to_dict("records"),
        "fdr_supported_region_lag_tests": supported.to_dict("records"),
        "fdr_supported_interaction_tests": interaction_supported.to_dict("records"),
        "multiple_testing": "BH-FDR across seven lags within each region-outcome family; interaction family across 7 lags x 2 outcomes",
        "inference": "Within-cay observational lag associations with cay-clustered SE; two-way clustering and leave-one-year-out reported; regional differences require interaction q and per-region MDE; not causal.",
    }
    (out / "区域ENSO滞后核查.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
