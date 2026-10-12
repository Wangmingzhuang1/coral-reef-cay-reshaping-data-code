"""区域分层的方向耦合：镜像主文图 3 合同（S2、A/B、90-365 d、风浪流共同完整区间）。

每个区域内在沙洲自身长轴/横轴坐标系中估计沿轴海流、沿轴波浪、横向海流、
横向波浪与相应方向位移的关联；沙洲固定效应以沙洲内去均值实现，标准误按
沙洲聚类；四个预设项在区域族内 BH-FDR。结果为观测关联，不作因果解释。
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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from 区域气候滞后模型 import (  # noqa: E402
    MIN_CAYS,
    MIN_INTERVALS,
    REGION_COLORS,
    apply_figure_style,
    attach_geography,
    build_geographic_mapping,
)

CORE_DAYS = (90, 365)
COVARIATES = ("log_interval", "midpoint_year", "season_sin", "season_cos")
AXES = (
    ("along", "current_along_mean", "wave_vector_along_mean", "centroid_along_shift_normalized_per_year"),
    ("cross", "current_cross_mean", "wave_vector_cross_mean", "centroid_cross_shift_normalized_per_year"),
)
TERM_LABELS = {
    ("along", "current"): "Along-axis current",
    ("along", "wave"): "Along-axis wave",
    ("cross", "current"): "Cross-axis current",
    ("cross", "wave"): "Cross-axis wave",
}


def load_sample() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "outputs" / "流场波浪沿轴横轴分解.csv", low_memory=False
    )
    numeric = [
        "time_interval_days",
        "area_t_m2",
        "current_along_mean",
        "current_cross_mean",
        "wave_vector_along_mean",
        "wave_vector_cross_mean",
        "centroid_along_shift_normalized_per_year",
        "centroid_cross_shift_normalized_per_year",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["time_t"] = pd.to_datetime(frame["time_t"], errors="coerce")
    frame["time_t1"] = pd.to_datetime(frame["time_t1"], errors="coerce")
    complete = pd.to_numeric(frame["flow_wave_wind_complete"], errors="coerce")
    frame = frame.loc[
        frame["sensor"].eq("sentinel2")
        & frame["quality_grade_t"].isin(["A", "B"])
        & frame["quality_grade_t1"].isin(["A", "B"])
        & frame["time_interval_days"].between(*CORE_DAYS)
        & frame["area_t_m2"].gt(0)
        & complete.eq(1)
        & frame[["time_t", "time_t1"]].notna().all(axis=1)
    ].copy()
    midpoint = frame["time_t"] + (frame["time_t1"] - frame["time_t"]) / 2
    frame["midpoint_year"] = midpoint.dt.year + (midpoint.dt.dayofyear - 1) / 365.2425
    frame["midpoint_calendar_year"] = midpoint.dt.year.astype(int)
    frame["season_sin"] = np.sin(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["season_cos"] = np.cos(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["log_interval"] = np.log(frame["time_interval_days"])
    observations = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    mapping = build_geographic_mapping(pd.concat([frame["reef_id"], observations["reef_id"]]), observations)
    frame = attach_geography(frame, mapping)
    frame["region"] = frame["analysis_region"]
    return frame


def fit_wave_region_interaction(sample, axis, current, wave, outcome):
    """Test GBR--Maldives heterogeneity directly; separate regional p values are not a test."""
    sub = sample.loc[sample.region.isin(["Great Barrier Reef", "Maldives"])].copy()
    cols = [outcome, current, wave, *COVARIATES]
    sub = sub.loc[sub[cols].notna().all(axis=1)].copy()
    sub["wave_x_maldives"] = sub[wave] * sub.region.eq("Maldives").astype(float)
    xcols = [current, wave, "wave_x_maldives", *COVARIATES]
    demeaned = sub[[outcome,*xcols]] - sub.groupby("sand_cay_id")[[outcome,*xcols]].transform("mean")
    informative=demeaned.abs().sum(axis=1).gt(0); sub=sub.loc[informative]; demeaned=demeaned.loc[informative]
    row={"axis":axis,"term":"Wave x Maldives","reference_region":"Great Barrier Reef","compared_region":"Maldives","n_intervals":int(len(sub)),"n_cays":int(sub.sand_cay_id.nunique()),"n_reefs":int(sub.reef_id.nunique())}
    if len(sub)<MIN_INTERVALS or sub.sand_cay_id.nunique()<MIN_CAYS: row["status"]="insufficient_sample"; return row
    fit=sm.OLS(demeaned[outcome],demeaned[xcols]).fit(cov_type="cluster",cov_kwds={"groups":sub.sand_cay_id,"use_correction":True})
    low,high=fit.conf_int().loc["wave_x_maldives"].astype(float); row.update({"status":"ok","coefficient":float(fit.params.wave_x_maldives),"ci95_low":low,"ci95_high":high,"two_sided_p_value":float(fit.pvalues.wave_x_maldives)})
    return row


def fit_axis_model(sub: pd.DataFrame, axis: str, current: str, wave: str, outcome: str):
    columns = [current, wave, *COVARIATES]
    valid = sub[[outcome, *columns]].notna().all(axis=1)
    work = sub.loc[valid].copy()
    grouped = work.groupby("sand_cay_id")
    demeaned = work[[outcome, *columns]] - grouped[[outcome, *columns]].transform("mean")
    informative = demeaned.abs().sum(axis=1).gt(0)
    work, demeaned = work.loc[informative], demeaned.loc[informative]
    rows = []
    for term, column in (("current", current), ("wave", wave)):
        row = {
            "axis": axis,
            "term": TERM_LABELS[(axis, term)],
            "n_intervals": int(len(work)),
            "n_cays": int(work["sand_cay_id"].nunique()),
        }
        if len(work) < MIN_INTERVALS or work["sand_cay_id"].nunique() < MIN_CAYS:
            row["status"] = "insufficient_sample"
            rows.append(row)
            continue
        fitted = sm.OLS(demeaned[outcome], demeaned[columns]).fit(
            cov_type="cluster",
            cov_kwds={"groups": work["sand_cay_id"], "use_correction": True},
        )
        low, high = fitted.conf_int().loc[column].astype(float)
        row.update(
            {
                "status": "ok",
                "coefficient": float(fitted.params[column]),
                "cluster_robust_se": float(fitted.bse[column]),
                "ci95_low": low,
                "ci95_high": high,
                "two_sided_p_value": float(fitted.pvalues[column]),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    sample = load_sample()
    rows: list[dict] = []
    for label, sub in [("ALL_S2", sample)] + [
        (region, group) for region, group in sample.groupby("region", observed=True)
    ]:
        for axis, current, wave, outcome in AXES:
            for row in fit_axis_model(sub, axis, current, wave, outcome):
                row["region"] = label
                rows.append(row)
    models = pd.DataFrame(rows)
    models["fdr_q_value"] = np.nan
    ok = models["status"].eq("ok")
    if ok.any():
        for _, index in models.loc[ok].groupby("region").groups.items():
            models.loc[index, "fdr_q_value"] = multipletests(
                models.loc[index, "two_sided_p_value"], method="fdr_bh"
            )[1]
    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    models.to_csv(out / "区域方向耦合模型.csv", index=False, encoding="utf-8-sig")
    interaction = pd.DataFrame([fit_wave_region_interaction(sample,*spec) for spec in AXES])
    interaction["fdr_q_value"] = np.nan
    ok_int = interaction.status.eq("ok")
    if ok_int.any(): interaction.loc[ok_int,"fdr_q_value"] = multipletests(interaction.loc[ok_int,"two_sided_p_value"],method="fdr_bh")[1]
    interaction.to_csv(out / "区域方向耦合交互检验.csv", index=False, encoding="utf-8-sig")

    apply_figure_style()
    import matplotlib.pyplot as plt

    fig, axes_plt = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for ax, axis in zip(axes_plt, ("along", "cross")):
        subset = models.loc[models["status"].eq("ok") & models["axis"].eq(axis)]
        terms = [TERM_LABELS[(axis, "current")], TERM_LABELS[(axis, "wave")]]
        positions = {term: index for index, term in enumerate(terms)}
        offsets = {
            region: 0.12 * (index - 1)
            for index, region in enumerate(sorted(subset["region"].unique()))
        }
        for region, group in subset.groupby("region"):
            xs = [positions[term] + offsets[region] for term in group["term"]]
            coef = group["coefficient"].to_numpy()
            ax.errorbar(
                xs,
                coef,
                yerr=[
                    coef - group["ci95_low"].to_numpy(),
                    group["ci95_high"].to_numpy() - coef,
                ],
                marker="o",
                capsize=3,
                linestyle="none",
                color=REGION_COLORS.get(region, "#666666"),
                label=region,
            )
        ax.axhline(0, color="#777777", lw=0.8, ls="--")
        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels([terms[i].replace("-axis", "") for i in range(len(terms))])
        ax.set_xlabel(f"{axis.capitalize()}-axis forcing component")
        ax.set_ylabel("Within-cay coefficient (95% CI)")
        ax.set_title(f"{axis.capitalize()}-axis displacement ~ forcing")
        ax.legend(frameon=False, title="Region")
    stem = (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r4_regional_directional_coupling"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)

    supported = models.loc[models["status"].eq("ok") & models["fdr_q_value"].lt(0.05)]
    audit = {
        "contract": "S2, A/B quality, 90-365 d core intervals, flow-wave-wind complete (mirrors main Figure 3)",
        "model": "within-cay demeaned OLS, cay-clustered robust SE, four pre-specified terms FDR-corrected within each region",
        "sample_counts": sample.groupby("region", observed=True)
        .agg(n_intervals=("transition_id", "size"), n_cays=("sand_cay_id", "nunique"))
        .reset_index()
        .to_dict("records"),
        "fdr_supported_terms": supported.to_dict("records"),
        "inference": "Observational directional association per region; not causal; OTHER/NH strata below gate reported as insufficient_sample.",
    }
    (out / "区域方向耦合核查.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
