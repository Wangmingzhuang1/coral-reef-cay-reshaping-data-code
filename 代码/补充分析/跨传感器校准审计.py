"""跨传感器校准审计：用同沙洲 ±30 d 的 S2-GE 同期配对估计测量偏移。

产物（可复用）：配对偏移表、校准系数 JSON（后续敏感性脚本直接读取）、
判定结论（GE 能否作为校准后敏感性层）。只读现有数据，不新增下载/标注。
预设判定规则：面积比中位偏离 1 不超过 10%、质心偏移中位不超过 0.25 等效半径、
且偏移对沙洲尺度的依赖斜率不显著 → 允许校准后敏感性层；否则仅描述层。
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

HERE = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from 区域气候滞后模型 import REGION_COLORS, apply_figure_style, region_code  # noqa: E402

MAX_PAIR_DAYS = 30
AREA_RATIO_TOL = 0.10
CENTROID_TOL_EQUIV_RADII = 0.25
RANDOM_SEED = 20260916


def load_pairs() -> pd.DataFrame:
    obs = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    numeric = ["sand_cay_area_m2", "sand_cay_perimeter_m", "sand_cay_relative_x", "sand_cay_relative_y", "pixel_size_m"]
    for column in numeric:
        obs[column] = pd.to_numeric(obs[column], errors="coerce")
    obs = obs.loc[
        obs["quality_grade"].isin(["A", "B"])
        & obs["sand_cay_area_m2"].gt(0)
        & obs["date"].notna()
    ].copy()
    s2 = obs[obs["sensor"].eq("sentinel2")]
    ge = obs[obs["sensor"].eq("google_earth")]
    pairs = s2.merge(ge, on=["reef_id", "sand_cay_id"], suffixes=("_s2", "_ge"))
    pairs["delta_days"] = (pairs["date_s2"] - pairs["date_ge"]).dt.days.abs()
    pairs = pairs[pairs["delta_days"].le(MAX_PAIR_DAYS)].copy()
    pairs["area_ratio"] = pairs["sand_cay_area_m2_s2"] / pairs["sand_cay_area_m2_ge"]
    pairs["perimeter_ratio"] = pairs["sand_cay_perimeter_m_s2"] / pairs["sand_cay_perimeter_m_ge"]
    pairs["equiv_radius_ge_m"] = np.sqrt(pairs["sand_cay_area_m2_ge"] / np.pi)
    dx = pairs["sand_cay_relative_x_s2"] - pairs["sand_cay_relative_x_ge"]
    dy = pairs["sand_cay_relative_y_s2"] - pairs["sand_cay_relative_y_ge"]
    pairs["centroid_offset_m"] = np.hypot(dx, dy)
    pairs["centroid_offset_equiv_radii"] = pairs["centroid_offset_m"] / pairs["equiv_radius_ge_m"]
    pairs["region"] = region_code(pairs["reef_id"])
    return pairs


def cay_bootstrap(values: pd.Series, groups: pd.Series, reps: int = 999):
    unique = list(groups.unique())
    if len(unique) < 5:
        return np.nan, np.nan
    lookup = {key: group.dropna().to_numpy() for key, group in values.groupby(groups)}
    rng = np.random.default_rng(RANDOM_SEED)
    medians = np.empty(reps)
    for index in range(reps):
        pick = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([lookup[key] for key in pick if len(lookup.get(key, []))])
        medians[index] = np.median(pooled) if len(pooled) else np.nan
    return float(np.nanpercentile(medians, 2.5)), float(np.nanpercentile(medians, 97.5))


def main() -> None:
    pairs = load_pairs()
    out = (PROJECT_ROOT / "outputs/气候检验")
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "跨传感器配对偏移.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for label, sub in [("ALL", pairs)] + [
        (region, group) for region, group in pairs.groupby("region", observed=True)
    ]:
        area_low, area_high = cay_bootstrap(sub["area_ratio"], sub["sand_cay_id"])
        cent_low, cent_high = cay_bootstrap(
            sub["centroid_offset_equiv_radii"], sub["sand_cay_id"]
        )
        summary_rows.append(
            {
                "stratum": label,
                "n_pairs": int(len(sub)),
                "n_cays": int(sub["sand_cay_id"].nunique()),
                "median_area_ratio": float(sub["area_ratio"].median()),
                "area_ratio_ci95_low": area_low,
                "area_ratio_ci95_high": area_high,
                "median_perimeter_ratio": float(sub["perimeter_ratio"].median()),
                "median_centroid_offset_equiv_radii": float(
                    sub["centroid_offset_equiv_radii"].median()
                ),
                "centroid_offset_ci95_low": cent_low,
                "centroid_offset_ci95_high": cent_high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "跨传感器校准汇总.csv", index=False, encoding="utf-8-sig")

    work = pairs.loc[
        pairs["area_ratio"].gt(0) & pairs["sand_cay_area_m2_ge"].gt(0)
    ].copy()
    work["log_ratio"] = np.log(work["area_ratio"])
    work["log_area_ge"] = np.log(work["sand_cay_area_m2_ge"])
    grouped = work.groupby("sand_cay_id")
    y = work["log_ratio"] - grouped["log_ratio"].transform("mean")
    x = work["log_area_ge"] - grouped["log_area_ge"].transform("mean")
    informative = x.abs().gt(0)
    fitted = sm.OLS(y[informative], sm.add_constant(x[informative])).fit(
        cov_type="cluster",
        cov_kwds={"groups": work.loc[informative, "sand_cay_id"], "use_correction": True},
    )
    slope = float(fitted.params["log_area_ge"])
    slope_p = float(fitted.pvalues["log_area_ge"])

    overall = summary[summary["stratum"].eq("ALL")].iloc[0]
    verdict_ok = (
        abs(overall["median_area_ratio"] - 1.0) <= AREA_RATIO_TOL
        and overall["median_centroid_offset_equiv_radii"] <= CENTROID_TOL_EQUIV_RADII
        and slope_p >= 0.05
    )
    audit = {
        "pair_rule": f"same cay, |date difference| <= {MAX_PAIR_DAYS} d, both quality A/B",
        "verdict_rule": (
            f"|median area ratio - 1| <= {AREA_RATIO_TOL} and median centroid offset "
            f"<= {CENTROID_TOL_EQUIV_RADII} equivalent radii and scale-dependence slope p >= 0.05"
        ),
        "verdict": "calibrated_sensitivity_allowed" if verdict_ok else "descriptive_only",
        "scale_dependence_slope_log_ratio_per_log_area": slope,
        "scale_dependence_slope_p_value": slope_p,
        "calibration_coefficients": {
            row["stratum"]: {
                "area_ratio_median": row["median_area_ratio"],
                "perimeter_ratio_median": row["median_perimeter_ratio"],
                "centroid_offset_equiv_radii_median": row["median_centroid_offset_equiv_radii"],
            }
            for _, row in summary.iterrows()
        },
        "summary": summary.to_dict("records"),
        "reuse_note": "后续敏感性脚本读取 跨传感器校准审计.json 的 calibration_coefficients 对 GE 指标做乘性校正；判定为 descriptive_only 时禁止入模型。",
    }
    (out / "跨传感器校准审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    apply_figure_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    ax = axes[0]
    for region, group in pairs.groupby("region", observed=True):
        ax.scatter(
            group["sand_cay_area_m2_ge"],
            group["area_ratio"],
            s=16,
            color=REGION_COLORS.get(region, "#666666"),
            label=region,
        )
    ax.axhline(1.0, color="#777777", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("GE cay area (m2)")
    ax.set_ylabel("S2 / GE area ratio")
    ax.set_title("Area ratio vs cay size")
    ax.legend(frameon=False, title="Region")
    ax = axes[1]
    for region, group in pairs.groupby("region", observed=True):
        ax.scatter(
            group["sand_cay_area_m2_ge"],
            group["centroid_offset_equiv_radii"],
            s=16,
            color=REGION_COLORS.get(region, "#666666"),
            label=region,
        )
    ax.axhline(CENTROID_TOL_EQUIV_RADII, color="#777777", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("GE cay area (m2)")
    ax.set_ylabel("Centroid offset (GE equivalent radii)")
    ax.set_title("Centroid offset vs cay size")
    stem = (PROJECT_ROOT / "outputs/figures/中间图") / "figure_r6_cross_sensor_calibration"
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
