"""Coordinate-mapped regional area and reshaping analysis.

Area is descriptive at cay x sensor level. Inference uses only Sentinel-2
90--365 d transitions, reduced to one median value per cay before modelling.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from 区域气候滞后模型 import REGION_COLORS, attach_geography, build_geographic_mapping  # noqa: E402

CORE_DAYS = (90, 365)
PRIMARY_REGIONS = ("Great Barrier Reef", "Maldives")
OUTCOMES = ("gross_mobility_fraction_per_year", "centroid_norm_shift_per_year")

def load_data():
    observations = pd.read_csv(ROOT / "outputs" / "沙洲观测主表.csv", low_memory=False)
    trajectories = pd.read_csv(ROOT / "outputs" / "沙洲面积主轨迹分类.csv")
    intervals = pd.read_csv(ROOT / "outputs" / "沙洲变化区间.csv", low_memory=False)
    reef_ids = pd.concat([observations.reef_id, trajectories.reef_id, intervals.reef_id])
    mapping = build_geographic_mapping(reef_ids, observations)
    return (attach_geography(observations, mapping), attach_geography(trajectories, mapping),
            attach_geography(intervals, mapping), mapping)

def area_structure(observations):
    work = observations.loc[observations.sensor.isin(["sentinel2", "google_earth"]) & observations.quality_grade.isin(["A", "B"]) & observations.sand_cay_area_m2.gt(0)]
    return work.groupby(["sensor", "sand_cay_id", "reef_id", "map_region", "analysis_region"], observed=True).agg(
        median_area_m2=("sand_cay_area_m2", "median"), n_observations=("sand_cay_area_m2", "size")
    ).reset_index()

def cay_level_core(intervals):
    work = intervals.copy()
    for column in ("time_interval_days", "area_t_m2", "gross_boundary_change_m2", "centroid_shift_m"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.loc[work.sensor.eq("sentinel2") & work.quality_grade_t.isin(["A", "B"]) & work.quality_grade_t1.isin(["A", "B"]) & work.time_interval_days.between(*CORE_DAYS) & work.area_t_m2.gt(0)].copy()
    years = work.time_interval_days / 365.2425
    work["gross_mobility_fraction_per_year"] = work.gross_boundary_change_m2 / work.area_t_m2 / years
    work["centroid_norm_shift_per_year"] = work.centroid_shift_m / np.sqrt(work.area_t_m2 / np.pi) / years
    return work.groupby(["sand_cay_id", "reef_id", "map_region", "analysis_region"], observed=True).agg(
        n_intervals=("transition_id", "size"), median_area_m2=("area_t_m2", "median"),
        median_interval_days=("time_interval_days", "median"),
        gross_mobility_fraction_per_year=("gross_mobility_fraction_per_year", "median"),
        centroid_norm_shift_per_year=("centroid_norm_shift_per_year", "median")
    ).reset_index()

def reef_bootstrap_difference(frame, outcome, reps=2000):
    data = frame.loc[frame.analysis_region.isin(PRIMARY_REGIONS)].copy(); reefs = sorted(data.reef_id.unique())
    rng = np.random.default_rng(20260917); values = []
    for _ in range(reps):
        sampled = pd.concat([data.loc[data.reef_id.eq(reef)] for reef in rng.choice(reefs, len(reefs), replace=True)], ignore_index=True)
        med = sampled.groupby("analysis_region")[outcome].median()
        if set(PRIMARY_REGIONS).issubset(med.index): values.append(float(med[PRIMARY_REGIONS[1]] - med[PRIMARY_REGIONS[0]]))
    return {"bootstrap_reps": reps, "reef_bootstrap_ci95_low": float(np.percentile(values, 2.5)), "reef_bootstrap_ci95_high": float(np.percentile(values, 97.5))}

def primary_models(cays):
    rows = []; data = cays.loc[cays.analysis_region.isin(PRIMARY_REGIONS)].copy()
    data["maldives"] = data.analysis_region.eq("Maldives").astype(int); data["log_median_area"] = np.log(data.median_area_m2)
    for outcome in OUTCOMES:
        work = data.dropna(subset=[outcome, "log_median_area", "median_interval_days"])
        if work.analysis_region.nunique() != 2: rows.append({"outcome": outcome, "status": "insufficient_regions"}); continue
        fit = sm.OLS(work[outcome], sm.add_constant(work[["maldives", "log_median_area", "median_interval_days"]])).fit(cov_type="cluster", cov_kwds={"groups": work.reef_id, "use_correction": True})
        low, high = fit.conf_int().loc["maldives"].astype(float)
        row = {"outcome": outcome, "status": "ok", "contrast": "Maldives - Great Barrier Reef", "n_cays": int(len(work)), "n_reefs": int(work.reef_id.nunique()), "n_gbr_cays": int(work.loc[work.maldives.eq(0), "sand_cay_id"].nunique()), "n_maldives_cays": int(work.loc[work.maldives.eq(1), "sand_cay_id"].nunique()), "coefficient": float(fit.params.maldives), "reef_cluster_se": float(fit.bse.maldives), "ci95_low": low, "ci95_high": high, "two_sided_p_value": float(fit.pvalues.maldives)}
        row.update(reef_bootstrap_difference(work, outcome)); rows.append(row)
    return pd.DataFrame(rows)

def main():
    observations, trajectories, intervals, mapping = load_data(); area = area_structure(observations); core = cay_level_core(intervals); models = primary_models(core)
    out = (PROJECT_ROOT / "outputs/气候检验"); out.mkdir(exist_ok=True)
    mapping.to_csv(out / "区域地理映射.csv", index=False, encoding="utf-8-sig")
    area.to_csv(out / "figure_1_cay_area_structure.csv", index=False, encoding="utf-8-sig")
    core.to_csv(out / "区域沙洲变化区间汇总.csv", index=False, encoding="utf-8-sig")
    models.to_csv(out / "区域沙洲变化检验.csv", index=False, encoding="utf-8-sig")
    eligible = trajectories.loc[trajectories.eligible.astype(str).str.lower().eq("true")]
    audit = {"mapping_contract": "registered standardized coordinates, then registered cay-centre median", "area_unit": "one cay x sensor series, A/B observation median", "inferential_unit": "one cay median across S2 90-365 day intervals", "primary_contrast": "Great Barrier Reef vs Maldives", "n_eligible_trajectories": int(len(eligible)), "model": "OLS adjusted for cay median area and median interval; reef-clustered SE; 2,000 reef resamples", "small_regions": "descriptive only"}
    (out / "区域沙洲变化核查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
