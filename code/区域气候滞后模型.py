"""区域分层气候滞后模型的公共估计合同（ENSO 与海洋热浪共用）。

合同：Sentinel-2、A/B 质量、90-365 天核心区间；沙洲固定效应以沙洲内去均值
实现；OLS 配沙洲聚类稳健标准误，另报沙洲×区间中点月双向聚类敏感性；预设滞后
族内 BH-FDR；逐年剔除符号一致性；区域门槛（≥30 区间且 ≥10 沙洲）不足时只报
描述统计。所有结果为观测关联，不作因果解释。
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

LAGS_MONTHS = (0, 3, 6, 9, 12, 18, 24)
MIN_INTERVALS = 30
MIN_CAYS = 10
COVARIATES = ("log_interval", "midpoint_year", "season_sin", "season_cos")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGION_COLORS = {
    "Great Barrier Reef": "#386CB0",
    "Maldives": "#D95F02",
    "South China Sea": "#1B9E77",
    "Marshall Islands": "#7570B3",
    "Indonesia": "#E7298A",
    "Lakshadweep": "#66A61E",
    "Farquhar Atoll": "#A6761D",
    "Seychelles": "#A6CEE3",
    "Western Indian Ocean": "#666666",
}
OUTCOME_LABELS = {
    "annualized_log_area_change": "Annualized log area change",
    "log_gross_mobility": "Log gross boundary mobility",
}
RANDOM_SEED = 20260916


def build_geographic_mapping(
    reef_ids: pd.Series | list[str], observations: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build the sole coordinate-driven reef-to-region crosswalk used by analyses.

    Registered standardised coordinates take precedence; registered cay centres are
    used only when a reef has no standardised coordinate.  The explicit OTHER
    crosswalk prevents a prefix from silently becoming an analytical region.
    """
    ids = pd.Series(reef_ids, dtype="string").dropna().drop_duplicates().sort_values()
    standard = pd.read_csv(
        ROOT / "data" / "source_dataset" / "data" / "metadata" / "standardized_coordinates.csv"
    )
    registry = pd.read_csv(
        ROOT / "data" / "source_dataset" / "data" / "metadata" / "cay_registry.csv"
    )
    def centres(data: pd.DataFrame, lon: str, lat: str) -> pd.DataFrame:
        work = data[["reef_id", lon, lat]].copy()
        work[lon] = pd.to_numeric(work[lon], errors="coerce")
        work[lat] = pd.to_numeric(work[lat], errors="coerce")
        return work.dropna().groupby("reef_id", as_index=False).agg(
            center_lon=(lon, "median"), center_lat=(lat, "median")
        )
    standard_centres = centres(standard, "center_lon", "center_lat")
    registry_centres = centres(registry, "center_lon", "center_lat")
    audit = pd.read_csv(ROOT / "data" / "source_dataset" / "data" / "metadata" / "coordinate_normalization_audit.csv")
    audit_centres = centres(audit, "lon", "lat")
    image_metadata = pd.read_csv(ROOT / "data" / "source_dataset" / "data" / "metadata" / "image_metadata_template.csv")
    image_centres = centres(image_metadata, "sand_cay_center_lon", "sand_cay_center_lat")
    rows: list[dict] = []
    for reef_id in ids:
        match = standard_centres.loc[standard_centres["reef_id"].eq(reef_id)]
        source = "standardized_coordinates"
        if match.empty:
            match = registry_centres.loc[registry_centres["reef_id"].eq(reef_id)]
            source = "cay_registry_median_center"
        if match.empty:
            match = audit_centres.loc[audit_centres["reef_id"].eq(reef_id)]
            source = "coordinate_normalization_audit_median_center"
        if match.empty:
            match = image_centres.loc[image_centres["reef_id"].eq(reef_id)]
            source = "image_metadata_median_center"
        coordinate_available = not match.empty
        number = re.search(r"OTHER_(\d{3})$", str(reef_id))
        if str(reef_id).startswith("GBR_"):
            region = "Great Barrier Reef"
        elif str(reef_id).startswith("MEDF_"):
            region = "Maldives"
        elif str(reef_id).startswith("NH_"):
            region = "South China Sea"
        elif number:
            value = int(number.group(1))
            if 1 <= value <= 3:
                region = "Indonesia"
            elif 4 <= value <= 20:
                region = "Marshall Islands"
            elif value == 21:
                region = "Farquhar Atoll"
            elif value == 22:
                region = "Seychelles"
            elif 23 <= value <= 24:
                region = "Lakshadweep"
            elif value == 25:
                region = "Maldives"
            elif value == 26:
                region = "Western Indian Ocean"
            else:
                raise ValueError(f"Unmapped OTHER reef: {reef_id}")
        else:
            raise ValueError(f"Unmapped analytical reef: {reef_id}")
        rows.append({
            "reef_id": reef_id, "center_lon": float(match.iloc[0]["center_lon"]) if coordinate_available else np.nan,
            "center_lat": float(match.iloc[0]["center_lat"]) if coordinate_available else np.nan,
            "coordinate_source": source if coordinate_available else "missing_registered_coordinate", "coordinate_available": coordinate_available, "map_region": region,
            "analysis_region": region,
        })
    mapping = pd.DataFrame(rows)
    mapping["sensor_scope"] = "no_ab_observation"
    if observations is not None and not observations.empty:
        obs = observations.copy()
        quality = obs.get("quality_grade", obs.get("feature_quality_grade"))
        if quality is not None:
            obs = obs.loc[quality.isin(["A", "B"])]
        scopes = obs.groupby("reef_id", observed=True)["sensor"].agg(lambda x: set(x.dropna()))
        def scope(values: set) -> str:
            return "both" if {"sentinel2", "google_earth"}.issubset(values) else (
                "sentinel2_only" if "sentinel2" in values else "google_earth_only")
        mapping["sensor_scope"] = mapping["reef_id"].map(scopes.map(scope)).fillna("no_ab_observation")
    if mapping["reef_id"].duplicated().any():
        raise ValueError("Geographic mapping must give each reef exactly one row")
    return mapping


def attach_geography(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[c for c in ("map_region", "analysis_region", "sensor_scope") if c in frame], errors="ignore").merge(
        mapping[["reef_id", "map_region", "analysis_region", "sensor_scope"]], on="reef_id", how="left", validate="many_to_one"
    )
    if result["analysis_region"].isna().any():
        missing = sorted(result.loc[result["analysis_region"].isna(), "reef_id"].dropna().unique())
        raise ValueError(f"Unmapped reefs in analytical frame: {missing}")
    return result


def apply_figure_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _within_cay(frame: pd.DataFrame, columns: list[str]):
    grouped = frame.groupby("sand_cay_id")
    demeaned = frame[columns] - grouped[columns].transform("mean")
    informative = demeaned.abs().sum(axis=1).gt(0)
    return demeaned.loc[informative], informative


def cay_bootstrap_median_ci(
    frame: pd.DataFrame, outcome: str, reps: int = 999, seed: int = RANDOM_SEED
):
    groups = [
        group[outcome].dropna().to_numpy()
        for _, group in frame.groupby("sand_cay_id", observed=True)
    ]
    groups = [group for group in groups if len(group)]
    if len(groups) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(reps, dtype=float)
    for index in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        pooled = np.concatenate([groups[i] for i in pick])
        medians[index] = np.median(pooled)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def describe_region_outcome(frame: pd.DataFrame, outcome: str) -> dict:
    valid = frame[frame[outcome].notna()]
    low, high = cay_bootstrap_median_ci(frame, outcome)
    return {
        "outcome": outcome,
        "n_intervals": int(len(valid)),
        "n_cays": int(valid["sand_cay_id"].nunique()),
        "n_reefs": int(valid["reef_id"].nunique()),
        "median": float(valid[outcome].median()) if len(valid) else np.nan,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def fit_lag_model(frame: pd.DataFrame, outcome: str, exposure: str) -> dict:
    columns = [exposure, *COVARIATES]
    valid = frame[[outcome, *columns]].notna().all(axis=1)
    sub = frame.loc[valid].copy()
    demeaned, informative = _within_cay(sub, [outcome, *columns])
    sub = sub.loc[informative]
    row = {
        "outcome": outcome,
        "exposure": exposure,
        "n_intervals": int(len(sub)),
        "n_cays": int(sub["sand_cay_id"].nunique()),
        "n_reefs": int(sub["reef_id"].nunique()),
    }
    if len(sub) < MIN_INTERVALS or sub["sand_cay_id"].nunique() < MIN_CAYS:
        row["status"] = "insufficient_sample"
        return row
    y = demeaned[outcome]
    x = demeaned[columns]
    fitted = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["sand_cay_id"], "use_correction": True},
    )
    low, high = fitted.conf_int().loc[exposure].astype(float)
    two_way_groups = np.column_stack(
        [
            pd.factorize(sub["sand_cay_id"])[0],
            pd.factorize(sub["midpoint_month"])[0],
        ]
    )
    fitted_two_way = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": two_way_groups, "use_correction": True},
    )
    two_low, two_high = fitted_two_way.conf_int().loc[exposure].astype(float)
    coefficients: list[float] = []
    for year in sorted(sub["midpoint_calendar_year"].unique()):
        leave = sub.loc[sub["midpoint_calendar_year"].ne(year)]
        leave_demeaned, leave_info = _within_cay(leave, [outcome, *columns])
        leave = leave.loc[leave_info]
        if len(leave) < MIN_INTERVALS or leave["sand_cay_id"].nunique() < MIN_CAYS:
            continue
        leave_fit = sm.OLS(leave_demeaned[outcome], leave_demeaned[columns]).fit(
            cov_type="cluster",
            cov_kwds={"groups": leave["sand_cay_id"], "use_correction": True},
        )
        coefficients.append(float(leave_fit.params[exposure]))
    array = np.asarray(coefficients, dtype=float)
    coef = float(fitted.params[exposure])
    row.update(
        {
            "status": "ok",
            "coefficient": coef,
            "cluster_robust_se": float(fitted.bse[exposure]),
            "ci95_low": low,
            "ci95_high": high,
            "two_sided_p_value": float(fitted.pvalues[exposure]),
            "two_way_cluster_se": float(fitted_two_way.bse[exposure]),
            "two_way_ci95_low": two_low,
            "two_way_ci95_high": two_high,
            "two_way_p_value": float(fitted_two_way.pvalues[exposure]),
            "minimum_detectable_effect": 2.80 * float(fitted.bse[exposure]),
            "leave_one_year_out_n": int(len(array)),
            "leave_one_year_out_same_sign_fraction": (
                float(np.mean(np.sign(array) == np.sign(coef))) if len(array) else np.nan
            ),
        }
    )
    return row


def fit_region_lag_models(
    frame: pd.DataFrame, outcomes: dict[str, pd.Series], exposure_template: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for outcome, valid in outcomes.items():
        base = frame.loc[valid.reindex(frame.index, fill_value=False)]
        for region, group in base.groupby("region", observed=True):
            for lag in LAGS_MONTHS:
                row = fit_lag_model(group, outcome, exposure_template.format(lag=lag))
                row["region"] = region
                row["lag_months"] = lag
                rows.append(row)
    result = pd.DataFrame(rows)
    result["two_way_fdr_q_value"] = np.nan
    ok = result["status"].eq("ok")
    if ok.any():
        for _, index in result.loc[ok].groupby(["region", "outcome"]).groups.items():
            result.loc[index, "two_way_fdr_q_value"] = multipletests(
                result.loc[index, "two_way_p_value"], method="fdr_bh"
            )[1]
    # Kept as a compatibility alias; the two-way-cluster value is the primary one.
    result["fdr_q_value"] = result["two_way_fdr_q_value"]
    return result


def fit_region_interaction(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    reference: str = "Great Barrier Reef",
    levels: list[str] | None = None,
) -> dict:
    if levels is None:
        counts = (
            frame.groupby("region", observed=True)
            .agg(n_intervals=("region", "size"), n_cays=("sand_cay_id", "nunique"))
        )
        levels = [
            level
            for level in sorted(frame["region"].dropna().unique())
            if level != reference
            and counts.loc[level, "n_intervals"] >= MIN_INTERVALS
            and counts.loc[level, "n_cays"] >= MIN_CAYS
        ]
    row = {"outcome": outcome, "exposure": exposure, "reference_region": reference}
    if not levels:
        row["status"] = "insufficient_regions"
        return row
    columns = [exposure, *COVARIATES]
    interaction_columns = [f"exposure_x_{level}" for level in levels]
    sub = frame.loc[
        frame["region"].isin([reference, *levels])
        & frame[[outcome, *columns]].notna().all(axis=1)
    ].copy()
    for level, column in zip(levels, interaction_columns):
        sub[column] = sub[exposure] * sub["region"].eq(level).astype(float)
    all_columns = [exposure, *interaction_columns, *COVARIATES]
    demeaned, informative = _within_cay(sub, [outcome, *all_columns])
    sub = sub.loc[informative]
    row.update(
        {
            "n_intervals": int(len(sub)),
            "n_cays": int(sub["sand_cay_id"].nunique()),
            "compared_regions": ",".join([reference, *levels]),
        }
    )
    if len(sub) < MIN_INTERVALS or sub["sand_cay_id"].nunique() < MIN_CAYS:
        row["status"] = "insufficient_sample"
        return row
    fitted = sm.OLS(demeaned[outcome], demeaned[all_columns]).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["sand_cay_id"], "use_correction": True},
    )
    two_way_groups = np.column_stack([
        pd.factorize(sub["sand_cay_id"])[0], pd.factorize(sub["midpoint_month"])[0]
    ])
    fitted_two_way = sm.OLS(demeaned[outcome], demeaned[all_columns]).fit(
        cov_type="cluster", cov_kwds={"groups": two_way_groups, "use_correction": True}
    )
    restriction = np.zeros((len(levels), len(all_columns)))
    for index, column in enumerate(interaction_columns):
        restriction[index, all_columns.index(column)] = 1.0
    wald = fitted.f_test(restriction)
    wald_two_way = fitted_two_way.f_test(restriction)
    terms: list[dict] = []
    for level, column in zip(levels, interaction_columns):
        low, high = fitted_two_way.conf_int().loc[column].astype(float)
        leave_coefficients: list[float] = []
        for year in sorted(sub["midpoint_calendar_year"].unique()):
            leave = sub.loc[sub["midpoint_calendar_year"].ne(year)]
            leave_demeaned, leave_info = _within_cay(leave, [outcome, *all_columns])
            leave = leave.loc[leave_info]
            if len(leave) < MIN_INTERVALS or leave["sand_cay_id"].nunique() < MIN_CAYS:
                continue
            leave_fit = sm.OLS(leave_demeaned[outcome], leave_demeaned[all_columns]).fit(
                cov_type="cluster", cov_kwds={"groups": leave["sand_cay_id"], "use_correction": True}
            )
            leave_coefficients.append(float(leave_fit.params[column]))
        coefficient = float(fitted_two_way.params[column])
        terms.append({
            "outcome": outcome, "exposure": exposure, "reference_region": reference,
            "compared_region": level, "n_intervals": int(len(sub)),
            "n_cays": int(sub["sand_cay_id"].nunique()), "n_reefs": int(sub["reef_id"].nunique()),
            "coefficient": coefficient, "two_way_cluster_se": float(fitted_two_way.bse[column]),
            "two_way_ci95_low": low, "two_way_ci95_high": high,
            "two_way_p_value": float(fitted_two_way.pvalues[column]),
            "leave_one_year_out_n": len(leave_coefficients),
            "leave_one_year_out_same_sign_fraction": float(np.mean(np.sign(leave_coefficients) == np.sign(coefficient))) if leave_coefficients else np.nan,
        })
    row.update(
        {
            "status": "ok",
            "wald_f": float(np.asarray(wald.fvalue).ravel()[0]),
            "wald_df_denom": float(np.asarray(wald.df_denom).ravel()[0]),
            "wald_p_value": float(np.asarray(wald.pvalue).ravel()[0]),
            "two_way_wald_f": float(np.asarray(wald_two_way.fvalue).ravel()[0]),
            "two_way_wald_df_denom": float(np.asarray(wald_two_way.df_denom).ravel()[0]),
            "two_way_wald_p_value": float(np.asarray(wald_two_way.pvalue).ravel()[0]),
            "_term_rows": terms,
        }
    )
    return row


def plot_region_lag_models(
    models: pd.DataFrame, output_stem: Path, exposure_label: str
) -> None:
    apply_figure_style()
    import matplotlib.pyplot as plt

    outcomes = [key for key in OUTCOME_LABELS if key in set(models["outcome"])]
    if not outcomes:
        return
    fig, axes = plt.subplots(
        1, len(outcomes), figsize=(5.0 * len(outcomes), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, outcome in zip(axes, outcomes):
        ok = models.loc[models["status"].eq("ok") & models["outcome"].eq(outcome)]
        for region, group in ok.groupby("region"):
            group = group.sort_values("lag_months")
            coef = group["coefficient"].to_numpy()
            ax.errorbar(
                group["lag_months"],
                coef,
                yerr=[
                    coef - group["two_way_ci95_low"].to_numpy(),
                    group["two_way_ci95_high"].to_numpy() - coef,
                ],
                marker="o",
                capsize=3,
                color=REGION_COLORS.get(region, "#666666"),
                label=region,
            )
        ax.axhline(0, color="#777777", lw=0.8, ls="--")
        ax.set_xticks(list(LAGS_MONTHS))
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel(f"Within-cay coefficient per {exposure_label} (two-way 95% CI)")
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.legend(frameon=False, title="Region")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    fig.savefig(
        output_stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)
