"""按地理区域与礁盘检验 Sentinel-2 沙洲质心移动是否随时间减弱。

迁移自 Kench et al. (2018) Figure 2 的证据逻辑：
1. 用相反方向变化的组成比例展示区域内部异质性；
2. 用区域尺度变化率展示净时间趋势。

本研究拥有重复质心观测，因此不使用两期端点率。主分析采用同一沙洲内固定效应，
并控制影像间隔和年内相位。礁盘筛查为探索性，不能替代区域层级推断。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REGIONAL_DIR = ROOT / "额外分析_区域与气候响应"
OUTPUT = HERE / "outputs"
FIGURES = HERE / "figures"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REGIONAL_DIR))

from 分析范围 import analysis_contract_digest, exclude_reefs  # noqa: E402
from 区域气候滞后模型 import (  # noqa: E402
    attach_geography,
    build_geographic_mapping,
)


PRIMARY_WINDOW = (30, 400)
CORE_WINDOW = (90, 365)
REGION_MIN_INTERVALS = 30
REGION_MIN_CAYS = 10
CAY_MIN_INTERVALS = 6
CAY_MIN_SPAN_YEARS = 3.0
REEF_MIN_INTERVALS = 12
REEF_MIN_SPAN_YEARS = 5.0
RANDOM_SEED = 20260920

REGION_ORDER = [
    "Great Barrier Reef",
    "Maldives",
    "Marshall Islands",
    "South China Sea",
    "Lakshadweep",
    "Indonesia",
    "Farquhar Atoll",
    "Seychelles",
    "Western Indian Ocean",
]
REGION_SHORT = {
    "Great Barrier Reef": "Great Barrier Reef",
    "Maldives": "Maldives",
    "Marshall Islands": "Marshall Islands",
    "South China Sea": "South China Sea",
    "Lakshadweep": "Lakshadweep",
    "Indonesia": "Indonesia",
    "Farquhar Atoll": "Farquhar Atoll",
    "Seychelles": "Seychelles",
    "Western Indian Ocean": "W. Indian Ocean",
}

DECLINE = "#3B6FB6"
INCREASE = "#D95F02"
NEUTRAL = "#777777"
DARK = "#222222"
LIGHT = "#D9D9D9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_fdr(frame: pd.DataFrame, p_column: str, gate_column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[gate_column].eq("inferential") & frame[p_column].notna()
    if valid.any():
        result.loc[valid] = multipletests(
            frame.loc[valid, p_column], method="fdr_bh"
        )[1]
    return result


def prepare_intervals() -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_path = ROOT / "outputs" / "沙洲变化区间.csv"
    observation_path = ROOT / "outputs" / "沙洲观测主表.csv"
    intervals = pd.read_csv(interval_path, low_memory=False)
    observations = pd.read_csv(observation_path, low_memory=False)
    intervals = exclude_reefs(intervals)

    numeric = [
        "time_interval_days",
        "centroid_shift_m",
        "area_t_m2",
        "vegetation_fraction_t",
    ]
    for column in numeric:
        intervals[column] = pd.to_numeric(intervals[column], errors="coerce")

    intervals = intervals.loc[
        intervals["sensor"].eq("sentinel2")
        & intervals["quality_grade_t"].isin(["A", "B"])
        & intervals["quality_grade_t1"].isin(["A", "B"])
        & intervals["time_interval_days"].between(*PRIMARY_WINDOW)
        & intervals["centroid_shift_m"].notna()
        & intervals["area_t_m2"].gt(0)
    ].copy()

    mapping = build_geographic_mapping(intervals["reef_id"], observations)
    intervals = attach_geography(intervals, mapping)
    intervals["time_t"] = pd.to_datetime(intervals["time_t"])
    intervals["time_t1"] = pd.to_datetime(intervals["time_t1"])
    intervals["midpoint"] = (
        intervals["time_t"]
        + (intervals["time_t1"] - intervals["time_t"]) / 2
    )
    intervals["midpoint_year"] = (
        intervals["midpoint"].dt.year
        + (intervals["midpoint"].dt.dayofyear - 1) / 365.2425
    )
    intervals["calendar_year"] = intervals["midpoint"].dt.year
    years = intervals["time_interval_days"] / 365.2425
    equivalent_radius = np.sqrt(intervals["area_t_m2"] / np.pi)
    intervals["centroid_norm_speed_per_year"] = (
        intervals["centroid_shift_m"] / equivalent_radius / years
    )
    intervals["centroid_speed_m_per_year"] = (
        intervals["centroid_shift_m"] / years
    )
    intervals["log1p_norm_speed"] = np.log1p(
        intervals["centroid_norm_speed_per_year"]
    )
    intervals["log1p_speed_m"] = np.log1p(
        intervals["centroid_speed_m_per_year"]
    )
    intervals["log_interval"] = np.log(intervals["time_interval_days"])
    phase = 2 * np.pi * intervals["midpoint"].dt.dayofyear / 365.2425
    intervals["season_sin"] = np.sin(phase)
    intervals["season_cos"] = np.cos(phase)
    return intervals, mapping


def within_fit(
    frame: pd.DataFrame,
    outcome: str,
    window: tuple[int, int],
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.loc[frame["time_interval_days"].between(*window)].copy()
    columns = [
        outcome,
        "midpoint_year",
        "log_interval",
        "season_sin",
        "season_cos",
    ]
    rows: list[dict] = []
    analytic_parts: list[pd.DataFrame] = []
    for region, group in work.groupby("analysis_region", observed=True):
        group = group.dropna(subset=columns).copy()
        counts = {
            "n_intervals": int(len(group)),
            "n_cays": int(group["sand_cay_id"].nunique()),
            "n_reefs": int(group["reef_id"].nunique()),
        }
        gate = (
            "inferential"
            if counts["n_intervals"] >= REGION_MIN_INTERVALS
            and counts["n_cays"] >= REGION_MIN_CAYS
            else "descriptive_only"
        )
        demeaned = group[columns] - group.groupby("sand_cay_id")[
            columns
        ].transform("mean")
        informative = demeaned.abs().sum(axis=1).gt(0)
        group = group.loc[informative].copy()
        demeaned = demeaned.loc[informative]
        row = {
            "specification": specification,
            "outcome": outcome,
            "region": region,
            "status": gate,
            **counts,
        }
        if len(group) < 8 or group["sand_cay_id"].nunique() < 1:
            row["status"] = "not_estimable"
            rows.append(row)
            continue
        model = sm.OLS(
            demeaned[outcome],
            demeaned[
                ["midpoint_year", "log_interval", "season_sin", "season_cos"]
            ],
        )
        if gate == "inferential":
            fit = model.fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": group["sand_cay_id"],
                    "use_correction": True,
                },
            )
            covariance = "sand_cay_cluster"
        else:
            fit = model.fit(cov_type="HC3")
            covariance = "HC3_descriptive"
        beta_year = float(fit.params["midpoint_year"])
        ci_low, ci_high = fit.conf_int().loc["midpoint_year"].astype(float)
        beta_decade = beta_year * 10
        row.update(
            {
                "covariance": covariance,
                "beta_log_per_decade": beta_decade,
                "ci95_low_log_per_decade": ci_low * 10,
                "ci95_high_log_per_decade": ci_high * 10,
                "percentage_scale": "1 + mobility",
                "percent_change_per_decade": 100 * np.expm1(beta_decade),
                "ci95_low_percent": 100 * np.expm1(ci_low * 10),
                "ci95_high_percent": 100 * np.expm1(ci_high * 10),
                "two_sided_p_value": float(fit.pvalues["midpoint_year"]),
            }
        )
        rows.append(row)
        analytic = group[
            [
                "transition_id",
                "analysis_region",
                "reef_id",
                "sand_cay_id",
                "midpoint",
                "midpoint_year",
                "calendar_year",
                "time_interval_days",
                "centroid_shift_m",
                "area_t_m2",
                "centroid_norm_speed_per_year",
                "centroid_speed_m_per_year",
            ]
        ].copy()
        analytic["specification"] = specification
        analytic_parts.append(analytic)
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = bh_fdr(
        result, "two_sided_p_value", "status"
    )
    analytic = (
        pd.concat(analytic_parts, ignore_index=True)
        if analytic_parts
        else pd.DataFrame()
    )
    return result, analytic


def regional_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    regions = ["Great Barrier Reef", "Maldives"]
    work = frame.loc[frame["analysis_region"].isin(regions)].copy()
    columns = [
        "log1p_norm_speed",
        "midpoint_year",
        "log_interval",
        "season_sin",
        "season_cos",
    ]
    work = work.dropna(subset=columns)
    work["time_gbr"] = (
        work["midpoint_year"]
        * work["analysis_region"].eq("Great Barrier Reef")
    )
    work["time_maldives"] = (
        work["midpoint_year"] * work["analysis_region"].eq("Maldives")
    )
    design = [
        "log1p_norm_speed",
        "time_gbr",
        "time_maldives",
        "log_interval",
        "season_sin",
        "season_cos",
    ]
    demeaned = work[design] - work.groupby("sand_cay_id")[design].transform(
        "mean"
    )
    informative = demeaned.abs().sum(axis=1).gt(0)
    work = work.loc[informative]
    demeaned = demeaned.loc[informative]
    fit = sm.OLS(
        demeaned["log1p_norm_speed"],
        demeaned[
            [
                "time_gbr",
                "time_maldives",
                "log_interval",
                "season_sin",
                "season_cos",
            ]
        ],
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["sand_cay_id"], "use_correction": True},
    )
    contrast = np.array([-10.0, 10.0, 0.0, 0.0, 0.0])
    test = fit.t_test(contrast)
    difference = float(np.asarray(test.effect).item())
    ci_low, ci_high = np.asarray(test.conf_int()).ravel()
    return pd.DataFrame(
        [
            {
                "contrast": "Maldives minus Great Barrier Reef",
                "scale": "difference in log1p normalized mobility trend per decade",
                "estimate": difference,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "two_sided_p_value": float(test.pvalue),
                "n_intervals": int(len(work)),
                "n_cays": int(work["sand_cay_id"].nunique()),
                "n_reefs": int(work["reef_id"].nunique()),
            }
        ]
    )


def cay_trends(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    predictors = [
        "midpoint_year",
        "log_interval",
        "season_sin",
        "season_cos",
    ]
    for (region, reef, cay), group in frame.groupby(
        ["analysis_region", "reef_id", "sand_cay_id"], observed=True
    ):
        group = group.dropna(subset=["log1p_norm_speed", *predictors]).copy()
        span = group["midpoint_year"].max() - group["midpoint_year"].min()
        if len(group) < CAY_MIN_INTERVALS or span < CAY_MIN_SPAN_YEARS:
            continue
        fit = sm.OLS(
            group["log1p_norm_speed"], sm.add_constant(group[predictors])
        ).fit()
        beta_decade = float(fit.params["midpoint_year"] * 10)
        rows.append(
            {
                "region": region,
                "reef_id": reef,
                "sand_cay_id": cay,
                "n_intervals": int(len(group)),
                "span_years": float(span),
                "beta_log_per_decade": beta_decade,
                "percent_change_per_decade": 100 * np.expm1(beta_decade),
                "trend_direction": (
                    "decreasing" if beta_decade < 0 else "increasing"
                ),
            }
        )
    return pd.DataFrame(rows)


def trend_composition(cays: pd.DataFrame) -> pd.DataFrame:
    if cays.empty:
        return pd.DataFrame()
    composition = (
        cays.groupby(["region", "trend_direction"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for column in ("decreasing", "increasing"):
        if column not in composition:
            composition[column] = 0
    composition["n_cays"] = composition.sum(axis=1)
    composition["decreasing_percent"] = (
        100 * composition["decreasing"] / composition["n_cays"]
    )
    composition["increasing_percent"] = (
        100 * composition["increasing"] / composition["n_cays"]
    )
    return composition.reset_index()


def reef_screen(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    predictors = [
        "midpoint_year",
        "log_interval",
        "season_sin",
        "season_cos",
    ]
    for (region, reef), group in frame.groupby(
        ["analysis_region", "reef_id"], observed=True
    ):
        group = group.dropna(subset=["log1p_norm_speed", *predictors]).copy()
        span = group["midpoint_year"].max() - group["midpoint_year"].min()
        if len(group) < REEF_MIN_INTERVALS or span < REEF_MIN_SPAN_YEARS:
            continue
        cay_dummies = pd.get_dummies(
            group["sand_cay_id"], prefix="cay", drop_first=True, dtype=float
        )
        x = pd.concat(
            [
                group[predictors].reset_index(drop=True),
                cay_dummies.reset_index(drop=True),
            ],
            axis=1,
        )
        fit = sm.OLS(
            group["log1p_norm_speed"].to_numpy(), sm.add_constant(x)
        ).fit(cov_type="HC3")
        beta = float(fit.params["midpoint_year"])
        ci_low, ci_high = fit.conf_int().loc["midpoint_year"].astype(float)
        rows.append(
            {
                "region": region,
                "reef_id": reef,
                "n_intervals": int(len(group)),
                "n_cays": int(group["sand_cay_id"].nunique()),
                "span_years": float(span),
                "covariance": "HC3_exploratory",
                "beta_log_per_decade": beta * 10,
                "ci95_low_log_per_decade": ci_low * 10,
                "ci95_high_log_per_decade": ci_high * 10,
                "percentage_scale": "1 + mobility",
                "percent_change_per_decade": 100 * np.expm1(beta * 10),
                "ci95_low_percent": 100 * np.expm1(ci_low * 10),
                "ci95_high_percent": 100 * np.expm1(ci_high * 10),
                "two_sided_p_value": float(fit.pvalues["midpoint_year"]),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_value"] = multipletests(
            result["two_sided_p_value"], method="fdr_bh"
        )[1]
        result["screen_class"] = np.select(
            [
                result["fdr_q_value"].lt(0.05)
                & result["ci95_high_percent"].lt(0),
                result["fdr_q_value"].lt(0.05)
                & result["ci95_low_percent"].gt(0),
                result["percent_change_per_decade"].lt(0),
            ],
            [
                "FDR-supported decrease",
                "FDR-supported increase",
                "candidate decrease",
            ],
            default="candidate increase",
        )
    return result


def annual_source(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["analysis_region", "reef_id", "sand_cay_id", "calendar_year"],
            observed=True,
        )
        .agg(
            n_intervals=("transition_id", "size"),
            median_norm_speed=(
                "centroid_norm_speed_per_year",
                "median",
            ),
            median_speed_m_per_year=(
                "centroid_speed_m_per_year",
                "median",
            ),
        )
        .reset_index()
    )


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    if not label:
        return
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def draw_composition_panel(ax_a: plt.Axes, composition: pd.DataFrame, available: list, panel: str = "a") -> None:
    """Panel: directional composition of within-cay mobility slopes by region."""
    comp = (
        composition.set_index("region").reindex(available).reset_index()
    )
    y = np.arange(len(comp))
    ax_a.barh(
        y,
        -comp["decreasing_percent"],
        color=DECLINE,
        height=0.66,
        label="Decreasing",
    )
    ax_a.barh(
        y,
        comp["increasing_percent"],
        color=INCREASE,
        height=0.66,
        label="Increasing",
    )
    ax_a.axvline(0, color=DARK, lw=0.7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(
        [REGION_SHORT.get(region, region) for region in comp["region"]]
    )
    ax_a.set_xlim(-105, 105)
    ax_a.set_xticks([-100, -50, 0, 50, 100])
    ax_a.set_xticklabels(["100", "50", "0", "50", "100"])
    ax_a.set_xlabel("Cay trajectories (%)")
    ax_a.invert_yaxis()
    for index, row in comp.iterrows():
        ax_a.text(
            102,
            index,
            f"n={int(row['n_cays'])}",
            ha="left",
            va="center",
            fontsize=6,
            color=NEUTRAL,
            clip_on=False,
        )
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.2,
    )
    panel_label(ax_a, panel)


def draw_regional_trend_panel(ax_b: plt.Axes, regional: pd.DataFrame, available: list, panel: str = "b", title: str = "") -> None:
    """Panel: within-cay mobility trends by region with the inference gate."""
    primary = regional.loc[
        regional["specification"].eq("primary_30_400d_normalized")
    ].copy()
    primary["order"] = primary["region"].map(
        {name: index for index, name in enumerate(available)}
    )
    primary = primary.sort_values("order")
    for index, row in enumerate(primary.itertuples()):
        filled = row.status == "inferential"
        color = (
            DECLINE
            if row.beta_log_per_decade < 0
            else INCREASE
        )
        if filled:
            ax_b.errorbar(
                row.beta_log_per_decade,
                index,
                xerr=np.array(
                    [
                        [
                            row.beta_log_per_decade
                            - row.ci95_low_log_per_decade
                        ],
                        [
                            row.ci95_high_log_per_decade
                            - row.beta_log_per_decade
                        ],
                    ]
                ),
                fmt="o",
                ms=5,
                mfc=color,
                mec=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.2,
            )
        else:
            ax_b.plot(
                row.beta_log_per_decade,
                index,
                marker="o",
                ms=5,
                mfc="white",
                mec=color,
                mew=1.2,
                linestyle="none",
            )
    ax_b.axvline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax_b.set_yticks(np.arange(len(primary)))
    ax_b.set_yticklabels(
        [REGION_SHORT.get(x, x) for x in primary["region"]]
    )
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Change in log(1 + mobility) per decade")
    filled_handle = mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor=NEUTRAL,
        markeredgecolor=NEUTRAL,
        markersize=4.5,
        label="Inference gate met",
    )
    open_handle = mpl.lines.Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=NEUTRAL,
        markersize=4.5,
        label="Descriptive only",
    )
    if title:
        ax_b.legend(
            handles=[filled_handle, open_handle],
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    else:
        ax_b.legend(
            handles=[filled_handle, open_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    if title:
        ax_b.set_title(title if not panel else f"{panel}  {title}", loc="left", fontweight="bold", pad=5)
    else:
        panel_label(ax_b, panel)


def draw_reef_screen_panel(ax_c: plt.Axes, reef: pd.DataFrame, panel: str = "c", note: bool = True) -> None:
    """Panel: exploratory reef-level mobility trend screen."""
    decline = reef.loc[
        reef["beta_log_per_decade"].lt(0)
    ].nsmallest(12, "beta_log_per_decade")
    decline = decline.sort_values(
        "beta_log_per_decade", ascending=True
    )
    y = np.arange(len(decline))
    for index, row in enumerate(decline.itertuples()):
        supported = (
            pd.notna(row.fdr_q_value) and row.fdr_q_value < 0.05
        )
        color = DECLINE if supported else NEUTRAL
        plot_low = max(row.ci95_low_log_per_decade, -2.0)
        plot_high = min(row.ci95_high_log_per_decade, 2.0)
        ax_c.errorbar(
            row.beta_log_per_decade,
            index,
            xerr=np.array(
                [
                    [
                        row.beta_log_per_decade
                        - plot_low
                    ],
                    [
                        plot_high
                        - row.beta_log_per_decade
                    ],
                ]
            ),
            fmt="o",
            ms=4.5,
            mfc=DECLINE if supported else "white",
            mec=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
        )
        if row.ci95_low_log_per_decade < -2:
            ax_c.plot(-2, index, marker="<", color=color, ms=4)
        if row.ci95_high_log_per_decade > 2:
            ax_c.plot(2, index, marker=">", color=color, ms=4)
    ax_c.axvline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(
        [
            f"{row.reef_id}  ({REGION_SHORT.get(row.region, row.region)})"
            for row in decline.itertuples()
        ]
    )
    ax_c.invert_yaxis()
    ax_c.set_xlim(-2.2, 2.2)
    ax_c.set_xlabel("Change in log(1 + mobility) per decade")
    if note:
        ax_c.text(
            0.99,
            0.02,
            "Open points: not FDR-supported; arrows: CI exceeds axis",
            transform=ax_c.transAxes,
            fontsize=5.8,
            color=NEUTRAL,
            ha="right",
            va="bottom",
        )
    panel_label(ax_c, panel)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    intervals, mapping = prepare_intervals()

    specs = [
        (
            "primary_30_400d_normalized",
            "log1p_norm_speed",
            PRIMARY_WINDOW,
        ),
        (
            "sensitivity_90_365d_normalized",
            "log1p_norm_speed",
            CORE_WINDOW,
        ),
        (
            "sensitivity_30_400d_absolute_m",
            "log1p_speed_m",
            PRIMARY_WINDOW,
        ),
    ]
    regional_parts = []
    analytic_parts = []
    for name, outcome, window in specs:
        result, analytic = within_fit(intervals, outcome, window, name)
        regional_parts.append(result)
        analytic_parts.append(analytic)
    regional = pd.concat(regional_parts, ignore_index=True)
    analytic = pd.concat(analytic_parts, ignore_index=True)
    contrast = regional_contrast(intervals)
    cays = cay_trends(intervals)
    composition = trend_composition(cays)
    reefs = reef_screen(intervals)
    annual = annual_source(intervals)

    mapping.to_csv(
        OUTPUT / "区域地理映射.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regional.to_csv(
        OUTPUT / "区域质心移动趋势.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast.to_csv(
        OUTPUT / "主要区域趋势差异.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cays.to_csv(
        OUTPUT / "沙洲质心移动趋势.csv",
        index=False,
        encoding="utf-8-sig",
    )
    composition.to_csv(
        OUTPUT / "区域趋势方向构成.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reefs.to_csv(
        OUTPUT / "礁盘质心移动趋势.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annual.to_csv(
        OUTPUT / "质心移动年度汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analytic.to_csv(
        OUTPUT / "区域趋势分析样本.csv",
        index=False,
        encoding="utf-8-sig",
    )


    primary = regional.loc[
        regional["specification"].eq("primary_30_400d_normalized")
    ]
    supported_regions = primary.loc[
        primary["status"].eq("inferential")
        & primary["fdr_q_value"].lt(0.05)
    ]
    decreasing_reefs = reefs.loc[
        reefs["screen_class"].eq("FDR-supported decrease")
    ]
    audit = {
        "analysis_question": (
            "Does annualized, size-normalized centroid mobility decline over "
            "time within the same cay, and which regions or reefs show that pattern?"
        ),
        "source_logic": (
            "Kench et al. (2018) Figure 2 regional change composition and "
            "atoll-level rates, adapted to repeated centroid trajectories."
        ),
        "primary_outcome": (
            "log1p(centroid displacement / starting equivalent radius / interval years)"
        ),
        "primary_model": (
            "within-cay fixed effects by demeaning; midpoint year + log interval "
            "+ annual sine/cosine; sand-cay clustered SE"
        ),
        "primary_window_days": list(PRIMARY_WINDOW),
        "region_inference_gate": {
            "minimum_intervals": REGION_MIN_INTERVALS,
            "minimum_cays": REGION_MIN_CAYS,
        },
        "cay_trend_screen": {
            "minimum_intervals": CAY_MIN_INTERVALS,
            "minimum_span_years": CAY_MIN_SPAN_YEARS,
            "role": "descriptive direction composition only",
        },
        "reef_screen": {
            "minimum_intervals": REEF_MIN_INTERVALS,
            "minimum_span_years": REEF_MIN_SPAN_YEARS,
            "covariance": "HC3",
            "multiple_comparison": "BH-FDR across all screened reefs",
            "role": "exploratory candidate screen; not regional inference",
        },
        "sample": {
            "intervals": int(len(intervals)),
            "cays": int(intervals["sand_cay_id"].nunique()),
            "reefs": int(intervals["reef_id"].nunique()),
            "regions": int(intervals["analysis_region"].nunique()),
            "region_counts": (
                intervals.groupby("analysis_region", observed=True)
                .agg(
                    intervals=("transition_id", "size"),
                    cays=("sand_cay_id", "nunique"),
                    reefs=("reef_id", "nunique"),
                )
                .reset_index()
                .to_dict("records")
            ),
        },
        "fdr_supported_regional_trends": supported_regions.to_dict("records"),
        "fdr_supported_decreasing_reefs": decreasing_reefs.to_dict("records"),
        "interpretation_boundary": (
            "A decline in interval centroid mobility is not proof of geomorphic "
            "stabilization, island persistence, or reduced habitability risk. "
            "Tide, image timing, sediment supply, vertical change and shoreline "
            "redistribution remain unmeasured or incompletely controlled."
        ),
        "analysis_contract_sha256": analysis_contract_digest(),
        "input_sha256": {
            "沙洲变化区间.csv": sha256(ROOT / "outputs" / "沙洲变化区间.csv"),
            "沙洲观测主表.csv": sha256(ROOT / "outputs" / "沙洲观测主表.csv"),
        },
        "software_reference": (
            "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. "
            "(2026). Scientific Agent Skills: A Library of Procedural Knowledge "
            "for Research Agents. arXiv:2609.00065. "
            "https://doi.org/10.48550/arXiv.2609.00065"
        ),
    }
    (OUTPUT / "区域质心移动趋势核查.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
