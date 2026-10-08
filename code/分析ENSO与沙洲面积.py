"""检验月尺度 Niño 3.4 与沙洲面积变化/边界重塑的预设滞后关联。

主分析只使用 Sentinel-2、A/B 质量、同沙洲相邻 90--365 天区间。
不同沙洲的异步面积不直接相加；共同面积曲线由沙洲固定效应和时间样条估计。
结果均为观测关联，不作因果解释。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from statsmodels.stats.multitest import multipletests

from 分析范围 import exclude_reefs


LAGS_MONTHS = (0, 3, 6, 9, 12, 18, 24)
CORE_DAYS = (90, 365)
NINO_SOURCE_URL = (
    "https://www.cpc.ncep.noaa.gov/data/indices/detrend.nino34.ascii.txt"
)


def load_nino34(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+")
    required = {"YR", "MON", "ANOM"}
    if not required.issubset(table.columns):
        raise ValueError(f"Niño 3.4 文件缺少字段：{sorted(required - set(table.columns))}")
    table = table.rename(
        columns={"YR": "year", "MON": "month", "ANOM": "nino34_anomaly_c"}
    )
    table["time"] = pd.to_datetime(
        dict(year=table["year"], month=table["month"], day=1), errors="coerce"
    )
    table["nino34_anomaly_c"] = pd.to_numeric(
        table["nino34_anomaly_c"], errors="coerce"
    )
    table = table.dropna(subset=["time", "nino34_anomaly_c"]).copy()
    table["month_period"] = table["time"].dt.to_period("M")
    if table["month_period"].duplicated().any():
        raise ValueError("Niño 3.4 月份存在重复")
    return table.sort_values("time").reset_index(drop=True)


def interval_weighted_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_months: int,
    lookup: dict[pd.Period, float],
) -> float:
    lag_start = start - pd.DateOffset(months=lag_months)
    lag_end = end - pd.DateOffset(months=lag_months)
    days = pd.date_range(lag_start + pd.Timedelta(days=1), lag_end, freq="D")
    if len(days) == 0:
        return np.nan
    values = pd.Series(days.to_period("M")).map(lookup)
    return float(values.mean()) if values.notna().all() else np.nan


def prepare_intervals(
    transitions: pd.DataFrame, nino: pd.DataFrame
) -> pd.DataFrame:
    frame = exclude_reefs(transitions).copy()
    frame["time_t"] = pd.to_datetime(frame["time_t"], errors="coerce")
    frame["time_t1"] = pd.to_datetime(frame["time_t1"], errors="coerce")
    numeric = [
        "time_interval_days",
        "area_t_m2",
        "area_t1_m2",
        "gross_boundary_change_m2",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["sensor"].eq("sentinel2")
        & frame["quality_grade_t"].isin(["A", "B"])
        & frame["quality_grade_t1"].isin(["A", "B"])
        & frame["time_interval_days"].between(*CORE_DAYS)
        & frame["area_t_m2"].gt(0)
        & frame["area_t1_m2"].gt(0)
        & frame[["time_t", "time_t1"]].notna().all(axis=1)
    ].copy()
    years = frame["time_interval_days"] / 365.2425
    frame["annualized_log_area_change"] = (
        np.log(frame["area_t1_m2"]) - np.log(frame["area_t_m2"])
    ) / years
    frame["gross_mobility_fraction_per_year"] = (
        frame["gross_boundary_change_m2"] / frame["area_t_m2"] / years
    )
    positive = frame.loc[
        frame["gross_mobility_fraction_per_year"].gt(0),
        "gross_mobility_fraction_per_year",
    ]
    floor = float(positive.quantile(0.01) / 2) if len(positive) else 1e-6
    frame["log_gross_mobility"] = np.log(
        frame["gross_mobility_fraction_per_year"].clip(lower=0) + floor
    )
    midpoint = frame["time_t"] + (frame["time_t1"] - frame["time_t"]) / 2
    frame["midpoint_year"] = midpoint.dt.year + (midpoint.dt.dayofyear - 1) / 365.2425
    # ENSO 是所有沙洲共享的时间暴露；保留日历月/年用于时间依赖敏感性检验。
    frame["midpoint_month"] = midpoint.dt.to_period("M").astype(str)
    frame["midpoint_calendar_year"] = midpoint.dt.year.astype(int)
    frame["season_sin"] = np.sin(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["season_cos"] = np.cos(2 * np.pi * midpoint.dt.dayofyear / 365.2425)
    frame["log_interval"] = np.log(frame["time_interval_days"])
    lookup = dict(zip(nino["month_period"], nino["nino34_anomaly_c"]))
    for lag in LAGS_MONTHS:
        frame[f"nino34_lag_{lag}m"] = [
            interval_weighted_index(start, end, lag, lookup)
            for start, end in zip(frame["time_t"], frame["time_t1"])
        ]
    return frame


def fit_lag_models(frame: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "annualized_log_area_change": frame["annualized_log_area_change"].notna(),
        "log_gross_mobility": frame["boundary_change_status"].eq("ok")
        & frame["log_gross_mobility"].notna(),
    }
    rows: list[dict[str, object]] = []
    for outcome, outcome_valid in outcomes.items():
        for lag in LAGS_MONTHS:
            exposure = f"nino34_lag_{lag}m"
            columns = [
                exposure,
                "log_interval",
                "midpoint_year",
                "season_sin",
                "season_cos",
            ]
            valid = outcome_valid & frame[columns].notna().all(axis=1)
            sub = frame.loc[valid].copy()
            grouped = sub.groupby("sand_cay_id")
            y = sub[outcome] - grouped[outcome].transform("mean")
            x = sub[columns] - grouped[columns].transform("mean")
            informative = x.abs().sum(axis=1).gt(0) & y.notna()
            sub, y, x = sub.loc[informative], y.loc[informative], x.loc[informative]
            if len(sub) < 30 or sub["sand_cay_id"].nunique() < 10:
                rows.append(
                    {
                        "outcome": outcome,
                        "lag_months": lag,
                        "status": "insufficient_sample",
                        "n_intervals": int(len(sub)),
                        "n_cays": int(sub["sand_cay_id"].nunique()),
                    }
                )
                continue
            fitted = sm.OLS(y, x).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": sub["sand_cay_id"],
                    "use_correction": True,
                },
            )
            low, high = fitted.conf_int().loc[exposure].astype(float)
            # 主模型按沙洲聚类。由于 Niño 3.4 在同一日历月对所有沙洲相同，
            # 另以沙洲和区间中点月作双向聚类，作为共同时间冲击的保守敏感性。
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
            two_way_low, two_way_high = fitted_two_way.conf_int().loc[exposure].astype(
                float
            )

            # 逐年剔除检验：每次重新做沙洲内去均值，避免只由某一异常年份驱动。
            leave_one_year_coefficients: list[float] = []
            for held_out_year in sorted(sub["midpoint_calendar_year"].unique()):
                leave = sub.loc[
                    sub["midpoint_calendar_year"].ne(held_out_year)
                ].copy()
                leave_grouped = leave.groupby("sand_cay_id")
                leave_y = leave[outcome] - leave_grouped[outcome].transform("mean")
                leave_x = leave[columns] - leave_grouped[columns].transform("mean")
                leave_informative = leave_x.abs().sum(axis=1).gt(0) & leave_y.notna()
                leave, leave_y, leave_x = (
                    leave.loc[leave_informative],
                    leave_y.loc[leave_informative],
                    leave_x.loc[leave_informative],
                )
                if len(leave) < 30 or leave["sand_cay_id"].nunique() < 10:
                    continue
                leave_fit = sm.OLS(leave_y, leave_x).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": leave["sand_cay_id"], "use_correction": True},
                )
                leave_one_year_coefficients.append(float(leave_fit.params[exposure]))
            loyo = np.asarray(leave_one_year_coefficients, dtype=float)
            rows.append(
                {
                    "outcome": outcome,
                    "lag_months": lag,
                    "status": "ok",
                    "n_intervals": int(len(sub)),
                    "n_cays": int(sub["sand_cay_id"].nunique()),
                    "n_reefs": int(sub["reef_id"].nunique()),
                    "coefficient_per_1c": float(fitted.params[exposure]),
                    "cluster_robust_se": float(fitted.bse[exposure]),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "two_sided_p_value": float(fitted.pvalues[exposure]),
                    "two_way_cluster_se": float(fitted_two_way.bse[exposure]),
                    "two_way_ci95_low": float(two_way_low),
                    "two_way_ci95_high": float(two_way_high),
                    "two_way_cluster_p_value": float(fitted_two_way.pvalues[exposure]),
                    "leave_one_year_out_n": int(len(loyo)),
                    "leave_one_year_out_min_coefficient": float(loyo.min()),
                    "leave_one_year_out_max_coefficient": float(loyo.max()),
                    "leave_one_year_out_same_sign_fraction": float(
                        np.mean(np.sign(loyo) == np.sign(fitted.params[exposure]) )
                    ),
                    "leave_one_year_out_all_same_sign": bool(
                        np.all(np.sign(loyo) == np.sign(fitted.params[exposure]))
                    ),
                    "adj_r_squared": float(fitted.rsquared_adj),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    ok = result["status"].eq("ok")
    for _, index in result.loc[ok].groupby("outcome").groups.items():
        result.loc[index, "fdr_q_value"] = multipletests(
            result.loc[index, "two_sided_p_value"], method="fdr_bh"
        )[1]
    result["two_way_cluster_fdr_q_value"] = np.nan
    for _, index in result.loc[ok].groupby("outcome").groups.items():
        result.loc[index, "two_way_cluster_fdr_q_value"] = multipletests(
            result.loc[index, "two_way_cluster_p_value"], method="fdr_bh"
        )[1]
    result["time_robust_supported"] = (
        result["status"].eq("ok")
        & result["fdr_q_value"].lt(0.05)
        & result["two_way_cluster_fdr_q_value"].lt(0.05)
        & result["leave_one_year_out_all_same_sign"].fillna(False)
    )
    return result


def fit_common_area_trend(
    observations: pd.DataFrame, nino: pd.DataFrame
) -> pd.DataFrame:
    frame = exclude_reefs(observations).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sand_cay_area_m2"] = pd.to_numeric(
        frame["sand_cay_area_m2"], errors="coerce"
    )
    frame = frame.loc[
        frame["sensor"].eq("sentinel2")
        & frame["quality_grade"].isin(["A", "B"])
        & frame["sand_cay_area_m2"].gt(0)
        & frame["date"].notna()
    ].copy()
    counts = frame.groupby("sand_cay_id")["date"].agg(["count", "min", "max"])
    eligible = counts.loc[
        counts["count"].ge(4)
        & (counts["max"] - counts["min"]).dt.days.ge(round(3 * 365.2425))
    ].index
    frame = frame.loc[frame["sand_cay_id"].isin(eligible)].copy()
    frame["decimal_year"] = (
        frame["date"].dt.year + (frame["date"].dt.dayofyear - 1) / 365.2425
    )
    frame["log_area"] = np.log(frame["sand_cay_area_m2"])
    spline = dmatrix(
        "bs(t, df=5, degree=3, include_intercept=False) - 1",
        {"t": frame["decimal_year"]},
        return_type="dataframe",
    )
    spline_info = spline.design_info
    spline.index = frame.index
    cay = pd.get_dummies(
        frame["sand_cay_id"], prefix="cay", drop_first=True, dtype=float
    )
    x = pd.concat(
        [
            pd.Series(1.0, index=frame.index, name="const"),
            cay,
            spline.set_axis(
                [f"time_spline_{i}" for i in range(spline.shape[1])], axis=1
            ),
        ],
        axis=1,
    )
    fitted = sm.OLS(frame["log_area"], x).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["sand_cay_id"], "use_correction": True},
    )
    grid_time = pd.date_range(
        frame["date"].min().to_period("M").to_timestamp(),
        frame["date"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    # Patsy 的 B 样条不外推；仅在首末真实观测日期之间绘制共同趋势。
    grid_time = grid_time[
        (grid_time >= frame["date"].min()) & (grid_time <= frame["date"].max())
    ]
    if len(grid_time) == 0:
        raise ValueError("共同趋势没有落在观测时间范围内的月尺度预测点")
    decimal_grid = grid_time.year + (grid_time.dayofyear - 1) / 365.2425
    grid_spline = np.asarray(
        build_design_matrices(
            [spline_info], {"t": np.asarray(decimal_grid, dtype=float)}
        )[0]
    )
    x_grid = pd.DataFrame(0.0, index=range(len(grid_time)), columns=x.columns)
    x_grid["const"] = 1.0
    spline_columns = [column for column in x.columns if column.startswith("time_spline_")]
    x_grid[spline_columns] = grid_spline
    contrast = x_grid.to_numpy() - x_grid.iloc[[0]].to_numpy()
    estimate = contrast @ np.asarray(fitted.params)
    covariance = np.asarray(fitted.cov_params())
    se = np.sqrt(np.einsum("ij,jk,ik->i", contrast, covariance, contrast))
    nino_lookup = nino.set_index("month_period")["nino34_anomaly_c"]
    return pd.DataFrame(
        {
            "time": grid_time,
            "common_log_area_change": estimate,
            "common_area_change_percent": np.expm1(estimate) * 100,
            "ci95_low_percent": np.expm1(estimate - 1.96 * se) * 100,
            "ci95_high_percent": np.expm1(estimate + 1.96 * se) * 100,
            "nino34_anomaly_c": pd.Series(grid_time.to_period("M")).map(
                nino_lookup
            ),
            "n_observations": len(frame),
            "n_cays": frame["sand_cay_id"].nunique(),
        }
    )


def plot_results(trend: pd.DataFrame, models: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(
        trend["time"],
        trend["ci95_low_percent"],
        trend["ci95_high_percent"],
        color="#386CB0",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        trend["time"], trend["common_area_change_percent"], color="#386CB0", lw=1.8
    )
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_ylabel("Common area change (%)")
    ax.set_title("Fixed-cohort common area trend and Niño 3.4")
    twin = ax.twinx()
    twin.plot(
        trend["time"], trend["nino34_anomaly_c"], color="#D95F02", lw=1.0, alpha=0.75
    )
    twin.axhline(0, color="#D95F02", lw=0.6, alpha=0.45)
    twin.set_ylabel("Niño 3.4 anomaly (°C)", color="#D95F02")
    twin.spines["top"].set_visible(False)

    ax = axes[1]
    colors = {
        "annualized_log_area_change": "#386CB0",
        "log_gross_mobility": "#1B9E77",
    }
    labels = {
        "annualized_log_area_change": "Log area change",
        "log_gross_mobility": "Gross mobility",
    }
    for outcome, group in models.loc[models["status"].eq("ok")].groupby("outcome"):
        group = group.sort_values("lag_months")
        ax.errorbar(
            group["lag_months"],
            group["coefficient_per_1c"],
            yerr=[
                group["coefficient_per_1c"] - group["two_way_ci95_low"],
                group["two_way_ci95_high"] - group["coefficient_per_1c"],
            ],
            marker="o",
            capsize=3,
            color=colors[outcome],
            label=labels[outcome],
        )
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(LAGS_MONTHS)
    ax.set_xlabel("ENSO lag (months)")
    ax.set_ylabel("Within-cay coefficient per 1°C (two-way 95% CI)")
    ax.set_title("Pre-specified lag associations: time-aware uncertainty")
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=300)
    fig.savefig(output.with_suffix(".tiff"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nino-file",
        type=Path,
        default=root / "data" / "environment" / "indices" / "detrend.nino34.ascii.txt",
    )
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=root / "outputs" / "沙洲变化区间.csv",
    )
    parser.add_argument(
        "--observation-csv",
        type=Path,
        default=root / "outputs" / "沙洲观测主表.csv",
    )
    parser.add_argument(
        "--trajectory-type-csv",
        type=Path,
        default=root / "outputs" / "沙洲净面积类型.csv",
        help="可选的长期净面积类型表；仅在 --net-area-types 有值时使用。",
    )
    parser.add_argument(
        "--net-area-types",
        nargs="+",
        default=None,
        help="仅分析指定净面积类型；该模式只作探索性分层，不替代全样本主模型。",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()

    nino = load_nino34(args.nino_file)
    transitions = pd.read_csv(args.transition_csv, low_memory=False)
    observations = pd.read_csv(args.observation_csv, low_memory=False)
    sample = prepare_intervals(transitions, nino)
    scope = "all eligible Sentinel-2 intervals"
    scope_counts: dict[str, int] | None = None
    if args.net_area_types:
        types = pd.read_csv(args.trajectory_type_csv, encoding="utf-8-sig")
        keys = ["reef_id", "sand_cay_id", "sensor", "reference_frame_id"]
        required = set(keys + ["net_area_type"])
        if not required.issubset(types.columns):
            raise ValueError(
                "长期类型表缺少字段："
                f"{sorted(required - set(types.columns))}"
            )
        selected = types.loc[
            types["net_area_type"].isin(args.net_area_types), keys + ["net_area_type"]
        ].drop_duplicates(keys)
        if selected.empty:
            raise ValueError(f"没有匹配的净面积类型：{args.net_area_types}")
        sample = sample.merge(selected, on=keys, how="inner", validate="many_to_one")
        observations = observations.merge(
            selected[keys], on=keys, how="inner", validate="many_to_one"
        )
        scope = "exploratory selected net-area types: " + ", ".join(args.net_area_types)
        scope_counts = {
            str(label): int(count)
            for label, count in sample["net_area_type"].value_counts().items()
        }
    models = fit_lag_models(sample)
    trend = fit_common_area_trend(observations, nino)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nino_out = args.output_dir / "ENSO月尺度指数.csv"
    sample_out = args.output_dir / "ENSO滞后分析样本.csv"
    models_out = args.output_dir / "ENSO滞后模型.csv"
    trend_out = args.output_dir / "ENSO共同面积趋势.csv"
    check_out = args.output_dir / "ENSO沙洲面积分析核查.json"
    nino.drop(columns="month_period").to_csv(nino_out, index=False, encoding="utf-8-sig")
    sample.to_csv(sample_out, index=False, encoding="utf-8-sig")
    models.to_csv(models_out, index=False, encoding="utf-8-sig")
    trend.to_csv(trend_out, index=False, encoding="utf-8-sig")
    plot_results(trend, models, Path(__file__).resolve().parent / '归档_非主线图件与代码' / 'figures' / "figure_7_enso_area")

    supported = models.loc[
        models["status"].eq("ok") & models["fdr_q_value"].lt(0.05)
    ]
    time_robust_supported = models.loc[models["time_robust_supported"]]
    summary = {
        "index": "NOAA CPC monthly detrended Nino 3.4 SST anomaly",
        "source_url": NINO_SOURCE_URL,
        "index_time_range": [
            str(nino["time"].min().date()),
            str(nino["time"].max().date()),
        ],
        "primary_sensor": "sentinel2",
        "analysis_scope": scope,
        "analysis_scope_interval_counts": scope_counts,
        "interval_window_days": list(CORE_DAYS),
        "lags_months": list(LAGS_MONTHS),
        "lag_definition": "daily overlap-weighted monthly index over (time_t-L, time_t1-L]",
        "interval_sample_rows": int(len(sample)),
        "interval_sample_cays": int(sample["sand_cay_id"].nunique()),
        "interval_sample_reefs": int(sample["reef_id"].nunique()),
        "common_trend_observations": int(trend["n_observations"].iloc[0]),
        "common_trend_cays": int(trend["n_cays"].iloc[0]),
        "fdr_supported_lag_tests": supported.to_dict("records"),
        "time_robust_supported_lag_tests": time_robust_supported.to_dict("records"),
        "multiple_testing": "BH-FDR across seven pre-specified lags within each outcome",
        "inference": (
            "Within-cay observational association with cay-clustered robust SE; "
            "two-way cay-by-midpoint-month clustering and leave-one-calendar-year-out "
            "are reported as common-time-exposure sensitivities; not causal. The common "
            "trend is a fixed-cohort model estimate, not a sum of asynchronous observed areas."
        ),
        "not_yet_included": [
            "region interaction",
            "ONI/RONI sensitivity",
            "wind-wave-current adjusted pathway model",
        ],
    }
    check_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
