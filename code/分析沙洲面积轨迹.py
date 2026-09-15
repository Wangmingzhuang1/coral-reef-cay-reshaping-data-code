"""识别已成洲沙洲的长期面积轨迹，并生成透明阈值下的诊断图。

分析单位为同一沙洲、同一传感器、同一固定参考框架的面积时间序列。
分类仅用于描述轨迹形态，不等同于沉积学发育阶段或因果机制。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from 分析范围 import exclude_reefs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

MIN_OBSERVATIONS = 4
MIN_SPAN_YEARS = 3.0
GROWTH_THRESHOLD = 0.25
CONTRACTION_THRESHOLD = -0.25
RANK_THRESHOLD = 0.40
LATE_CV_THRESHOLD = 0.15
EXTREME_STEP_RATIO = 2.0

# 2026-09-07 人工复核结论：键为 (sand_cay_id, sensor)，
# 值为 (复核后类别, 复核判定, 依据)。
# 判定口径：待复核轨迹的大幅度跳变经序列签名与抽查影像核验后，
# 确认为沙洲真实大幅度动态；少数叠加水位/眩光/雾霭调制者单独标记。
REVIEW_DATE = "2026-09-07"
REVIEWED_PRIMARY_TRACKS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("GBR_008_cay_001", "sentinel2"): (
        "Complex/non-monotonic",
        "real_with_tide_optical_modulation",
        "双峰大幅振荡；低值期部分受水色/分割调制，沙洲动态幅度真实",
    ),
    ("GBR_009_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "孤立低值一期，其余稳定于20-27k m2，真实动态平衡",
    ),
    ("GBR_037_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "55-130k m2反复振荡、净面积稳定，真实大幅边界重塑",
    ),
    ("GBR_038_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "39-101k m2反复振荡、净面积稳定，真实大幅边界重塑",
    ),
    ("GBR_039_cay_001", "sentinel2"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "基线约60k m2并反复深降至13-31k，非单调真实大幅变化",
    ),
    ("GBR_053_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "孤立低值两期(18.6/20.8k)，其余稳定，真实动态平衡",
    ),
    ("GBR_055_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "两期孤立低值，其余稳定于60-66k m2，真实动态平衡",
    ),
    ("GBR_061_cay_002", "sentinel2"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "11-47k m2反复振荡、无单调趋势，真实大幅变化",
    ),
    ("MEDF_002_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "30-96k m2反复振荡、净面积稳定，真实大幅重塑",
    ),
    ("MEDF_004_cay_001", "sentinel2"): (
        "Sustained growth",
        "confirmed_real_large_amplitude",
        "目视核验：砂体沿礁缘持续连通扩大，真实持续增长",
    ),
    ("MEDF_007_cay_001", "sentinel2"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "小沙洲5.6-53.9k m2反复振荡，真实大幅变化",
    ),
    ("MEDF_008_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "单期低值(10.4k)，其余稳定于21-25k m2，真实动态平衡",
    ),
    ("MEDF_012_cay_001", "google_earth"): (
        "Contraction",
        "confirmed_real_large_amplitude",
        "2.6k持续降至0.4k m2，真实持续收缩",
    ),
    ("MEDF_013_cay_001", "google_earth"): (
        "Contraction",
        "confirmed_real_large_amplitude",
        "2.0k持续降至0.3-0.8k m2，真实持续收缩",
    ),
    ("MEDF_019_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "孤立低值数期(40-58k)，其余稳定于85-100k m2，真实动态平衡",
    ),
    ("MEDF_025_cay_001", "google_earth"): (
        "Contraction",
        "confirmed_real_large_amplitude",
        "目视核验：2016年后亮砂带持续缩短，真实持续收缩",
    ),
    ("MEDF_026_cay_001", "sentinel2"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "39-114k m2全程反复振荡、无稳定增长段，真实大幅变化",
    ),
    ("MEDF_026_cay_002", "sentinel2"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "2022年后骤降至5-25k并部分恢复，非单调真实变化",
    ),
    ("MEDF_029_cay_001", "google_earth"): (
        "Contraction",
        "real_with_tide_optical_modulation",
        "目视核验：2014年后砂体持续变小；2016太阳眩光放大个别降幅，总体真实收缩",
    ),
    ("MEDF_032_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "先升后回落至初始水平、净面积稳定，真实大幅重塑",
    ),
    ("MEDF_037_cay_001", "google_earth"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "目视核验：2013-2014扩大后持续萎缩，非单调真实变化",
    ),
    ("MEDF_038_cay_001", "google_earth"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "2.2-4.8k m2反复振荡、净面积稳定，真实动态平衡",
    ),
    ("MEDF_050_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "单期低值(14.6k)，其余稳定于45-60k m2，真实动态平衡",
    ),
    ("NH_015_cay_001", "google_earth"): (
        "Complex/non-monotonic",
        "confirmed_real_large_amplitude",
        "目视核验：2018-11 短暂砂舌扩大后回退，真实脉冲变化",
    ),
    ("NH_JY_010_cay_001", "google_earth"): (
        "Contraction",
        "confirmed_real_large_amplitude",
        "目视核验：2.2k连续缩至0.7k m2并维持新平衡，真实收缩",
    ),
    ("NH_NE_008_cay_001", "google_earth"): (
        "Complex/non-monotonic",
        "real_with_tide_optical_modulation",
        "2014-10-04同框架双沙洲同步增大，为水位调制的出露脉冲；沙洲动态真实",
    ),
    ("NH_NE_008_cay_002", "google_earth"): (
        "Complex/non-monotonic",
        "real_with_tide_optical_modulation",
        "2014-10-04同框架双沙洲同步增大，为水位调制的出露脉冲；沙洲动态真实",
    ),
    ("NH_TX_013_cay_001", "google_earth"): (
        "Contraction",
        "confirmed_real_large_amplitude",
        "2014-2015由1.1k阶跃缩至0.4-0.6k m2后稳定，真实收缩至新平衡",
    ),
    ("NH_YSZ_011_cay_001", "google_earth"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "孤立高值一期(1.2k)，其余稳定于0.5-0.8k m2，真实动态平衡",
    ),
    ("NH_YSZ_016_cay_001", "google_earth"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "孤立高低值(2.4/0.7k)，其余稳定，真实动态平衡",
    ),
    ("OTHER_004_cay_001", "sentinel2"): (
        "Stable-area reworking",
        "confirmed_real_large_amplitude",
        "伴随振荡的缓慢下降、净变化-15%，真实大幅重塑",
    ),
    ("OTHER_023_cay_001", "google_earth"): (
        "Stable-area reworking",
        "real_with_tide_optical_modulation",
        "目视核验：沙嘴尺度稳定于43-69k m2；2014-10至12雾霭致低估，沙洲真实稳定",
    ),
    ("OTHER_024_cay_001", "sentinel2"): (
        "Complex/non-monotonic",
        "real_with_tide_optical_modulation",
        "目视核验：高低水位出露面积双峰切换，真实但受水位调制的大幅变化",
    ),
}

CATEGORY_ORDER = [
    "Growth then stable",
    "Sustained growth",
    "Stable-area reworking",
    "Contraction",
    "Complex/non-monotonic",
    "Requires review",
]

COLORS = {
    "Growth then stable": "#1B9E77",
    "Sustained growth": "#66A61E",
    "Stable-area reworking": "#386CB0",
    "Contraction": "#D95F02",
    "Complex/non-monotonic": "#7570B3",
    "Requires review": "#BDBDBD",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def trajectory_metrics(group: pd.DataFrame) -> dict[str, float | int | str]:
    group = group.sort_values("date")
    area = group["sand_cay_area_m2"].to_numpy(dtype=float)
    time_years = (
        (group["date"] - group["date"].iloc[0]).dt.days.to_numpy(dtype=float)
        / 365.2425
    )
    first_n = min(2, len(group))
    late_n = min(3, len(group))
    start_area = float(np.median(area[:first_n]))
    end_area = float(np.median(area[-late_n:]))
    relative_change = (end_area - start_area) / start_area
    late_mean = float(np.mean(area[-late_n:]))
    late_cv = (
        float(np.std(area[-late_n:], ddof=1) / late_mean)
        if late_n >= 2 and late_mean > 0
        else np.nan
    )
    rho = float(spearmanr(time_years, area).statistic) if len(area) >= 3 else np.nan
    adjacent_ratio = np.maximum(area[1:] / area[:-1], area[:-1] / area[1:])
    max_step_ratio = float(np.max(adjacent_ratio)) if len(adjacent_ratio) else 1.0
    return {
        "n_observations": int(len(group)),
        "start_date": group["date"].iloc[0].date().isoformat(),
        "end_date": group["date"].iloc[-1].date().isoformat(),
        "span_years": float(time_years[-1]),
        "start_area_m2": start_area,
        "end_area_m2": end_area,
        "relative_change": float(relative_change),
        "late_cv": late_cv,
        "spearman_rho": rho,
        "max_adjacent_area_ratio": max_step_ratio,
    }


def classify(
    row: pd.Series,
    growth_threshold: float = GROWTH_THRESHOLD,
    late_cv_threshold: float = LATE_CV_THRESHOLD,
) -> str:
    if row["max_adjacent_area_ratio"] > EXTREME_STEP_RATIO:
        return "Requires review"
    if (
        row["relative_change"] >= growth_threshold
        and row["spearman_rho"] >= RANK_THRESHOLD
    ):
        if row["late_cv"] <= late_cv_threshold:
            return "Growth then stable"
        return "Sustained growth"
    if (
        row["relative_change"] <= CONTRACTION_THRESHOLD
        and row["spearman_rho"] <= -RANK_THRESHOLD
    ):
        return "Contraction"
    if abs(row["relative_change"]) < growth_threshold:
        return "Stable-area reworking"
    return "Complex/non-monotonic"


def build_tracks(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["reef_id", "sand_cay_id", "sensor", "reference_frame_id"]
    metric_rows: list[dict[str, object]] = []
    point_rows: list[pd.DataFrame] = []
    for key, group in observations.groupby(keys, dropna=False):
        group = group.sort_values("date").drop_duplicates("date", keep="first")
        metrics = trajectory_metrics(group)
        metrics.update(track_shape_metrics(group))
        metrics.update(dict(zip(keys, key)))
        metric_rows.append(metrics)
        points = group[
            keys
            + [
                "image_id",
                "date",
                "sand_cay_area_m2",
                "sand_cay_perimeter_m",
                "major_axis_length_m",
                "minor_axis_length_m",
                "sand_cay_major_axis_angle",
                "sand_cay_centroid_x",
                "sand_cay_centroid_y",
                "pixel_size_m",
                "vegetation_fraction",
                "quality_grade",
            ]
        ].copy()
        points["track_id"] = "|".join(str(value) for value in key)
        point_rows.append(points)
    metrics = pd.DataFrame(metric_rows)
    points = pd.concat(point_rows, ignore_index=True)
    metrics["eligible"] = (
        metrics["n_observations"].ge(MIN_OBSERVATIONS)
        & metrics["span_years"].ge(MIN_SPAN_YEARS)
    )
    metrics["trajectory_class"] = "Not eligible"
    metrics.loc[metrics["eligible"], "trajectory_class"] = metrics.loc[
        metrics["eligible"]
    ].apply(classify, axis=1)
    metrics["track_id"] = metrics[keys].astype(str).agg("|".join, axis=1)
    return metrics, points


def choose_primary_tracks(metrics: pd.DataFrame) -> pd.DataFrame:
    eligible = metrics.loc[metrics["eligible"]].copy()
    eligible = eligible.sort_values(
        ["sand_cay_id", "n_observations", "span_years"],
        ascending=[True, False, False],
    )
    return eligible.drop_duplicates("sand_cay_id", keep="first").copy()


def _relative_change(values: np.ndarray) -> float:
    first_n = min(2, len(values))
    late_n = min(3, len(values))
    start = float(np.median(values[:first_n]))
    end = float(np.median(values[-late_n:]))
    return (end - start) / start if start > 0 else np.nan


def _max_adjacent_ratio(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    ratio = np.maximum(values[1:] / values[:-1], values[:-1] / values[1:])
    return float(np.max(ratio))


def _wrap_axis_deg(delta: np.ndarray) -> np.ndarray:
    return (delta + 90.0) % 180.0 - 90.0


def track_shape_metrics(group: pd.DataFrame) -> dict[str, float]:
    """周长、长短轴、主轴方向与质心的轨迹级多指标摘要。"""
    group = group.sort_values("date")
    perimeter = group["sand_cay_perimeter_m"].to_numpy(dtype=float)
    major = group["major_axis_length_m"].to_numpy(dtype=float)
    minor = group["minor_axis_length_m"].to_numpy(dtype=float)
    angle = group["sand_cay_major_axis_angle"].to_numpy(dtype=float)
    area = group["sand_cay_area_m2"].to_numpy(dtype=float)
    pixel = group["pixel_size_m"].to_numpy(dtype=float)
    cx = group["sand_cay_centroid_x"].to_numpy(dtype=float)
    cy = group["sand_cay_centroid_y"].to_numpy(dtype=float)
    elongation = np.where(minor > 0, major / np.maximum(minor, 1e-9), np.nan)
    result = {
        "perimeter_relative_change": _relative_change(perimeter),
        "perimeter_max_adjacent_ratio": _max_adjacent_ratio(perimeter),
        "major_axis_relative_change": _relative_change(major),
        "minor_axis_relative_change": _relative_change(minor),
        "elongation_relative_change": _relative_change(elongation),
    }
    if len(angle) >= 2 and np.all(np.isfinite(angle)):
        steps = _wrap_axis_deg(np.diff(angle))
        result["orientation_max_step_deg"] = float(np.max(np.abs(steps)))
        result["orientation_net_deg"] = float(
            _wrap_axis_deg(np.array([float(np.sum(steps))]))[0]
        )
    else:
        result["orientation_max_step_deg"] = np.nan
        result["orientation_net_deg"] = np.nan
    start_area = float(np.median(area[: min(2, len(area))]))
    scale = np.sqrt(start_area / np.pi) if start_area > 0 else np.nan
    if len(cx) >= 2 and np.all(np.isfinite(cx)) and np.all(np.isfinite(pixel)):
        dx = np.diff(cx) * pixel[1:]
        dy = np.diff(cy) * pixel[1:]
        step = np.hypot(dx, dy)
        net = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) * pixel[-1])
        result["centroid_path_normalized"] = float(np.sum(step)) / scale
        result["centroid_net_normalized"] = net / scale
        result["centroid_max_step_normalized"] = float(np.max(step)) / scale
    else:
        result["centroid_path_normalized"] = np.nan
        result["centroid_net_normalized"] = np.nan
        result["centroid_max_step_normalized"] = np.nan
    return result


MULTIMETRIC_ORDER = [
    "Persistent reworking",
    "Net growth",
    "Net contraction",
    "Complex",
]


def classify_multimetric(row: pd.Series) -> str:
    """四类多指标清单：以复核后面积类别为骨架，多指标列作为证据。"""
    reviewed = row["trajectory_class_reviewed"]
    if reviewed in ("Sustained growth", "Growth then stable"):
        return "Net growth"
    if reviewed == "Contraction":
        return "Net contraction"
    if reviewed == "Stable-area reworking":
        return "Persistent reworking"
    return "Complex"


def interpolate_track(
    points: pd.DataFrame, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = points.sort_values("date")
    t = (points["date"] - points["date"].iloc[0]).dt.days.to_numpy(dtype=float)
    if t[-1] <= 0:
        return grid, np.full_like(grid, np.nan)
    t = t / t[-1]
    area = points["sand_cay_area_m2"].to_numpy(dtype=float)
    baseline = float(np.median(area[: min(2, len(area))]))
    return grid, np.interp(grid, t, area / baseline)


def sensitivity_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for growth in (0.15, 0.20, 0.25, 0.30):
        for late_cv in (0.10, 0.15, 0.20):
            labels = primary.apply(
                classify,
                axis=1,
                growth_threshold=growth,
                late_cv_threshold=late_cv,
            )
            rows.append(
                {
                    "growth_threshold": growth,
                    "late_cv_threshold": late_cv,
                    "growth_then_stable_n": int(
                        labels.eq("Growth then stable").sum()
                    ),
                    "sustained_growth_n": int(labels.eq("Sustained growth").sum()),
                    "eligible_cays": int(len(primary)),
                }
            )
    return pd.DataFrame(rows)


def apply_review(primary: pd.DataFrame) -> pd.DataFrame:
    """把人工复核结论写回主轨迹表；规则类别保留以便追溯。"""
    primary = primary.copy()
    reviewed: list[str] = []
    verdicts: list[str] = []
    bases: list[str] = []
    dates: list[str] = []
    for _, row in primary.iterrows():
        entry = REVIEWED_PRIMARY_TRACKS.get((row["sand_cay_id"], row["sensor"]))
        if entry is None:
            reviewed.append(row["trajectory_class"])
            verdicts.append("not_required")
            bases.append("")
            dates.append("")
        else:
            reviewed.append(entry[0])
            verdicts.append(entry[1])
            bases.append(entry[2])
            dates.append(REVIEW_DATE)
    primary["trajectory_class_reviewed"] = reviewed
    primary["review_verdict"] = verdicts
    primary["review_basis"] = bases
    primary["reviewed_date"] = dates
    primary["multimetric_class"] = primary.apply(classify_multimetric, axis=1)
    return primary


def make_figure(
    metrics: pd.DataFrame, primary: pd.DataFrame, points: pd.DataFrame
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax_a, ax_b, ax_c = axes

    # a: temporal support for all same-sensor, same-frame tracks.
    ax_a.scatter(
        metrics["span_years"],
        metrics["n_observations"],
        s=17,
        color="#BDBDBD",
        alpha=0.65,
        linewidth=0,
    )
    eligible = metrics.loc[metrics["eligible"]]
    ax_a.scatter(
        eligible["span_years"],
        eligible["n_observations"],
        s=20,
        color="#386CB0",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    ax_a.axvline(MIN_SPAN_YEARS, color="#777777", linestyle="--", linewidth=0.7)
    ax_a.axhline(MIN_OBSERVATIONS, color="#777777", linestyle="--", linewidth=0.7)
    ax_a.text(
        0.03,
        0.96,
        f"{len(eligible)} tracks / {eligible['sand_cay_id'].nunique()} cays eligible",
        transform=ax_a.transAxes,
        va="top",
        fontsize=7,
    )
    ax_a.set_xlabel("Observation span (years)")
    ax_a.set_ylabel("Observations per track")
    ax_a.set_title("Longitudinal support", loc="left")
    panel_label(ax_a, "a")

    # b: one primary trajectory per cay prevents double weighting.
    counts = (
        primary["trajectory_class"]
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
    )
    y = np.arange(len(counts))[::-1]
    ax_b.barh(
        y,
        counts.to_numpy(),
        height=0.62,
        color=[COLORS[name] for name in counts.index],
    )
    for yi, value in zip(y, counts):
        ax_b.text(value + 0.5, yi, str(int(value)), va="center", fontsize=7)
    ax_b.set_yticks(y, counts.index)
    ax_b.set_xlabel("Primary cay trajectories")
    ax_b.set_title("Diagnostic trajectory classes", loc="left")
    panel_label(ax_b, "b")

    # c: display every trajectory classified as growth-then-stable.
    selected = primary.loc[
        primary["trajectory_class"].eq("Growth then stable")
    ].copy()
    grid = np.linspace(0, 1, 41)
    curves: list[np.ndarray] = []
    for track_id in selected["track_id"]:
        track_points = points.loc[points["track_id"].eq(track_id)].copy()
        _, curve = interpolate_track(track_points, grid)
        curves.append(curve)
        ax_c.plot(
            grid,
            curve,
            color=COLORS["Growth then stable"],
            linewidth=0.65,
            alpha=0.28,
        )
    if curves:
        matrix = np.vstack(curves)
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanquantile(matrix, 0.25, axis=0)
        q75 = np.nanquantile(matrix, 0.75, axis=0)
        ax_c.fill_between(
            grid,
            q25,
            q75,
            color=COLORS["Growth then stable"],
            alpha=0.18,
            linewidth=0,
        )
        ax_c.plot(
            grid,
            median,
            color=COLORS["Growth then stable"],
            linewidth=2.0,
            label="Median and IQR",
        )
    ax_c.axhline(1, color="#777777", linestyle="--", linewidth=0.7)
    ax_c.text(
        0.03,
        0.96,
        f"n={len(selected)} cay" if len(selected) == 1 else f"n={len(selected)} cays",
        transform=ax_c.transAxes,
        va="top",
        fontsize=7,
    )
    ax_c.set_xlabel("Relative observation time")
    ax_c.set_ylabel("Area relative to initial value")
    ax_c.set_title("A rare growth-to-stability candidate", loc="left")
    panel_label(ax_c, "c")

    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.21, top=0.86, wspace=0.56
    )
    stem = Path(__file__).resolve().parent / '归档_非主线图件与代码' / 'figures' / "figure_3_area_trajectory_patterns"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    observations = exclude_reefs(
        pd.read_csv(OUTPUT_DIR / "沙洲观测主表.csv")
    )
    observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations["sand_cay_area_m2"] = pd.to_numeric(
        observations["sand_cay_area_m2"], errors="coerce"
    )
    observations = observations.loc[
        observations["sensor"].isin(["sentinel2", "google_earth"])
        & observations["quality_grade"].isin(["A", "B"])
        & observations["date"].notna()
        & observations["sand_cay_area_m2"].gt(0)
    ].copy()

    metrics, points = build_tracks(observations)
    primary = choose_primary_tracks(metrics)
    primary = apply_review(primary)
    sensitivity = sensitivity_table(primary)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        OUTPUT_DIR / "沙洲面积轨迹指标.csv", index=False, encoding="utf-8-sig"
    )
    primary.to_csv(
        OUTPUT_DIR / "沙洲面积主轨迹分类.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity.to_csv(
        OUTPUT_DIR / "沙洲面积轨迹阈值敏感性.csv",
        index=False,
        encoding="utf-8-sig",
    )
    points.loc[points["track_id"].isin(primary["track_id"])].to_csv(
        SOURCE_DIR / "figure3_area_trajectory_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary.to_csv(
        SOURCE_DIR / "figure3_area_trajectory_classes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_figure(metrics, primary, points)
    summary = {
        "analysis_unit": (
            "same sand cay × same sensor × same fixed reference frame trajectory"
        ),
        "excluded_reefs": ["GBR_066"],
        "eligibility": {
            "minimum_observations": MIN_OBSERVATIONS,
            "minimum_span_years": MIN_SPAN_YEARS,
        },
        "classification_rule": {
            "growth_threshold": GROWTH_THRESHOLD,
            "rank_threshold": RANK_THRESHOLD,
            "late_cv_threshold": LATE_CV_THRESHOLD,
            "extreme_step_ratio_for_review": EXTREME_STEP_RATIO,
        },
        "eligible_tracks": int(metrics["eligible"].sum()),
        "eligible_cays": int(metrics.loc[metrics["eligible"], "sand_cay_id"].nunique()),
        "primary_cay_trajectories": int(len(primary)),
        "class_counts": {
            str(key): int(value)
            for key, value in primary["trajectory_class"].value_counts().items()
        },
        "review": {
            "review_date": REVIEW_DATE,
            "reviewed_tracks": int(primary["review_verdict"].ne("not_required").sum()),
            "verdict_counts": {
                str(key): int(value)
                for key, value in primary["review_verdict"].value_counts().items()
            },
            "reviewed_class_counts": {
                str(key): int(value)
                for key, value in primary["trajectory_class_reviewed"]
                .value_counts()
                .items()
            },
            "policy": (
                "Extreme area steps were reviewed against series signatures and "
                "spot-checked imagery; they are retained as real large-amplitude "
                "cay dynamics, with tide/optical modulation flagged separately."
            ),
            "multimetric_class_counts": {
                str(key): int(value)
                for key, value in primary["multimetric_class"]
                .value_counts()
                .items()
            },
        },
        "interpretation_warning": (
            "Trajectory classes are transparent descriptive rules, not validated "
            "geomorphic development stages. Sensor, registration, tide and image "
            "quality remain alternative explanations for apparent area changes."
        ),
    }
    (OUTPUT_DIR / "沙洲面积轨迹核查.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
