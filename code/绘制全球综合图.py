"""组合主文图7与补充验证图。

主文图7：可迁移的尺寸/过程/测量规则与区域异质的时间趋势；
补充图S4：标准化沙洲中位数的跨源一致性；
补充图S6：质心移动性方向构成与探索性礁盘筛查。
面板绘制函数统一来自三个分析模块，本脚本只负责组合与导出。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
morph = importlib.import_module("分析全球形态重塑谱系")
uni = importlib.import_module("分析全球普适性与可迁移性")
trend = importlib.import_module("分析区域质心移动趋势")

MAIN_FIGURES = ROOT / "outputs" / "figures"


def draw_figure_7() -> None:
    """Main Figure 7: transferable rules plus regionally heterogeneous trends."""
    morph.configure_style()
    regions = pd.read_csv(HERE / "outputs" / "全球形态重塑区域汇总.csv")
    intervals = uni.load_intervals()
    uni_cays = uni.cay_table(intervals)
    scaling = uni.fit_scaling(uni_cays)
    veg_parts = []
    for source in ["sentinel2", "google_earth"]:
        v = uni.vegetation_gradient(intervals.loc[intervals["sensor"].eq(source)]).copy()
        v["sensor"] = source
        veg_parts.append(v)
    vegetation = pd.concat(veg_parts, ignore_index=True)
    regional = pd.read_csv(HERE / "outputs" / "区域质心移动趋势.csv")
    available = [r for r in morph.REGION_ORDER if r in set(regional["region"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.965, bottom=0.09, wspace=0.30, hspace=0.45)
    fig.text(0.022, 0.72, "Transferable constraints", rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
    fig.text(0.022, 0.27, "Region-specific responses", rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax_a, ax_b, ax_c, ax_d = axes.flat
    uni.draw_scaling_panel(ax_a, uni_cays, scaling, "a")
    uni.draw_vegetation_panel(ax_b, vegetation, "b")
    morph.draw_regional_spectra_condensed(ax_c, regions, "c")
    trend.draw_regional_trend_panel(ax_d, regional, available, "d", title="Long-term mobility trends show no common direction")
    morph.save_figure(fig, MAIN_FIGURES / "figure_7_global_transferability")


def draw_supplementary_cross_source() -> None:
    """Supplementary figure: cross-source agreement of standardized cay medians."""
    morph.configure_style()
    intervals = uni.load_intervals()
    agreement, pivot = uni.cross_source_agreement(intervals)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    uni.draw_agreement_panel(ax, agreement, pivot, "")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.14)
    morph.save_figure(fig, HERE / "figures" / "figure_s4_cross_source_agreement")


def draw_supplementary_mobility_screen() -> None:
    """Supplementary figure: directional composition and exploratory reef screen."""
    trend.apply_style()
    composition = pd.read_csv(HERE / "outputs" / "区域趋势方向构成.csv")
    reef = pd.read_csv(HERE / "outputs" / "礁盘质心移动趋势.csv")
    available = [r for r in trend.REGION_ORDER if r in set(composition["region"])]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.0, 1.45]})
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.16, wspace=0.95)
    trend.draw_composition_panel(ax_a, composition, available, "a")
    trend.draw_reef_screen_panel(ax_b, reef, "b", note=False)
    morph.save_figure(fig, HERE / "figures" / "figure_s6_mobility_screen")


def main() -> None:
    MAIN_FIGURES.mkdir(parents=True, exist_ok=True)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    draw_figure_7()
    draw_supplementary_cross_source()
    draw_supplementary_mobility_screen()


if __name__ == "__main__":
    main()
