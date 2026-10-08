"""Generate the asymmetric Figure 1 regional context panel set.

Claim: study coverage and sensor-specific size distributions provide the spatial
context for area-near-stable yet persistently displaced cay trajectories.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from 区域气候滞后模型 import REGION_COLORS, apply_figure_style  # noqa: E402
type_mod = importlib.import_module("分析沙洲类型与区域差异")
GLOBAL = ROOT / "全球意义"
sys.path.insert(0, str(GLOBAL))
morph_mod = importlib.import_module("分析全球形态重塑谱系")

DISPLAY_ORDER = ["Great Barrier Reef", "Maldives", "South China Sea", "Marshall Islands", "Other"]
DISPLAY_COLORS = {**REGION_COLORS, "Other": "#7A7A7A"}
SMALL_REGION_GATE = 5

def save(fig, stem):
    stem.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})

def display_region(frame, small_regions):
    return frame["map_region"].where(~frame["map_region"].isin(small_regions), "Other")

def draw_map_panel(axm, mapping):
    """Panel a: Indo-Pacific study regions and sensor coverage."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    axm.set_extent([40, 180, -30, 25], crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.LAND, facecolor="#F4F4F4", zorder=0)
    axm.coastlines(linewidth=.35, color="#666666")
    present = mapping.loc[mapping.coordinate_available.astype(bool)].copy()
    for region, group in present.groupby("map_region", observed=True):
        color = REGION_COLORS.get(region, "#666666")
        s2 = group.sensor_scope.isin(["both", "sentinel2_only"])
        ge = group.sensor_scope.isin(["both", "google_earth_only"])
        axm.scatter(group.loc[s2, "center_lon"], group.loc[s2, "center_lat"], s=25, color=color, edgecolor="white", linewidth=.35, transform=ccrs.PlateCarree(), zorder=3)
        axm.scatter(group.loc[ge, "center_lon"], group.loc[ge, "center_lat"], s=39, facecolor="none", edgecolor=color, linewidth=.85, transform=ccrs.PlateCarree(), zorder=4)
    labels = {"Great Barrier Reef": (151, -22), "Maldives": (76, 1), "South China Sea": (114, 18), "Marshall Islands": (169, 14), "Lakshadweep": (73, 12)}
    for name, (x, y) in labels.items():
        axm.text(x, y, name, fontsize=6.7, transform=ccrs.PlateCarree(), ha="center", va="center", zorder=5)
    axm.text(.012, .973, "a  Study regions", transform=axm.transAxes, fontweight="bold", fontsize=8.5, va="top")
    axm.text(.012, .025, "Filled: Sentinel-2     Open: Google Earth", transform=axm.transAxes, fontsize=6.2, va="bottom")


def draw_area_panel(axa, areas, regions):
    """Panel b: sensor-specific cay area structure by region."""
    for sensor, offset, marker in [("sentinel2", -.18, "o"), ("google_earth", .18, "s")]:
        part = areas.loc[areas.sensor.eq(sensor)]
        for index, region in enumerate(regions):
            values = part.loc[part.plot_region.eq(region), "median_area_m2"].dropna().to_numpy()
            jitter = np.random.default_rng(20260917 + index + int(offset * 10)).uniform(-.055, .055, len(values))
            axa.scatter(np.full(len(values), index + offset) + jitter, values, s=11, marker=marker, color=DISPLAY_COLORS[region], alpha=.78, zorder=2)
            if len(values) >= 3:
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                axa.vlines(index + offset, q1, q3, color="#202020", lw=.8, zorder=3)
                axa.hlines(med, index + offset - .075, index + offset + .075, color="#202020", lw=1.0, zorder=3)
            if len(values): axa.text(index + offset, max(values) * 1.28, f"n={len(values)}", ha="center", va="bottom", fontsize=5.2)
    axa.set_yscale("log"); axa.set_ylim(areas.median_area_m2.min() * 0.72, areas.median_area_m2.max() * 2.4)
    axa.set_ylabel("Median cay area (m²)", fontsize=6.6)
    axa.set_xticks(range(len(regions))); axa.set_xticklabels([r.replace(" ", "\n") for r in regions], fontsize=5.6)
    axa.set_title("b  Cay area structure\nCircle: Sentinel-2; square: Google Earth", loc="left", fontweight="bold", fontsize=7.2)


def draw_track_panel(axc, tracks, regions):
    """Panel c: near-stable endpoint area coexisting with continued displacement."""
    from matplotlib.lines import Line2D
    for sensor, marker in [("sentinel2", "o"), ("google_earth", "s")]:
        part = tracks.loc[tracks.sensor.eq(sensor)]
        for region, group in part.groupby("plot_region", observed=True):
            stable = group.strict_morphological_stability.astype(bool)
            axc.scatter(group.absolute_relative_area_change_percent, group.centroid_path_normalized, s=15, marker=marker, color=DISPLAY_COLORS.get(region, "#777777"), alpha=.78, linewidth=.2, zorder=2)
            axc.scatter(group.loc[stable, "absolute_relative_area_change_percent"], group.loc[stable, "centroid_path_normalized"], s=31, marker=marker, facecolor="none", edgecolor="black", linewidth=.65, zorder=3)
    axc.axvline(25, color="#666666", lw=.65, ls="--"); axc.axhline(1, color="#666666", lw=.65, ls="--")
    axc.set_xlabel("Absolute endpoint area change (%)", fontsize=6.6); axc.set_ylabel("Cumulative centroid path (equivalent radii)", fontsize=6.6)
    axc.set_title("c  Near-stable area and continued displacement", loc="left", fontweight="bold", fontsize=8, pad=15)
    color_handles = [Line2D([], [], marker="o", color=DISPLAY_COLORS[r], linestyle="None", label=r, markersize=4) for r in regions]
    axc.legend(handles=color_handles, loc="upper right", fontsize=4.9, ncol=1, handletextpad=.35, labelspacing=.2)


def main():
    apply_figure_style()
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    out, figdir = HERE / "outputs", HERE / "figures"
    mapping = pd.read_csv(out / "区域地理映射.csv")
    areas = pd.read_csv(out / "figure_1_cay_area_structure.csv")
    tracks = pd.read_csv(ROOT / "outputs" / "沙洲面积主轨迹分类.csv")
    tracks = type_mod.add_type_columns(tracks.loc[tracks.eligible.astype(str).str.lower().eq("true")]).merge(
        mapping[["reef_id", "map_region"]], on="reef_id", how="left", validate="many_to_one"
    )
    counts = areas.groupby("map_region", observed=True)["sand_cay_id"].nunique()
    small_regions = set(counts.loc[counts.lt(SMALL_REGION_GATE)].index)
    areas["plot_region"] = display_region(areas, small_regions)
    tracks["plot_region"] = display_region(tracks, small_regions)
    tracks["absolute_relative_area_change_percent"] = tracks.relative_change.abs() * 100
    tracks.to_csv(out / "figure_1_trajectory_relation.csv", index=False, encoding="utf-8-sig")
    areas.to_csv(out / "figure_1_cay_area_structure.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(out / "figure_1_region_map.csv", index=False, encoding="utf-8-sig")
    morph_cays = pd.read_csv(GLOBAL / "outputs" / "全球形态重塑沙洲汇总.csv")
    morph_regions = pd.read_csv(GLOBAL / "outputs" / "全球形态重塑区域汇总.csv")

    # Keep the visible geographic frame of panel a aligned with the outer
    # boundaries of the two panel rows below. Robinson axes preserve their
    # map aspect, so the top row needs enough height rather than a stretched map.
    fig = plt.figure(figsize=(183 / 25.4, 203 / 25.4))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.72, 0.80, 0.95],
        left=0.06, right=0.985, top=0.985, bottom=0.075, hspace=0.34, wspace=0.32,
    )
    axm = fig.add_subplot(grid[0, :], projection=ccrs.Robinson(central_longitude=120))
    draw_map_panel(axm, mapping)
    regions = [r for r in DISPLAY_ORDER if r in set(areas.plot_region)]
    axb = fig.add_subplot(grid[1, 0]); draw_area_panel(axb, areas, regions)
    axc = fig.add_subplot(grid[1, 1]); draw_track_panel(axc, tracks, regions)
    axd = fig.add_subplot(grid[2, 0]); morph_mod.draw_signatures_net_panel(axd, morph_cays, morph_regions, "d")
    axe = fig.add_subplot(grid[2, 1]); morph_mod.draw_signatures_balance_panel(axe, morph_regions, "e")
    save(fig, figdir / "figure_1_regional_context")
    plt.close(fig)

if __name__ == "__main__": main()
