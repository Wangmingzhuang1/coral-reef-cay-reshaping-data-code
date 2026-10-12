# 珊瑚礁沙洲时空演变的年度重复性与植被相关移动性差异

This repository archives the current analysis code, derived measurement/statistical tables,
and figure source data for a multi-temporal study of coral-reef sand cays.

Current archive: **v2.0.0, 12 October 2026**. Repository: https://github.com/Wangmingzhuang1/coral-reef-cay-reshaping-data-code
Versioned record: https://github.com/Wangmingzhuang1/coral-reef-cay-reshaping-data-code/releases/tag/v2.0.0

## Scope and primary results

- Long-term area analysis: 93 records, including 63 near-stable, 17 net-gain and 13 net-loss records.
- Comparable position analysis: 90 records, including 60 near-stable records.
- Annual geometry reconstruction: 52 cays; five position/shape indicators show corrected support.
- Environmental direction models: 760 intervals from 46 cays.
- Vegetation–path correlations: 53 Sentinel-2 and 37 Google Earth records, analysed separately.
- Within-interval shoreline contrasts: 1,012 comparisons from 62 cays.
- High-speed distribution under the accepted initial-cover labels: 521 bare intervals (94 high-speed)
  and 889 vegetated intervals (47 high-speed); 18.0% versus 5.3%. The threshold is the pooled p90.
- Bare-interval high-wave model: 497 intervals from 34 cays, with reef-clustered uncertainty.

Earlier exploratory analyses and sensitivity results are retained as supporting records.
The current main-result tables are mapped below; older English figure-number filenames reflect
earlier figure arrangements and do not define the current main-figure numbering.

## Files and main-figure mapping

| Current figure / analysis | Current source tables |
| --- | --- |
| Fig. 1: area outcomes and actual examples | `outputs/figures/source_data/表A_沙洲级面积路径状态.csv`; `四区图谱位置面积序列.csv` in the same source-data folder |
| Fig. 2: position and boundary differences | `outputs/审稿问题复核/原始网格一致长期位置描述.csv`; `outputs/figures/source_data/主图2_连续空间轨迹.csv` |
| Fig. 3: annual geometry | `outputs/审稿问题复核/单一原始网格留一年汇总.csv`; `outputs/审稿问题复核/单一原始网格留一年逐景对照.csv`; `outputs/figures/source_data/figure_2_harmonic_relative_error_reduction.csv` |
| Fig. 4: wave/current direction and exposure | `outputs/审稿问题复核/单一原始网格方向模型.csv`; `单一原始网格高波浪日模型.csv`; `覆盖分组风浪分析时段.csv` in the same folder |
| Fig. 5: observed cover-state changes | `outputs/figures/source_data/六向状态转换与保持.csv`; `表面状态相邻观测对.csv` in the same source-data folder |
| Fig. 6: vegetation and mobility | `outputs/figures/source_data/主图6_覆盖与累计路径统计.csv`; `outputs/植被扇区对照.csv`; `outputs/植被扇区对照统计.csv`; `outputs/审稿问题复核/单一原始网格高移动速率对照.csv` |

`代码/` contains Python modules grouped by scientific task. `outputs/` contains source measurements,
derived tables and all retained statistical comparisons. `额外分析_空间时间覆盖联合规律/` contains
the secondary joint-analysis code and results. CSV files use UTF-8 with BOM for Windows Excel.
`文件清单.csv` records every packaged file's SHA-256 and flags path-sanitized copies.

## Reproduce checks from the public archive

Download the release asset `数据代码.zip` and extract it into a new directory. Open that directory
as the working directory, install Python 3.11 or later and the packages in `requirements.txt`, then run:

```text
python 代码/控制验证/复核审稿主要问题.py --check-derived
```

This command checks the past-only training dates, two seven-response BH families, 1,490 state pairs,
90 position-comparable records, shoreline-contrast eligibility, the accepted vegetation grouping,
the high-speed threshold/counts, the high-wave model family, and every manifest SHA-256.
The files can also be read directly on GitHub; use the ZIP asset when byte-for-byte checks are required.

## Observation definitions and access to original inputs

Coordinates are image-based planform positions; geometry units and normalizations are named in
the table columns and defined by the analysis functions. `sand_cay_id`, `reef_id`, `sensor`,
`reference_frame_id`, dates and observation IDs link the measurement and comparison tables.
Green fraction describes visible vegetation inferred from rendered RGB images. Accepted initial
cover `none` defines bare intervals; `sparse`, `partial` and `dominant` define vegetated intervals.
This definition differs from the historical `stage` grouping retained for older exploratory analyses.

The public archive contains derived-data checks; recomputing masks, spectral diagnostics,
background alignment or the full measurement pipeline additionally requires original imagery,
per-date masks, annotation metadata and environmental input records. Those files are not in this archive.
The source-data layout expected by the scripts is `../data/source_dataset` relative to the project root.
Manuscript-generation and local delivery-audit scripts also require local manuscript/frozen-archive files.

Original imagery and environment sources:
- Sentinel-2: [Copernicus Data Space](https://dataspace.copernicus.eu/).
- Google Earth historical imagery: accessed through Google Earth under its applicable terms.
- ERA5: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).
- GLORYS12V1 and WAVERYS: [Copernicus Marine Service](https://marine.copernicus.eu/).
- IBTrACS: [NOAA NCEI](https://www.ncei.noaa.gov/products/international-best-track-archive).

No third-party original imagery, per-date masks, manuscript files or reviewer correspondence are published.
Personal absolute paths in packaged copies are replaced with relative paths or `{USER_HOME}`;
the authoritative local measurement files remain unchanged.

## Licences and citation

Code: MIT (`LICENSE`). Author-derived tables: CC BY 4.0 (`DATA_LICENSE`).
Third-party data remain subject to their original providers' terms.

To cite this archive, include the study title above, the version v2.0.0, the date 12 October 2026,
and https://github.com/Wangmingzhuang1/coral-reef-cay-reshaping-data-code/releases/tag/v2.0.0. No DOI is assigned to this release.
This release supersedes the previous repository layout; prior code/data remain recoverable in GitHub history.
