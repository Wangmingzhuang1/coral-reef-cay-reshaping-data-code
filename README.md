# Coral reef cay reworking: derived data, figure source data and analysis code

中文说明：本仓库发布论文《珊瑚礁沙洲的持续重塑与平面相对稳定共存》的派生数据表、图源数据、环境强迫区间聚合表与分析代码。原始影像与沙洲掩膜因许可与体积原因不在此发布（见下文 Data not included）。

## Contents

- 'data/analysis_tables/' - derived analysis tables produced by the study pipeline (trajectory classification, seasonal recurrence pairs and harmonic validation, directional coupling models and sensitivities, vegetation-mobility analyses, continuous-time state models, typhoon event pairing, ENSO lag models, wind increment tests, error propagation and detectability outputs).
- 'data/figure_source_data/' - one CSV per main/supplementary figure panel group, sufficient to regenerate every plot element.
- 'data/environment_forcing/' - interval-aggregated forcing tables (ERA5 wind, GLORYS current, WAVERYS wave) extracted per reef frame and joined to observation intervals (t, t1].
- 'code/' - analysis and figure scripts (Chinese filenames). Entry points: 分析沙洲面积轨迹.py, 分析Sentinel2季节形态重复性.py, 对齐流场波浪与沙洲变化.py, 分析植被形态台风关系.py, 拟合Sentinel2三状态连续时间模型.py, 分析台风事件前后沙洲响应.py, 分析ENSO与沙洲面积.py, 稳健性扩展检验.py, 误差传播与补充检验.py, 绘制当前结果图.py.

## Reproduction order

1. Build observation and change tables from the archived mask registry (registry not distributed; see below): 构建沙洲观测与变化表.py.
2. Run analyses in the order listed in 'code/'; each script writes its tables to outputs/ and its figure source data to outputs/figures/source_data/.
3. Render figures: 绘制当前结果图.py (matplotlib, Times New Roman labels; exports SVG/PDF/PNG/TIFF).

Scripts expect the derived tables in 'data/' of this repository plus the non-distributed mask archive; path columns are sanitized to 'local://' prefixes here.

## Targeted reviewer checks (Supplementary S4.4–S4.6 and S5-4)

Current results are in `data/analysis_tables/稳健性扩展/`:

- `面积近稳轨迹采样分层.csv`: criterion decomposition and splits by observation count and record length.
- `等数量季节模板核验.csv`: held-out-year comparison with equal same/opposite-quarter training counts.
- `高波浪日季节与比例核验.csv`: count and fraction models with interval-duration and annual-season controls.
- `高波浪日核验区间.csv`: the 1,228 derived intervals used to refit these models, with 54 cays and 50 reef clusters.
- `审稿意见针对性核验.json`: model formulas, uncertainty, sample sizes, test-family definitions and the WAVERYS variable metadata inspected from the original NetCDF.

These checks can be rerun from the published derived tables without imagery, masks or daily forcing files. From the repository root, first prepare the working layout:

```python
from pathlib import Path
import shutil
shutil.copytree('data/analysis_tables', 'outputs', dirs_exist_ok=True)
for name in ('稳健性扩展检验.py', '构建沙洲观测与变化表.py'):
    shutil.copy2(Path('code') / name, name)
```

Then run:

```text
python 稳健性扩展检验.py --review-comments-from-tables
```

Dependencies: Python, numpy, pandas, scipy, statsmodels, opencv-python, scikit-image, xarray and h5netcdf. The existing `--review-comments-only` mode constructs the interval inputs from the full local archive; the default mode retains the original mask-perturbation checks. Both archive-based modes require the corresponding non-distributed source files and original directory layout.

Direction models use VSDX/VSDY (surface Stokes drift, positive east/north), together with GLORYS uo/vo. VMDR is a wave-from direction and is not used to construct regression vectors. The published metadata documents this variable distinction. The checks do not estimate an independent registration RMSE or a cumulative-position-error null model.

## Source data and provenance

- Sentinel-2 L2A reflectance: Copernicus Data Space (https://dataspace.copernicus.eu), tiles and dates as listed in the study registry.
- Google Earth historical imagery: fixed-camera, north-up screenshots exported under Google Earth terms; pixel sizes recorded per reference frame.
- ERA5 (10 m wind): Copernicus Climate Data Store, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels . GLORYS12V1 (current): Copernicus Marine Service, https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030 . WAVERYS (wave): Copernicus Marine Service, https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_WAV_001_032 . IBTrACS: NOAA NCEI, https://www.ncei.noaa.gov/products/international-best-track-archive .
- All outlines are visible waterlines at acquisition time; no tidal normalization was applied.

## Global transferability layer

Scripts in 'code/': 分析全球形态重塑谱系.py (cross-source standardization and regional spectra), 分析全球普适性与可迁移性.py (size scaling, vegetation gradient, cross-source agreement), 分析区域质心移动趋势.py (regional mobility trends and exploratory reef screen), 绘制全球综合图.py (Figure 7 and Supplementary Figures S4/S6), 核查MEDF053上升候选.py (per-scene diagnostics of the single-reef increasing candidate), and 绘制研究区域与区域结果图.py with helpers 区域气候滞后模型.py and 分析沙洲类型与区域差异.py (Figure 1 regional-context panels).

Tables in 'data/analysis_tables/': 全球形态重塑区间汇总.csv (2,159 standardized intervals), 全球形态重塑沙洲汇总.csv, 全球形态重塑区域汇总.csv, 全球形态重塑传感器调整模型.csv (source-term diagnostics), 全球普适性标度拟合.csv, 全球普适性沙洲汇总.csv, 全球植被梯度汇总.csv, 全球跨源一致性.csv, 全球跨源沙洲中位数pivot.csv, 区域质心移动趋势.csv, 区域趋势方向构成.csv, 礁盘质心移动趋势.csv, 区域趋势分析样本.csv, 区域地理映射.csv; audit files 全球形态重塑核查.json, 全球普适性核查.json, 区域质心移动趋势核查.json.

Reproduction order: 分析全球形态重塑谱系.py, then 分析全球普适性与可迁移性.py, then 分析区域质心移动趋势.py, then 绘制全球综合图.py. The global scripts retain their original layout: place them under `全球意义/`, place the global tables under `全球意义/outputs/`, and place 沙洲变化区间.csv and 沙洲观测主表.csv under the root `outputs/`. Shared root-level helper scripts are included in `code/`. Full mask-dependent analyses require the original local archive.

## Data not included

Raw imagery, per-date masks and reef-frame rasters are not distributed (third-party imagery terms and archive size). Mask-derived metrics are fully provided in 'data/analysis_tables/沙洲观测主表.csv' and related tables; mask path columns are sanitized to 'local://' placeholders.

## License

Code: MIT (see 'LICENSE'). Data tables: CC-BY-4.0. Cite the associated manuscript when reusing.

## Version

v1.2, 2026-10-08. Synchronizes the current manuscript-derived tables and figure source data, including the 2,159 cross-source intervals, activity-envelope metrics, geometry decomposition and targeted reviewer checks. The current sample-selection audit is in `data/analysis_tables/研究方法核查.json`.
