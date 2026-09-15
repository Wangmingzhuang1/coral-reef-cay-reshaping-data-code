# Coral reef cay reworking: derived data, figure source data and analysis code

中文说明：本仓库发布论文《面积近稳掩盖了珊瑚礁沙洲的持续重塑》的派生数据表、图源数据、环境强迫区间聚合表与分析代码。原始影像与沙洲掩膜因许可与体积原因不在此发布（见下文 Data not included）。

## Contents

- 'data/analysis_tables/' - derived analysis tables produced by the study pipeline (trajectory classification, seasonal recurrence pairs and harmonic validation, directional coupling models and sensitivities, vegetation-mobility analyses, continuous-time state models, typhoon event pairing, ENSO lag models, wind increment tests, error propagation and detectability outputs), plus 'figure_manifest.json' describing figure order and evidence levels.
- 'data/figure_source_data/' - one CSV per main/supplementary figure panel group, sufficient to regenerate every plot element.
- 'data/environment_forcing/' - interval-aggregated forcing tables (ERA5 wind, GLORYS current, WAVERYS wave) extracted per reef frame and joined to observation intervals (t, t1].
- 'code/' - analysis and figure scripts (Chinese filenames). Entry points: 分析沙洲面积轨迹.py, 分析Sentinel2季节形态重复性.py, 对齐流场波浪与沙洲变化.py, 分析植被形态台风关系.py, 拟合Sentinel2三状态连续时间模型.py, 分析台风事件前后沙洲响应.py, 分析ENSO与沙洲面积.py, 稳健性扩展检验.py, 误差传播与补充检验.py, 绘制当前结果图.py.

## Reproduction order

1. Build observation and change tables from the archived mask registry (registry not distributed; see below): 构建沙洲观测与变化表.py.
2. Run analyses in the order listed in 'code/'; each script writes its tables to outputs/ and its figure source data to outputs/figures/source_data/.
3. Render figures: 绘制当前结果图.py (matplotlib, Times New Roman labels; exports SVG/PDF/PNG/TIFF).

Scripts expect the derived tables in 'data/' of this repository plus the non-distributed mask archive; path columns are sanitized to 'local://' prefixes here.

## Source data and provenance

- Sentinel-2 L2A reflectance: Copernicus Data Space (https://dataspace.copernicus.eu), tiles and dates as listed in the study registry.
- Google Earth historical imagery: fixed-camera, north-up screenshots exported under Google Earth terms; pixel sizes recorded per reference frame.
- ERA5 (wind), GLORYS (current), WAVERYS (wave): Copernicus Marine Service; IBTrACS: NOAA; tide model pre-registration FES2022b (tide sensitivity check not executed, data not obtained at analysis time).
- All outlines are visible waterlines at acquisition time; no tidal normalization was applied.

## Data not included

Raw imagery, per-date masks and reef-frame rasters are not distributed (third-party imagery terms and archive size). Mask-derived metrics are fully provided in 'data/analysis_tables/沙洲观测主表.csv' and related tables; mask path columns are sanitized to 'local://' placeholders.

## License

Code: MIT (see 'LICENSE'). Data tables: CC-BY-4.0. Cite the associated manuscript when reusing.

## Version

v1.0, 2026-09-15. Matches manuscript analysis contract '2026-09-15-real-cay-axis-1'.
