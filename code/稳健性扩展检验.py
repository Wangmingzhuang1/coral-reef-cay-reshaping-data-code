"""边界判读与质心位置的必要敏感性检验（补充材料 S4 的证据来源）。"""
from __future__ import annotations
import json
import argparse
import sys
from pathlib import Path
import numpy as np
from 构建沙洲观测与变化表 import read_mask
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
EXT = OUT / '稳健性扩展'
EXT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260915)


def perturb(mask, rng, shift=False):
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mode = rng.integers(0, 3)
    if mode == 1:
        m = cv2.erode(m, kernel, iterations=1)
    elif mode == 2:
        m = cv2.dilate(m, kernel, iterations=1)
    if shift:
        dy, dx = rng.integers(-1, 2, size=2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), borderValue=0)
    return m > 0


def rework_fraction(m0, m1):
    a = int(m0.sum())
    if a == 0:
        return np.nan
    return (int((m0 & ~m1).sum()) + int((m1 & ~m0).sum())) / a


def iou(m0, m1):
    inter = int((m0 & m1).sum())
    union = int((m0 | m1).sum())
    return inter / union if union else np.nan


def block_error_propagation():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs['date'] = pd.to_datetime(obs.date)
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    mask_lookup = obs.drop_duplicates(['sand_cay_id', 'sensor', 'date']).set_index(['sand_cay_id', 'sensor', 'date']).mask_path.to_dict()
    iv = pd.read_csv(OUT / '沙洲变化区间.csv')
    iv = iv[(iv.sensor.eq('sentinel2')) & iv.time_interval_days.between(30, 800)].copy()
    iv['time_t'] = pd.to_datetime(iv.time_t)
    iv['time_t1'] = pd.to_datetime(iv.time_t1)
    rows = []
    for _, r in iv.sample(min(len(iv), 260), random_state=7).iterrows():
        p0 = mask_lookup.get((r.sand_cay_id, r.sensor, r.time_t))
        p1 = mask_lookup.get((r.sand_cay_id, r.sensor, r.time_t1))
        if p0 is None or p1 is None or not Path(p0).is_file() or not Path(p1).is_file():
            continue
        m0, m1 = read_mask(p0), read_mask(p1)
        if m0.shape != m1.shape:
            continue
        obs_rework = rework_fraction(m0, m1)
        do_shift = r.sensor == 'google_earth'
        nulls = [rework_fraction(perturb(m0, RNG, do_shift), perturb(m0, RNG, do_shift)) for _ in range(15)]
        nulls = [v for v in nulls if not np.isnan(v)]
        if not nulls:
            continue
        p_exceed = float(np.mean([v < obs_rework for v in nulls]))
        rows.append({'sand_cay_id': r.sand_cay_id, 'sensor': r.sensor, 'time_t': str(r.time_t.date()), 'area_px': int(m0.sum()), 'observed_reworking': obs_rework, 'null_p95': float(np.quantile(nulls, .95)), 'p_exceed': p_exceed})
    df = pd.DataFrame(rows)
    df.to_csv(EXT / '误差传播超越概率.csv', index=False, encoding='utf-8-sig')
    cay_area = df.groupby('sand_cay_id').area_px.median()
    tert = pd.qcut(cay_area, 3, labels=['small', 'medium', 'large'])
    df['size_stratum'] = df.sand_cay_id.map(tert)
    track = df.groupby('sand_cay_id').p_exceed.median()
    stable = pd.read_csv(OUT / '沙洲面积主轨迹分类.csv')
    stable_ids = set(stable.loc[stable.relative_change.abs().lt(.25), 'sand_cay_id'])
    strat_rows = []
    for stratum in ['small', 'medium', 'large']:
        cays = [c for c in track.index if c in stable_ids and tert.get(c) == stratum]
        st = track[cays]
        sub = df[df.size_stratum.eq(stratum)]
        strat_rows.append({'stratum': stratum, 'n_stable_tracks': len(st), 'fraction_p_exceed_ge_095': float((st >= .95).mean()) if len(st) else np.nan, 'median_observed_reworking': float(sub.observed_reworking.median()), 'median_null_p95': float(sub.null_p95.median())})
    iou_rows = []
    masks = obs[obs.mask_path.map(lambda p: Path(p).is_file())].sample(240, random_state=11)
    for _, r in masks.iterrows():
        m = read_mask(r.mask_path)
        do_shift = r.sensor == 'google_earth'
        for _ in range(8):
            v = 1 - iou(m, perturb(m, RNG, do_shift))
            if not np.isnan(v):
                iou_rows.append({'sensor': r.sensor, 'area_px': int(m.sum()), 'iou_error_change': v})
    iou_df = pd.DataFrame(iou_rows)
    iou_df['size_stratum'] = pd.qcut(iou_df.area_px, 3, labels=['small', 'medium', 'large'])
    iou_strat = iou_df.groupby('size_stratum', observed=True).iou_error_change.agg(['median', lambda s: s.quantile(.95)]).rename(columns={'<lambda_0>': 'p95'})
    iou_df.to_csv(EXT / 'IoU误差零分布.csv', index=False, encoding='utf-8-sig')
    df.to_csv(EXT / '误差传播超越概率.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(strat_rows).to_csv(EXT / '误差传播尺寸分层.csv', index=False, encoding='utf-8-sig')
    iou_strat.reset_index().to_csv(EXT / 'IoU误差尺寸分层.csv', index=False, encoding='utf-8-sig')
    return {'pairs': int(len(df)), 'strata': strat_rows, 'iou_strata': iou_strat.reset_index().to_dict('records')}


def block_centroid_error_and_perturbed_phase():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs['date'] = pd.to_datetime(obs.date)
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    rows = []
    masks = obs[obs.mask_path.map(lambda p: Path(p).is_file())].sample(240, random_state=21)
    for _, r in masks.iterrows():
        m = read_mask(r.mask_path)
        a = m.sum()
        if a < 50:
            continue
        rad = np.sqrt(a / np.pi)
        ys, xs = np.nonzero(m)
        c0 = np.array([xs.mean(), ys.mean()])
        for _ in range(6):
            mp = perturb(m, RNG, False)
            ys2, xs2 = np.nonzero(mp)
            if len(xs2) == 0:
                continue
            c1 = np.array([xs2.mean(), ys2.mean()])
            rows.append({'area_px': int(a), 'centroid_error_radii': float(np.linalg.norm(c1 - c0) / rad)})
    cen = pd.DataFrame(rows)
    cen['size_stratum'] = pd.qcut(cen.area_px, 3, labels=['small', 'medium', 'large'])
    cen_strat = cen.groupby('size_stratum', observed=True).centroid_error_radii.agg(['median', lambda s: s.quantile(.95)]).rename(columns={'<lambda_0>': 'p95'})
    cen.to_csv(EXT / '质心误差等效半径.csv', index=False, encoding='utf-8-sig')
    cen_strat.reset_index().to_csv(EXT / '质心误差尺寸分层.csv', index=False, encoding='utf-8-sig')
    pair = pd.read_csv(OUT / 'Sentinel2季节形态配对.csv')
    mask_by_image = obs.drop_duplicates('image_id').set_index('image_id').mask_path.to_dict()
    need = [c for c in ['sand_cay_id', 'left_image_id', 'right_image_id', 'phase_distance', 'gap_years', 'iou'] if c in pair.columns]
    if len(need) < 6:
        return {'centroid_strata': cen_strat.reset_index().to_dict('records'), 'perturbed_phase': {'status': 'skipped_columns'}}
    sub = pair[need].dropna()
    import statsmodels.api as sm
    def fit_beta(frame, col):
        d = frame[['phase_distance', 'gap_years']].copy()
        d['gap2'] = d.gap_years ** 2
        d = pd.concat([d, pd.get_dummies(frame.sand_cay_id, drop_first=True, dtype=float)], axis=1)
        model = sm.OLS(frame[col], sm.add_constant(d)).fit(cov_type='cluster', cov_kwds={'groups': frame.sand_cay_id})
        return float(model.params['phase_distance']), float(model.bse['phase_distance'])
    beta0, se0 = fit_beta(sub, 'iou')
    samp = sub.sample(min(len(sub), 900), random_state=31).copy()
    iou_p = []
    for _, r in samp.iterrows():
        p0 = mask_by_image.get(r.left_image_id)
        p1 = mask_by_image.get(r.right_image_id)
        if p0 is None or p1 is None or not Path(p0).is_file() or not Path(p1).is_file():
            iou_p.append(np.nan)
            continue
        m0, m1 = read_mask(p0), read_mask(p1)
        iou_p.append(iou(perturb(m0, RNG, False), perturb(m1, RNG, False)) if m0.shape == m1.shape else np.nan)
    samp['iou_pert'] = iou_p
    samp = samp.dropna(subset=['iou_pert'])
    beta1, se1 = fit_beta(samp, 'iou_pert')
    out = {'beta_original': beta0, 'se_original': se0, 'beta_perturbed': beta1, 'se_perturbed': se1, 'n_perturbed_pairs': int(len(samp)), 'centroid_strata': cen_strat.reset_index().to_dict('records')}
    pd.DataFrame([{'metric': 'phase_beta_original', 'value': beta0, 'se': se0}, {'metric': 'phase_beta_perturbed', 'value': beta1, 'se': se1}]).to_csv(EXT / '扰动掩膜相位系数.csv', index=False, encoding='utf-8-sig')
    return out












def block_reviewer_checks(from_tables=False):
    """仅核验采样分层、等数量季节模板及高波浪日的窗口/季节口径。"""
    from statsmodels.stats.multitest import multipletests
    import xarray as xr
    import statsmodels.formula.api as smf

    recorded = json.loads((EXT / '审稿意见针对性核验.json').read_text(encoding='utf-8')) if from_tables else None

    def fit_clustered(formula, data):
        return smf.ols(formula, data=data).fit(
            cov_type='cluster', cov_kwds={'groups': data['reef_id'], 'use_correction': True})

    tracks = pd.read_csv(OUT / '沙洲面积主轨迹分类.csv')
    near = tracks.loc[tracks.relative_change.abs().lt(.25)].copy()
    flags = pd.DataFrame({
        'path': near.centroid_path_normalized.gt(1),
        'area_cv': near.late_cv.gt(.15),
        'area_ratio': near.max_adjacent_area_ratio.gt(1.5),
    }, index=near.index)
    near['exceeds_stability'] = flags.any(axis=1)
    strata = []
    for column in ('n_observations', 'span_years'):
        median = float(near[column].median())
        for label, subset in (
            ('at_or_below_median', near.loc[near[column].le(median)]),
            ('above_median', near.loc[near[column].gt(median)]),
        ):
            strata.append({'variable': column, 'median_split': median, 'group': label,
                           'n_tracks': len(subset),
                           'n_exceeds_stability': int(subset.exceeds_stability.sum()),
                           'fraction': float(subset.exceeds_stability.mean())})
    pd.DataFrame(strata).to_csv(EXT / '面积近稳轨迹采样分层.csv', index=False, encoding='utf-8-sig')

    template = pd.read_csv(OUT / 'Sentinel2季节模板交叉验证.csv')
    equal = template.loc[template.n_same_quarter_training.eq(template.n_opposite_quarter_training)]
    values = equal.groupby('sand_cay_id').gain_same_vs_opposite.mean().to_numpy()
    rng = np.random.default_rng(20261008)
    boot = rng.choice(values, size=(1999, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    template_result = {'n_images': len(equal), 'n_cays': len(values),
                       'mean_gain': float(values.mean()), 'ci95_low': float(low),
                       'ci95_high': float(high), 'positive_cays': int((values > 0).sum()),
                       'bootstrap': 1999, 'seed': 20261008,
                       'scope': 'same versus opposite quarter training counts equal; held-out year unchanged'}
    pd.DataFrame([template_result]).to_csv(EXT / '等数量季节模板核验.csv', index=False, encoding='utf-8-sig')

    if from_tables:
        bare = pd.read_csv(EXT / '高波浪日核验区间.csv', parse_dates=['midpoint'])
    else:
        archive = ROOT / '额外分析_区域与气候响应' / '归档_裸沙质心轨迹周期与突发'
        sys.path.insert(0, str(archive))
        from 分析裸沙质心轨迹风浪对照 import build_usable
        from 分析裸沙质心轨迹周期与突发 import build_steps
        from 分析沙洲形状与质心迁移 import load_observations
    
        observations = load_observations()
        steps = build_steps(observations)
        forcing = pd.read_csv(OUT / '流场波浪区间特征.csv', low_memory=False)
        daily = pd.read_csv(ROOT / 'data/environment/harmonized/daily/reef_daily_features.csv',
                            usecols=['reef_id', 'time', 'VHM0', 'u10', 'v10'], low_memory=False)
        daily['time'] = pd.to_datetime(daily.time, format='ISO8601')
        usable, _ = build_usable(observations, steps, forcing, daily)
        bare = usable.loc[usable.stratum.eq('bare')].dropna(subset=['event_storm_days']).copy()
    assert bare.waverys_complete.eq(1).all(), '高波浪日比例需要完整日覆盖。'
    bare['high_wave_fraction'] = bare.event_storm_days / bare.time_interval_days
    assert bare.high_wave_fraction.between(0, 1).all()
    phase = 2 * np.pi * bare.midpoint.dt.dayofyear / 365.2425
    bare['season_sin'], bare['season_cos'] = np.sin(phase), np.cos(phase)
    bare[['transition_id', 'reef_id', 'sand_cay_id', 'midpoint', 'time_interval_days',
          'speed_norm', 'log_speed', 'event_storm_days', 'waverys_complete',
          'high_wave_fraction', 'season_sin', 'season_cos']].to_csv(
              EXT / '高波浪日核验区间.csv', index=False, encoding='utf-8-sig')
    rows = []
    specs = [
        ('count_duration', 'event_storm_days', ''),
        ('count_duration_season', 'event_storm_days', ' + season_sin + season_cos'),
        ('fraction_duration_season', 'high_wave_fraction', ' + season_sin + season_cos'),
    ]
    for label, term, season in specs:
        formula = f'log_speed ~ {term} + time_interval_days{season} + C(sand_cay_id)'
        fit = fit_clustered(formula, bare)
        low, high = fit.conf_int().loc[term]
        rows.append({'model': label, 'term': term, 'estimate': float(fit.params[term]),
                     'ci95_low': float(low), 'ci95_high': float(high),
                     'p_value': float(fit.pvalues[term]), 'n_intervals': len(bare),
                     'n_cays': bare.sand_cay_id.nunique(), 'n_reefs': bare.reef_id.nunique(),
                     'cluster': 'reef_id', 'formula': formula})
    if from_tables:
        reference = recorded['high_wave_models'][0]
        reference_estimate, reference_q = reference['estimate'], reference['fdr_q_value']
    else:
        reference = pd.read_csv(archive / 'outputs/裸沙质心轨迹风浪对照模型.csv')
        reference = reference.loc[reference.analysis.eq('E4_event_storm_days_bare')].iloc[0]
        reference_estimate, reference_q = reference.estimate, reference.fdr_q_value
    assert np.isclose(rows[0]['estimate'], reference_estimate, rtol=1e-7)
    # 原模型保留原检验族的 q；两项新增敏感性检验组成独立的校正组。
    rows[0]['fdr_q_value'] = float(reference_q)
    for row, q in zip(rows[1:], multipletests([r['p_value'] for r in rows[1:]], method='fdr_bh')[1]):
        row['fdr_q_value'] = float(q)
    pd.DataFrame(rows).to_csv(EXT / '高波浪日季节与比例核验.csv', index=False, encoding='utf-8-sig')

    if from_tables:
        metadata = recorded['waverys_metadata']
        metadata_source = recorded['waverys_metadata_source']
    else:
        source = next((ROOT / 'data/environment/raw/waverys').rglob('*.nc'))
        with source.open('rb') as handle, xr.open_dataset(handle, engine='h5netcdf') as dataset:
            metadata = {key: {
                **{name: str(value) for name, value in dataset[key].attrs.items()
                   if name in ('standard_name', 'long_name', 'units')},
                **{name: float(dataset[key].encoding.get(name, 0))
                   for name in ('add_offset', 'scale_factor')},
            } for key in ('VSDX', 'VSDY', 'VMDR')}
        metadata_source = str(source.relative_to(ROOT))
    result = {'trajectory_counts': {'near_stable': len(near),
              'exceeds_stability': int(near.exceeds_stability.sum()),
              'path': int(flags.path.sum()), 'area_cv': int(flags.area_cv.sum()),
              'area_ratio': int(flags.area_ratio.sum()),
              'path_and_area': int((flags.path & (flags.area_cv | flags.area_ratio)).sum()),
              'path_only': int((flags.path & ~flags.area_cv & ~flags.area_ratio).sum())},
              'sampling_strata': strata, 'equal_training_template': template_result,
              'high_wave_models': rows, 'waverys_metadata_source': metadata_source,
              'waverys_metadata': metadata,
              'scope': 'No independent registration RMSE or cumulative-error null model estimated.'}
    (EXT / '审稿意见针对性核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='边界误差与审稿意见的有限核验。')
    parser.add_argument('--review-comments-only', action='store_true', help='只核验现有轨迹、模板及高波浪日口径。')
    parser.add_argument('--review-comments-from-tables', action='store_true',
                        help='从公开的派生区间表复算本轮核验，无需影像及日环境数据。')
    args = parser.parse_args()
    if args.review_comments_only or args.review_comments_from_tables:
        block_reviewer_checks(from_tables=args.review_comments_from_tables)
        return
    # 当前论文只需要边界误差和质心/轮廓敏感性；必需检验失败应直接报错。
    summary = {
        "error_propagation": block_error_propagation(),
        "centroid_error_perturbed_phase": block_centroid_error_and_perturbed_phase(),
    }
    (EXT / '稳健性扩展汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
