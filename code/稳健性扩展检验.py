"""误差传播、方差分解与补充敏感性检验（补充材料 S4 的证据来源）。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
EXT = OUT / '稳健性扩展'
EXT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260915)


def read_mask(path):
    buf = np.fromfile(Path(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img > 0


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


def block_variance_components():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs['date'] = pd.to_datetime(obs.date)
    obs['doy'] = obs.date.dt.dayofyear
    metrics = ['sand_cay_centroid_x_norm', 'sand_cay_centroid_y_norm', 'sand_cay_compactness', 'sand_cay_elongation', 'sand_cay_major_axis_angle', 'sand_cay_area_m2', 'vegetation_fraction']
    rows = []
    for metric in metrics:
        sub = obs[['sand_cay_id', 'doy', metric]].dropna()
        sub = sub.groupby('sand_cay_id').filter(lambda g: len(g) >= 4)
        seas, within, between = [], [], []
        for cay, g in sub.groupby('sand_cay_id'):
            y = g[metric].to_numpy()
            t = 2 * np.pi * g.doy.to_numpy() / 365.2425
            X = np.column_stack([np.ones(len(y)), np.sin(t), np.cos(t)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            within_var = y.var(ddof=1)
            seas_var = pred.var(ddof=1)
            seas.append(seas_var / within_var if within_var > 0 else np.nan)
            within.append(within_var)
            between.append((y.mean() - sub[metric].mean()) ** 2)
        within = np.mean(within)
        between = np.mean(between) * 1
        rows.append({'metric': metric, 'median_seasonal_share_of_within': float(np.nanmedian(seas)), 'between_over_within': float(between / within) if within > 0 else np.nan, 'n_cays': int(sub.sand_cay_id.nunique())})
    df = pd.DataFrame(rows)
    df.to_csv(EXT / '季节与年际方差分解.csv', index=False, encoding='utf-8-sig')
    return df.set_index('metric').median_seasonal_share_of_within.round(3).to_dict()


def block_transition_seasonality():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs['date'] = pd.to_datetime(obs.date)
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    state_map = {'none': 'low', 'sparse': 'low', 'partial': 'partial', 'dominant': 'dominant'}
    obs['state'] = obs.automatic_vegetation_state.map(state_map)
    obs = obs.dropna(subset=['state'])
    coords = pd.read_csv(OUT / '潮位影像时刻与坐标.csv')
    cay_lat = coords.groupby('sand_cay_id').sand_cay_center_lat.mean()
    events, exposure = [], []
    for cay, g in obs.groupby('sand_cay_id'):
        g = g.sort_values('date')
        south = bool(cay_lat.get(cay, 0) < 0)
        dates = g.date.to_list()
        states = g.state.to_list()
        for i in range(len(dates) - 1):
            mid = dates[i] + (dates[i + 1] - dates[i]) / 2
            season = 'NovApr' if (((mid.month % 12) in (10, 11, 0, 1, 2, 3)) != south) else 'MayOct'
            exposure.append({'state': states[i], 'season': season, 'days': (dates[i + 1] - dates[i]).days})
            if states[i] != states[i + 1]:
                events.append({'from': states[i], 'to': states[i + 1], 'season': season})
    ev = pd.DataFrame(events)
    ex = pd.DataFrame(exposure).groupby(['state', 'season']).days.sum()
    out = []
    for (fr, to), grp in ev.groupby(['from', 'to']):
        tot = len(grp)
        tot_e = ex.get((fr, 'NovApr'), 0) + ex.get((fr, 'MayOct'), 0)
        if tot_e == 0:
            continue
        for season in ['NovApr', 'MayOct']:
            e = ex.get((fr, season), 0)
            out.append({'from': fr, 'to': to, 'season': season, 'observed': int((grp.season == season).sum()), 'expected': float(tot * e / tot_e)})
    df = pd.DataFrame(out)
    df['ratio'] = df.observed / df.expected.replace(0, np.nan)
    from scipy import stats as st
    from statsmodels.stats.multitest import multipletests
    keys = list(df.groupby(['from', 'to']).groups.keys())
    ps = []
    for k in keys:
        grp = df[df[['from', 'to']].eq(pd.Series(k, index=['from', 'to'])).all(axis=1)]
        o = grp.observed.to_numpy()
        e = grp.expected.to_numpy()
        chi = ((o - e) ** 2 / np.where(e > 0, e, 1)).sum()
        ps.append(st.chi2.sf(chi, 1))
    qs = multipletests(ps, method='fdr_bh')[1]
    df['p_value'] = np.repeat(ps, 2)
    df['fdr_q_value'] = np.repeat(qs, 2)
    df.to_csv(EXT / '状态转换季节异质性.csv', index=False, encoding='utf-8-sig')
    sig = df[df.fdr_q_value.lt(.05)][['from', 'to', 'season', 'observed', 'expected', 'ratio']]
    return {'transitions_tested': int(len(keys)), 'seasonal_significant': sig.to_dict('records')}


def block_establishment_pretrend():
    ev = pd.read_csv(OUT / '植被建立沙洲内前后对照.csv')
    ev['establishment_date'] = pd.to_datetime(ev.establishment_date)
    iv = pd.read_csv(OUT / '沙洲变化区间.csv')
    iv['time_t1'] = pd.to_datetime(iv.time_t1)
    rows = []
    for _, e in ev.iterrows():
        g = iv[(iv.sand_cay_id.eq(e.sand_cay_id)) & (iv.sensor.eq(e.sensor)) & (iv.time_t1.le(e.establishment_date))].sort_values('time_t1')
        pre = np.nan
        if len(g) >= 2:
            half = len(g) // 2
            early = g.centroid_shift_m_per_year.iloc[:half].median()
            late = g.centroid_shift_m_per_year.iloc[half:].median()
            if early > 0 and late > 0:
                pre = float(np.log(late / early))
        rows.append({'sand_cay_id': e.sand_cay_id, 'sensor': e.sensor, 'establishment_date': str(e.establishment_date.date()), 'log_ratio_after_before': e.log_ratio_after_before, 'pre_trend_log_ratio': pre, 'n_before_intervals': len(g)})
    df = pd.DataFrame(rows)
    df.to_csv(EXT / '植被建立前趋势.csv', index=False, encoding='utf-8-sig')
    pre = df.pre_trend_log_ratio.dropna()
    return {'n_events': int(len(df)), 'median_log_ratio_after_before': float(df.log_ratio_after_before.median()), 'iqr_after_before': [float(df.log_ratio_after_before.quantile(.25)), float(df.log_ratio_after_before.quantile(.75))], 'n_with_pretrend': int(len(pre)), 'median_pre_trend': float(pre.median()) if len(pre) else None, 'iqr_pre_trend': [float(pre.quantile(.25)), float(pre.quantile(.75))] if len(pre) else None}


def block_mde():
    m = pd.read_csv(OUT / '强迫植被沙洲内模型.csv')
    rows = []
    for spec, term, outcome in [('within_cay_fixed_effects', 'current_cross_mean_z', 'cross_shift'), ('within_cay_fixed_effects', 'current_cross_mean_z', 'along_shift'), ('between_within_random_intercept', 'veg_within_z', 'log_gross_mobility')]:
        r = m[(m.specification.eq(spec)) & (m.term.eq(term)) & (m.outcome.eq(outcome))]
        if r.empty:
            continue
        r = r.iloc[0]
        se = (r.ci95_high - r.ci95_low) / 3.92
        rows.append({'outcome': outcome, 'term': term, 'se': float(se), 'mde_80_power': float(2.80 * se), 'observed': float(r.coefficient)})
    df = pd.DataFrame(rows)
    df.to_csv(EXT / '空结果最小可检测效应.csv', index=False, encoding='utf-8-sig')
    return df.to_dict('records')


def block_moran():
    core = pd.read_csv(OUT / '流场波浪沿轴横轴分解.csv')
    core = core[(core.sensor.eq('sentinel2')) & core.time_interval_days.between(90, 365)]
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    # 坐标来自潮位影像时刻与坐标表（礁盘级均值）
    coords = pd.read_csv(OUT / '潮位影像时刻与坐标.csv').groupby('reef_id')[['sand_cay_center_lon', 'sand_cay_center_lat']].mean().rename(columns={'sand_cay_center_lon': 'lon', 'sand_cay_center_lat': 'lat'})
    terms = [c for c in ['current_along_mean', 'wave_vector_along_mean', 'current_cross_mean', 'wave_vector_cross_mean'] if c in core.columns]
    resp = 'centroid_along_shift_normalized_per_year'
    if resp not in core.columns or not terms:
        return {'status': 'skipped_missing_columns'}
    sub = core[terms + [resp, 'sand_cay_id', 'reef_id']].dropna()
    X = sub[terms].to_numpy()
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    y = sub[resp].to_numpy()
    dummies = pd.get_dummies(sub.sand_cay_id).to_numpy()
    A = np.column_stack([X, dummies])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    reef_res = pd.Series(resid, index=sub.reef_id.to_numpy()).groupby(level=0).mean()
    reef_res = reef_res.reindex(coords.index).dropna()
    P = coords.loc[reef_res.index].to_numpy()
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    W = 1 / np.where(D > 0, D, np.inf)
    np.fill_diagonal(W, 0)
    z = reef_res.to_numpy()
    z = z - z.mean()
    n = len(z)
    def moran(z):
        return n / W.sum() * (z[:, None] * z[None, :] * W).sum() / (z ** 2).sum()
    obs_I = moran(z)
    perm = np.array([moran(z[RNG.permutation(n)]) for _ in range(999)])
    p = float(np.mean(perm >= obs_I))
    pd.DataFrame({'reef_id': reef_res.index, 'mean_residual': reef_res.to_numpy()}).to_csv(EXT / '方向模型残差礁盘均值.csv', index=False, encoding='utf-8-sig')
    return {'morans_I': float(obs_I), 'permutation_p': p, 'n_reefs': n}


def main():
    summary = {}
    for name, fn in [('error_propagation', block_error_propagation), ('centroid_error_perturbed_phase', block_centroid_error_and_perturbed_phase), ('variance_components', block_variance_components), ('transition_seasonality', block_transition_seasonality), ('establishment_pretrend', block_establishment_pretrend), ('mde', block_mde), ('moran', block_moran)]:
        try:
            summary[name] = fn()
        except Exception as exc:
            summary[name] = {'error': f'{type(exc).__name__}: {exc}'}
    (EXT / '稳健性扩展汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
