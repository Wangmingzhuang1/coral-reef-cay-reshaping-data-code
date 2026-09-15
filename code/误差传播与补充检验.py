"""测量误差传播、方差分解、季节 CTMC、建立事件前趋势、MDE 与残差空间自相关补充检验。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.linalg import expm
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.optimize import minimize
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
RNG = np.random.default_rng(20260915)
D_PAIR = 200
D_IOU = 60


def read_mask(path):
    return np.asarray(Image.open(path)) > 0


def perturb(mask, rng):
    m = binary_erosion(mask) if rng.random() < .5 else binary_dilation(mask)
    dx, dy = rng.integers(-1, 2, size=2)
    return np.roll(np.roll(m, dx, axis=0), dy, axis=1)


def centroid_area(mask):
    yy, xx = np.nonzero(mask)
    if len(yy) == 0:
        return None
    return (yy.mean(), xx.mean()), len(yy)


def error_propagation():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    obs['date'] = pd.to_datetime(obs.date)
    tracks = pd.read_csv(OUT / '沙洲面积主轨迹分类.csv')
    rows = []
    iou_deficits = []
    for tr in tracks.itertuples():
        seq = obs[obs.sand_cay_id.eq(tr.sand_cay_id) & obs.sensor.eq(tr.sensor)].sort_values('date')
        seq = seq[seq.mask_path.map(lambda p: Path(p).is_file())]
        if len(seq) < 4:
            continue
        masks = [read_mask(p) for p in seq.mask_path]
        ca = [centroid_area(m) for m in masks]
        radii = [np.sqrt(a / np.pi) for _, a in ca]
        obs_path = float(tr.centroid_path_normalized)
        obs_net = float(abs(tr.relative_change)) * 100
        null_paths = np.zeros(D_PAIR)
        for i in range(len(masks) - 1):
            r0 = radii[i]
            shifts = []
            for _ in range(D_PAIR):
                c1, _ = centroid_area(perturb(masks[i], RNG))
                c2, _ = centroid_area(perturb(masks[i], RNG))
                shifts.append(np.hypot(c1[0] - c2[0], c1[1] - c2[1]) / r0)
            null_paths += np.asarray(shifts)
        first, last = masks[0], masks[-1]
        a0 = ca[0][1]
        null_nets = []
        for _ in range(D_PAIR):
            _, af = centroid_area(perturb(last, RNG))
            _, a0p = centroid_area(perturb(first, RNG))
            null_nets.append(abs(af - a0p) / a0p * 100)
        null_nets = np.asarray(null_nets)
        for m in masks:
            for _ in range(max(2, D_IOU // len(masks))):
                p1, p2 = perturb(m, RNG), perturb(m, RNG)
                inter = np.logical_and(p1, p2).sum()
                union = np.logical_or(p1, p2).sum()
                if union:
                    iou_deficits.append(1 - inter / union)
        rows.append({'sand_cay_id': tr.sand_cay_id, 'sensor': tr.sensor, 'n_obs': len(masks),
                     'observed_path_radii': obs_path, 'null_path_mean': float(null_paths.mean()),
                     'null_path_p95': float(np.quantile(null_paths, .95)),
                     'p_null_ge_observed_path': float((null_paths >= obs_path).mean()),
                     'observed_net_pct': obs_net, 'null_net_p95': float(np.quantile(null_nets, .95)),
                     'p_null_ge_observed_net': float((null_nets >= obs_net).mean()),
                     'net_area_type': ('Net-area stable' if abs(tr.relative_change) < .25 else ('Moderate net change' if abs(tr.relative_change) < .5 else 'Major net change'))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / '误差传播零分布.csv', index=False, encoding='utf-8-sig')
    iou_deficits = np.asarray(iou_deficits)
    iou_sum = pd.DataFrame({'statistic': ['median', 'p95', 'p99', 'n'],
                            'iou_deficit': [float(np.median(iou_deficits)), float(np.quantile(iou_deficits, .95)), float(np.quantile(iou_deficits, .99)), float(len(iou_deficits))]})
    iou_sum.to_csv(OUT / 'IoU噪声零带.csv', index=False, encoding='utf-8-sig')
    stable = df[df.net_area_type.eq('Net-area stable')]
    return {'tracks': len(df), 'stable_tracks': len(stable),
            'stable_path_p_lt_05': float((stable.p_null_ge_observed_path < .05).mean()),
            'iou_deficit_p95': float(np.quantile(iou_deficits, .95))}


def variance_components():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    obs['date'] = pd.to_datetime(obs.date)
    obs['doy'] = obs.date.dt.dayofyear
    obs['sin'] = np.sin(2 * np.pi * obs.doy / 365.2425)
    obs['cos'] = np.cos(2 * np.pi * obs.doy / 365.2425)
    metrics = {'centroid_x': 'sand_cay_centroid_x_norm', 'centroid_y': 'sand_cay_centroid_y_norm',
               'compactness': 'sand_cay_compactness', 'elongation': 'sand_cay_elongation',
               'major_axis_angle': 'sand_cay_major_axis_angle', 'log_area': None}
    obs['log_area'] = np.log(obs.sand_cay_area_m2)
    rows = []
    for name, col in metrics.items():
        col = col or 'log_area'
        d = obs[['sand_cay_id', col, 'sin', 'cos']].dropna()
        d = d[d.sand_cay_id.map(d.sand_cay_id.value_counts().ge(6))]
        if d.empty:
            continue
        X = sm.add_constant(d[['sin', 'cos']])
        model = sm.MixedLM(d[col], X, groups=d.sand_cay_id).fit(reml=True)
        pred = model.fe_params['const'] + d.sin * model.fe_params['sin'] + d.cos * model.fe_params['cos']
        within_harmonic = float(pred.groupby(d.sand_cay_id.values).transform(lambda s: s - s.mean()).var())
        var_cay = float(model.cov_re.iloc[0, 0])
        var_resid = float(model.scale)
        rows.append({'metric': name, 'harmonic_within_var': within_harmonic, 'between_cay_var': var_cay,
                     'residual_var': var_resid,
                     'seasonal_share_within': within_harmonic / (within_harmonic + var_resid),
                     'n_obs': len(d), 'n_cays': d.sand_cay_id.nunique()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / '季节方差分量.csv', index=False, encoding='utf-8-sig')
    return df[['metric', 'seasonal_share_within']].round(3).to_dict('records')


def _loglik(params, pairs, n_states=3):
    q = np.zeros((n_states, n_states))
    k = 0
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                q[i, j] = np.exp(params[k])
                k += 1
    np.fill_diagonal(q, -q.sum(axis=1))
    ll = 0.0
    for dt, P_target in pairs:
        P = expm(q * dt)
        ll += np.log(max(P[P_target], 1e-12))
    return -ll


def seasonal_ctmc():
    obs = pd.read_csv(OUT / '沙洲观测主表.csv')
    obs = obs[(obs.sensor.eq('sentinel2')) & obs.quality_grade.isin(['A', 'B'])].copy()
    obs['date'] = pd.to_datetime(obs.date)
    state_map = {'none': 0, 'sparse': 0, 'partial': 1, 'dominant': 2}
    obs['state'] = obs.automatic_vegetation_state.map(state_map)
    obs = obs.dropna(subset=['state'])
    coords = pd.read_csv(ROOT / 'data' / 'source_dataset' / 'data' / 'metadata' / 'standardized_coordinates.csv')
    lat_col = [c for c in coords.columns if 'lat' in c.lower()]
    lat = coords.set_index(coords.columns[0])[lat_col[0]] if lat_col else None
    pairs = []
    for cay, g in obs.sort_values('date').groupby('sand_cay_id'):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            dt_days = (g.date[i + 1] - g.date[i]).days
            if 30 <= dt_days <= 365:
                mid_month = (g.date[i] + pd.Timedelta(days=dt_days / 2)).month
                latv = float(lat.loc[g.reef_id[i]]) if lat is not None and g.reef_id[i] in lat.index else 0.0
                warm = (4 <= mid_month <= 9) if latv >= 0 else (mid_month <= 3 or mid_month >= 10)
                pairs.append((dt_days / 365.25, (int(g.state[i]), int(g.state[i + 1])), g.date[i + 1].year, warm))
    hom = [(dt, ij) for dt, ij, _, _ in pairs]
    x0 = np.log(np.full(6, .3))
    res_hom = minimize(_loglik, x0, args=(hom,), method='L-BFGS-B')
    ll_hom = -res_hom.fun
    ll_seas = 0.0
    loyo = {'hom': 0.0, 'seas': 0.0}
    years = sorted({y for _, _, y, _ in pairs})
    for warm in (True, False):
        sub = [(dt, ij) for dt, ij, _, w in pairs if w == warm]
        if len(sub) < 30:
            ll_seas += -_loglik(res_hom.x, sub)
            continue
        r = minimize(_loglik, x0, args=(sub,), method='L-BFGS-B')
        ll_seas += -r.fun
    for y in years:
        train = [(dt, ij) for dt, ij, yy, _ in pairs if yy != y]
        test = [(dt, ij) for dt, ij, yy, _ in pairs if yy == y]
        if not test or len(train) < 60:
            continue
        rh = minimize(_loglik, x0, args=(train,), method='L-BFGS-B')
        loyo['hom'] += -_loglik(rh.x, test)
        ls = 0.0
        for warm in (True, False):
            tr = [(dt, ij) for dt, ij, yy, w in pairs if yy != y and w == warm]
            te = [(dt, ij) for dt, ij, yy, w in pairs if yy == y and w == warm]
            if not te:
                continue
            if len(tr) >= 30:
                rw = minimize(_loglik, x0, args=(tr,), method='L-BFGS-B')
                ls += -_loglik(rw.x, te)
            else:
                ls += -_loglik(rh.x, te)
        loyo['seas'] += ls
    lr = 2 * (ll_seas - ll_hom)
    p = float(stats.chi2.sf(lr, 6))
    out = pd.DataFrame([{'model': 'homogeneous', 'loglik': ll_hom, 'loyo_loglik': loyo['hom']},
                        {'model': 'seasonal_piecewise', 'loglik': ll_seas, 'loyo_loglik': loyo['seas']},
                        {'model': 'lr_test', 'loglik': lr, 'loyo_loglik': p}])
    out.to_csv(OUT / '分段齐次CTMC比较.csv', index=False, encoding='utf-8-sig')
    return {'n_pairs': len(pairs), 'lr_stat': float(lr), 'lr_p': p, 'loyo_hom': loyo['hom'], 'loyo_seas': loyo['seas']}


def establishment_pretrend():
    events = pd.read_csv(OUT / '植被建立沙洲内前后对照.csv')
    intervals = pd.read_csv(OUT / '沙洲变化区间.csv')
    mob_col = 'gross_mobility_fraction_per_year' if 'gross_mobility_fraction_per_year' in intervals.columns else 'centroid_shift_normalized_per_year'
    intervals['time_t'] = pd.to_datetime(intervals.time_t)
    intervals['time_t1'] = pd.to_datetime(intervals.time_t1)
    intervals['mid'] = intervals.time_t + (intervals.time_t1 - intervals.time_t) / 2
    rows = []
    for ev in events.itertuples():
        g = intervals[intervals.sand_cay_id.eq(ev.sand_cay_id) & intervals.sensor.eq(ev.sensor)]
        pre = g[g.mid.lt(ev.establishment_date)][['mid', mob_col]].dropna()
        post = g[g.mid.ge(ev.establishment_date)][['mid', mob_col]].dropna()
        pre_slope = np.nan
        if len(pre) >= 2:
            x = (pre.mid - pre.mid.iloc[0]).dt.days / 365.25
            pre_slope = float(stats.linregress(x, pre[mob_col]).slope)
        post_slope = np.nan
        if len(post) >= 2:
            x = (post.mid - post.mid.iloc[0]).dt.days / 365.25
            post_slope = float(stats.linregress(x, post[mob_col]).slope)
        rows.append({'sensor': ev.sensor, 'sand_cay_id': ev.sand_cay_id, 'establishment_date': str(ev.establishment_date),
                     'n_pre_intervals': len(pre), 'n_post_intervals': len(post),
                     'pre_slope_per_year': pre_slope, 'post_slope_per_year': post_slope,
                     'log_ratio_after_before': ev.log_ratio_after_before})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / '植被建立前趋势.csv', index=False, encoding='utf-8-sig')
    usable = df[df.n_pre_intervals.ge(2) & df.pre_slope_per_year.notna()]
    boot = [np.median(RNG.choice(usable.pre_slope_per_year, size=len(usable), replace=True)) for _ in range(2000)] if len(usable) else [np.nan]
    return {'n_events': len(df), 'n_with_pretrend': len(usable),
            'median_pre_slope': float(usable.pre_slope_per_year.median()) if len(usable) else np.nan,
            'pre_slope_ci': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            'median_log_ratio': float(df.log_ratio_after_before.median())}


def mde_nulls():
    rows = []
    m = pd.read_csv(OUT / '强迫植被沙洲内模型.csv')
    fam_dir = 4
    for term, label in [('current_cross_mean_z', 'cross current'), ('veg_within_z', 'within-cay vegetation')]:
        r = m[(m.specification.eq('within_cay_fixed_effects')) & m.term.eq(term)]
        if r.empty:
            r = m[m.term.eq(term)]
        r = r.iloc[0]
        se = (r.ci95_high - r.ci95_low) / 3.92
        k = fam_dir if 'current' in term else 2
        alpha = .05 / k
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(.8)
        rows.append({'test': label, 'se': se, 'alpha_family': alpha, 'mde_80_power': z * se, 'estimate': r.coefficient})
    w = pd.read_csv(OUT / '风场增量检验模型比较.csv')
    w = w[w.model.eq('increment')]
    if not w.empty:
        r = w.iloc[0]
        for col, label in [('joint_wind_cluster_fdr_q_value', None)]:
            pass
        se_cols = [c for c in w.columns if 'se' in c.lower()]
        if se_cols:
            se = float(w[se_cols[0]].iloc[0])
            z = stats.norm.ppf(1 - .05 / 3 / 2) + stats.norm.ppf(.8)
            rows.append({'test': 'wind increment joint', 'se': se, 'alpha_family': .05 / 3, 'mde_80_power': z * se, 'estimate': np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / '空结果最小可检测效应.csv', index=False, encoding='utf-8-sig')
    return df.round(3).to_dict('records')


def moran_residuals():
    d = pd.read_csv(OUT / '流场波浪沿轴横轴分解.csv')
    d = d[(d.sensor.eq('sentinel2')) & d.flow_wave_complete].copy()
    terms = ['current_along_mean', 'wave_vector_along_mean', 'current_cross_mean', 'wave_vector_cross_mean']
    y = 'centroid_along_shift_normalized_per_year'
    d = d[[y, 'sand_cay_id', 'reef_id'] + terms].dropna()
    for c in [y] + terms:
        d[c + '_d'] = d.groupby('sand_cay_id')[c].transform(lambda s: s - s.mean())
    X = sm.add_constant(d[[c + '_d' for c in terms]])
    res = sm.OLS(d[y + '_d'], X).fit()
    d['resid'] = res.resid
    reef = d.groupby('reef_id').resid.mean()
    coords = pd.read_csv(ROOT / 'data' / 'source_dataset' / 'data' / 'metadata' / 'standardized_coordinates.csv')
    id_col = coords.columns[0]
    lat_col = [c for c in coords.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in coords.columns if 'lon' in c.lower()][0]
    cc = coords.set_index(id_col)[[lat_col, lon_col]]
    cc = cc.loc[[r for r in reef.index if r in cc.index]]
    reef = reef.loc[cc.index]
    lat = np.deg2rad(cc[lat_col].to_numpy())
    lon = np.deg2rad(cc[lon_col].to_numpy())
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    w = 1 / dist
    np.fill_diagonal(w, 0)
    z = reef.to_numpy()
    z = z - z.mean()
    n = len(z)
    s0 = w.sum()
    I = n / s0 * (z[:, None] * z[None, :] * w).sum() / (z ** 2).sum()
    ei = -1 / (n - 1)
    s1 = ((w + w.T).sum() ** 2) / 4
    s2 = ((w.sum(axis=1) + w.sum(axis=0)) ** 2).sum()
    a1 = n * ((w + w.T).sum() ** 2) / 4
    b1 = n * s2
    var = (n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2) - (n * (n - 3) * s0 ** 2)) / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2) if n > 3 else np.nan
    sd = np.sqrt(max(var - ei ** 2, 1e-12))
    zscore = (I - ei) / sd
    p = float(2 * (1 - stats.norm.cdf(abs(zscore))))
    pd.DataFrame([{'n_reefs': n, 'moran_I': float(I), 'expected_I': float(ei), 'z': float(zscore), 'p': p}]).to_csv(OUT / '方向模型残差空间自相关.csv', index=False, encoding='utf-8-sig')
    return {'n_reefs': n, 'moran_I': float(I), 'p': p}


if __name__ == '__main__':
    summary = {'error_propagation': error_propagation(),
               'variance_components': variance_components(),
               'seasonal_ctmc': seasonal_ctmc(),
               'establishment': establishment_pretrend(),
               'mde': mde_nulls(),
               'moran': moran_residuals()}
    (OUT / '补充检验汇总.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
