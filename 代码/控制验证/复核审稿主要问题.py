"""审稿新增证据：原始波段诊断、过去训练的预测对照及质心平移分解。

不覆盖标注、掩膜、主观测表或原留一年结果。光谱阈值为诊断情景，非真值。
"""
import sys, json, hashlib, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '公共'))
from 项目路径 import ROOT, configure_imports
configure_imports()
import cv2
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from scipy.stats import spearmanr
from 构建沙洲观测与变化表 import read_mask, select_analysis_observations
from 分析Sentinel2季节形态重复性 import harmonic_errors, METRICS, fdr_bh, paired_wilcoxon, cluster_bootstrap_ci, build_harmonic_validation, build_template_validation

OUT = ROOT / 'outputs/审稿问题复核'
DATA = ROOT.parent / 'data/source_dataset'

def save(frame, name):
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')

def spectral_check(obs):
    metadata = pd.read_csv(DATA / 'data/metadata/sentinel2_annotation_metadata.csv').drop_duplicates('image_id').set_index('image_id')
    rows = []
    for row in obs[obs.sensor.eq('sentinel2')].itertuples():
        base = dict(image_id=row.image_id, sand_cay_id=row.sand_cay_id, date=str(row.date.date()), vegetation_fraction=row.vegetation_fraction, area_m2=row.sand_cay_area_m2)
        if row.image_id not in metadata.index:
            rows.append(base | dict(status='missing_metadata')); continue
        raw = Path(str(metadata.loc[row.image_id, 'source_geotiff']))
        raw = raw if raw.is_absolute() else DATA / raw
        if not raw.is_file():
            rows.append(base | dict(status='missing_raster')); continue
        mask = read_mask(row.mask_path)
        with rasterio.open(raw) as ds:
            names = ds.descriptions
            if 'B04' not in names or 'B08' not in names or mask.shape != ds.shape:
                rows.append(base | dict(status='band_or_grid_mismatch')); continue
            red, nir = (ds.read(names.index(b) + 1).astype(float) for b in ['B04', 'B08'])
            clear = mask & (red > 0) & (nir > 0)
            if 'SCL' in names:
                clear &= ~np.isin(ds.read(names.index('SCL') + 1), [0, 1, 3, 8, 9, 10, 11])
            detail = dict(status='ok', clear_fraction=float(clear.sum() / mask.sum()), n_clear=int(clear.sum()), raw_path=str(raw.relative_to(DATA)), crs=str(ds.crs), grid=str(tuple(ds.transform)), raster_width=ds.width, raster_height=ds.height,
                          radiometric_offset_documented=bool('BOA_ADD_OFFSET' in str(ds.tags())))
            # 裁切文件没有保留处理基线/偏移元数据，两个偏移情景均报告。
            for offset in [0, 1000]:
                valid = clear & (red > offset) & (nir > offset)
                values = (nir[valid] - red[valid]) / (nir[valid] + red[valid] - 2 * offset)
                detail[f'valid_fraction_offset{offset}'] = float(valid.sum() / mask.sum())
                detail[f'median_index_offset{offset}'] = float(np.median(values)) if len(values) else np.nan
                for threshold in [.2, .3, .4]:
                    detail[f'fraction_index_gt{threshold}_offset{offset}'] = float(np.mean(values > threshold)) if len(values) else np.nan
            rows.append(base | detail)
    table = pd.DataFrame(rows); save(table, '原始红光近红外逐景诊断')
    summaries = []
    for offset in [0, 1000]:
        g = table[(table.status == 'ok') & (table[f'valid_fraction_offset{offset}'] >= .8)].copy()
        for threshold in [.2, .3, .4]:
            column = f'fraction_index_gt{threshold}_offset{offset}'
            means = g.groupby('sand_cay_id')[['vegetation_fraction', column]].mean().dropna()
            rho, p = spearmanr(means.vegetation_fraction, means[column])
            # 同沙洲中心化仅为方向诊断，重复观测不用于独立样本显著性。
            centered = g[['vegetation_fraction', column]] - g.groupby('sand_cay_id')[['vegetation_fraction', column]].transform('mean')
            within = spearmanr(centered.vegetation_fraction, centered[column]).statistic
            summaries.append(dict(offset_dn=offset, threshold=threshold, n_images=len(g), n_cays=len(means), between_cay_rho=rho, between_cay_p=p, within_cay_centered_rho=within))
    summary = pd.DataFrame(summaries); save(summary, '颜色与原始波段一致性诊断')
    from 提取沙洲颜色组成 import read_image, calibrated_colour_pixels
    cases=[]
    for row in obs[(obs.sand_cay_id=='GBR_101_cay_001')&obs.sensor.eq('sentinel2')&obs.date.isin(pd.to_datetime(['2018-04-26','2019-01-11','2019-09-03']))].itertuples():
        image=read_image(Path(str(metadata.loc[row.image_id,'image_path'])),cv2.IMREAD_COLOR);mask=read_mask(row.mask_path)
        calibrated=calibrated_colour_pixels(image,mask);unbalanced=calibrated_colour_pixels(image,mask,apply_white_balance=False)
        cases.append(dict(image_id=row.image_id,date=str(row.date.date()),sensor=row.sensor,white_balance_green_fraction=float(calibrated['vegetation'].mean()),unbalanced_green_fraction=float(unbalanced['vegetation'].mean()),gain_r=float(calibrated['gains'][0]),gain_g=float(calibrated['gains'][1]),gain_b=float(calibrated['gains'][2])))
    cases=pd.DataFrame(cases).merge(table,on=['image_id','date'],suffixes=('','_diagnostic'),validate='one_to_one')
    save(cases,'图3案例颜色与波段诊断')
    # 栅格原点的一致性只验证采样网格，绝不充当实测配准精度。
    table['grid_key'] = table[['grid','crs','raster_width','raster_height']].astype(str).agg('|'.join,axis=1)
    grids = table[table.status.eq('ok')].groupby('sand_cay_id').agg(n=('image_id','size'), n_grids=('grid_key','nunique'), n_crs=('crs','nunique'))
    save(grids.reset_index(), 'Sentinel2原始栅格网格核验')
    return dict(n=len(table), statuses=table.status.value_counts().to_dict(), offset_metadata_available=int(table.radiometric_offset_documented.fillna(False).sum()), summary=summary.to_dict('records'))

def forward_check(obs):
    frame = obs[obs.sensor.eq('sentinel2')].copy()
    grids = pd.read_csv(OUT / 'Sentinel2原始栅格网格核验.csv')
    frame = frame[frame.sand_cay_id.isin(grids.loc[grids.n_grids.eq(1), 'sand_cay_id'])].copy()
    frame['sand_cay_compactness'] = frame.compactness_primary_crofton
    frame['year'] = frame.date.dt.year; frame['doy'] = frame.date.dt.dayofyear
    frame['time_years'] = (frame.date - frame.date.min()).dt.days / 365.2425
    rows = []
    for metric in METRICS:
        for cay, group in frame.dropna(subset=[metric]).groupby('sand_cay_id'):
            group = group.sort_values('date')
            for year, test in group.groupby('year'):
                train = group[group.year < year]
                # 至少三个过去年度、九景；整个目标年度不进入训练。
                if train.year.nunique() < 3 or len(train) < 9: continue
                base, seasonal = harmonic_errors(train, test, metric)
                if metric == 'sand_cay_major_axis_angle':
                    persistence = 2 * np.abs(np.sin(np.deg2rad(test[metric].to_numpy() - train[metric].iloc[-1])))
                else:
                    scale = train[metric].quantile(.75) - train[metric].quantile(.25)
                    if not np.isfinite(scale) or scale <= 0: scale = train[metric].std()
                    if not np.isfinite(scale) or scale <= 0: scale = 1.
                    persistence = np.abs(test[metric].to_numpy() - train[metric].iloc[-1]) / scale
                for (_, target), b, s, p in zip(test.iterrows(), base, seasonal, persistence):
                    rows.append(dict(metric=metric, sand_cay_id=cay, image_id=target.image_id, date=str(target.date.date()), target_year=int(year), training_last_date=str(train.date.max().date()), n_training=len(train), baseline_error=b, seasonal_error=s, persistence_error=p, gain_vs_linear=b-s, gain_vs_persistence=p-s))
    v = pd.DataFrame(rows); save(v, '仅过去训练年度预测逐景对照')
    summaries = []
    for comparison, gain, error in [('linear', 'gain_vs_linear', 'baseline_error'), ('persistence', 'gain_vs_persistence', 'persistence_error')]:
        family = []
        for metric, g in v.groupby('metric', sort=False):
            means = g.groupby('sand_cay_id')[[gain, error, 'seasonal_error']].mean()
            low, high = cluster_bootstrap_ci(g, gain, 'sand_cay_id', 2000)
            family.append(dict(metric=metric, comparison=comparison, n_images=len(g), n_cays=len(means), mean_gain=float(means[gain].mean()), ci95_low=low, ci95_high=high, relative_error_reduction=float(1-means.seasonal_error.sum()/means[error].sum()), wilcoxon_p_value=paired_wilcoxon(means[gain])))
        s = pd.DataFrame(family); s['fdr_q_value'] = fdr_bh(s.wilcoxon_p_value); summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True); save(summary, '仅过去训练年度预测汇总')
    assert (pd.to_datetime(v.training_last_date).dt.year < v.target_year).all()
    return summary.to_dict('records')

def strict_grid_check(obs):
    grids=pd.read_csv(OUT/'Sentinel2原始栅格网格核验.csv')
    f=obs[obs.sensor.eq('sentinel2')&obs.sand_cay_id.isin(grids.loc[grids.n_grids.eq(1),'sand_cay_id'])].copy()
    f['sand_cay_compactness']=f.compactness_primary_crofton
    f['year']=f.date.dt.year;f['doy']=f.date.dt.dayofyear;f['quarter']=f.date.dt.quarter
    f['time_years']=(f.date-f.date.min()).dt.days/365.2425
    v,s=build_harmonic_validation(f,2000);save(v,'单一原始网格留一年逐景对照');save(s,'单一原始网格留一年汇总')
    masks={int(i):read_mask(x) for i,x in f.mask_path.items()}
    v,t=build_template_validation(f,masks,2000);save(v,'单一原始网格年度模板逐景对照');save(t,'单一原始网格年度模板汇总')
    pairs=pd.read_csv(ROOT/'outputs/Sentinel2季节形态配对.csv');pairs=pairs[pairs.sand_cay_id.isin(f.sand_cay_id)]
    design=pairs[['phase_distance','gap_years']].copy();design['gap_years_squared']=design.gap_years**2
    design=pd.concat([design,pd.get_dummies(pairs.sand_cay_id,drop_first=True,dtype=float)],axis=1)
    model=sm.OLS(pairs.iou,sm.add_constant(design)).fit(cov_type='cluster',cov_kwds={'groups':pairs.sand_cay_id})
    phase=dict(n_pairs=len(pairs),n_cays=pairs.sand_cay_id.nunique(),coefficient=float(model.params.phase_distance),ci=model.conf_int().loc['phase_distance'].tolist(),p=float(model.pvalues.phase_distance))
    return dict(n_images=len(f),n_cays=f.sand_cay_id.nunique(),excluded_cays=grids.loc[grids.n_grids.gt(1),'sand_cay_id'].tolist(),harmonic=s.to_dict('records'),template=t.to_dict('records'),phase=phase)

def geometry_check(obs):
    d = pd.read_csv(ROOT / 'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv')
    rows = []
    for p in d.itertuples():
        g = obs[(obs.sand_cay_id==p.sand_cay_id)&(obs.sensor==p.sensor)&(obs.reference_frame_id==p.reference_frame_id)].sort_values('date')
        a,b = read_mask(g.mask_path.iloc[0]), read_mask(g.mask_path.iloc[-1])
        if a.shape != b.shape: continue
        delta = g[['sand_cay_centroid_x','sand_cay_centroid_y']].iloc[-1].to_numpy(float)-g[['sand_cay_centroid_x','sand_cay_centroid_y']].iloc[0].to_numpy(float)
        # 加留边防止移位裁切，最近邻不改变像元面积；这是几何对齐，不是背景配准。
        pad = int(np.ceil(np.abs(delta).max())) + 2
        aa,bb = np.pad(a,pad), np.pad(b,pad)
        matrix = np.float32([[1,0,-delta[0]],[0,1,-delta[1]]])
        aligned = cv2.warpAffine(bb.astype('uint8'),matrix,(bb.shape[1],bb.shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
        assert aligned.sum()==b.sum()
        rows.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,raw_exchange_fraction=float(np.logical_xor(aa,bb).sum()/aa.sum()),centroid_aligned_exchange_fraction=float(np.logical_xor(aa,aligned).sum()/aa.sum()),raw_iou=float((aa&bb).sum()/(aa|bb).sum()),centroid_aligned_iou=float((aa&aligned).sum()/(aa|aligned).sum()),shift_x_px=float(delta[0]),shift_y_px=float(delta[1])))
    table = pd.DataFrame(rows); save(table,'质心几何对齐后轮廓差异')
    # 绿色绝对面积与总面积分开，保留全部转换方向与阴性证据。
    b = pd.read_csv(ROOT/'outputs/figures/source_data/表B_区间级状态边界位移环境.csv')
    pairs=[]
    for (cay,sensor,reference),g in obs.groupby(['sand_cay_id','sensor','reference_frame_id']):
        g=g.sort_values('date')
        for (_,a),(_,z) in zip(g.iloc[:-1].iterrows(),g.iloc[1:].iterrows()):
            gap=(z.date-a.date).days
            if gap<30 or gap>365:continue
            ga=a.vegetation_fraction*a.sand_cay_area_m2;gz=z.vegetation_fraction*z.sand_cay_area_m2
            pairs.append(dict(sand_cay_id=cay,sensor=sensor,date0=str(a.date.date()),date1=str(z.date.date()),fraction_change=z.vegetation_fraction-a.vegetation_fraction,green_area_change_m2=gz-ga,total_area_change_m2=z.sand_cay_area_m2-a.sand_cay_area_m2,green_area_contribution=(gz-ga)/z.sand_cay_area_m2,denominator_contribution=ga*(1/z.sand_cay_area_m2-1/a.sand_cay_area_m2)))
    pairs=pd.DataFrame(pairs)
    assert np.allclose(pairs.fraction_change,pairs.green_area_contribution+pairs.denominator_contribution)
    save(pairs,'绿色比例变化分子与分母分解')
    grids=pd.read_csv(OUT/'Sentinel2原始栅格网格核验.csv')
    excluded=set(grids.loc[grids.n_grids.gt(1),'sand_cay_id'])
    positions=d[~(d.sensor.eq('sentinel2')&d.sand_cay_id.isin(excluded))].copy()
    save(positions,'原始网格一致长期位置描述')
    near=positions[positions.net_area_bin.eq('Near-stable area')]
    from 核查空间轨迹与覆盖转换 import track
    support=pd.read_csv(ROOT/'outputs/P0证据核查/核心与小尺寸案例配准支持.csv')
    backgrounds=[]
    for p in d[d.sand_cay_id.isin(support.sand_cay_id)].itertuples():
        g=track(p);masks=[read_mask(x) for x in g.resolved_mask_path]
        if len({m.shape for m in masks})!=1:continue
        union=np.logical_or.reduce(masks);valid=cv2.dilate(union.astype('uint8'),np.ones((81,81),np.uint8))==0
        valid[:20]=False;valid[-20:]=False;valid[:,:20]=False;valid[:,-20:]=False
        backgrounds.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,width=valid.shape[1],height=valid.shape[0],buffer_radius_m=40*p.pixel_size_m if hasattr(p,'pixel_size_m') else 40*g.pixel_size_m.iloc[0],candidate_background_fraction=float(valid.mean()),candidate_background_pixels=int(valid.sum())))
    save(pd.DataFrame(backgrounds),'背景筛查有效搜索范围')
    # 标签接受子集与直接测得绿色比例的相邻区间分别保留。
    b['date0']=pd.to_datetime(b.time_t).dt.strftime('%Y-%m-%d');b['date1']=pd.to_datetime(b.time_t1).dt.strftime('%Y-%m-%d')
    lookup=obs.copy();lookup['date_key']=lookup.date.dt.strftime('%Y-%m-%d')
    keys=['sensor','sand_cay_id','reference_frame_id']
    endpoints=lookup[keys+['date_key','vegetation_fraction','sand_cay_area_m2']]
    accepted=b.merge(endpoints.rename(columns={'date_key':'date0','vegetation_fraction':'f0','sand_cay_area_m2':'a0'}),on=keys+['date0'],validate='many_to_one')
    accepted=accepted.merge(endpoints.rename(columns={'date_key':'date1','vegetation_fraction':'f1','sand_cay_area_m2':'a1'}),on=keys+['date1'],validate='many_to_one')
    accepted['fraction_change']=accepted.f1-accepted.f0
    accepted['green_area_change_m2']=accepted.f1*accepted.a1-accepted.f0*accepted.a0
    accepted['green_area_contribution']=accepted.green_area_change_m2/accepted.a1
    accepted['denominator_contribution']=accepted.f0*accepted.a0*(1/accepted.a1-1/accepted.a0)
    assert np.allclose(accepted.fraction_change,accepted.green_area_contribution+accepted.denominator_contribution)
    assert len(accepted)==len(b)
    save(accepted,'接受状态对绿色比例分解')
    changes=accepted[accepted.state_t0.ne(accepted.state_t1)]
    decomposition_summary=dict(n_state_changes=len(changes),n_fraction_and_green_area_opposite=int((changes.fraction_change*changes.green_area_change_m2<0).sum()),n_denominator_dominant=int((changes.denominator_contribution.abs()>changes.green_area_contribution.abs()).sum()))
    (OUT/'覆盖比例分解汇总.json').write_text(json.dumps(decomposition_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(n_geometry=len(table), cases=table[table.sand_cay_id.isin(['MEDF_008_cay_001','MEDF_022_cay_001'])].to_dict('records'), n_cover_intervals=len(pairs),n_accepted_intervals=len(accepted),n_position_tracks=len(positions),n_near_stable=len(near),n_envelope_le1=int(near.centroid_envelope_normalized.le(1).sum()),median_envelope=float(near.centroid_envelope_normalized.median()),iqr_envelope=near.centroid_envelope_normalized.quantile([.25,.75]).tolist(),median_net_over_path=float((near.centroid_net_normalized/near.centroid_path_normalized).median()),n_ratio_lt_half=int((near.centroid_net_normalized/near.centroid_path_normalized).lt(.5).sum()))

def association_check(obs):
    grids=pd.read_csv(OUT/'Sentinel2原始栅格网格核验.csv');excluded=set(grids.loc[grids.n_grids.gt(1),'sand_cay_id'])
    from 分析已成洲沙洲形态响应 import unified_forcing_vegetation_model
    a=pd.read_csv(ROOT/'outputs/已成洲沙洲定量分析样本.csv')
    a=a[~(a.sensor.eq('sentinel2')&a.sand_cay_id.isin(excluded))]
    decomposition=pd.read_csv(ROOT/'outputs/流场波浪沿轴横轴分解.csv')
    models,diagnostics,_=unified_forcing_vegetation_model(a,decomposition,fit_random_intercepts=False)
    save(models,'单一原始网格方向模型');save(diagnostics,'单一原始网格方向模型诊断')
    from 分析植被形态台风关系 import load_trajectory_frame,trajectory_rank_associations
    trajectories=load_trajectory_frame(ROOT/'outputs')
    trajectories=trajectories[~(trajectories.sensor.eq('sentinel2')&trajectories.sand_cay_id.isin(excluded))]
    rank=trajectory_rank_associations(trajectories);save(rank,'单一原始网格覆盖路径秩相关')
    from 分析裸沙质心轨迹周期与突发 import load_observations,build_steps
    from 分析裸沙质心轨迹风浪对照 import load_forcing_inputs,build_usable,fit_forcing_models
    observations=load_observations();observations=observations[~observations.sand_cay_id.isin(excluded)]
    steps=build_steps(observations, grouping='cover');forcing,daily=load_forcing_inputs();usable,_=build_usable(observations,steps,forcing,daily)
    wave=fit_forcing_models(usable);save(wave,'单一原始网格高波浪日模型')
    directions=models[(models.specification=='within_cay_fixed_effects')&models.outcome.isin(['along_shift','cross_shift'])&models.term.isin(['current_along_mean_z','current_cross_mean_z','wave_vector_along_mean_z','wave_vector_cross_mean_z'])]
    return dict(direction=directions.to_dict('records'),cover_path=rank[rank.outcome.eq('centroid_path_normalized')].to_dict('records'),high_wave=wave[wave.analysis.eq('E4_event_storm_days_bare')].to_dict('records'))

def check_derived():
    """无原始影像时，复核随包派生表的样本、时间隔离、统计校正及比例恒等式。"""
    forward=pd.read_csv(OUT/'仅过去训练年度预测逐景对照.csv')
    assert forward.image_id.nunique()==1029 and forward.sand_cay_id.nunique()==56
    assert (pd.to_datetime(forward.training_last_date)<pd.to_datetime(forward.date)).all()
    assert forward.groupby('metric').image_id.nunique().eq(1029).all()
    summary=pd.read_csv(OUT/'仅过去训练年度预测汇总.csv')
    for comparison, group in summary.groupby('comparison'):
        assert len(group)==7
        assert np.allclose(fdr_bh(group.wilcoxon_p_value),group.fdr_q_value)
    pairs=pd.read_csv(OUT/'接受状态对绿色比例分解.csv')
    assert len(pairs)==1490
    assert np.allclose(pairs.fraction_change,pairs.green_area_contribution+pairs.denominator_contribution,atol=1e-12)
    changed=pairs[pairs.state_t0.ne(pairs.state_t1)]
    assert len(changed)==172
    assert int((changed.fraction_change*changed.green_area_change_m2<0).sum())==9
    assert int((changed.denominator_contribution.abs()>changed.green_area_contribution.abs()).sum())==20
    grids=pd.read_csv(OUT/'Sentinel2原始栅格网格核验.csv')
    assert set(grids.loc[grids.n_grids.gt(1),'sand_cay_id'])=={'GBR_037_cay_001','GBR_068_cay_001','GBR_095_cay_001','NH_015_cay_001'}
    positions=pd.read_csv(OUT/'原始网格一致长期位置描述.csv')
    assert len(positions)==90
    from 原始网格核验 import single_grid_cays
    sectors=pd.read_csv(ROOT/'outputs/植被扇区对照.csv')
    audit=pd.read_csv(OUT/'图6b原始网格样本审计.csv')
    assert sectors.sand_cay_id.isin(single_grid_cays()).all()
    assert not sectors.transition_id.duplicated().any()
    assert set(sectors.transition_id)==set(audit.loc[audit.single_original_grid_sequence,'transition_id'])
    stats=pd.read_csv(ROOT/'outputs/植被扇区对照统计.csv')
    for name,subset in [('all',sectors),('background',sectors[~sectors.pulse])]:
        row=stats.loc[stats.stratum.eq(name)].iloc[0]
        assert row.n_transitions==len(subset) and row.n_cays==subset.sand_cay_id.nunique()
        assert np.isclose(row.mean_mobility_difference,subset.mobility_difference.mean())
        assert row.bootstrap_unit=='cay_with_all_intervals' and row.wilcoxon_unit=='cay_mean_difference'
    lag=pd.read_csv(ROOT/'outputs/植被扇区因果滞后检验.csv')
    tested=lag[lag.test.isin(['M7_forward','M9_reverse'])]
    assert len(tested)==4 and np.allclose(fdr_bh(tested.p_value),tested.fdr_q_value)
    tail_path=OUT/'单一原始网格高移动速率对照.csv'
    if tail_path.exists():
        tail=pd.read_csv(tail_path).iloc[0]
        speed=pd.read_csv(OUT/'单一原始网格移动速率区间.csv')
        assert speed.sand_cay_id.isin(single_grid_cays()).all()
        assert speed.cover_state_t.isin(['none','sparse','partial','dominant']).all()
        assert speed.stratum.eq('bare').equals(speed.cover_state_t.eq('none'))
        assert speed.grouping_definition.eq('accepted_initial_cover_none_vs_sparse_partial_dominant').all()
        labels=select_analysis_observations(pd.read_csv(ROOT/'outputs/植被状态审校主表.csv',low_memory=False))
        labels['date']=pd.to_datetime(labels.date)
        audit=speed.assign(time_t=pd.to_datetime(speed.time_t)).merge(
            labels[['sensor','sand_cay_id','reference_frame_id','date','analysis_vegetation_state']].rename(columns={'date':'time_t'}),
            on=['sensor','sand_cay_id','reference_frame_id','time_t'],how='left',validate='many_to_one')
        assert audit.cover_state_t.equals(audit.analysis_vegetation_state)
        threshold=float(speed.speed_norm.quantile(.9))
        assert np.isclose(threshold,tail.pooled_p90_speed_radii_per_year)
        fractions=[]
        for stratum in ['bare','vegetated']:
            subset=speed[speed.stratum.eq(stratum)]
            high=int(subset.speed_norm.ge(threshold).sum())
            assert high==tail['n_'+stratum+'_high_speed']
            assert len(subset)==tail['n_'+stratum+'_intervals']
            assert np.isclose(high/len(subset),tail[stratum+'_median'])
            fractions.append(high/len(subset))
        assert np.isclose(fractions[0]-fractions[1],tail.estimate)
        assert tail.n_reefs==speed.reef_id.nunique() and pd.isna(tail.fdr_q_value)
        wave=pd.read_csv(OUT/'单一原始网格高波浪日模型.csv')
        assert len(wave)==7 and np.allclose(fdr_bh(wave.p_value),wave.fdr_q_value)
        usable=pd.read_csv(OUT/'覆盖分组风浪分析时段.csv')
        assert set(usable.transition_id).issubset(set(speed.transition_id))
        current=wave[wave.analysis.eq('E4_event_storm_days_bare')].iloc[0]
        bare=usable[usable.stratum.eq('bare') & usable.event_storm_days.notna()]
        assert current.n_observations==len(bare) and current.n_cays==bare.sand_cay_id.nunique()
        seasonal=pd.read_csv(OUT/'覆盖分组高波浪季节与比例核验.csv')
        assert np.isclose(seasonal.iloc[0].estimate,current.estimate)
        assert np.allclose(fdr_bh(seasonal.iloc[1:].p_value),seasonal.iloc[1:].fdr_q_value)
    if (ROOT/'文件清单.csv').exists():
        manifest=pd.read_csv(ROOT/'文件清单.csv')
        for row in manifest.itertuples():
            path=ROOT/row.relative_path
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256,row.relative_path
    print('派生表复核通过：前向日期隔离、两组七指标FDR、1490状态对、90条位置记录及图6b逐区间网格资格/统计单位；有清单时逐文件验证SHA-256。')

def mobility_check(obs):
    """按既有p90定义与礁盘联合抽样复核网格一致样本，不改变历史FDR族。"""
    from 原始网格核验 import single_grid_cays
    from 分析裸沙质心轨迹周期与突发 import load_observations, build_steps, episodicity_tests
    from 分析裸沙主文升级检验 import extreme_contrast
    observations = load_observations()
    observations = observations[observations.sand_cay_id.isin(single_grid_cays())]
    steps = build_steps(observations, grouping='cover')
    assert steps.sand_cay_id.isin(single_grid_cays()).all()
    _, detail = episodicity_tests(steps)
    row = extreme_contrast(steps, detail)[1]
    threshold = float(steps.speed_norm.quantile(.9))
    row['pooled_p90_speed_radii_per_year'] = threshold
    for stratum in ['bare', 'vegetated']:
        d = steps[steps.stratum.eq(stratum)]
        row['n_' + stratum + '_intervals'] = len(d)
        row['n_' + stratum + '_high_speed'] = int(d.speed_norm.ge(threshold).sum())
    row['fdr_q_value'] = np.nan
    row['evidence_role'] = 'Accepted initial cover none versus sparse/partial/dominant; grid-consistent distribution contrast; nominal bootstrap p, no inherited historical q; pooled threshold held fixed.'
    save(pd.DataFrame([row]), '单一原始网格高移动速率对照')
    steps.to_csv(OUT/'单一原始网格移动速率区间.csv', index=False, encoding='utf-8-sig')
    return row


def cover_group_check(obs):
    """使用图5的已审校期初覆盖标签，复算既有风浪、季节控制及区域剔除检验。"""
    from 原始网格核验 import single_grid_cays
    from 分析裸沙质心轨迹周期与突发 import load_observations, build_steps
    from 分析裸沙质心轨迹风浪对照 import load_forcing_inputs, build_usable, fit_forcing_models, fit_clustered
    observations = load_observations()
    observations = observations[observations.sand_cay_id.isin(single_grid_cays())]
    steps = build_steps(observations, grouping='cover')
    forcing, daily = load_forcing_inputs()
    usable, _ = build_usable(observations, steps, forcing, daily)
    models = fit_forcing_models(usable)
    save(models, '单一原始网格高波浪日模型')
    save(usable, '覆盖分组风浪分析时段')
    bare = usable[usable.stratum.eq('bare')].dropna(subset=['event_storm_days']).copy()
    assert bare.waverys_complete.eq(1).all()
    bare['high_wave_fraction'] = bare.event_storm_days / bare.time_interval_days
    assert bare.high_wave_fraction.between(0,1).all()
    phase = 2*np.pi*bare.midpoint.dt.dayofyear/365.2425
    bare['season_sin'], bare['season_cos'] = np.sin(phase), np.cos(phase)
    rows=[]
    for name,term,season in [('count_duration','event_storm_days',''),
                             ('count_duration_season','event_storm_days',' + season_sin + season_cos'),
                             ('fraction_duration_season','high_wave_fraction',' + season_sin + season_cos')]:
        formula=f'log_speed ~ {term} + time_interval_days{season} + C(sand_cay_id)'
        fit=fit_clustered(formula,bare);low,high=fit.conf_int().loc[term]
        rows.append(dict(model=name,term=term,estimate=float(fit.params[term]),ci95_low=float(low),
                         ci95_high=float(high),p_value=float(fit.pvalues[term]),n_intervals=len(bare),
                         n_cays=bare.sand_cay_id.nunique(),n_reefs=bare.reef_id.nunique(),formula=formula))
    reference=models[models.analysis.eq('E4_event_storm_days_bare')].iloc[0]
    assert np.isclose(rows[0]['estimate'],reference.estimate)
    rows[0]['fdr_q_value']=float(reference.fdr_q_value)
    for row,q in zip(rows[1:],fdr_bh(pd.Series([r['p_value'] for r in rows[1:]]))):
        row['fdr_q_value']=float(q)
    save(pd.DataFrame(rows),'覆盖分组高波浪季节与比例核验')
    bare['region']=bare.reef_id.str.extract(r'^([^_]+)')[0]
    leave=[]
    for excluded_region in sorted(bare.region.unique()):
        d=bare[bare.region.ne(excluded_region)]
        fit=fit_clustered('log_speed ~ event_storm_days + time_interval_days + C(sand_cay_id)',d)
        low,high=fit.conf_int().loc['event_storm_days']
        leave.append(dict(excluded_region=excluded_region,estimate=float(fit.params['event_storm_days']),
                          ci95_low=float(low),ci95_high=float(high),p_value=float(fit.pvalues['event_storm_days']),
                          n_intervals=len(d),n_cays=d.sand_cay_id.nunique(),n_reefs=d.reef_id.nunique()))
    save(pd.DataFrame(leave),'覆盖分组高波浪逐区域剔除')
    tail=mobility_check(obs)
    result=dict(definition='accepted_initial_cover_none_vs_sparse_partial_dominant',
                all_intervals=len(steps),all_cays=steps.sand_cay_id.nunique(),
                high_wave=models[models.analysis.eq('E4_event_storm_days_bare')].to_dict('records'),
                seasonal=rows,leave_region_out=leave,mobility=tail)
    (OUT/'覆盖分组复核结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['all','spectral','forward','geometry','grid','association','mobility','cover'],default='all');parser.add_argument('--check-derived',action='store_true');args=parser.parse_args()
    if args.check_derived:
        check_derived();return
    OUT.mkdir(exist_ok=True)
    master=ROOT/'outputs/沙洲观测主表.csv';obs=select_analysis_observations(pd.read_csv(master))
    result=json.loads((OUT/'审稿复核结果.json').read_text(encoding='utf-8')) if (OUT/'审稿复核结果.json').exists() else {}
    for name,fun in [('spectral',spectral_check),('forward',forward_check),('geometry',geometry_check),('grid',strict_grid_check),('association',association_check),('mobility',mobility_check),('cover',cover_group_check)]:
        if args.part in ['all',name]:
            result[name]=fun(obs);print(name,json.dumps(result[name],ensure_ascii=False),flush=True)
    if 'cover' in result:
        result['mobility']=result['cover']['mobility']
        if 'association' in result:
            result['association']['high_wave']=result['cover']['high_wave']
    result['master_sha256']=hashlib.sha256(master.read_bytes()).hexdigest()
    result['scope']='No independent ground control or blind relabelling; spectral offset and thresholds are diagnostic scenarios; forward comparisons each use seven-response BH families.'
    (OUT/'审稿复核结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
