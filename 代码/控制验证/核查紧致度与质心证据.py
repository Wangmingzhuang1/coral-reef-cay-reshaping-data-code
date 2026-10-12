"""P0同对象紧致度与质心采样/误差情景；P1匹配支持清单。原观测指标不覆盖。"""
import sys,json,itertools,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT
from 构建沙洲观测与变化表 import read_mask,primary_compactness,select_analysis_observations
import numpy as np
import pandas as pd
from 分析Sentinel2季节形态重复性 import build_harmonic_validation,fdr_bh
OUT=ROOT/'outputs/P0证据核查';OUT.mkdir(exist_ok=True)
def save(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def run(compactness_only=False,spatial_only=False):
    master=pd.read_csv(ROOT/'outputs/沙洲观测主表.csv')
    if not spatial_only:
        geometry=[]
        selected_ids=set(select_analysis_observations(master).image_id)
        for i,row in master.iterrows():
            mask=read_mask(row.mask_path)
            if mask is None:
                if row.image_id in selected_ids:raise ValueError('有效观测缺真实掩膜：'+row.image_id)
                geometry.append({key:np.nan for key in ['compactness_primary_crofton','compactness_primary_external','compactness_area_pixels','compactness_perimeter_crofton_pixels','compactness_perimeter_external_pixels']})
                continue
            z=primary_compactness(mask)
            assert np.isclose(z['compactness_area_pixels'],row.sand_cay_area_pixels)
            geometry.append(z)
            if i%400==0:print('紧致度逐景复算',i,flush=True)
        g=pd.DataFrame(geometry,index=master.index)
        for key in g:master[key]=g[key]
        master['compactness_method']='compactness_primary_crofton: largest 8-connected component, four-direction Crofton, no clipping; sand_cay_compactness: unchanged legacy field'
        master.to_csv(ROOT/'outputs/沙洲观测主表.csv',index=False,encoding='utf-8-sig')
        save(master[['sensor','sand_cay_id','reference_frame_id','date','image_id','quality_grade','sand_cay_compactness',*list(g.columns)]],'逐景紧致度同对象口径')
        frame=select_analysis_observations(master[master.sensor.eq('sentinel2')]);frame['year']=frame.date.dt.year;frame['doy']=frame.date.dt.dayofyear;frame['time_years']=(frame.date-frame.date.min()).dt.days/365.2425
        validations=[];summaries=[]
        for method,column in [('legacy','sand_cay_compactness'),('primary_external','compactness_primary_external'),('primary_crofton','compactness_primary_crofton')]:
            f=frame.copy();f['sand_cay_compactness']=f[column]
            v,s=build_harmonic_validation(f,2000,metrics=['sand_cay_compactness'])
            # 保持原七响应BH家族；其他六项使用相同既有检验结果。
            old=pd.read_csv(ROOT/'outputs/Sentinel2季节谐波检验汇总.csv');pvals=old.wilcoxon_p_value.copy();pvals.loc[old.metric.eq('sand_cay_compactness')]=s.wilcoxon_p_value.iloc[0]
            q=float(fdr_bh(pvals).loc[old.metric.eq('sand_cay_compactness')].iloc[0]);means=v.groupby('sand_cay_id')[['baseline_error','seasonal_error']].mean();gain=float((means.baseline_error.sum()-means.seasonal_error.sum())/means.baseline_error.sum())
            s=s.iloc[0].to_dict();s.update(method=method,relative_error_reduction=gain,fdr_q_seven_metrics=q,fdr_q_value=q);summaries.append(s);v['method']=method;validations.append(v)
        save(pd.DataFrame(summaries),'紧致度年度同折对照');save(pd.concat(validations),'紧致度年度逐景对照')
        # 不重复计算轮廓模板；仅同步七响应谐波，维持年度样本和训练/留出规则。
        current=frame.copy();current['sand_cay_compactness']=current.compactness_primary_crofton
        v,s=build_harmonic_validation(current,2000)
        v.to_csv(ROOT/'outputs/Sentinel2季节谐波交叉验证.csv',index=False,encoding='utf-8-sig');s.to_csv(ROOT/'outputs/Sentinel2季节谐波检验汇总.csv',index=False,encoding='utf-8-sig')
        base={'compactness_definition':'same largest 8-connected component area and four-direction Crofton perimeter; raw stored C preserved; primary external perimeter sensitivity','n_recomputed':int(master.compactness_primary_crofton.notna().sum()),'n_unavailable_outside_analysis':int(master.compactness_primary_crofton.isna().sum()),'selected_C_gt1':int(select_analysis_observations(master).compactness_primary_crofton.gt(1).sum()),'annual_comparison':summaries,'master_sha256':hashlib.sha256((ROOT/'outputs/沙洲观测主表.csv').read_bytes()).hexdigest()}
        annual_audit=ROOT/'outputs/Sentinel2季节形态重复性核查.json'
        if annual_audit.exists():
            j=json.loads(annual_audit.read_text(encoding='utf-8'));j['harmonic_tests']=s.to_dict(orient='records');j['compactness_definition']=base['compactness_definition'];annual_audit.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
        if compactness_only:
            base['spatial_status']='pending downstream continuous-track audit'
            (OUT/'核查结果.json').write_text(json.dumps(base,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(base,ensure_ascii=False,indent=2),flush=True)
            return
    else:
        base=json.loads((OUT/'核查结果.json').read_text(encoding='utf-8'))
        assert base['master_sha256']==hashlib.sha256((ROOT/'outputs/沙洲观测主表.csv').read_bytes()).hexdigest(),'紧致度阶段与当前主表不一致'
        summaries=base['annual_comparison']
    d=pd.read_csv(ROOT/'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv')
    obs=select_analysis_observations(master);samples=[];errors=[];support=[];rng=np.random.default_rng(20261009)
    for p in d.itertuples(index=False):
        h=obs[(obs.sand_cay_id==p.sand_cay_id)&(obs.sensor==p.sensor)&(obs.reference_frame_id==p.reference_frame_id)].sort_values('date').reset_index(drop=True)
        scale=float(h.pixel_size_m.iloc[0]);radius=np.sqrt(p.start_area_m2/np.pi);xy=h[['sand_cay_centroid_x','sand_cay_centroid_y']].to_numpy(dtype=float)
        assert h.pixel_size_m.nunique()==1 and h.reference_frame_id.nunique()==1
        for frequency in ['all','month','quarter','year']:
            indices=list(range(len(h))) if frequency=='all' else sorted(set(h.groupby(h.date.dt.to_period({'month':'M','quarter':'Q','year':'Y'}[frequency])).head(1).index.tolist()+[0,len(h)-1]))
            z=xy[indices];path=float(np.linalg.norm(np.diff(z,axis=0),axis=1).sum()*scale/radius);net=float(np.linalg.norm(z[-1]-z[0])*scale/radius);envelope=float(np.linalg.norm(z-z.mean(axis=0),axis=1).max()*scale/radius)
            samples.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,region=p.region,frequency=frequency,n=len(z),path_r=path,net_r=net,envelope_r=envelope))
        for sigma in [.5,1,2]:
            noise=rng.normal(0,sigma,(2000,len(h),2));null_path=np.linalg.norm(np.diff(noise,axis=1),axis=2).sum(axis=1)*scale/radius;null_net=np.linalg.norm(noise[:,-1]-noise[:,0],axis=1)*scale/radius;noisy=xy[None,:,:]+noise;path=np.linalg.norm(np.diff(noisy,axis=1),axis=2).sum(axis=1)*scale/radius
            errors.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,sigma_pixels=sigma,sigma_m=sigma*scale,initial_radius_m=radius,noise_only_path_p95_r=float(np.quantile(null_path,.95)),noise_only_net_p95_r=float(np.quantile(null_net,.95)),observed_path_r=p.centroid_path_normalized,observed_net_r=p.centroid_net_normalized,perturbed_path_p025_r=float(np.quantile(path,.025)),perturbed_path_p975_r=float(np.quantile(path,.975)),scope='hypothetical independent isotropic errors per date; scenario ranges, not measured registration uncertainty or confidence intervals'))
        support.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,region=p.region,n=len(h),initial_area_m2=p.start_area_m2,initial_radius_m=radius,pixel_size_m=scale,one_pixel_over_radius=scale/radius,minimum_minor_pixels=float(h.sand_cay_minor_axis_length.min()),median_interval_days=float(h.date.diff().dt.days.median()),net_r=p.centroid_net_normalized,envelope_r=p.centroid_envelope_normalized,path_r=p.centroid_path_normalized,independent_ground_control_available=False))
    sample=pd.DataFrame(samples);err=pd.DataFrame(errors);sup=pd.DataFrame(support);save(sample,'主序列端点固定采样敏感性');save(err,'主序列质心误差传播情景');save(sup,'主序列尺寸与观测支持')
    assert sample.groupby('sand_cay_id').net_r.apply(lambda x:x.max()-x.min()).max()<1e-10
    # 全部同区域匹配候选报告，另按ID选择互不复用沙洲的对照；不按反差大小选择。
    pairs=pd.read_csv(ROOT/'outputs/轨迹与转换核查/相似面积记录候选配对.csv');pairs=pairs[pairs.same_region].sort_values(['cay0','cay1']);lookup=d.set_index('sand_cay_id');rows=[];used=set()
    for p in pairs.itertuples():
        a=lookup.loc[p.cay0];b=lookup.loc[p.cay1];keep=p.cay0 not in used and p.cay1 not in used
        if keep:used.update([p.cay0,p.cay1])
        rows.append(dict(cay0=p.cay0,cay1=p.cay1,region=a.region,area_ratio=p.area_ratio,span_ratio=p.span_ratio,observation_ratio=p.observation_ratio,net_area_gap=p.net_area_gap,envelope_absolute_difference_r=abs(a.centroid_envelope_normalized-b.centroid_envelope_normalized),net_absolute_difference_r=abs(a.centroid_net_normalized-b.centroid_net_normalized),non_reused_by_id=keep))
    matched=pd.DataFrame(rows);save(matched,'全部同区域可比记录差异')
    # 真实影像背景筛查覆盖四区图1、可比展示对及各区最小面积个体；不校正/覆盖原质心。
    from 核查空间轨迹与覆盖转换 import registration_background_screen,track
    ids=set(pd.read_csv(ROOT/'outputs/figures/source_data/四区图谱时相.csv').sand_cay_id) if (ROOT/'outputs/figures/source_data/四区图谱时相.csv').exists() else set()
    ids.update(['GBR_055_cay_001','OTHER_008_cay_002','MEDF_015_cay_001','NH_015_cay_001','MEDF_008_cay_001','MEDF_022_cay_001'])
    ids.update(d[d.region.isin(['Great Barrier Reef','Marshall Islands','Maldives','South China Sea'])].sort_values(['start_area_m2','sand_cay_id']).groupby('region').head(1).sand_cay_id)
    screens=[];screen_support=[]
    for p in d[d.sand_cay_id.isin(ids)].itertuples(index=False):
        if p.sand_cay_id=='NH_015_cay_001':
            screen=pd.read_csv(ROOT/'outputs/轨迹与转换核查/NH_015背景特征配准筛查.csv')
        else:
            h=track(p);masks=[read_mask(x) for x in h.resolved_mask_path]
            if len({m.shape for m in masks})!=1:
                screen_support.append(dict(sand_cay_id=p.sand_cay_id,region=p.region,status='raster shape mismatch',n=len(h),accepted=0));continue
            xy=h[['sand_cay_centroid_x','sand_cay_centroid_y']].to_numpy(dtype=float)
            actual=np.array([[np.where(m)[1].mean(),np.where(m)[0].mean()] for m in masks]);assert np.allclose(xy,actual,atol=1e-6)
            screen=registration_background_screen(h,xy,masks)
        screen['sand_cay_id']=p.sand_cay_id;screen['region']=p.region;screens.append(screen)
        screen_support.append(dict(sand_cay_id=p.sand_cay_id,region=p.region,status='candidate background screen only',n=p.n_observations,accepted=int(screen.screen_accepted.sum()),adjacent_pairs=len(screen)))
        print('背景筛查',p.sand_cay_id,int(screen.screen_accepted.sum()),'/',len(screen),flush=True)
    save(pd.concat(screens,ignore_index=True),'核心与小尺寸案例背景筛查');save(pd.DataFrame(screen_support),'核心与小尺寸案例配准支持')
    summary={'compactness_definition':'same largest 8-connected component area and four-direction Crofton perimeter; raw stored C preserved; primary external perimeter sensitivity','n_recomputed':int(master.compactness_primary_crofton.notna().sum()),'n_unavailable_outside_analysis':int(master.compactness_primary_crofton.isna().sum()),'selected_C_gt1':int(select_analysis_observations(master).compactness_primary_crofton.gt(1).sum()),'annual_comparison':summaries,'centroid_support':{'tracks':len(d),'sampling_rules':4,'scenario_sigma_pixels':[.5,1,2],'registration_scope':'no independent ground control; NH_015 prior background screen remains a candidate check','noise_only_scope':'cannot subtract hypothetical noise from observed path'},'matched_support':{'all_same_region_pairs':len(matched),'unique_cays':len(set(matched.cay0)|set(matched.cay1)),'regions':matched.groupby('region').size().to_dict(),'non_reused_pairs':int(matched.non_reused_by_id.sum()),'median_envelope_difference_r':float(matched.envelope_absolute_difference_r.median()),'non_reused_median_envelope_difference_r':float(matched.loc[matched.non_reused_by_id,'envelope_absolute_difference_r'].median())}}
    summary.update(master_sha256=hashlib.sha256((ROOT/'outputs/沙洲观测主表.csv').read_bytes()).hexdigest(),continuous_track_sha256=hashlib.sha256((ROOT/'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary_geometry_code_sha256=hashlib.sha256((ROOT/'代码/公共/构建沙洲观测与变化表.py').read_bytes()).hexdigest())
    (OUT/'核查结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--compactness-only',action='store_true');mode.add_argument('--spatial-only',action='store_true');args=parser.parse_args();run(args.compactness_only,args.spatial_only)
