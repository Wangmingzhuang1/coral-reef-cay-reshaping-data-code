"""空间轨迹、转换前后支持及NH_015配准筛查；不建立新地貌类型或因果模型。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from 构建沙洲观测与变化表 import select_analysis_observations,read_mask

SRC=ROOT/'outputs/figures/source_data'
OUT=ROOT/'outputs/轨迹与转换核查';OUT.mkdir(parents=True,exist_ok=True)
OBS=select_analysis_observations(pd.read_csv(ROOT.parent/'归档/重构前/第一阶段审计/有效观测与图像掩膜清单.csv'))
OBS.date=pd.to_datetime(OBS.date)
def save(d,name):d.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def track(p):return OBS[(OBS.sand_cay_id==p.sand_cay_id)&(OBS.sensor==p.sensor)&(OBS.reference_frame_id==p.reference_frame_id)].sort_values('date').copy()

def continuous():
    d=pd.read_csv(SRC/'表A_沙洲级面积路径状态.csv');rows=[]
    for p in d.itertuples():
        g=track(p);m0=read_mask(g.iloc[0].resolved_mask_path);m1=read_mask(g.iloc[-1].resolved_mask_path)
        same=m0.shape==m1.shape and np.isclose(g.iloc[0].pixel_size_m,g.iloc[-1].pixel_size_m)
        row=p._asdict();row['endpoint_geometry_comparable']=same
        row['net_to_path_ratio']=p.centroid_net_normalized/p.centroid_path_normalized
        if same:
            loss=int((m0&~m1).sum());gain=int((m1&~m0).sum());area=int(m0.sum())
            row.update(endpoint_loss_fraction=loss/area,endpoint_gain_fraction=gain/area,endpoint_exchange_fraction=(loss+gain)/area,endpoint_net_fraction=(gain-loss)/area)
        else:row.update(endpoint_loss_fraction=np.nan,endpoint_gain_fraction=np.nan,endpoint_exchange_fraction=np.nan,endpoint_net_fraction=np.nan)
        rows.append(row)
    d=pd.DataFrame(rows);save(d,'连续空间轨迹与首末边界交换')
    # 同源、近稳面积、尺寸/跨度/观测数可比；优先同区域并选择最大活动包络反差；不以接近零的净位移作比值排序，透明记录全部候选。
    near=d[d.net_area_bin.eq('Near-stable area') & d.endpoint_geometry_comparable & d.sensor.eq('sentinel2')]
    pairs=[]
    for i,a in near.iterrows():
        for j,b in near.loc[near.index>i].iterrows():
            if a.reef_id==b.reef_id:continue
            ar=max(a.start_area_m2,b.start_area_m2)/min(a.start_area_m2,b.start_area_m2)
            span=max(a.span_years,b.span_years)/min(a.span_years,b.span_years)
            density=max(a.n_observations,b.n_observations)/min(a.n_observations,b.n_observations)
            if ar<=1.5 and span<=1.5 and density<=1.5 and abs(a.relative_change-b.relative_change)<=.10:
                pairs.append(dict(cay0=a.sand_cay_id,cay1=b.sand_cay_id,area_ratio=ar,span_ratio=span,observation_ratio=density,net_area_gap=abs(a.relative_change-b.relative_change),net_displacement_contrast=max(a.centroid_net_normalized,b.centroid_net_normalized)/max(min(a.centroid_net_normalized,b.centroid_net_normalized),1e-9),same_region=a.region==b.region,envelope_contrast=abs(a.centroid_envelope_normalized-b.centroid_envelope_normalized)))
    matched=pd.DataFrame(pairs).sort_values(['same_region','envelope_contrast','cay0','cay1'],ascending=[False,False,True,True]);save(matched,'相似面积记录候选配对')
    chosen=d[d.sand_cay_id.isin(matched.iloc[0][['cay0','cay1']])].copy();save(chosen,'相似面积不同空间活动展示对')
    return {'primary':len(d),'endpoint_comparable':int(d.endpoint_geometry_comparable.sum()),'matched_candidates':len(matched),'same_region_candidates':int(matched.same_region.sum()),'display_pair':chosen.sand_cay_id.tolist(),'display_scope':'descriptive contrast selected for illustration, not a hypothesis test'}

def flanking():
    b=pd.read_csv(SRC/'表B_区间级状态边界位移环境.csv');b.time_t=pd.to_datetime(b.time_t);b.time_t1=pd.to_datetime(b.time_t1)
    rank={'low_cover':0,'partial_cover':1,'dominant_cover':2}
    strict=pd.read_csv(SRC/'严格标签一致状态对.csv')
    strictkeys=set(zip(strict.sand_cay_id,pd.to_datetime(strict.date),pd.to_datetime(strict.next_date)))
    lookup=OBS.set_index(['sensor','sand_cay_id','reference_frame_id','date'])
    rows=[]
    for keys,g in b.groupby(['sensor','sand_cay_id','reference_frame_id']):
        g=g.sort_values('time_t').reset_index(drop=True)
        for i in range(1,len(g)-1):
            pre,event,post=g.iloc[i-1],g.iloc[i],g.iloc[i+1]
            if event.state_t0==event.state_t1:continue
            continuous=pre.time_t1==event.time_t and event.time_t1==post.time_t
            stable=pre.state_t0==pre.state_t1 and post.state_t0==post.state_t1
            if not(continuous and stable and pre.mask_geometry_comparable and event.mask_geometry_comparable and post.mask_geometry_comparable):continue
            before=lookup.loc[(*keys,pre.time_t)];after=lookup.loc[(*keys,post.time_t)]
            phase0=pre.time_t+(pre.time_t1-pre.time_t)/2;phase1=post.time_t+(post.time_t1-post.time_t)/2
            phasegap=abs(phase0.dayofyear-phase1.dayofyear);phasegap=min(phasegap,365.2425-phasegap)
            area0=float(before.sand_cay_area_m2);area1=float(after.sand_cay_area_m2)
            strict_all=all((keys[1],x.time_t,x.time_t1) in strictkeys for x in [pre,event,post])
            forces=['current_along_mean','current_cross_mean','wave_vector_along_mean','wave_vector_cross_mean','event_storm_days']
            complete=all(pd.notna(x[v]) for x in [pre,post] for v in forces)
            direction=1 if rank[event.state_t1]>rank[event.state_t0] else -1
            row=dict(sand_cay_id=keys[1],reef_id=event.reef_id,reference_frame_id=keys[2],from_state=event.state_t0,to_state=event.state_t1,direction='Increasing cover' if direction==1 else 'Decreasing cover',pre_start=pre.time_t,event_start=event.time_t,event_end=event.time_t1,post_end=post.time_t1,pre_days=pre.interval_days,post_days=post.interval_days,season_gap_days=phasegap,area_ratio=area1/area0,strict_labels=strict_all,environment_complete=complete,pre_speed=pre.centroid_speed_normalized,post_speed=post.centroid_speed_normalized,pre_exchange=pre.boundary_exchange_m2/area0/(pre.interval_days/365.2425),post_exchange=post.boundary_exchange_m2/area1/(post.interval_days/365.2425))
            row['speed_log_change']=np.log(row['post_speed']+.01)-np.log(row['pre_speed']+.01)
            row['exchange_log_change']=np.log1p(row['post_exchange'])-np.log1p(row['pre_exchange'])
            row['direction_adjusted_speed_change']=direction*row['speed_log_change'];row['direction_adjusted_exchange_change']=direction*row['exchange_log_change']
            row['observation_matched']=phasegap<=45 and .5<=row['post_days']/row['pre_days']<=2 and 2/3<=row['area_ratio']<=1.5
            rows.append(row)
    e=pd.DataFrame(rows).sort_values(['sand_cay_id','event_start']).reset_index(drop=True)
    # Chronological greedy selection uses dates alone, not observed effect sizes; no image/interval reuse within cay.
    used={};keep=[]
    for x in e.itertuples():
        dates={x.pre_start,x.event_start,x.event_end,x.post_end};occupied=used.setdefault(x.sand_cay_id,set())
        keep.append(not bool(dates&occupied))
        if keep[-1]:occupied.update(dates)
    e['non_reused']=keep;save(e,'状态转换前后候选与排除标记')
    specs={'eligible_flanks':e,'non_reused':e[e.non_reused],'strict_non_reused':e[e.non_reused&e.strict_labels],'observation_matched':e[e.non_reused&e.observation_matched],'environment_complete_matched':e[e.non_reused&e.observation_matched&e.environment_complete]}
    for tolerance in [90,120]:
        relaxed=e.non_reused & e.season_gap_days.le(tolerance) & e.post_days.div(e.pre_days).between(.5,2) & e.area_ratio.between(2/3,1.5)
        specs['observation_matched_'+str(tolerance)+'d']=e[relaxed]
    support=[];contrasts=[];rng=np.random.default_rng(20261009)
    for label,t in specs.items():
        support.append(dict(specification=label,n_events=len(t),n_cays=t.sand_cay_id.nunique(),n_reefs=t.reef_id.nunique(),increasing=int(t.direction.eq('Increasing cover').sum()),decreasing=int(t.direction.eq('Decreasing cover').sum()),environment_complete=int(t.environment_complete.sum())))
        for direction in ['Increasing cover','Decreasing cover']:
            for outcome in ['speed_log_change','exchange_log_change']:
                v=t[t.direction.eq(direction)].groupby('sand_cay_id')[outcome].mean().dropna().to_numpy()
                if len(v)>=3:
                    boot=np.median(rng.choice(v,(1999,len(v)),replace=True),axis=1);lo,hi=np.quantile(boot,[.025,.975]);center=np.median(v)
                else:center=lo=hi=np.nan
                contrasts.append(dict(specification=label,direction=direction,outcome=outcome,n_cays=len(v),n_events=int(t.direction.eq(direction).sum()),cay_equal_median=center,ci95_low=lo,ci95_high=hi,inference='descriptive cay-bootstrap interval; no causal test; no multivariable model'))
    save(pd.DataFrame(support),'转换前后有效样本层级');save(pd.DataFrame(contrasts),'转换前后描述性对照')
    return {'support':support,'adjusted_model':'not fitted: small clustered events, sparse full environment overlap and seasonal matches; no evidence of a stable coupled mechanism is inferred'}

def registration_background_screen(g,coords,masks,*,buffer_m=None,edge_m=None,compact_features=False):
    scale=float(g.pixel_size_m.iloc[0])
    if len({m.shape for m in masks})!=1:
        raise ValueError('栅格形状不同，不能匹配背景')
    union=np.logical_or.reduce(masks)
    radius=40 if buffer_m is None else int(np.ceil(buffer_m/scale))
    edge=20 if edge_m is None else max(1,int(np.ceil(edge_m/scale)))
    exclude=cv2.dilate(union.astype('uint8'),np.ones((2*radius+1,2*radius+1),np.uint8))
    valid=((exclude==0)*255).astype('uint8');valid[:edge]=0;valid[-edge:]=0;valid[:,:edge]=0;valid[:,-edge:]=0
    options=dict(edgeThreshold=5,patchSize=15) if compact_features else {}
    orb=cv2.ORB_create(nfeatures=5000,fastThreshold=7,**options);features=[];clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    clearance=cv2.distanceTransform(valid,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    for path in g.resolved_image_path:
        gray=cv2.cvtColor(np.array(Image.open(path).convert('RGB')),cv2.COLOR_RGB2GRAY)
        keypoints,descriptors=orb.detectAndCompute(clahe.apply(gray),valid)
        if compact_features and descriptors is not None:
            # 不只排除中心：按金字塔层级及旋转块外接圆要求完整描述块位于背景。
            keep=[i for i,kp in enumerate(keypoints)
                  if clearance[int(round(kp.pt[1])),int(round(kp.pt[0]))]
                  >=np.ceil(orb.getPatchSize()*np.sqrt(2)/2*orb.getScaleFactor()**kp.octave)]
            keypoints=[keypoints[i] for i in keep]
            descriptors=descriptors[keep] if keep else None
        features.append((keypoints,descriptors))
    bf=cv2.BFMatcher(cv2.NORM_HAMMING);rows=[]
    for i in range(len(g)-1):
        kp0,d0=features[i];kp1,d1=features[i+1];row=dict(date0=str(g.date.iloc[i].date()),date1=str(g.date.iloc[i+1].date()),raw_step_m=float(np.linalg.norm(coords[i+1]-coords[i])*scale),screen_accepted=False,background_fraction=float((valid>0).mean()),buffer_m=radius*scale,edge_m=edge*scale,n_features_t=len(kp0),n_features_t1=len(kp1))
        if d0 is not None and d1 is not None:
            good=[pair[0] for pair in bf.knnMatch(d1,d0,k=2) if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
            if len(good)>=8:
                src=np.float32([kp1[m.queryIdx].pt for m in good]);dst=np.float32([kp0[m.trainIdx].pt for m in good]);a,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000,confidence=.999)
                if a is not None:
                    inside=inliers.ravel().astype(bool);points=src[inside];fit=cv2.transform(src[:,None,:],a)[:,0,:];res=np.linalg.norm(fit[inside]-dst[inside],axis=1);factor=float(np.hypot(a[0,0],a[1,0]));rotation=float(np.degrees(np.arctan2(a[1,0],a[0,0])));cells=len(set(zip((points[:,0]/(union.shape[1]/4)).astype(int),(points[:,1]/(union.shape[0]/4)).astype(int))))
                    aligned=a[:,:2]@coords[i+1]+a[:,2];correction=(aligned-coords[i+1])*scale
                    row.update(n_matches=len(good),n_inliers=int(inside.sum()),inlier_fraction=float(inside.mean()),coverage_cells=cells,scale_factor=factor,rotation_deg=rotation,median_residual_px=float(np.median(res)),background_correction_m=float(np.linalg.norm(correction)),screen_aligned_step_m=float(np.linalg.norm(aligned-coords[i])*scale),screen_accepted=bool(inside.sum()>=30 and inside.mean()>=.5 and cells>=6 and .98<=factor<=1.02 and abs(rotation)<=1))
        rows.append(row)
    return pd.DataFrame(rows)

def registration():
    p=pd.read_csv(SRC/'表A_沙洲级面积路径状态.csv');p=p[p.sand_cay_id.eq('NH_015_cay_001')].iloc[0];g=track(p)
    scale=g.pixel_size_m.iloc[0];radius=np.sqrt(p.start_area_m2/np.pi)
    coords=g[['sand_cay_centroid_x','sand_cay_centroid_y']].to_numpy();masks=[read_mask(x) for x in g.resolved_mask_path]
    recomputed=np.array([[np.nonzero(m)[1].mean(),np.nonzero(m)[0].mean()] for m in masks])
    assert np.allclose(coords,recomputed,atol=1e-6)
    screen=registration_background_screen(g,coords,masks);save(screen,'NH_015背景特征配准筛查')
    # Sampling sensitivity excludes repeated within-month captures only; no measurement is overwritten.
    sampled=[]
    for freq in ['all','month','quarter','year']:
        t=g.copy()
        if freq!='all':
            t['period']=t.date.dt.to_period({'month':'M','quarter':'Q','year':'Y'}[freq]);t=t.groupby('period',sort=True).first().reset_index()
        xy=t[['sand_cay_centroid_x','sand_cay_centroid_y']].to_numpy();sampled.append(dict(sampling=freq,n=len(t),path_r=np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()*scale/radius,net_r=np.linalg.norm(xy[-1]-xy[0])*scale/radius,envelope_r=np.linalg.norm(xy-xy.mean(axis=0),axis=1).max()*scale/radius))
    save(pd.DataFrame(sampled),'NH_015采样敏感性')
    rng=np.random.default_rng(20261009);scenarios=[]
    for sigma in [1,2,5]:
        noise=rng.normal(0,sigma,(5000,len(g),2));null=np.linalg.norm(np.diff(noise,axis=1),axis=2).sum(axis=1)*scale/radius;perturbed=coords+noise;path=np.linalg.norm(np.diff(perturbed,axis=1),axis=2).sum(axis=1)*scale/radius
        scenarios.append(dict(sigma_pixels=sigma,n_simulations=5000,noise_only_median_r=float(np.median(null)),noise_only_p95_r=float(np.quantile(null,.95)),observed_perturbed_p025_r=float(np.quantile(path,.025)),observed_perturbed_p975_r=float(np.quantile(path,.975)),scope='hypothetical independent isotropic positional error per date; not measured registration uncertainty or a confidence interval'))
    save(pd.DataFrame(scenarios),'NH_015位置误差情景')
    crop=pd.read_csv(ROOT/'data/source_dataset/data/metadata/google_earth_crop_manifest.csv');crop=crop[crop.reef_id.eq('NH_015')];save(crop,'NH_015裁剪参考元数据')
    accepted=screen[screen.screen_accepted]
    return {'n_dates':len(g),'n_frames':g.reference_frame_id.nunique(),'n_pixel_sizes':g.pixel_size_m.nunique(),'mask_centroid_reproduction':True,'n_background_accepted':len(accepted),'n_adjacent_pairs':len(screen),'raw_path_r':float(p.centroid_path_normalized),'net_r':float(p.centroid_net_normalized),'envelope_r':float(p.centroid_envelope_normalized),'accepted_raw_sum_r':float(accepted.raw_step_m.sum()/radius),'accepted_screen_aligned_sum_r':float(accepted.screen_aligned_step_m.sum()/radius),'background_screen_scope':'candidate submerged-reef texture, not independently surveyed stable control points; no mask correction or overwrite','largest_steps':screen.nlargest(5,'raw_step_m').to_dict('records'),'crop_rows':len(crop),'crop_rectangles':int(crop[['crop_xmin','crop_ymin','crop_xmax','crop_ymax']].drop_duplicates().shape[0])}

if __name__=='__main__':
    summary={'continuous':continuous(),'flanking':flanking(),'NH_015':registration()}
    (OUT/'证据核查.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
