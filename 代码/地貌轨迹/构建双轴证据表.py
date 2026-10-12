
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import numpy as np,pandas as pd,json
from 构建沙洲观测与变化表 import select_analysis_observations,read_mask
ROOT = PROJECT_ROOT;SRC=ROOT/'outputs/figures/source_data';BASE=ROOT.parent/"归档/重构前"
primary=pd.read_csv(SRC/'长期主序列双定义.csv')
surface=select_analysis_observations(pd.read_csv(ROOT/'outputs/植被状态审校主表.csv'))
joint=surface.merge(primary[['sand_cay_id','sensor','reference_frame_id']],on=['sand_cay_id','sensor','reference_frame_id'],how='inner')
median=joint.groupby('sand_cay_id').vegetation_fraction.median();primary['median_green_fraction']=primary.sand_cay_id.map(median)
freq=pd.read_csv(SRC/'双轴联合沙洲级观测.csv');mode=freq.sort_values(['within_cay_fraction','analysis_vegetation_state'],ascending=[False,True]).drop_duplicates('sand_cay_id').set_index('sand_cay_id').analysis_vegetation_state
primary['most_observed_surface_state']=primary.sand_cay_id.map(mode)
pairs=pd.read_csv(SRC/'表面状态相邻观测对.csv');change=pairs[pairs.analysis_cover_state_3!=pairs.next_state].groupby('sand_cay_id').size()
primary['s2_observed_transition_count']=primary.sand_cay_id.map(change).fillna(0).astype(int)
primary['s2_state_pair_count']=primary.sand_cay_id.map(pairs.groupby('sand_cay_id').size()).fillna(0).astype(int)
primary['state_pair_scope']='Sentinel-2 30–365 d; may differ from primary source; zero is not absence of change'
primary.to_csv(SRC/'表A_沙洲级面积路径状态.csv',index=False,encoding='utf-8-sig')
obs=pd.read_csv(BASE/'第一阶段审计/有效观测与图像掩膜清单.csv');obs=select_analysis_observations(obs);obs.date=pd.to_datetime(obs.date)
lookup={(r.sensor,r.sand_cay_id,r.reference_frame_id,str(r.date.date())):r for r in obs.itertuples()}
rows=[]
for p in pairs.itertuples():
    key=(p.sensor,p.sand_cay_id,p.reference_frame_id)
    early=lookup[(*key,str(p.date)[:10])];late=lookup[(*key,str(p.next_date)[:10])]
    m0=read_mask(early.resolved_mask_path);m1=read_mask(late.resolved_mask_path);scale=early.pixel_size_m
    valid_frame=m0.shape==m1.shape and np.isclose(scale,late.pixel_size_m)
    if valid_frame:
        erosion=np.sum(m0&~m1)*scale**2;gain=np.sum(m1&~m0)*scale**2;area=np.sum(m0)*scale**2
        dx=(late.sand_cay_centroid_x-early.sand_cay_centroid_x)*scale;dy=-(late.sand_cay_centroid_y-early.sand_cay_centroid_y)*scale
    else:
        erosion=gain=area=dx=dy=np.nan
    rows.append(dict(sensor=p.sensor,sand_cay_id=p.sand_cay_id,reef_id=p.reef_id,reference_frame_id=p.reference_frame_id,time_t=str(p.date)[:10],time_t1=str(p.next_date)[:10],interval_days=p.interval_days,state_t0=p.analysis_cover_state_3,state_t1=p.next_state,region=p.region,area_change_m2=gain-erosion,erosion_m2=erosion,accretion_m2=gain,boundary_exchange_m2=gain+erosion,centroid_dx_m=dx,centroid_dy_m=dy,centroid_speed_normalized=np.hypot(dx,dy)/np.sqrt(area/np.pi)/(p.interval_days/365.2425),quality_t0=early.quality_grade,quality_t1=late.quality_grade,green_fraction_t0=early.vegetation_fraction,green_fraction_t1=late.vegetation_fraction))
result=pd.DataFrame(rows)
result['mask_geometry_comparable']=result.boundary_exchange_m2.notna()
result.loc[~result.mask_geometry_comparable].to_csv(SRC/'状态对几何尺寸不一致待复核.csv',index=False,encoding='utf-8-sig')
force=pd.read_csv(ROOT/'outputs/流场波浪沿轴横轴分解.csv')
keys=['sensor','sand_cay_id','reference_frame_id','time_t','time_t1']
for key in ['time_t','time_t1']:force[key]=force[key].str[:10]
terms=['current_along_mean','current_cross_mean','wave_vector_along_mean','wave_vector_cross_mean']
result=result.merge(force[keys+[x for x in terms if x in force]],on=keys,how='left',validate='one_to_one')
wave=pd.read_csv(ROOT/'outputs/稳健性扩展/高波浪日核验区间.csv')
changes=pd.read_csv(ROOT/'outputs/沙洲变化区间.csv')[['transition_id']+keys]
for key in ['time_t','time_t1']:changes[key]=changes[key].str[:10]
wave=wave[['transition_id','event_storm_days']].merge(changes,on='transition_id',how='left');result=result.merge(wave[keys+['event_storm_days']],on=keys,how='left',validate='one_to_one')
result['forcing_scope']='Exact original adjacent interval only; missing values require fresh environment aggregation'
result.to_csv(SRC/'表B_区间级状态边界位移环境.csv',index=False,encoding='utf-8-sig')
print({'table_A':len(primary),'table_B':len(result),'exact_high_wave_intervals':int(result.event_storm_days.notna().sum())})
