"""补图S8：连续伸长率的早晚差异与真实轮廓；不推断趋势或分类。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT
from 构建沙洲观测与变化表 import select_analysis_observations,read_mask
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=ROOT/'outputs/figures/补充图';src=ROOT/'outputs/figures/source_data';out.mkdir(exist_ok=True)
    all_tracks=pd.read_csv(ROOT/'outputs/形状审查/主序列形状分布与窗口变化.csv')
    d=all_tracks[~all_tracks.anchor_overlap].copy()
    assert len(d)==90 and d.E_direction_preserved_1px.sum()==53
    all_tracks.to_csv(src/'补图S8_长期形状与季节支持.csv',index=False,encoding='utf-8-sig')
    quarter=pd.read_csv(ROOT/'outputs/形状审查/同季度早晚形状位置.csv')
    quarter.to_csv(src/'补图S8_同季度形状位置对照.csv',index=False,encoding='utf-8-sig')
    candidates=pd.read_csv(ROOT/'outputs/形状审查/轮廓候选案例及选择条件.csv')
    chosen=candidates.sort_values(['sensor','abs_change'],ascending=[True,False]).groupby('sensor').head(1)
    chosen.to_csv(src/'补图S8_真实轮廓案例.csv',index=False,encoding='utf-8-sig')
    obs=select_analysis_observations(pd.read_csv(ROOT.parent/'归档/重构前/第一阶段审计/有效观测与图像掩膜清单.csv'));obs.date=pd.to_datetime(obs.date)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':7,'axes.titlesize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),layout='constrained');a,b,c,e=axes.flat
    for sensor,color,marker,label in [('sentinel2','#22536D','o','Sentinel-2; n=55'),('google_earth','#BC7137','^','Google Earth; n=35')]:
        g=d[d.sensor.eq(sensor)]
        a.vlines(g.delta_log_area,g.delta_E_scenario_min,g.delta_E_scenario_max,color=color,alpha=.22,lw=.6)
        a.scatter(g.delta_log_area,g.delta_E,s=12,color=color,marker=marker,alpha=.8,label=label)
        z=g[g.E_direction_preserved_1px];a.scatter(z.delta_log_area,z.delta_E,s=24,facecolors='none',edgecolors='black',linewidth=.6)
        q=quarter[quarter.sensor.eq(sensor)]
        b.scatter(q.original_delta_E,q.delta_E,s=12,color=color,marker=marker,alpha=.8)
        z=q[q.E_direction_preserved_1px & q.original_E_direction_preserved_1px & q.same_direction_as_original]
        b.scatter(z.original_delta_E,z.delta_E,s=24,facecolors='none',edgecolors='black',linewidth=.6)
    a.axhline(0,color='#888888',lw=.6);a.axvline(0,color='#888888',lw=.6);a.legend(frameon=False,fontsize=6)
    a.set(xlabel='Late / early log area ratio',ylabel='Late − early elongation')
    a.text(.03,.03,'Ranges: ±1-pixel scenarios, not CIs',transform=a.transAxes,fontsize=6)
    b.axhline(0,color='#888888',lw=.6);b.axvline(0,color='#888888',lw=.6);b.set(xlabel='Original-window elongation change',ylabel='Same-quarter elongation change')
    b.text(.97,.97,'63 series; 38 with the same sign\n18 preserve direction in both windows',ha='right',va='top',transform=b.transAxes,fontsize=6)
    contour_source=[]
    for ax,p in zip([c,e],chosen.itertuples()):
        g=obs[(obs.sand_cay_id==p.sand_cay_id)&(obs.sensor==p.sensor)&(obs.reference_frame_id==p.reference_frame_id)].sort_values('date').reset_index(drop=True)
        radius=np.sqrt(p.start_area_m2/np.pi)
        for i in [0,1,len(g)-3,len(g)-2,len(g)-1]:
            x=g.iloc[i];m=read_mask(x.resolved_mask_path);y,z=np.where(m);cx=z.mean();cy=y.mean()
            contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                xy=contour[:,0,:];xy=np.vstack([xy,xy[0]])
                ax.plot((xy[:,0]-cx)*x.pixel_size_m/radius,-(xy[:,1]-cy)*x.pixel_size_m/radius,color='#22536D' if i<2 else '#BC7137',alpha=.65,lw=.7)
            contour_source.append(dict(sand_cay_id=p.sand_cay_id,sensor=p.sensor,date=str(x.date.date()),image_id=x.image_id,window='early' if i<2 else 'late',pixel_size_m=x.pixel_size_m,initial_radius_m=radius,recentered_x_pixel=cx,recentered_y_pixel=cy,mask_path=x.resolved_mask_path))
        ax.set_aspect('equal');ax.set(xlabel='Centroid-centred E-W (r)',ylabel='Centroid-centred N-S (r)',title=p.sand_cay_id+' | '+('S2' if p.sensor=='sentinel2' else 'GE'))
        ax.text(.03,.03,f'Window E: {p.E_early:.2f} → {p.E_late:.2f}',transform=ax.transAxes,fontsize=6)
    pd.DataFrame(contour_source).to_csv(src/'补图S8_逐景轮廓来源.csv',index=False,encoding='utf-8-sig')
    for ax,label in zip(axes.flat,'abcd'):ax.text(-.14,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=10)
    for ext in ['png','pdf','svg','tiff']:fig.savefig(out/('图S8_长期伸长率与边界敏感性.'+ext),dpi=350)
    plt.close(fig)
    print('S8完成；93条汇总、90条不重叠窗口、10景真实轮廓来源')

if __name__=='__main__':main()
