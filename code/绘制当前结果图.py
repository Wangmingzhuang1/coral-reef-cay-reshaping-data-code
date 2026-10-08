"""按当前合同绘制论文结果图。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from 提取沙洲颜色组成 import vegetation_pixels
from 构建沙洲观测与变化表 import read_mask, normalize_cay_mask
import pandas as pd
import cv2
from scipy import stats
from PIL import Image
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'; FIG=OUT/'figures'; SRC=FIG/'source_data'
FIG.mkdir(exist_ok=True); SRC.mkdir(exist_ok=True)
C={'navy':'#1F4E6D','blue':'#5AA6D1','teal':'#2B9D8F','orange':'#ED8B27','red':'#C44E52','grey':'#77818B','light':'#D9E1E8','dark':'#263640'}
mpl.rcParams.update({'font.family':'Times New Roman','font.size':7.2,'axes.labelsize':7.5,'axes.titlesize':9.4,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.75,'figure.dpi':180})

def clean(ax): ax.spines[['top','right']].set_visible(False); ax.tick_params(width=.7,length=3)
def tag(ax,s,y=1.02): ax.text(-.14,y,s,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')
def save(fig,name):
    for ext,kw in [('svg',{}),('pdf',{}),('png',{'dpi':360}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}})]: fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',**kw)

def binned_mobility_summary(frame, n_bins=6):
    """Use equal-count bins to show a trajectory-level association without a point-cloud panel."""
    data = frame[['vegetation_fraction_median', 'centroid_path_normalized']].dropna().copy()
    data['bin'] = pd.qcut(data['vegetation_fraction_median'], q=min(n_bins, len(data)), duplicates='drop')
    return data.groupby('bin', observed=True).agg(
        vegetation_fraction_median=('vegetation_fraction_median', 'median'),
        mobility_median=('centroid_path_normalized', 'median'),
        mobility_q25=('centroid_path_normalized', lambda x: x.quantile(.25)),
        mobility_q75=('centroid_path_normalized', lambda x: x.quantile(.75)),
        n=('centroid_path_normalized', 'size'),
    ).reset_index(drop=True)


def read_raster(path, flags):
    """Read a raster from a Windows path without relying on ASCII filenames."""
    buffer = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f'Unreadable raster: {path}')
    return image


def colour_proxy_pixels(image_bgr, cay_mask):
    return vegetation_pixels(image_bgr, cay_mask)


def crop_to_mask(mask, padding_fraction=.14, minimum_padding=10):
    yy, xx = np.where(mask > 0)
    padding = max(minimum_padding, int(max(xx.max() - xx.min(), yy.max() - yy.min()) * padding_fraction))
    return (max(0, yy.min() - padding), min(mask.shape[0], yy.max() + padding + 1), max(0, xx.min() - padding), min(mask.shape[1], xx.max() + padding + 1))

def fig1():
    d=pd.read_csv(OUT/'沙洲面积主轨迹分类.csv');d['absolute_net_area_change_pct']=d.relative_change.abs()*100;d['cumulative_centroid_path_radii']=d.centroid_path_normalized;d['net_area_type']=np.select([d.relative_change.abs().lt(.25),d.relative_change.abs().lt(.50)],['Net-area stable','Moderate net change'],default='Major net change');d['strict_morphological_stability']=d.net_area_type.eq('Net-area stable')&d.late_cv.le(.15)&d.max_adjacent_area_ratio.le(1.50)&d.centroid_path_normalized.le(1.0); d.to_csv(SRC/'figure_1_trajectory_states.csv',index=False,encoding='utf-8-sig')
    morph=d.late_cv.le(.15)&d.max_adjacent_area_ratio.le(1.50)&d.centroid_path_normalized.le(1.0)
    threshold=pd.DataFrame([{'net_area_cutoff_pct':cut,'n_area_stable':int((sel:=d.absolute_net_area_change_pct.lt(cut)).sum()),'n_morphologically_stable':int((sel&morph).sum()),'reworking_fraction':float(1-(sel&morph).sum()/sel.sum())} for cut in np.arange(10,51,5)])
    threshold.to_csv(SRC/'figure_1_threshold_stability.csv',index=False,encoding='utf-8-sig')
    order=['Net-area stable','Moderate net change','Major net change']; pal=dict(zip(order,[C['navy'],C['orange'],C['red']]))
    fig=plt.figure(figsize=(7.2,5.35)); gs=fig.add_gridspec(2,2,width_ratios=[1.15,0.85],height_ratios=[1.05,1.0],wspace=.35,hspace=.40); a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); cc=fig.add_subplot(gs[1,0]); dd=fig.add_subplot(gs[1,1])
    strict=d[d.strict_morphological_stability]
    d['na_bin']=pd.cut(d.absolute_net_area_change_pct,bins=[-1,10,25,50,1e5],labels=['0–10','10–25','25–50','≥50'])
    bs=d.groupby('na_bin',observed=True).cumulative_centroid_path_radii.agg(med='median',q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75),n='size').reset_index()
    bs.to_csv(SRC/'figure_1_path_by_net_area_bin.csv',index=False,encoding='utf-8-sig')
    xb=np.arange(len(bs))
    a.fill_between(xb,bs.q25,bs.q75,color=C['light'],alpha=.85,zorder=1)
    a.plot(xb,bs.med,color=C['navy'],lw=2,marker='o',ms=4,zorder=3)
    a.axhline(1,color=C['grey'],ls='--',lw=.8); a.text(0.02,1.15,'1-radius threshold',color=C['grey'],fontsize=6.5,ha='left')
    for xx,r in zip(xb,bs.itertuples()): a.text(xx,r.med*1.15,f'n={r.n}',ha='center',fontsize=6.2,color=C['dark'])
    n_stable=int(d.absolute_net_area_change_pct.lt(25).sum()); n_rework=int((d.absolute_net_area_change_pct.lt(25)&~morph).sum())
    a.text(.98,.40,f'n={len(d)} primary tracks\n{n_stable} area-stable, {n_rework} still reworking',transform=a.transAxes,ha='right',va='center',fontsize=6.4,color=C['navy'])
    a.set_yscale('log'); a.set(xticks=xb,xticklabels=bs.na_bin.astype(str),xlabel='Absolute net area change (%)',ylabel='Cumulative centroid path (initial equivalent radii)'); clean(a); tag(a,'a',1.08)
    n=d.net_area_type.value_counts().reindex(order); y=np.arange(3)[::-1]; b.barh(y,n,color=[pal[k] for k in order],height=.62); b.set(yticks=y,yticklabels=['Stable\n<25%','Moderate\n25-<50%','Major\n≥50%'],xlim=(0,70),xlabel='Primary trajectories (n)'); [b.text(v+1.2,yy,str(int(v)),va='center',fontsize=7.5) for yy,v in zip(y,n)]; clean(b);tag(b,'b')
    cc.fill_between(threshold.net_area_cutoff_pct,threshold.reworking_fraction*100,65,color=C['light'],alpha=.8);cc.plot(threshold.net_area_cutoff_pct,threshold.reworking_fraction*100,color=C['navy'],lw=1.8);cc.axvline(25,color=C['orange'],ls='--',lw=1);r25=threshold[threshold.net_area_cutoff_pct.eq(25)].iloc[0];cc.text(26,67,f'25%: {r25.reworking_fraction*100:.1f}% reworking\n({int(r25.n_morphologically_stable)}/{int(r25.n_area_stable)} strictly stable)',fontsize=6.1,color=C['dark'],va='bottom');cc.set(xlabel='Net-area cutoff (%)',ylabel='Area-stable tracks still reworking (%)',ylim=(65,90));clean(cc);tag(cc,'c')
    mm=pd.read_csv(OUT/'沙洲面积轨迹指标.csv'); mm.to_csv(SRC/'figure_1_longitudinal_support.csv',index=False,encoding='utf-8-sig'); elig=mm[mm.eligible]; dd.hexbin(mm.span_years,mm.n_observations,gridsize=14,mincnt=1,cmap=mpl.colors.LinearSegmentedColormap.from_list('track_density',['#EEF5F8',C['blue'],C['navy']]),linewidths=0); dd.axvline(3,color=C['orange'],ls='--',lw=.9); dd.axhline(4,color=C['orange'],ls='--',lw=.9); dd.text(.98,.96,f'{len(elig)} tracks / {elig.sand_cay_id.nunique()} cays eligible',transform=dd.transAxes,ha='right',va='top',fontsize=6.4,color=C['navy']); dd.set(xlabel='Observation span (years)',ylabel='Observations per track'); clean(dd);tag(dd,'d')
    fig.subplots_adjust(left=.10,right=.99,bottom=.11,top=.96);save(fig,'figure_1_dynamic_equilibrium');plt.close(fig)
    return {'primary_trajectories':len(d),'net_area_stable':int(n.iloc[0]),'strict_stability':len(strict),'threshold_reworking_range_pct':[float(threshold.reworking_fraction.min()*100),float(threshold.reworking_fraction.max()*100)],'eligible_tracks':int(len(elig)),'eligible_cays':int(elig.sand_cay_id.nunique())}

def fig2_seasonal():
    """以曲线、条带和掩膜图组呈现年度重复性，避免把该证据压缩为点图。"""
    pairwise=pd.read_csv(OUT/'Sentinel2季节形态配对.csv')
    design=pairwise[['phase_distance','gap_years']].copy()
    design['gap_years_squared']=design.gap_years**2
    design=pd.concat([design,pd.get_dummies(pairwise['sand_cay_id'],drop_first=True,dtype=float)],axis=1)
    model=sm.OLS(pairwise['iou'],sm.add_constant(design)).fit(cov_type='cluster',cov_kwds={'groups':pairwise['sand_cay_id']})
    beta=float(model.params['phase_distance']); se=float(model.bse['phase_distance'])
    phase_days=np.linspace(0,182.5,121); phase_distance=1-np.cos(2*np.pi*phase_days/365.2425)
    phase_curve=pd.DataFrame({'annual_phase_separation_days':phase_days,'phase_distance':phase_distance,'adjusted_iou_change_from_same_phase':beta*phase_distance,'ci95_low':(beta-1.96*se)*phase_distance,'ci95_high':(beta+1.96*se)*phase_distance})
    phase_curve.to_csv(SRC/'figure_2_seasonal_phase_curve.csv',index=False,encoding='utf-8-sig')

    harmonic=pd.read_csv(OUT/'Sentinel2季节谐波交叉验证.csv')
    cay=harmonic.groupby(['metric','sand_cay_id']).agg(baseline_error=('baseline_error','mean'),seasonal_error=('seasonal_error','mean')).reset_index()
    rng=np.random.default_rng(20260914); rows=[]
    order=['vegetation_fraction','sand_cay_area_m2','sand_cay_centroid_x_norm','sand_cay_centroid_y_norm','sand_cay_compactness','sand_cay_elongation','sand_cay_major_axis_angle']
    labels={'vegetation_fraction':'Vegetation proxy','sand_cay_area_m2':'Area','sand_cay_centroid_x_norm':'Centroid E-W','sand_cay_centroid_y_norm':'Centroid N-S','sand_cay_compactness':'Compactness','sand_cay_elongation':'Elongation','sand_cay_major_axis_angle':'Major-axis angle'}
    q_lookup=pd.read_csv(OUT/'Sentinel2季节谐波检验汇总.csv').set_index('metric')['fdr_q_value']
    for metric in order:
        values=cay[cay.metric.eq(metric)][['baseline_error','seasonal_error']].dropna().to_numpy()
        estimate=(values[:,0].sum()-values[:,1].sum())/values[:,0].sum()
        boot=np.array([(sample[:,0].sum()-sample[:,1].sum())/sample[:,0].sum() for sample in (values[rng.choice(len(values),size=len(values),replace=True)] for _ in range(2000))])
        rows.append({'metric':metric,'label':labels[metric],'n_cays':len(values),'mean_relative_error_reduction':float(estimate),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'fdr_q_value':float(q_lookup.loc[metric])})
    harmonic_visual=pd.DataFrame(rows)
    harmonic_visual.to_csv(SRC/'figure_2_harmonic_relative_error_reduction.csv',index=False,encoding='utf-8-sig')

    template=pd.read_csv(OUT/'Sentinel2季节模板交叉验证.csv')
    obs=pd.read_csv(OUT/'沙洲观测主表.csv')
    obs=obs[(obs.sensor.eq('sentinel2'))&~obs.quality_grade.eq('C')].copy();obs['date']=pd.to_datetime(obs.date);obs['quarter']=obs.date.dt.quarter
    identity=obs[['sand_cay_id','reef_id']].drop_duplicates().drop_duplicates('sand_cay_id'); template=template.merge(identity,on='sand_cay_id',how='left');template['region']=template.reef_id.str.split('_').str[0]
    recurrent=template.groupby(['sand_cay_id','reef_id','region']).agg(mean_same_vs_opposite=('gain_same_vs_opposite','mean'),n_heldout=('image_id','size')).reset_index()
    def build_case_records(exemplar):
        cay_obs=obs[obs.sand_cay_id.eq(exemplar.sand_cay_id)].copy()
        cay_obs['year']=cay_obs.date.dt.year
        cay_obs['mask_path']=cay_obs.mask_path.fillna('')
        masked=cay_obs[cay_obs.mask_path.apply(lambda p: Path(p).is_file())]
        counts=masked.groupby('quarter').year.nunique().sort_values(ascending=False)
        if counts.empty:
            return []
        selected_quarter=int(counts.index[0])
        same_season=masked[masked.quarter.eq(selected_quarter)].sort_values('date').groupby('year',as_index=False).first()
        if len(same_season)<4:
            return []
        return [{'region':exemplar.region,'sand_cay_id':exemplar.sand_cay_id,'quarter':selected_quarter,'rank':rank,'date':record.date.strftime('%Y-%m-%d'),'mask_path':record.mask_path,'centroid_x':record.sand_cay_centroid_x,'centroid_y':record.sand_cay_centroid_y,'pixel_size_m':record.pixel_size_m} for rank,record in enumerate(same_season.itertuples(),start=1)]
    exemplars=[];collected={}
    for region,forced in [('GBR',None),('MEDF',None),('NH','NH_015_cay_001')]:
        pool=recurrent[recurrent.region.eq(region)]
        if forced: pool=pool[pool.sand_cay_id.eq(forced)]
        candidates=pool.sort_values(['mean_same_vs_opposite','n_heldout'],ascending=False)
        chosen=None; fallback=None
        for candidate in candidates.itertuples():
            records=build_case_records(candidate)
            if not records:
                continue
            uniform=int(pd.Series([np.asarray(Image.open(record['mask_path'])).shape for record in records]).value_counts().iloc[0])
            if fallback is None:
                fallback=(candidate,records)
            if uniform>=6:
                chosen=(candidate,records); break
        if chosen is None:
            chosen=fallback
        if chosen:
            exemplars.append(chosen[0]); collected[region]=chosen[1]
    atlas=pd.DataFrame(exemplars)
    atlas.to_csv(SRC/'figure_2_seasonal_mask_atlas_selection.csv',index=False,encoding='utf-8-sig')

    fig=plt.figure(figsize=(7.2,5.4))
    a=fig.add_axes([.13,.50,.322,.43]); b=fig.add_axes([.575,.50,.322,.43])
    a.plot(phase_curve.annual_phase_separation_days,phase_curve.adjusted_iou_change_from_same_phase,color=C['navy'],lw=1.7)
    a.fill_between(phase_curve.annual_phase_separation_days,phase_curve.ci95_low,phase_curve.ci95_high,color=C['blue'],alpha=.25,lw=0)
    a.axhline(0,color=C['grey'],ls='--',lw=.75);a.axvline(91.3,color=C['grey'],ls=':',lw=.7)
    a.set(xlim=(0,182.5),xticks=[0,45,91,137,182],xlabel='Annual phase separation (days)',ylabel='Adjusted mask IoU change')
    a.text(.97,.06,f'{int(pairwise.sand_cay_id.nunique())} cays; {len(pairwise):,} pairs\nclustered 95% CI',transform=a.transAxes,ha='right',va='bottom',fontsize=6.1,color=C['grey']);clean(a);tag(a,'a')

    y=np.arange(len(harmonic_visual))[::-1]; cols=np.where(harmonic_visual.fdr_q_value.lt(.05),C['teal'],C['light'])
    b.barh(y,harmonic_visual.mean_relative_error_reduction,color=cols,height=.64,edgecolor='none')
    b.errorbar(harmonic_visual.mean_relative_error_reduction,y,xerr=np.vstack([harmonic_visual.mean_relative_error_reduction-harmonic_visual.ci95_low,harmonic_visual.ci95_high-harmonic_visual.mean_relative_error_reduction]),fmt='none',ecolor=C['dark'],capsize=2,lw=.8)
    b.axvline(0,color=C['grey'],ls='--',lw=.75);b.set(yticks=y,yticklabels=harmonic_visual.label,xlabel='LOYO error reduction (%)')
    b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1,decimals=0));b.tick_params(axis='y',labelsize=6.2);clean(b);tag(b,'b')

    cases=[record for region in atlas.region for record in collected[region]]
    case_table=pd.DataFrame(cases)
    case_table.drop(columns=['mask_path'],errors='ignore').to_csv(SRC/'figure_2_time_coloured_outline_cases.csv',index=False,encoding='utf-8-sig')
    case_table['date_parsed']=pd.to_datetime(case_table['date'])
    date_norm=mpl.colors.Normalize(vmin=mdates.date2num(case_table.date_parsed.min()),vmax=mdates.date2num(case_table.date_parsed.max()))
    time_cmap=mpl.colormaps['viridis']
    panel_side=.323; panel_w=.2425; panel_xs=[.13,.4375,.745]; panel_y=.115
    for column,(region,group) in enumerate(case_table.groupby('region',sort=False)):
        rendered=[]
        for record in group.sort_values('rank').itertuples():
            mask=read_mask(record.mask_path)
            rendered.append((record,mask))
        shapes=pd.Series([mask.shape for _,mask in rendered]).value_counts()
        target_shape=shapes.index[0]
        rendered=[(record,mask) for record,mask in rendered if mask.shape==target_shape]
        union=np.logical_or.reduce([mask for _,mask in rendered])
        y0,y1,x0,x1=crop_to_mask(union,padding_fraction=.08,minimum_padding=10)
        frame_height,frame_width=rendered[0][1].shape
        side=min(max(y1-y0,x1-x0),frame_height,frame_width)
        center_y=(y0+y1)/2;center_x=(x0+x1)/2
        y0=int(round(center_y-side/2));y0=max(0,min(y0,frame_height-side));y1=y0+side
        x0=int(round(center_x-side/2));x0=max(0,min(x0,frame_width-side));x1=x0+side
        overlay=fig.add_axes([panel_xs[column],panel_y,panel_w,panel_side])
        for record,mask in rendered:
            colour=time_cmap(date_norm(mdates.date2num(record.date_parsed)))
            crop=mask[y0:y1,x0:x1]
            rgba=np.zeros(crop.shape+(4,),dtype=float)
            rgba[...,0],rgba[...,1],rgba[...,2]=colour[0],colour[1],colour[2]
            rgba[...,3]=np.where(crop,.08,0.0)
            overlay.imshow(rgba,interpolation='nearest',zorder=1)
            overlay.contour(crop,levels=[.5],colors=[colour],linewidths=1.05,zorder=2)
            overlay.scatter(record.centroid_x-x0,record.centroid_y-y0,s=16,color=colour,edgecolor='white',linewidth=.35,zorder=4)
        for (earlier,_),(later,_) in zip(rendered[:-1],rendered[1:]):
            overlay.annotate('',xy=(later.centroid_x-x0,later.centroid_y-y0),xytext=(earlier.centroid_x-x0,earlier.centroid_y-y0),arrowprops=dict(arrowstyle='-|>',color=C['dark'],lw=.75,alpha=.72,shrinkA=4,shrinkB=5),zorder=3)
        ps=float(group.pixel_size_m.iloc[0]); side_px=float(side)
        bar_m=min([100,200,300,500,1000,2000], key=lambda c: abs(np.log(c/max(1.0,0.30*side_px*ps))))
        bar_px=bar_m/ps; ybar=side_px*0.955; x1b=side_px*0.96; x0b=x1b-bar_px
        overlay.plot([x0b,x1b],[ybar,ybar],color=C['dark'],lw=1.3,solid_capstyle='butt',zorder=6)
        overlay.plot([x0b,x0b],[ybar-side_px*0.012,ybar+side_px*0.012],color=C['dark'],lw=1.0,zorder=6)
        overlay.plot([x1b,x1b],[ybar-side_px*0.012,ybar+side_px*0.012],color=C['dark'],lw=1.0,zorder=6)
        overlay.text((x0b+x1b)/2,ybar-side_px*0.025,f'{bar_m} m',ha='center',va='bottom',fontsize=5.6,color=C['dark'],zorder=6)
        first_date=rendered[0][0].date_parsed.strftime('%Y-%m-%d')
        last_date=rendered[-1][0].date_parsed.strftime('%Y-%m-%d')
        overlay.text(.03,.97,region,transform=overlay.transAxes,ha='left',va='top',fontsize=7.2,weight='bold',color=C['dark'])
        overlay.text(.97,.97,f'n={len(rendered)}',transform=overlay.transAxes,ha='right',va='top',fontsize=5.8,color=C['grey'])
        overlay.text(.03,.03,f'{first_date} → {last_date}',transform=overlay.transAxes,ha='left',va='bottom',fontsize=5.6,color=C['grey'])
        overlay.set_aspect('equal');overlay.set_xticks([]);overlay.set_yticks([])
        for spine in overlay.spines.values(): spine.set_color(C['light']);spine.set_linewidth(.55)
        if column==0:
            tag(overlay,'c')
    colour_axis=fig.add_axes([.43,.045,.26,.012])
    bar=fig.colorbar(mpl.cm.ScalarMappable(norm=date_norm,cmap=time_cmap),cax=colour_axis,orientation='horizontal')
    tick_years=list(range(int(case_table.date_parsed.dt.year.min()),int(case_table.date_parsed.dt.year.max())+1,2))
    bar.set_ticks([mdates.date2num(pd.Timestamp(year,1,1)) for year in tick_years]);bar.set_ticklabels([str(year) for year in tick_years])
    bar.ax.tick_params(labelsize=6,length=2,width=.6);bar.outline.set_linewidth(.6)
    colour_axis.text(1.03,.5,'Observation date',transform=colour_axis.transAxes,va='center',fontsize=6.2,color=C['dark'])
    save(fig,'figure_2_seasonal_recurrence');plt.close(fig)
    return {'pairwise_cays':int(pairwise.sand_cay_id.nunique()),'pairwise_masks':int(len(pairwise)),'harmonic_cays':int(harmonic.sand_cay_id.nunique()),'atlas_cays':atlas.sand_cay_id.tolist()}

def fig3_directional():
    m=pd.read_csv(OUT/'强迫植被沙洲内模型.csv'); spec=[('along_shift','current_along_mean_z','Along current → along shift'),('along_shift','wave_vector_along_mean_z','Along wave → along shift'),('cross_shift','current_cross_mean_z','Cross current → cross shift'),('cross_shift','wave_vector_cross_mean_z','Cross wave → cross shift')]
    rows=[]
    for outcome,term,label in spec:
        r=m[(m.specification.eq('within_cay_fixed_effects'))&(m.outcome.eq(outcome))&(m.term.eq(term))].iloc[0].copy();r['label']=label;rows.append(r)
    main=pd.DataFrame(rows);main.to_csv(SRC/'figure_3_directional_models.csv',index=False,encoding='utf-8-sig')
    core=pd.read_csv(OUT/'流场波浪沿轴横轴分解.csv'); bg=core[core.sensor.eq('sentinel2')&core.time_interval_days.between(90,365)&core[['typhoon_strong_count','typhoon_r34_count']].fillna(0).sum(axis=1).eq(0)][['current_along_mean','major_axis_change_m_per_year']].dropna().copy();bg['current_rank']=bg.current_along_mean.rank(pct=True);bg['axis_rank']=bg.major_axis_change_m_per_year.rank(pct=True);bg.to_csv(SRC/'figure_3_long_axis_background.csv',index=False,encoding='utf-8-sig')
    rs=pd.read_csv(OUT/'流场波浪背景区间方向秩相关.csv'); long=rs[(rs.sensor.eq('sentinel2'))&(rs.driver.eq('current_along_mean'))&(rs.outcome.eq('major_axis_change_m_per_year'))].iloc[0];rho,_=stats.spearmanr(bg.current_along_mean,bg.major_axis_change_m_per_year)
    reef=m[m.specification.eq('within_cay_reef_cluster_sensitivity')]
    windows=pd.read_csv(OUT/'方向模型时间窗敏感性.csv')
    perturb=pd.read_csv(OUT/'方向模型掩膜扰动敏感性.csv')
    leave=pd.read_csv(OUT/'方向模型逐沙洲剔除敏感性汇总.csv')
    robust_rows=[]
    settings=[('Primary',main),('Reef cluster',reef),('60–365 d',windows[windows.window_days.eq('60-365')]),('120–365 d',windows[windows.window_days.eq('120-365')]),('−1 pixel',perturb[perturb.perturbation_pixels.eq(-1)]),('+1 pixel',perturb[perturb.perturbation_pixels.eq(1)])]
    for setting,frame in settings:
        for outcome,term,label in spec:
            found=frame[(frame.outcome.eq(outcome))&(frame.term.eq(term))]
            if not found.empty:
                r=found.iloc[0];robust_rows.append({'setting':setting,'outcome':outcome,'term':term,'label':label,'coefficient':r.coefficient,'ci95_low':r.ci95_low,'ci95_high':r.ci95_high,'direction_retained':bool(np.sign(r.coefficient)==np.sign(main[(main.outcome.eq(outcome))&(main.term.eq(term))].coefficient.iloc[0]))})
    for outcome,term,label in spec:
        found=leave[(leave.outcome.eq(outcome))&(leave.term.eq(term))]
        if not found.empty:
            r=found.iloc[0];robust_rows.append({'setting':'Leave-one-cay','outcome':outcome,'term':term,'label':label,'coefficient':np.nan,'ci95_low':np.nan,'ci95_high':np.nan,'direction_retained':bool(r.all_same_sign_as_full)})
    robustness=pd.DataFrame(robust_rows);robustness.to_csv(SRC/'figure_3_s2_robustness.csv',index=False,encoding='utf-8-sig')
    fig=plt.figure(figsize=(7.2,6.3)); gs=fig.add_gridspec(3,2,width_ratios=[1.0,1.05],height_ratios=[1.05,1.35,1.05],wspace=.42,hspace=.42)
    a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,0]); d=fig.add_subplot(gs[1,1]); e=fig.add_subplot(gs[2,:])
    ang=28.; ux,uy=np.cos(np.deg2rad(ang)),np.sin(np.deg2rad(ang)); vx,vy=-uy,ux; PURPLE='#8E7CC3'
    a.add_patch(mpl.patches.Ellipse((0,0),2.3,1.15,angle=ang,facecolor='#EFE6D4',edgecolor='#8A6D3B',lw=1.2))
    a.plot([-1.6*ux,1.6*ux],[-1.6*uy,1.6*uy],color=C['grey'],lw=.8); a.plot([-1.5*vx,1.5*vx],[-1.5*vy,1.5*vy],color=C['grey'],lw=.8,ls=':')
    a.annotate('',xy=(.95*ux,.95*uy),xytext=(-.55*ux,-.55*uy),arrowprops=dict(arrowstyle='-|>',color=C['navy'],lw=1.6)); a.text(-.60*ux-.05,-.60*uy-.16,'Current',color=C['navy'],fontsize=6.6,ha='right')
    a.annotate('',xy=(.50*ux+.30*vx,.50*uy+.30*vy),xytext=(-.10*ux-.75*vx,-.10*uy-.75*vy),arrowprops=dict(arrowstyle='-|>',color=C['teal'],lw=1.4)); a.text(-.10*ux-.95*vx,-.10*uy-1.0*vy,'Waves',color=C['teal'],fontsize=6.6)
    a.annotate('',xy=(.62*ux,.62*uy),xytext=(.20*ux,.20*uy),arrowprops=dict(arrowstyle='-|>',color=C['navy'],lw=.9,ls='--',alpha=.8))
    a.annotate('',xy=(.30*vx,.30*vy),xytext=(.10*vx,.10*vy),arrowprops=dict(arrowstyle='-|>',color=C['teal'],lw=.9,ls='--',alpha=.8))
    a.text(1.35*ux,1.25*uy,'Major axis (along)',fontsize=6.3,color=C['dark']); a.text(1.35*vx,1.45*vy,'Cross axis',fontsize=6.3,color=C['grey'])
    a.text(-2.3,-1.62,'Forcing averaged over (t, t1] and projected on the cay axes',ha='left',fontsize=6.0,color=C['grey'])
    a.set(xlim=(-2.35,2.6),ylim=(-1.75,1.65)); a.set_aspect('equal'); a.axis('off'); tag(a,'a')
    b.add_patch(mpl.patches.Ellipse((-.45,-.28),2.0,1.0,angle=ang,facecolor='none',edgecolor=C['light'],lw=1.3,ls='--'))
    b.add_patch(mpl.patches.Ellipse((0,0),2.0,1.0,angle=ang,facecolor='#EFE6D4',edgecolor='#8A6D3B',lw=1.2))
    b.plot([-1.6*ux,1.6*ux],[-1.6*uy,1.6*uy],color=C['grey'],lw=.8); b.plot([-1.5*vx,1.5*vx],[-1.5*vy,1.5*vy],color=C['grey'],lw=.8,ls=':')
    b.scatter([-.45],[-.28],s=16,color=C['grey'],zorder=3); b.scatter([0],[0],s=16,color=C['navy'],zorder=3)
    b.annotate('',xy=(-.02,-.01),xytext=(-.43,-.27),arrowprops=dict(arrowstyle='-|>',color=C['navy'],lw=1.5)); b.text(0.10,-1.30,'Along-axis shift',color=C['navy'],fontsize=6.6)
    b.annotate('',xy=(.30*vx+.02,.30*vy+.02),xytext=(.06*vx,.06*vy),arrowprops=dict(arrowstyle='-|>',color=PURPLE,lw=1.4)); b.text(0.15,0.95,'Cross-axis shift',color=PURPLE,fontsize=6.6)
    b.text(1.20,0.15,'outline at t1',color='#8A6D3B',fontsize=6.2); b.text(-2.30,-1.30,'outline at t',color=C['grey'],fontsize=6.2)
    b.set(xlim=(-2.35,2.6),ylim=(-1.75,1.65)); b.set_aspect('equal'); b.axis('off'); tag(b,'b')
    y=np.arange(len(main))[::-1];c.errorbar(main.coefficient,y,xerr=np.vstack([main.coefficient-main.ci95_low,main.ci95_high-main.coefficient]),fmt='none',color=C['grey'],capsize=2,lw=1);c.scatter(main.coefficient,y,s=44,color=np.where(main.fdr_q_value.lt(.05),C['navy'],C['light']),edgecolor='white',lw=.4,zorder=3)
    for yy,r in zip(y,main.itertuples()):c.text(r.ci95_high+.04,yy,f'q={r.fdr_q_value:.3g}',va='center',fontsize=6,color=C['dark'])
    c.axvline(0,color=C['grey'],ls='--',lw=.8);c.set(yticks=y,yticklabels=['Along current','Along wave','Cross current','Cross wave'],xlabel='Standardized coefficient (95% CI)');clean(c);tag(c,'c')
    hb=d.hexbin(bg.current_rank,bg.axis_rank,gridsize=18,mincnt=1,cmap=mpl.colors.LinearSegmentedColormap.from_list('density',['#EEF5F8',C['blue'],C['navy']]),linewidths=0);qb=pd.qcut(bg.current_rank,9,duplicates='drop');bi=bg.groupby(qb,observed=True).agg(x=('current_rank','median'),y=('axis_rank','median')).reset_index(drop=True);d.plot(bi.x,bi.y,color=C['orange'],lw=1.6);d.text(.04,.96,f'n={len(bg)}\nρ={rho:.3f}; FDR q={long.fdr_q_value:.3f}',transform=d.transAxes,va='top',fontsize=6.4,color=C['navy']);d.set(xlabel='Rank-normalized along-axis current',ylabel='Rank-normalized major-axis change');clean(d);tag(d,'d',1.12)
    grid=robustness.pivot(index='label',columns='setting',values='direction_retained').reindex(index=main.label,columns=['Primary','Reef cluster','60–365 d','120–365 d','−1 pixel','+1 pixel','Leave-one-cay'])
    retained=grid.fillna(False).astype(bool); primary_supported=main.set_index('label').fdr_q_value.lt(.05)
    display=np.zeros(retained.shape,dtype=int)
    for i,label in enumerate(retained.index):
        display[i,retained.iloc[i].to_numpy()]=2 if bool(primary_supported.loc[label]) else 1
    e.imshow(display,cmap=mpl.colors.ListedColormap([C['red'],C['light'],C['teal']]),vmin=0,vmax=2,aspect='auto')
    e.set(xticks=np.arange(len(grid.columns)),xticklabels=grid.columns,yticks=np.arange(len(grid.index)),yticklabels=['Along current','Along wave','Cross current','Cross wave'])
    e.tick_params(axis='x',rotation=45,labelsize=6);[e.text(j,i,'S' if display[i,j]==2 else ('D' if display[i,j]==1 else 'F'),ha='center',va='center',color='white' if display[i,j]==2 else C['dark'],fontsize=7,weight='bold') for i in range(len(grid.index)) for j in range(len(grid.columns))];clean(e);tag(e,'e')
    fig.subplots_adjust(left=.13,right=.99,bottom=.14,top=.96);save(fig,'figure_3_directional_coupling');plt.close(fig)
    return {'within_cay_intervals':int(main.n_intervals.iloc[0]),'long_axis_background_intervals':len(bg),'s2_robustness_checks':int(len(robustness))}



def fig5_surface_state_mobility():
    """Main Figure 5: vegetation-associated mobility and reversible surface states."""
    cays = pd.read_csv(SRC/'figure5_vegetation_stability_cays.csv')
    s2c = cays[cays.sensor.eq('sentinel2')].copy()
    corr = pd.read_csv(OUT/'植被稳定性轨迹秩相关.csv')
    sector = pd.read_csv(OUT/'植被扇区对照统计.csv')
    sector = sector[sector.stratum.isin(['all','background','window_90_540','window_120_730'])].copy()
    sector['label'] = sector.stratum.map({'all':'All intervals','background':'Background','window_90_540':'90–540 d','window_120_730':'120–730 d'})
    est = pd.read_csv(OUT/'Sentinel2三状态连续时间模型估计.csv')
    states = est[(est.variant.eq('primary_30_365d')) & (est.record_type.eq('state_summary'))].copy()
    states['label'] = states.from_state.map({'low_cover':'Low cover','partial_cover':'Partial cover','dominant_cover':'Dominant cover'})
    transitions = pd.read_csv(OUT/'Sentinel2三状态连续时间观测转移计数.csv')
    transitions = transitions[transitions.from_state.ne(transitions.to_state)].copy()

    s2c.to_csv(SRC/'figure_5_vegetation_trajectory_cays.csv', index=False, encoding='utf-8-sig')
    corr[corr.sensor.eq('sentinel2')].to_csv(SRC/'figure_5_vegetation_trajectory_correlations.csv', index=False, encoding='utf-8-sig')
    sector.to_csv(SRC/'figure_5_within_image_sector_contrast.csv', index=False, encoding='utf-8-sig')
    states.to_csv(SRC/'figure_5_conditional_holding_times.csv', index=False, encoding='utf-8-sig')
    transitions.to_csv(SRC/'figure_5_observed_state_transitions.csv', index=False, encoding='utf-8-sig')

    colour_all = pd.read_csv(OUT/'沙洲颜色组成.csv')
    colour_all = colour_all[(colour_all.sensor.eq('google_earth')) & colour_all.status.eq('ok') & colour_all.quality_grade.isin(['A','B'])]
    colour_all = colour_all[colour_all.image_path.map(lambda path: Path(path).is_file()) & colour_all.mask_path.map(lambda path: Path(path).is_file())]
    examples_spec = [
        ('MEDF_053_cay_001','2019-02-14','Low proxy'),
        ('MEDF_040_cay_001','2011-02-22','Moderate proxy'),
        ('GBR_095_cay_001','2025-07-14','High proxy'),
    ]
    examples = []
    for cay_id, date, label in examples_spec:
        subset = colour_all[colour_all.sand_cay_id.eq(cay_id) & colour_all.date.eq(date)]
        if subset.empty:
            raise ValueError(f'No Google Earth colour-proxy observation for {cay_id} on {date}')
        record = subset.iloc[0].copy()
        record['example_label'] = label
        examples.append(record)
    examples = pd.DataFrame(examples)
    examples[['image_id','sand_cay_id','reef_id','date','automatic_vegetation_state','vegetation_fraction','example_label']].to_csv(
        SRC/'figure_5_colour_proxy_examples.csv', index=False, encoding='utf-8-sig'
    )

    fig = plt.figure(figsize=(7.2, 7.35))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0, .86], wspace=.38, hspace=.42)
    a, b, c, d = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))

    summary_rows = []
    for sensor in ('sentinel2', 'google_earth'):
        summary = binned_mobility_summary(cays[cays.sensor.eq(sensor)])
        summary['sensor'] = sensor
        summary_rows.append(summary)
    binned = pd.concat(summary_rows, ignore_index=True)
    binned.to_csv(SRC/'figure_5_vegetation_binned_trajectories.csv', index=False, encoding='utf-8-sig')
    correlations = corr[corr.outcome.eq('centroid_path_normalized')]
    for sensor, colour, label, linestyle in [
        ('sentinel2', C['teal'], 'Sentinel-2', '-'),
        ('google_earth', C['orange'], 'Google Earth', '--'),
    ]:
        sub = binned[binned.sensor.eq(sensor)]
        a.fill_between(sub.vegetation_fraction_median, sub.mobility_q25, sub.mobility_q75, color=colour, alpha=.13, zorder=1)
        a.plot(sub.vegetation_fraction_median, sub.mobility_median, color=colour, lw=1.9, ls=linestyle, label=label, zorder=3)
    a.set_yscale('log')
    a.set(xlabel='Median vegetation proxy fraction', ylabel='Cumulative centroid path (radii)')
    annotation = []
    for sensor, label in [('sentinel2','S2'), ('google_earth','GE')]:
        row = correlations[correlations.sensor.eq(sensor)].iloc[0]
        annotation.append(f'{label}: n={int((cays.sensor == sensor).sum())}, ρ={row.spearman_rho:.3f}, q={row.fdr_q_value:.3g}')
    a.text(.04, .05, '\n'.join(annotation), transform=a.transAxes, va='bottom', fontsize=5.6, color=C['dark'])
    a.legend(frameon=False, fontsize=5.8, loc='upper right')
    clean(a); tag(a, 'a'); a.set_title('Vegetation-associated mobility gradient', loc='left', fontsize=8.4)

    sector = sector.set_index('stratum').loc[['all','background','window_90_540','window_120_730']].reset_index()
    y = np.arange(len(sector))[::-1]
    b.errorbar(sector.mean_mobility_difference, y, xerr=np.vstack([sector.mean_mobility_difference-sector.bootstrap_ci_low, sector.bootstrap_ci_high-sector.mean_mobility_difference]), fmt='none', color=C['grey'], lw=1, capsize=2)
    b.scatter(sector.mean_mobility_difference, y, s=40, color=C['teal'], edgecolor='white', lw=.4, zorder=3)
    b.axvline(0, color=C['grey'], ls='--', lw=.8)
    b.set(yticks=y, yticklabels=sector.label, xlabel='Vegetated minus bare mobility')
    clean(b); tag(b, 'b'); b.set_title('Within-image surface contrast', loc='left', fontsize=8.4)

    pos = {'low_cover':(.14,.57), 'partial_cover':(.50,.57), 'dominant_cover':(.86,.57)}
    labels = {'low_cover':'Low', 'partial_cover':'Partial', 'dominant_cover':'Dominant'}
    years = states.set_index('from_state').mean_holding_years.to_dict()
    for state, (x, yy) in pos.items():
        c.scatter(x, yy, s=790, color=C['teal'], edgecolor='white', lw=1.0, zorder=3)
        c.text(x, yy, f"{labels[state]}\n{years[state]:.2f} y", ha='center', va='center', fontsize=5.7, color='white', weight='bold')
    for left, right in [('low_cover','partial_cover'), ('partial_cover','dominant_cover')]:
        forward = int(transitions[(transitions.from_state.eq(left)) & (transitions.to_state.eq(right))].n_pairs.iloc[0])
        reverse = int(transitions[(transitions.from_state.eq(right)) & (transitions.to_state.eq(left))].n_pairs.iloc[0])
        x1, y1 = pos[left]; x2, y2 = pos[right]
        c.annotate('', xy=(x2-.08,y2+.04), xytext=(x1+.08,y1+.04), arrowprops=dict(arrowstyle='->', color=C['navy'], lw=1.1, connectionstyle='arc3,rad=.24'))
        c.annotate('', xy=(x1+.08,y1-.04), xytext=(x2-.08,y2-.04), arrowprops=dict(arrowstyle='->', color=C['grey'], lw=1.1, connectionstyle='arc3,rad=.24'))
        c.text((x1+x2)/2, .75, f'{forward}/{reverse}', ha='center', fontsize=6.8, color=C['dark'])
    c.text(.50, .16, 'labels are forward/reverse observed transitions', ha='center', fontsize=5.4, color=C['grey'])
    c.set(xlim=(0,1), ylim=(0,1)); c.axis('off'); tag(c, 'c'); c.set_title('Reversible surface states', loc='left', fontsize=8.4)

    order = ['Low cover','Partial cover','Dominant cover']
    st = states.set_index('label').loc[order].reset_index()
    y = np.arange(3)[::-1]
    d.errorbar(st.mean_holding_years, y, xerr=np.vstack([st.mean_holding_years-st.ci_low, st.ci_high-st.mean_holding_years]), fmt='none', color=C['dark'], lw=1, capsize=2)
    d.scatter(st.mean_holding_years, y, s=42, color=C['navy'], edgecolor='white', lw=.4, zorder=3)
    d.set(yticks=y, yticklabels=order, xlabel='Model-estimated holding time (years)')
    clean(d); tag(d, 'd'); d.set_title('Observed-state duration', loc='left', fontsize=8.4)

    example_grid = gs[2, :].subgridspec(1, 3, wspace=.06)
    for index, record in enumerate(examples.itertuples()):
        ax = fig.add_subplot(example_grid[0, index])
        image_bgr = read_raster(record.image_path, cv2.IMREAD_COLOR)
        cay_mask = read_mask(record.mask_path)
        if image_bgr.shape[:2] != cay_mask.shape:
            raise ValueError(f'Image-mask mismatch for {record.image_id}')
        proxy = colour_proxy_pixels(image_bgr, cay_mask)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_crop = cay_mask > 0
        ys, xs = np.nonzero(mask_crop)
        himg, wimg = rgb.shape[:2]
        span = max(ys.max()-ys.min()+1, (xs.max()-xs.min()+1)/1.05) * 2.4
        ch = min(span, himg)
        cw = min(ch*1.05, wimg)
        if cw < ch*1.05:
            ch = cw/1.05
        ccy, ccx = (ys.min()+ys.max())/2, (xs.min()+xs.max())/2
        y0 = int(min(max(ccy-ch/2, 0), himg-ch))
        x0 = int(min(max(ccx-cw/2, 0), wimg-cw))
        y1, x1 = int(y0+ch), int(x0+cw)
        rgb = rgb[y0:y1, x0:x1]
        mask_crop = mask_crop[y0:y1, x0:x1]
        proxy_crop = proxy[y0:y1, x0:x1]
        ax.imshow(rgb)
        ax.contour(mask_crop, levels=[.5], colors=['white'], linewidths=.85)
        overlay = np.zeros((*proxy_crop.shape, 4), dtype=float)
        overlay[proxy_crop] = [.17, .62, .56, .75]
        ax.imshow(overlay)
        ax.set_title(record.example_label, fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index == 0:
            tag(ax, 'e')

    fig.subplots_adjust(left=.11, right=.99, bottom=.035, top=.96)
    save(fig, 'figure_5_surface_state_mobility')
    plt.close(fig)
    return {'s2_trajectory_cays':len(s2c), 'ge_trajectory_cays':int(cays.sensor.eq('google_earth').sum()), 'sector_pairs':int(sector.loc[sector.stratum.eq('all'), 'n_transitions'].iloc[0]), 'state_pairs':int(states.n_pairs.iloc[0]), 'state_changes':int(transitions.n_pairs.sum())}


def fig_s1_typhoon():
    au=json.loads((OUT/'台风事件前后分析核查.json').read_text(encoding='utf-8'));pf=au['pairing_funnel'];p=pd.read_csv(OUT/'台风事件前后主分析样本.csv').sort_values('gross_mobility_fraction',ascending=False).copy();p['pair_number']=np.arange(1,len(p)+1);p.to_csv(SRC/'figure_4_strict_event_pairs.csv',index=False,encoding='utf-8-sig');f=pd.DataFrame({'stage':['Direct events','Within image span','Pre-event image','Post-event image','Same frame','Isolated pairs'],'n':[pf['direct_event_cay_records'],pf['event_cay_records_inside_observation_span'],pf['event_cay_records_with_pre_image'],pf['event_cay_records_with_post_image'],pf['event_cay_records_with_same_frame_pair'],pf['isolated_same_frame_pairs']]});f.to_csv(SRC/'figure_4_event_pairing_funnel.csv',index=False,encoding='utf-8-sig')
    fig=plt.figure(figsize=(7.2,4.2));gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.18,1.28],hspace=.45,wspace=.48)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);cc=fig.add_subplot(gs[0,2]);dd=fig.add_subplot(gs[1,0:2]);ee=fig.add_subplot(gs[1,2])
    y=np.arange(len(f))[::-1];a.hlines(y,1,f.n,color=C['light'],lw=4);a.scatter(f.n,y,s=60,color=[C['grey']]*4+[C['orange'],C['teal']]);a.set_xscale('log');a.set(yticks=y,yticklabels=f.stage,xlabel='Event-cay records (log scale)');[a.text(v*1.22,yy,f'{int(v):,}',va='center',fontsize=6.7) for yy,v in zip(y,f.n)];clean(a)
    b.scatter(p.net_area_fraction*100,p.gross_mobility_fraction*100,s=58,c=plt.cm.tab20(pd.Categorical(p.sid).codes),edgecolor='white',lw=.45);b.axvline(0,color=C['grey'],ls='--',lw=.8);b.set(xlabel='Net area change (%)',ylabel='Gross boundary reworking (%)');clean(b)
    x=p.pair_number.to_numpy();e=p.erosion_fraction.to_numpy()*100;de=p.deposition_fraction.to_numpy()*100;net=p.net_area_fraction.to_numpy()*100;cc.bar(x,e,width=.64,color=C['red'],label='Erosion');cc.bar(x,de,width=.64,bottom=e,color=C['teal'],label='Deposition');cc.plot(x,net,'o',color=C['dark'],ms=3.7,label='Net area change',zorder=4);cc.axhline(0,color=C['grey'],lw=.75);cc.set(xticks=x,xlabel='Strict event pair, ordered by reworking',ylabel='Area fraction (%)');cc.legend(frameon=False,loc='upper right',fontsize=6.5);clean(cc)
    cmpdf=pd.read_csv(OUT/'背景与台风脉冲对比.csv');row=cmpdf[cmpdf.sensor.eq('sentinel2')&cmpdf.outcome.eq('gross_mobility_fraction_per_year')].iloc[0]
    dec=pd.read_csv(OUT/'流场波浪沿轴横轴分解.csv')[['transition_id','current_along_mean','wave_hs_p90']]
    samp=pd.read_csv(OUT/'已成洲沙洲定量分析样本.csv');samp=samp[samp.sensor.eq('sentinel2')].merge(dec,on='transition_id',how='inner');samp=samp[samp[['current_along_mean','wave_hs_p90']].notna().all(axis=1)].copy()
    samp['pulse']=(samp[['typhoon_strong_count','typhoon_r34_count']].fillna(0).sum(axis=1)>0)
    data=[samp.loc[~samp.pulse,'gross_mobility_fraction_per_year'].to_numpy(),samp.loc[samp.pulse,'gross_mobility_fraction_per_year'].to_numpy()]
    bp=dd.boxplot(list(data),widths=.5,patch_artist=True,boxprops=dict(lw=.8),medianprops=dict(color=C['dark']),showfliers=False)
    for patch,col in zip(bp['boxes'],[C['blue'],C['orange']]):patch.set_facecolor(col);patch.set_alpha(.55)
    rng=np.random.default_rng(7)
    for i,vals in enumerate(data):dd.scatter(np.full(len(vals),i+1)+rng.uniform(-.14,.14,len(vals)),vals,s=7,color=[C['blue'],C['orange']][i],alpha=.45,linewidth=0)
    dd.set_yscale('log');dd.set(xticks=[1,2],xticklabels=[f'Background\nn={len(data[0])}',f'Typhoon pulse\nn={len(data[1])}'],ylabel='Gross reworking per year (log)')
    dd.text(.98,.96,f'medians {row.median_background:.2f} vs {row.median_pulse:.2f}\nMann-Whitney p={row.mannwhitney_p_value:.3f}; FDR q={row.fdr_q_value:.3f}',transform=dd.transAxes,ha='right',va='top',fontsize=6.2,color=C['dark']);clean(dd);tag(dd,'d')
    rec=pd.read_csv(OUT/'事件后恢复.csv');rec=rec[rec.sensor.eq('sentinel2')].copy();rec.to_csv(SRC/'figure_4_post_event_recovery.csv',index=False,encoding='utf-8-sig')
    ee.hist(rec.post_to_background_ratio,bins=12,color='#8E7CC3',edgecolor='white',lw=.4)
    ee.set(xlim=(-0.8,6.2));far=rec.post_to_background_ratio.gt(6)
    if int(far.sum()):ee.text(.40,.62,f'{int(far.sum())} event at {rec.post_to_background_ratio.max():.1f}x',transform=ee.transAxes,va='top',fontsize=6.2,color=C['dark'])
    ee.axvline(1,color=C['dark'],ls='--',lw=.9);ee.axvline(1.5,color=C['grey'],ls=':',lw=.9)
    ee.set(xlabel='Post-event mobility / cay background',ylabel='Pulse events')
    ee.text(.98,.96,f'n={len(rec)}, median={rec.post_to_background_ratio.median():.3f}\n{rec.post_to_background_ratio.le(1.5).mean()*100:.1f}% within 1.5x',transform=ee.transAxes,ha='right',va='top',fontsize=6.2,color=C['dark']);clean(ee);tag(ee,'e')
    for ax,label in ((a,'a'),(b,'b'),(cc,'c')):ax.text(-.18,1.15,label,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')
    fig.subplots_adjust(left=.10,right=.99,bottom=.10,top=.95);save(fig,'figure_4_event_scale_typhoon');plt.close(fig)
    return {'strict_pairs':len(p),'storms':int(p.sid.nunique()),'cays':int(p.sand_cay_id.nunique()),'pulse_intervals':int(len(data[1])),'recovery_events':int(len(rec))}

def fig_s2_enso():
    m=pd.read_csv(OUT/'ENSO滞后模型.csv');co=pd.read_csv(OUT/'ENSO共同面积趋势.csv');m.to_csv(SRC/'figure_5_enso_lag_models.csv',index=False,encoding='utf-8-sig');co.to_csv(SRC/'figure_5_common_area_trend.csv',index=False,encoding='utf-8-sig');m['label']=m.outcome.map({'annualized_log_area_change':'Net area change','log_gross_mobility':'Boundary reworking'});co['date']=pd.to_datetime(co['time']);fig,(a,b,cc)=plt.subplots(1,3,figsize=(7.2,2.75),gridspec_kw={'width_ratios':[1.15,1,1]})
    a.plot(co.date,co.common_area_change_percent,color=C['navy'],lw=1.5,label='Fixed-cohort common area trend');a.fill_between(co.date,co.ci95_low_percent,co.ci95_high_percent,color=C['light']);z=a.twinx();z.plot(co.date,co.nino34_anomaly_c,color=C['orange'],lw=1.1,label='Niño 3.4 anomaly');z.set_ylabel('Niño 3.4 anomaly (°C)',color=C['orange']);z.tick_params(axis='y',colors=C['orange']);a.xaxis.set_major_locator(mdates.YearLocator(2));a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));a.set(xlabel='Year',ylabel='Common area change (%)');a.legend(loc='upper left',frameon=False);z.legend(loc='lower left',frameon=False);clean(a)
    for ax,label in [(b,'Net area change'),(cc,'Boundary reworking')]:
        x=m[m.label.eq(label)].sort_values('lag_months');y=np.arange(len(x))[::-1];col=np.where(x.time_robust_supported,C['teal'],C['light']);ax.errorbar(x.coefficient_per_1c,y,xerr=np.vstack([x.coefficient_per_1c-x.two_way_ci95_low,x.two_way_ci95_high-x.coefficient_per_1c]),fmt='none',ecolor=C['grey'],lw=.9,capsize=2);ax.scatter(x.coefficient_per_1c,y,s=36,color=col,edgecolor=C['dark'],lw=.4,zorder=3)
        for yy,r in zip(y,x.itertuples()):
            if r.time_robust_supported:ax.text(r.two_way_ci95_high+.008,yy,f'q={r.two_way_cluster_fdr_q_value:.3f}',va='center',fontsize=6,color=C['teal'])
        ax.axvline(0,color=C['grey'],ls='--',lw=.8);ax.set(yticks=y,yticklabels=[str(int(v)) for v in x.lag_months],xlabel='Coefficient per 1 °C Niño 3.4');clean(ax)
    b.set_ylabel('Lag (months)')
    for ax,label in ((a,'a'),(b,'b'),(cc,'c')):ax.text(-.18,1.15,label,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')
    fig.subplots_adjust(left=.09,right=.995,bottom=.16,top=.88,wspace=.58);save(fig,'figure_5_enso_lagged_association');plt.close(fig);return {'area_intervals':1011,'mobility_intervals':987,'robust_lags_months':[0,3,24]}

def fig_s3_wind():
    w=pd.read_csv(OUT/'风场增量检验模型比较.csv');coverage=w[w.model.eq('coverage')].iloc[0];w=w[w.model.eq('increment')].copy();w['delta_adj_r2']=w.adj_r2_flow_wave_wind-w.adj_r2_flow_wave;w['outcome_label']=w.term.map({'log_gross_mobility':'Gross reworking','along_shift':'Along-axis shift','cross_shift':'Cross-axis shift'});w['scope_label']=w.analysis_scope.map({'all_intervals_pulse_adjusted':'All intervals\n(pulse adjusted)','no_typhoon_pulse_intervals':'No typhoon-pulse\nintervals'});w.to_csv(SRC/'figure_s1_wind_increment.csv',index=False,encoding='utf-8-sig');fig,(a,b,cc)=plt.subplots(1,3,figsize=(7.2,2.75),gridspec_kw={'width_ratios':[.78,1.05,1.05]})
    audit=json.loads((OUT/'流场波浪覆盖核查.json').read_text(encoding='utf-8'));wind_complete=int(audit['complete_flow_wave_wind_sentinel2_intervals']);with_typhoon=int(audit['wind_typhoon_day_partition']['wind_complete_sentinel2_intervals_with_local_typhoon_days']);without_typhoon=wind_complete-with_typhoon
    a.bar(0,without_typhoon,color=C['blue'],width=.55,label='No local typhoon days');a.bar(0,with_typhoon,bottom=without_typhoon,color=C['orange'],width=.55,label='Local typhoon days');a.text(0,without_typhoon/2,f'{without_typhoon}',color='white',ha='center',va='center',fontsize=8);a.text(.34,without_typhoon+with_typhoon/2,f'{with_typhoon}',color=C['orange'],ha='left',va='center',fontsize=6.7);a.text(0,wind_complete+18,f'{wind_complete} wind-complete S2 intervals',ha='center',fontsize=6.7);a.set(xlim=(-.7,.7),ylim=(0,max(850,wind_complete+100)),xticks=[],ylabel='Intervals');clean(a)
    order=['Gross reworking','Along-axis shift','Cross-axis shift'];pos=np.arange(3);styles={'All intervals\n(pulse adjusted)':(.16,'o',C['navy']),'No typhoon-pulse\nintervals':(-.16,'s',C['teal'])}
    for scope,(off,marker,col) in styles.items():
        x=w[w.scope_label.eq(scope)].set_index('outcome_label').loc[order];b.scatter(pos+off,x.joint_wind_cluster_fdr_q_value,marker=marker,s=34,color=col,edgecolor='white',lw=.45,label=scope.replace('\n',' '));[b.text(xx,v+.035,f'{v:.3f}',ha='left' if off>0 else 'right',fontsize=6.2,color=col) for xx,v in zip(pos+off,x.joint_wind_cluster_fdr_q_value)]
    b.axhline(.05,color=C['red'],ls='--',lw=.8);b.text(-.46,.058,'FDR threshold = 0.05',color=C['red'],fontsize=6);b.set(ylim=(0,.58),xticks=pos,xticklabels=['Gross\nreworking','Along-axis\nshift','Cross-axis\nshift'],ylabel='Joint wind test, FDR q');clean(b)
    for i,scope in enumerate(styles):x=w[w.scope_label.eq(scope)].set_index('outcome_label').loc[order];cc.bar(pos+(i-.5)*.30,x.delta_adj_r2,width=.30,color=styles[scope][2]);cc.axhline(0,color=C['grey'],lw=.75);cc.set(xticks=pos,xticklabels=['Gross\nreworking','Along-axis\nshift','Cross-axis\nshift'],ylabel='Change in adjusted R²');clean(cc)
    cc.title.set_fontsize(8.6)
    for ax,label in ((a,'a'),(b,'b'),(cc,'c')):ax.text(-.19,1.12,label,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')
    ha_,la_=a.get_legend_handles_labels();hb_,lb_=b.get_legend_handles_labels()
    fig.legend(ha_+hb_,la_+lb_,loc='lower center',bbox_to_anchor=(.5,.02),ncol=4,frameon=False,fontsize=6.4)
    fig.subplots_adjust(left=.10,right=.99,bottom=.24,top=.92,wspace=.48);save(fig,'figure_s1_wind_increment');plt.close(fig);return {'wind_complete_s2_intervals':wind_complete,'model_intervals':int(coverage.n_intervals),'model_cays':int(coverage.n_cays)}

def main():
    s={'figure_1':fig1(),'figure_2':fig2_seasonal(),'figure_3':fig3_directional(),'figure_5':fig5_surface_state_mobility(),'figure_s1':fig_s1_typhoon(),'figure_s2':fig_s2_enso(),'figure_s3':fig_s3_wind()}
    o={'analysis_contract_version':'2026-09-15-real-cay-axis-1','core_conclusion':'Net-area stability can coexist with persistent reworking. Sentinel-2 masks contain a repeatable annual geometric component in position and outline, while area and the colour-derived vegetation proxy do not show a repeatable annual predictive contribution. Along-axis current, along-axis wave and cross-axis wave are positively associated with the corresponding displacement in the real cay coordinate system. These are observational, tide-unadjusted associations and do not identify sediment flux or a unique physical cause.','figure_order':[{'figure':'figure_1_dynamic_equilibrium','manuscript_label':'Figure 1','role':'Long-term geomorphic phenomenon','evidence_level':'descriptive trajectory evidence'},{'figure':'figure_2_seasonal_recurrence','manuscript_label':'Figure 2','role':'Annual recurrence of tide-unadjusted observed masks','evidence_level':'within-cay observational recurrence with leave-one-year-out validation'},{'figure':'figure_3_directional_coupling','manuscript_label':'Figure 3','role':'Directional flow-wave association','evidence_level':'within-cay association with reported clustering, time-window, boundary and leave-one-cay sensitivities'},{'figure':'figure_4_vegetation_mobility','manuscript_label':'Figure 4','role':'Vegetation-mobility association','evidence_level':'between-cay, within-image and descriptive cross-sensor association'},{'figure':'figure_5_state_reversibility','manuscript_label':'Figure 5','role':'Reversible surface states','evidence_level':'model-estimated continuous-time holding times'},{'figure':'figure_4_event_scale_typhoon','manuscript_label':'Supplementary Figure S1','role':'Strict typhoon event cases','evidence_level':'case-level observational evidence'},{'figure':'figure_5_enso_lagged_association','manuscript_label':'Supplementary Figure S2','role':'Interannual lagged association','evidence_level':'time-sensitivity-supported observational association'},{'figure':'figure_s1_wind_increment','manuscript_label':'Supplementary Figure S3','role':'Wind increment sensitivity','evidence_level':'FDR-controlled model sensitivity'}],'labels':'English; Times New Roman','exports':['SVG editable text','PDF editable text','PNG preview','600 dpi LZW TIFF'],'summaries':s}
    (FIG/'figure_manifest.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)
    )
    print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
