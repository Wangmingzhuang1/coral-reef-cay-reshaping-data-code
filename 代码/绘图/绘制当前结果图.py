"""按当前合同绘制论文结果图。"""
from __future__ import annotations
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
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
ROOT = PROJECT_ROOT; OUT=ROOT/'outputs'; FIG=OUT/'figures'; SRC=FIG/'source_data'
ROOT = PROJECT_ROOT; OUT=ROOT/'outputs'; FIG=OUT/'figures'; SRC=FIG/'source_data'
ROOT = PROJECT_ROOT; OUT=ROOT/'outputs'; FIG=OUT/'figures'; SRC=FIG/'source_data'
ROOT = PROJECT_ROOT; OUT=ROOT/'outputs'; FIG=OUT/'figures'; SRC=FIG/'source_data'
FIG.mkdir(exist_ok=True); SRC.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True); SRC.mkdir(exist_ok=True)
C={'navy':'#1F4E6D','blue':'#5AA6D1','teal':'#2B9D8F','orange':'#ED8B27','red':'#C44E52','grey':'#77818B','light':'#D9E1E8','dark':'#263640'}
mpl.rcParams.update({'font.family':'Times New Roman','font.size':7.2,'axes.labelsize':7.5,'axes.titlesize':9.4,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.75,'figure.dpi':180})
def clean(ax): ax.spines[['top','right']].set_visible(False); ax.tick_params(width=.7,length=3)


def tag(ax,s,y=1.02): ax.text(-.14,y,s,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')


def save(fig,name):
    for ext,kw in [('svg',{}),('pdf',{}),('png',{'dpi':360}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}})]: fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',**kw)


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


def fig2_seasonal():
    """以曲线、条带和掩膜图组呈现年度重复性，避免把该证据压缩为点图。"""
    pairwise=pd.read_csv(OUT/'Sentinel2季节形态配对.csv')
    grids=pd.read_csv(OUT/'审稿问题复核/Sentinel2原始栅格网格核验.csv')
    grid_ids=set(grids.loc[grids.n_grids.eq(1),'sand_cay_id'])
    pairwise=pairwise[pairwise.sand_cay_id.isin(grid_ids)].copy()
    design=pairwise[['phase_distance','gap_years']].copy()
    design['gap_years_squared']=design.gap_years**2
    design=pd.concat([design,pd.get_dummies(pairwise['sand_cay_id'],drop_first=True,dtype=float)],axis=1)
    model=sm.OLS(pairwise['iou'],sm.add_constant(design)).fit(cov_type='cluster',cov_kwds={'groups':pairwise['sand_cay_id']})
    beta=float(model.params['phase_distance']); se=float(model.bse['phase_distance'])
    phase_days=np.linspace(0,182.5,121); phase_distance=1-np.cos(2*np.pi*phase_days/365.2425)
    phase_curve=pd.DataFrame({'annual_phase_separation_days':phase_days,'phase_distance':phase_distance,'adjusted_iou_change_from_same_phase':beta*phase_distance,'ci95_low':(beta-1.96*se)*phase_distance,'ci95_high':(beta+1.96*se)*phase_distance})
    phase_curve.to_csv(SRC/'figure_2_seasonal_phase_curve.csv',index=False,encoding='utf-8-sig')

    harmonic=pd.read_csv(OUT/'审稿问题复核/单一原始网格留一年逐景对照.csv')
    cay=harmonic.groupby(['metric','sand_cay_id']).agg(baseline_error=('baseline_error','mean'),seasonal_error=('seasonal_error','mean')).reset_index()
    rng=np.random.default_rng(20260914); rows=[]
    order=['vegetation_fraction','sand_cay_area_m2','sand_cay_centroid_x_norm','sand_cay_centroid_y_norm','sand_cay_compactness','sand_cay_elongation','sand_cay_major_axis_angle']
    labels={'vegetation_fraction':'Vegetation proxy','sand_cay_area_m2':'Area','sand_cay_centroid_x_norm':'Centroid E-W','sand_cay_centroid_y_norm':'Centroid N-S','sand_cay_compactness':'Compactness','sand_cay_elongation':'Elongation','sand_cay_major_axis_angle':'Major-axis angle'}
    q_lookup=pd.read_csv(OUT/'审稿问题复核/单一原始网格留一年汇总.csv').set_index('metric')['fdr_q_value']
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
    for region,forced in [('GBR',None),('MEDF',None),('OTHER',None)]:
        pool=recurrent[recurrent.region.eq(region)]
        pool=pool[pool.sand_cay_id.isin(grid_ids)]
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
    a=fig.add_axes([.13,.12,.322,.38]); b=fig.add_axes([.575,.12,.322,.38])
    a.plot(phase_curve.annual_phase_separation_days,phase_curve.adjusted_iou_change_from_same_phase,color=C['navy'],lw=1.7)
    a.fill_between(phase_curve.annual_phase_separation_days,phase_curve.ci95_low,phase_curve.ci95_high,color=C['blue'],alpha=.25,lw=0)
    a.axhline(0,color=C['grey'],ls='--',lw=.75);a.axvline(91.3,color=C['grey'],ls=':',lw=.7)
    a.set(xlim=(0,182.5),xticks=[0,45,91,137,182],xlabel='Annual phase separation (days)',ylabel='Adjusted mask IoU change')
    a.text(.97,.06,f'{int(pairwise.sand_cay_id.nunique())} cays; {len(pairwise):,} pairs\nclustered 95% CI',transform=a.transAxes,ha='right',va='bottom',fontsize=6.1,color=C['grey']);clean(a);tag(a,'b')

    y=np.arange(len(harmonic_visual))[::-1]; cols=np.where(harmonic_visual.fdr_q_value.lt(.05),C['teal'],C['light'])
    b.barh(y,harmonic_visual.mean_relative_error_reduction,color=cols,height=.64,edgecolor='none')
    b.errorbar(harmonic_visual.mean_relative_error_reduction,y,xerr=np.vstack([harmonic_visual.mean_relative_error_reduction-harmonic_visual.ci95_low,harmonic_visual.ci95_high-harmonic_visual.mean_relative_error_reduction]),fmt='none',ecolor=C['dark'],capsize=2,lw=.8)
    b.axvline(0,color=C['grey'],ls='--',lw=.75);b.set(yticks=y,yticklabels=harmonic_visual.label,xlabel='LOYO error reduction (%)')
    b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1,decimals=0));b.tick_params(axis='y',labelsize=6.2);clean(b);tag(b,'c')

    cases=[record for region in atlas.region for record in collected[region]]
    case_table=pd.DataFrame(cases)
    case_table.drop(columns=['mask_path'],errors='ignore').to_csv(SRC/'figure_2_time_coloured_outline_cases.csv',index=False,encoding='utf-8-sig')
    case_table['date_parsed']=pd.to_datetime(case_table['date'])
    date_norm=mpl.colors.Normalize(vmin=mdates.date2num(case_table.date_parsed.min()),vmax=mdates.date2num(case_table.date_parsed.max()))
    time_cmap=mpl.colormaps['viridis']
    panel_side=.323; panel_w=.2425; panel_xs=[.13,.4375,.745]; panel_y=.625
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
        case_id=group.sand_cay_id.iloc[0]
        quarter=int(group.quarter.iloc[0])
        overlay.text(.03,.97,f'{region}\n{case_id} | Q{quarter}',transform=overlay.transAxes,ha='left',va='top',fontsize=6.7,weight='bold',color=C['dark'])
        overlay.text(.97,.97,f'n={len(rendered)}',transform=overlay.transAxes,ha='right',va='top',fontsize=5.8,color=C['grey'])
        overlay.text(.03,.03,f'{first_date} → {last_date}',transform=overlay.transAxes,ha='left',va='bottom',fontsize=5.6,color=C['grey'])
        overlay.set_aspect('equal');overlay.set_xticks([]);overlay.set_yticks([])
        for spine in overlay.spines.values(): spine.set_color(C['light']);spine.set_linewidth(.55)
        if column==0:
            tag(overlay,'a')
    colour_axis=fig.add_axes([.43,.57,.26,.012])
    bar=fig.colorbar(mpl.cm.ScalarMappable(norm=date_norm,cmap=time_cmap),cax=colour_axis,orientation='horizontal')
    tick_years=list(range(int(case_table.date_parsed.dt.year.min()),int(case_table.date_parsed.dt.year.max())+1,2))
    bar.set_ticks([mdates.date2num(pd.Timestamp(year,1,1)) for year in tick_years]);bar.set_ticklabels([str(year) for year in tick_years])
    bar.ax.tick_params(labelsize=6,length=2,width=.6);bar.outline.set_linewidth(.6)
    colour_axis.text(1.03,.5,'Observation date',transform=colour_axis.transAxes,va='center',fontsize=6.2,color=C['dark'])
    save(fig,'figure_2_seasonal_recurrence');plt.close(fig)
    return {'pairwise_cays':int(pairwise.sand_cay_id.nunique()),'pairwise_masks':int(len(pairwise)),'harmonic_cays':int(harmonic.sand_cay_id.nunique()),'atlas_cays':atlas.sand_cay_id.tolist()}



if __name__ == "__main__":
    print(fig2_seasonal())
