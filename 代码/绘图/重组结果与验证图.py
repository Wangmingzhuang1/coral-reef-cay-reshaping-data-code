"""按科学问题重组面板；补图只承载独立控制，不复制整张原结果图。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT, configure_imports
configure_imports()
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from 绘制当前结果图 import C,clean,tag

OUT=ROOT/'outputs'; SRC=OUT/'figures/source_data'; FIG=OUT/'figures'
REGIONS=['Great Barrier Reef','Marshall Islands','Maldives','South China Sea']
SHORT=['GBR','Marshall','Maldives','SCS']
COLORS=['#417998','#D89445','#55966B','#9A647F']
def read(p):return pd.read_csv(OUT/p)
def save(fig,name):
    folder=FIG/('补充图' if name.startswith('图S') else '主图');folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg','tiff']:
        kw={'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {'dpi':360}
        fig.savefig(folder/(name+'.'+ext),bbox_inches='tight',**kw)
    plt.close(fig)
def source(data,name):data.to_csv(SRC/(name+'.csv'),index=False,encoding='utf-8-sig')

def trajectories():
    d=read('审稿问题复核/原始网格一致长期位置描述.csv');source(d,'主图2_连续空间轨迹')
    selected=read('轨迹与转换核查/相似面积不同空间活动展示对.csv');source(selected,'主图2_可比面积展示对')
    matched=read('P0证据核查/全部同区域可比记录差异.csv');source(matched,'主图2_全部匹配差异')
    fig=plt.figure(figsize=(7.2,5.6));grid=fig.add_gridspec(2,12,height_ratios=[1.3,1],hspace=.7,wspace=1.5)
    bins=['Near-stable area','Net gain','Net loss'];colors=[C['navy'],C['teal'],C['red']]
    ax=fig.add_subplot(grid[0,:3])
    for kind,color in zip(bins,colors):
        p=d[d.net_area_bin==kind];ax.scatter(p.centroid_envelope_normalized,p.centroid_net_normalized,s=15,color=color,alpha=.65,label=kind)
    for row in selected.itertuples():
        ax.scatter(row.centroid_envelope_normalized,row.centroid_net_normalized,s=45,facecolors='none',edgecolors='black',lw=.8)
        ax.annotate(row.reef_id,(row.centroid_envelope_normalized,row.centroid_net_normalized),xytext=(3,5),textcoords='offset points',fontsize=5)
    ax.set(xscale='log',yscale='log',xlabel='Position envelope (r)',ylabel='Net displacement (r)');ax.set_title('90 grid-consistent tracks',fontsize=8);ax.legend(fontsize=5,frameon=False,loc='upper center',bbox_to_anchor=(1.35,1.37),ncol=3,title='Net-area outcome only',title_fontsize=5.5);clean(ax);tag(ax,'a')
    ax.text(.03,.03,'32 comparable pairs; 27 cays\nGBR and Maldives',transform=ax.transAxes,fontsize=5.5,va='bottom')
    ax=fig.add_subplot(grid[0,4:8]);v=ax.scatter(d.centroid_path_normalized,d.centroid_net_normalized,c=d.n_observations,cmap='viridis',s=17)
    lim=[.001,d.centroid_path_normalized.max()*1.2];ax.plot(lim,lim,ls=':',color='#777');ax.set(xscale='log',yscale='log',xlabel='Cumulative path (r)',ylabel='Net displacement (r)');clean(ax);tag(ax,'b')
    nh=d[d.sand_cay_id.eq('NH_015_cay_001')].iloc[0];ax.scatter([nh.centroid_path_normalized],[nh.centroid_net_normalized],s=45,facecolors='none',edgecolors=C['orange']);ax.annotate('NH_015*',(nh.centroid_path_normalized,nh.centroid_net_normalized),xytext=(-34,10),textcoords='offset points',fontsize=5.5)
    cb=fig.colorbar(v,ax=ax,pad=.03,fraction=.06);cb.set_label('Observations',fontsize=6)
    ax=fig.add_subplot(grid[0,9:]);valid=d[d.endpoint_geometry_comparable]
    for kind,color in zip(bins,colors):
        p=valid[valid.net_area_bin==kind];ax.scatter(p.centroid_net_normalized,p.endpoint_exchange_fraction*100,s=15,color=color,alpha=.65)
    for row in selected.itertuples():ax.scatter(row.centroid_net_normalized,row.endpoint_exchange_fraction*100,s=45,facecolors='none',edgecolors='black',lw=.8)
    ax.set(xscale='log',yscale='log',xlabel='Net displacement (r)',ylabel='First–last boundary exchange (%)');ax.set_title('90 comparable endpoints',fontsize=8);clean(ax);tag(ax,'c')
    r=read('尺度与跨源/全球形态重塑区域汇总.csv').set_index('analysis_region').reindex(REGIONS).reset_index();source(r,'主图2_区域独立指标')
    metrics=[('standardized_net_area_change','Net area (%)',100),('standardized_gross_boundary_reworking','Boundary exchange (%)',100),('standardized_centroid_translation','Centroid translation (r)',1),('standardized_axial_adjustment','Axial adjustment (r)',1)]
    for i,(key,title,factor) in enumerate(metrics):
        ax=fig.add_subplot(grid[1,3*i:3*(i+1)]);y=np.arange(4)
        for j,row in enumerate(r.itertuples()):
            value=getattr(row,key)*factor;lo=getattr(row,key+'_ci95_low')*factor;hi=getattr(row,key+'_ci95_high')*factor
            ax.errorbar(value,j,xerr=[[value-lo],[hi-value]],fmt='o',color=COLORS[j],capsize=2,ms=4)
        ax.set(yticks=y,yticklabels=[f'{x}; n={int(z)}' for x,z in zip(SHORT,r.n_cays)] if i==0 else [],xlabel=title);ax.invert_yaxis();clean(ax);tag(ax,chr(ord('d')+i))
    fig.subplots_adjust(left=.12,right=.97,top=.85,bottom=.12)
    save(fig,'图2_长期轨迹与空间调整')

def context():
    # 地图已归入主图1；此处只保留来源尺寸和跨源测量控制。
    fig,(ax,b)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1.15]})
    d=read('气候检验/figure_1_cay_area_structure.csv');source(d,'补图S1_来源尺寸支持')
    for sensor,offset,color,marker,label in [('sentinel2',-.13,C['navy'],'o','Sentinel-2'),('google_earth',.13,C['orange'],'s','Google Earth')]:
        for i,reg in enumerate(REGIONS):
            p=d[(d.sensor==sensor)&(d.map_region==reg)];ax.scatter(np.repeat(i+offset,len(p)),p.median_area_m2,s=9,alpha=.45,color=color,marker=marker,label=label if i==0 else None)
    ax.set(xticks=range(4),xticklabels=SHORT,yscale='log',ylabel='Cay median area (m²)');ax.legend(fontsize=6,frameon=False);clean(ax);tag(ax,'a')
    import 检验尺寸几何与跨源一致性 as uni
    intervals=uni.load_intervals();agreement,pivot=uni.cross_source_agreement(intervals);source(agreement,'补图S1_跨源诊断');uni.draw_agreement_panel(b,agreement,pivot,'b')
    fig.subplots_adjust(left=.1,right=.98,bottom=.2,top=.88,wspace=.4);save(fig,'图S1_样本来源与跨源控制')

def track_support():
    d=read('沙洲面积轨迹指标.csv');p=pd.read_csv(SRC/'长期主序列双定义.csv');source(d,'补图S2_采样支持')
    rows=[]
    for cutoff in np.arange(10,51,5):
        sel=p.relative_change.abs()<cutoff/100;morph=(p.late_cv<=.15)&(p.max_adjacent_area_ratio<=1.5)&(p.centroid_path_normalized<=1)
        rows.append({'cutoff_percent':cutoff,'n_area_group':int(sel.sum()),'n_within_three_thresholds':int((sel&morph).sum())})
    t=pd.DataFrame(rows);source(t,'补图S2_阈值敏感性')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7));a,b,c,e=axes.ravel();a.plot(t.cutoff_percent,t.n_area_group,'o-',label='Net-area group',color=C['navy']);a.plot(t.cutoff_percent,t.n_within_three_thresholds,'s-',label='Within three shape thresholds',color=C['teal']);a.axvline(25,ls=':',color='#888');a.set(xlabel='Net-area cutoff (%)',ylabel='Primary tracks (n)');a.legend(fontsize=6,frameon=False);tag(a,'a')
    b.scatter(d.span_years,d.n_observations,s=10,alpha=.4,color=C['navy']);b.axvline(3,ls=':',color=C['orange']);b.axhline(4,ls=':',color=C['orange']);b.set(xlabel='Observed span (years)',ylabel='Valid observations per track');tag(b,'b')
    nh=read('轨迹与转换核查/NH_015采样敏感性.csv');source(nh,'补图S2_NH采样敏感性');c.bar(range(len(nh)),nh.path_r,color=C['navy']);c.set(xticks=range(len(nh)),xticklabels=['All','Monthly','Quarterly','Yearly'],ylabel='NH_015 observed path (r)')
    for i,row in enumerate(nh.itertuples()):c.text(i,row.path_r+.5,f'n={row.n}',ha='center',fontsize=6)
    tag(c,'c');check=read('轨迹与转换核查/NH_015背景特征配准筛查.csv');source(check,'补图S2_NH背景筛查')
    dates=pd.to_datetime(check.date1);e.plot(dates,check.raw_step_m,color='#BBB',lw=.6);e.scatter(dates,check.raw_step_m,c=[C['teal'] if x else '#AAA' for x in check.screen_accepted],s=18);e.set(xlabel='Observation year',ylabel='NH_015 adjacent displacement (m)');e.text(.03,.94,'9/29 background checks supported',transform=e.transAxes,fontsize=6,va='top');e.xaxis.set_major_locator(mpl.dates.YearLocator(3));e.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'));tag(e,'d')
    for ax in [a,b,c,e]:clean(ax)
    fig.tight_layout(pad=2.1);save(fig,'图S2_轨迹阈值与采样支持')

def annual_controls():
    d=read('Sentinel2季节模板检验汇总.csv');e=read('稳健性扩展/等数量季节模板核验.csv');source(d,'补图S3_年度模板');source(e,'补图S3_等训练数')
    fig=plt.figure(figsize=(7.2,5.0));grid=fig.add_gridspec(2,2,height_ratios=[1,1.4]);a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);c=fig.add_subplot(grid[1,:])
    a.errorbar(d.mean_gain,range(len(d)),xerr=np.vstack([d.mean_gain-d.ci95_low,d.ci95_high-d.mean_gain]),fmt='o',color=C['navy'],capsize=2);a.set(yticks=range(len(d)),yticklabels=['Same vs all seasons','Same vs opposite season'],xlabel='Held-out Brier-error gain');a.axvline(0,ls=':',color='#777');tag(a,'a')
    row=e.iloc[0];b.errorbar([row.mean_gain],[0],xerr=[[row.mean_gain-row.ci95_low],[row.ci95_high-row.mean_gain]],fmt='o',color=C['teal'],capsize=3);b.axvline(0,ls=':',color='#777');b.set(yticks=[0],yticklabels=['Equal training count'],xlabel='Held-out Brier-error gain');b.text(.98,.8,f'{int(row.n_images)} images; {int(row.n_cays)} cays\n{int(row.positive_cays)} positive',ha='right',transform=b.transAxes,fontsize=7);tag(b,'b')
    for ax in [a,b]:clean(ax)
    forward=read('审稿问题复核/仅过去训练年度预测汇总.csv');source(forward,'补图S3_仅过去训练预测')
    metrics=['sand_cay_centroid_x_norm','sand_cay_centroid_y_norm','sand_cay_compactness','sand_cay_elongation','sand_cay_major_axis_angle']
    for comparison,offset,color,label in [('linear',-.13,C['navy'],'Seasonal vs linear'),('persistence',.13,C['orange'],'Seasonal vs persistence')]:
        g=forward[forward.comparison.eq(comparison)].set_index('metric').loc[metrics]
        y=np.arange(len(g))+offset;c.errorbar(g.mean_gain,y,xerr=np.vstack([g.mean_gain-g.ci95_low,g.ci95_high-g.mean_gain]),fmt='o',color=color,capsize=2,ms=3,label=label)
    c.axvline(0,ls=':',color='#777');c.set(yticks=range(5),yticklabels=['Centroid E-W','Centroid N-S','Compactness','Elongation','Major-axis angle'],xlabel='Forward error gain (scaled units)');c.invert_yaxis();c.legend(fontsize=6,frameon=False,loc='upper right');clean(c);tag(c,'c')
    fig.tight_layout(pad=2.2);save(fig,'图S3_年度预测对照')

def direction_controls():
    w=read('方向模型时间窗敏感性.csv');p=read('方向模型掩膜扰动敏感性.csv');loo=read('方向模型逐沙洲剔除敏感性汇总.csv')
    source(w,'补图S4_时间窗');source(p,'补图S4_像素扰动');source(loo,'补图S4_剔除沙洲')
    terms=[('along_shift','current_along_mean_z','Along current'),('along_shift','wave_vector_along_mean_z','Along wave'),('cross_shift','current_cross_mean_z','Cross current'),('cross_shift','wave_vector_cross_mean_z','Cross wave')]
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.0))
    for ax,frame,key,title in [(axes[0],w,'window_days','a  Time windows'),(axes[1],p,'perturbation_pixels','b  Boundary perturbation')]:
        conditions=list(frame[key].drop_duplicates());offsets=np.linspace(-.22,.22,len(conditions))
        for k,condition in enumerate(conditions):
            for i,(outcome,term,_) in enumerate(terms):
                row=frame[(frame[key]==condition)&(frame.outcome==outcome)&(frame.term==term)].iloc[0]
                ax.errorbar(row.coefficient,i+offsets[k],xerr=[[row.coefficient-row.ci95_low],[row.ci95_high-row.coefficient]],fmt='o',color=['#417998','#D89445','#55966B'][k],ms=3,capsize=1,label=str(condition) if i==0 else None)
        ax.axvline(0,color='#777',ls=':');ax.set(yticks=range(4),yticklabels=[x[2] for x in terms] if ax==axes[0] else [],xlabel='Coefficient (95% CI)');ax.set_title(title,loc='left',fontsize=8);ax.legend(fontsize=5.5,frameon=False,loc='upper left',bbox_to_anchor=(0,-.23));ax.invert_yaxis();clean(ax)
    ax=axes[2]
    for i,(outcome,term,_) in enumerate(terms):
        row=loo[(loo.outcome==outcome)&(loo.term==term)].iloc[0];ax.plot([row.coefficient_min,row.coefficient_max],[i,i],lw=3,color=C['navy']);ax.text(row.coefficient_max+.02,i,f'{int(row.n_ci_excluding_zero)}/{int(row.n_deletions)}',fontsize=6,va='center')
    ax.set(yticks=range(4),yticklabels=[],xlabel='Leave-one-cay coefficient range');ax.axvline(0,color='#777',ls=':');ax.invert_yaxis();ax.set_title('c  Cay omission',loc='left',fontsize=8);clean(ax)
    fig.subplots_adjust(left=.15,right=.96,top=.85,bottom=.3,wspace=.4);save(fig,'图S4_方向关联敏感性')

def state_controls():
    d=pd.read_csv(SRC/'严格标签一致转换敏感性.csv');changes=d[d.analysis_cover_state_3!=d.next_state].copy();source(changes,'补图S5_严格标签变化');cal=read('Sentinel2三状态连续时间模型校准.csv');cal=cal[(cal.from_state=='dominant_cover')&(cal.to_state=='low_cover')];source(cal,'补图S5_状态模型校准')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7));a,b,c,e=axes.ravel();labels=[x.replace('_cover','').title()+' → '+y.replace('_cover','').title() for x,y in zip(changes.analysis_cover_state_3,changes.next_state)]
    a.barh(range(len(changes)),changes.n_pairs,color=C['navy']);a.set(yticks=range(len(changes)),yticklabels=labels,xlabel='Strict-label state-change pairs');a.invert_yaxis();tag(a,'a')
    x=np.arange(len(cal));b.bar(x-.18,cal.observed_pairs,.36,color=C['teal'],label='Observed');b.bar(x+.18,cal.expected_pairs,.36,color=C['orange'],label='CTMC expected');b.set(xticks=x,xticklabels=cal.interval_bin,ylabel='Dominant-to-low pairs');b.legend(fontsize=6,frameon=False);tag(b,'b')
    support=read('轨迹与转换核查/转换前后有效样本层级.csv');source(support,'补图S5_前后比较样本支持');q=support.set_index('specification').loc[['eligible_flanks','non_reused','strict_non_reused','observation_matched']];c.bar(range(4),q.n_events,color=[C['navy'],C['teal'],C['orange'],'#AAA']);c.set(xticks=range(4),xticklabels=['Stable\nflanks','No image\nreuse','Strict\nlabels','Matched\nobservations'],ylabel='Eligible state-change windows (n)')
    for i,x in enumerate(q.n_events):c.text(i,x+.4,str(x),ha='center',fontsize=7)
    tag(c,'c');contrasts=read('轨迹与转换核查/转换前后描述性对照.csv');source(contrasts,'补图S5_前后描述对照');z=contrasts[contrasts.specification.eq('non_reused')]
    labels=[]
    for i,(direction,outcome,label) in enumerate([('Increasing cover','speed_log_change','Increase: speed'),('Decreasing cover','speed_log_change','Decrease: speed'),('Increasing cover','exchange_log_change','Increase: exchange'),('Decreasing cover','exchange_log_change','Decrease: exchange')]):
        row=z[(z.direction==direction)&(z.outcome==outcome)].iloc[0];e.errorbar(row.cay_equal_median,i,xerr=[[row.cay_equal_median-row.ci95_low],[row.ci95_high-row.cay_equal_median]],fmt='o',color=C['teal'] if direction=='Increasing cover' else C['orange'],capsize=2);labels.append(f'{label}; n={int(row.n_cays)}')
    e.set(yticks=range(4),yticklabels=labels,xlabel='After minus before (log-scale)');e.invert_yaxis();e.axvline(0,ls=':',color='#777');e.set_title('Unadjusted within-cay contrasts',fontsize=8);tag(e,'d')
    for ax in [a,b,c,e]:clean(ax)
    fig.tight_layout(pad=2.2);save(fig,'图S5_状态判读与模型校准')

def scale_controls():
    d=read('尺度与跨源/全球普适性沙洲汇总.csv');g=read('尺度与跨源/全球尺寸几何分解.csv');source(d,'补图S6_几何控制对象');source(g,'补图S6_几何分解')
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3));a.scatter(d.area_m2,d.gross,s=13,alpha=.55,color=C['navy'],label='Observed reworking');a.scatter(d.area_m2,d.floor,s=10,alpha=.3,color=C['orange'],marker='^',label='±1-pixel reference');a.set(xscale='log',yscale='log',xlabel='Cay median area (m²)',ylabel='Relative boundary exchange / year');a.legend(fontsize=6,frameon=False);tag(a,'a')
    p=g[g.stratum=='ALL'];b.errorbar(p.slope,range(len(p)),xerr=np.vstack([p.slope-p.ci95_low,p.ci95_high-p.slope]),fmt='o',capsize=2,color=C['teal']);b.set(yticks=range(len(p)),yticklabels=['Relative exchange','Perimeter / area','Boundary displacement'],xlabel='Log-log size slope (95% CI)');b.axvline(0,color='#777',ls=':');tag(b,'b')
    for ax in [a,b]:clean(ax)
    fig.tight_layout(pad=2.2);save(fig,'图S6_尺寸几何与可检测性')

def alternative_controls():
    wind=read('风场增量检验模型比较.csv');wind=wind[wind.analysis_scope.isin(['all_intervals_pulse_adjusted','no_typhoon_pulse_intervals'])];event=read('台风事件匹配对照统计.csv');event=event[event.specification=='all_events'];source(wind,'补图S7_风场增量');source(event,'补图S7_事件对照')
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.0));order=['log_gross_mobility','along_shift','cross_shift']
    for scope,col,offset,label in [('all_intervals_pulse_adjusted',C['navy'],-.12,'All intervals'),('no_typhoon_pulse_intervals',C['orange'],.12,'No typhoon pulse')]:
        p=wind[wind.analysis_scope==scope].set_index('term').reindex(order);a.scatter(p.joint_wind_cluster_fdr_q_value,np.arange(3)+offset,color=col,label=label,s=25)
    a.axvline(.05,ls=':',color='#777');a.set(yticks=range(3),yticklabels=['Boundary exchange','Along shift','Cross shift'],xlabel='Joint-wind FDR q');a.legend(fontsize=6,frameon=False);tag(a,'a')
    outcome=['erosion_fraction','deposition_fraction','net_area_fraction','gross_mobility_fraction'];p=event.set_index('outcome').reindex(outcome).dropna(subset=['storm_median_difference']);b.errorbar(p.storm_median_difference,range(len(p)),xerr=np.vstack([p.storm_median_difference-p.bootstrap_ci95_low,p.bootstrap_ci95_high-p.storm_median_difference]),fmt='o',color=C['teal'],capsize=2);b.set(yticks=range(len(p)),yticklabels=[x.replace('_fraction','').replace('_',' ').title() for x in p.index],xlabel='Matched event minus background');b.axvline(0,ls=':',color='#777');tag(b,'b')
    for ax in [a,b]:clean(ax)
    fig.tight_layout(pad=2.2);save(fig,'图S7_风场与台风替代检验')

if __name__=='__main__':
    trajectories();context();track_support();annual_controls();direction_controls();state_controls();scale_controls();alternative_controls()
    print('主图2与七幅独立验证补图完成；原整图不再作为补图。')
