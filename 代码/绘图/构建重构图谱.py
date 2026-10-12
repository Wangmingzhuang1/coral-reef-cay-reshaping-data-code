"""从已质控原始影像与掩膜生成观测图谱；不写入原始数据。"""

import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from 构建沙洲观测与变化表 import select_analysis_observations,read_mask
from 拟合Sentinel2三状态连续时间模型 import make_pairs

ROOT = PROJECT_ROOT
BASE=ROOT.parent/"归档/重构前"
OUT=ROOT/'outputs/figures'
SRC=OUT/'source_data'
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COL=['#2B6F8E','#BC7B23','#B94445','#716086']
ST=['none','sparse','partial','dominant']
STCOL=['#D7C7A5','#A9C99E','#589982','#266B5B']
ORDER=['Near-stable area','Net gain','Net loss','Complex / uncertain']
def save(df,name):df.to_csv(SRC/(name+'.csv'),index=False,encoding='utf-8-sig')
def figure(fig,name):
    for ext in ['png','pdf','svg','tiff']:fig.savefig(OUT/(name+'.'+ext),dpi=350,facecolor='white',bbox_inches='tight')
    plt.close(fig)
def tag(ax,t):ax.text(-.08,1.08,t,transform=ax.transAxes,fontweight='bold',fontsize=11)
def track(p):return obs[(obs.sand_cay_id==p.sand_cay_id)&(obs.sensor==p.sensor)&(obs.reference_frame_id==p.reference_frame_id)].sort_values('date').copy()
def crop(masks,margin=.18):
    union=np.logical_or.reduce(masks); yy,xx=np.nonzero(union)
    pad=max(10,int(max(np.ptp(xx),np.ptp(yy))*margin))
    return max(0,xx.min()-pad),min(union.shape[1],xx.max()+pad+1),max(0,yy.min()-pad),min(union.shape[0],yy.max()+pad+1)
def image(ax,q,box,highlight=False):
    x0,x1,y0,y1=box;m=read_mask(q.resolved_mask_path)
    im=np.asarray(Image.open(q.resolved_image_path).convert('RGB'))
    assert im.shape[:2]==m.shape
    ax.imshow(im[y0:y1,x0:x1],interpolation='nearest' if highlight else None)
    if highlight:ax.contour(m[y0:y1,x0:x1],levels=[.5],colors='black',linewidths=1.1)
    ax.contour(m[y0:y1,x0:x1],levels=[.5],colors='white',linewidths=.55 if highlight else .6)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    scale=float(q.pixel_size_m);width=(x1-x0)*scale
    length=max(10,round(width/4/10)*10)
    ax.plot([.07,.07+length/width],[.10,.10],transform=ax.transAxes,color='white',lw=2)
    ax.text(.07,.13,f'{length:g} m',transform=ax.transAxes,color='white',fontsize=7)
    ax.text(.92,.88,'N ↑',transform=ax.transAxes,ha='right',color='white',fontsize=8)
    ax.set_title(f'{q.date:%Y-%m-%d}\n{q.quality_grade}; {scale:.2f} m/pixel',fontsize=8,pad=2)
    return m

SRC.mkdir(parents=True,exist_ok=True)
obs=pd.read_csv(BASE/'第一阶段审计/有效观测与图像掩膜清单.csv')
obs=select_analysis_observations(obs)
obs['date']=pd.to_datetime(obs.date)
primary=pd.read_csv(ROOT/'outputs/沙洲面积主轨迹分类.csv')
geo=pd.read_csv(BASE/'第一阶段审计/完整区域地理映射.csv').set_index('reef_id').map_region.to_dict()
primary['region']=primary.reef_id.map(geo)
# 保留审查分类；净面积分箱独立计算，不强行改名。
primary['net_area_bin']=np.select([primary.relative_change.abs()<.25,primary.relative_change>=.25,primary.relative_change<=-.25],['Near-stable area','Net gain','Net loss'],default='Unresolved')
primary['reviewed_area_class']=primary.multimetric_class
save(primary,'长期主序列双定义')

# 四案例按目标轨迹特征分层，再按观测数、跨度、ID选取。
spec=[('Great Barrier Reef','Near-stable area'),('Marshall Islands','Net gain'),('Maldives','Net loss'),('South China Sea','Net gain')]
cases=[]
for region,kind in spec:
    candidates=primary[(primary.region==region)&(primary.net_area_bin==kind)].sort_values(['n_observations','span_years','sand_cay_id'],ascending=[False,False,True])
    cases.append(candidates.iloc[0])
fig=plt.figure(figsize=(9.6,12.8));gs=fig.add_gridspec(4,6,width_ratios=[1,1,1,1,1,1],hspace=.88,wspace=.58)
quantitative_axes=[]
# 研究区位置是四区真实图谱的入口，直接取已核查礁盘坐标。
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
locations=pd.read_csv(ROOT/'outputs/气候检验/figure_1_region_map.csv')
mapax=fig.add_axes([.075,.675,.88,.304],projection=ccrs.PlateCarree())
mapax.set_extent([65,178,-27,25],crs=ccrs.PlateCarree())
mapax.add_feature(cfeature.LAND,facecolor='#EFEFEF')
mapax.coastlines(linewidth=.4,color='#777')
grid=mapax.gridlines(draw_labels=True,xlocs=[75,105,135,165],ylocs=[-20,0,20],linewidth=.25,color='#AAA',alpha=.5)
grid.top_labels=False;grid.right_labels=False;grid.xlabel_style={'size':7};grid.ylabel_style={'size':7}
handles=[]
for k,(region,_) in enumerate(spec):
    dloc=locations[locations.map_region.eq(region)]
    mapax.scatter(dloc.center_lon,dloc.center_lat,s=10,color=COL[k],transform=ccrs.PlateCarree(),zorder=3)
    handles.append(Line2D([],[],marker='o',linestyle='none',color=COL[k],markersize=4,label=region))
mapax.set_title('a  Study regions',loc='left',fontweight='bold',fontsize=10)
mapax.legend(handles=handles,loc='lower left',ncol=2,fontsize=7.5,frameon=True,facecolor='white',edgecolor='none',framealpha=.9,columnspacing=1.2,labelspacing=.45)
save(locations,'主图1_区域位置')
fig.subplots_adjust(bottom=.07,top=.61,left=.075,right=.955)
example_rows=[];coordinates=[]
for i,p in enumerate(cases):
    d=track(p);masks=[read_mask(v) for v in d.resolved_mask_path];assert all(m is not None for m in masks)
    mid=d.date.min()+(d.date.max()-d.date.min())/2
    selected=[d.iloc[0],d.loc[(d.date-mid).abs().idxmin()],d.iloc[-1]]
    box=crop([read_mask(q.resolved_mask_path) for q in selected],margin=.10)
    for j,q in enumerate(selected):
        ax=fig.add_subplot(gs[i,j]);image(ax,q,box,highlight=True)
        if j==0:fig.text(.075,gs[i,0].get_position(fig).y1+.027,f'{chr(98+i)}  {p.region}\n{p.sand_cay_id} | {"S2" if p.sensor=="sentinel2" else "GE"}',fontsize=8.5,weight='bold')
        r=q.to_dict();r.update(display_xmin=int(box[0]),display_xmax=int(box[1]),display_ymin=int(box[2]),display_ymax=int(box[3]),position=['early','middle','late'][j],net_area_bin=p.net_area_bin,long_term_delta=p.relative_change,n_observations=p.n_observations,span_years=p.span_years);example_rows.append(r)
    x0,x1,y0,y1=box;early=masks[0];late=masks[-1];change=np.zeros(early.shape,dtype=int);change[early&late]=1;change[early&~late]=2;change[late&~early]=3
    ax=fig.add_subplot(gs[i,3]);ax.imshow(change[y0:y1,x0:x1],cmap=ListedColormap(['#FFFFFF','#D3D7D7','#C34F51','#328F89']),vmin=0,vmax=3);ax.set_xticks([]);ax.set_yticks([]);ax.set_box_aspect(1);ax.set_aspect('equal',adjustable='datalim');ax.set_axis_off();quantitative_axes.append(ax)
    area=float(d.iloc[0].sand_cay_area_m2);s=float(d.iloc[0].pixel_size_m)
    er=np.sum(early&~late)*s*s;ac=np.sum(late&~early)*s*s
    ax.text(.03,.03,f'Loss {er/area*100:.1f}%\nGain {ac/area*100:.1f}%',transform=ax.transAxes,fontsize=7,va='bottom',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
    # 位置使用真实掩膜质心字段，北向为图像y反向。
    dx=(d.sand_cay_centroid_x-d.iloc[0].sand_cay_centroid_x)*s;dy=-(d.sand_cay_centroid_y-d.iloc[0].sand_cay_centroid_y)*s
    ax=fig.add_subplot(gs[i,4])
    if p.sand_cay_id=='NH_015_cay_001':
        indexes=[0,d.index.get_loc(selected[1].name),len(d)-1]
        ax.scatter(dx.iloc[indexes],dy.iloc[indexes],c=matplotlib.dates.date2num(d.iloc[indexes].date),cmap='viridis',s=22,zorder=3)
        ax.set_xlabel('East (m)',fontsize=8)
    else:
        ax.plot(dx,dy,color='#A4ADB5',lw=.7);ax.scatter(dx,dy,c=matplotlib.dates.date2num(d.date),cmap='viridis',s=12,zorder=3)
        ax.set_xlabel('East (m)',fontsize=8.5)
    ax.scatter(dx.iloc[0],dy.iloc[0],marker='s',color='black',s=22);ax.scatter(dx.iloc[-1],dy.iloc[-1],marker='*',color='#B94445',s=40);ax.set_box_aspect(1);ax.set_aspect('equal',adjustable='datalim');ax.set_ylabel('North (m)');quantitative_axes.append(ax)
    ax=fig.add_subplot(gs[i,5]);ax.plot(d.date,d.sand_cay_area_m2/10000,'o-',ms=2,lw=.7,color=COL[i]);ax.set(ylabel='Visible area (ha)');ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(3));ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'));ax.set_box_aspect(1);quantitative_axes.append(ax);ax.text(.03,.97,f'ΔA={p.relative_change:+.1%}',transform=ax.transAxes,fontsize=8,va='top');ax.tick_params(axis='x',rotation=30)
    for k,(_,q) in enumerate(d.iterrows()):coordinates.append(dict(sand_cay_id=p.sand_cay_id,date=q.date,east_m=dx.iloc[k],north_m=dy.iloc[k],area_m2=q.sand_cay_area_m2,mask_path=q.resolved_mask_path,image_path=q.resolved_image_path))
fig.subplots_adjust(bottom=.07,top=.61,left=.075,right=.955)
# 保持四区案例与面板，保证按Word页面缩放后日期、坐标和案例标签可读。
for axis in fig.axes:
    for text in [axis.title,axis.xaxis.label,axis.yaxis.label,*axis.get_xticklabels(),*axis.get_yticklabels(),*axis.texts]:
        text.set_fontsize(max(8.5,text.get_fontsize()))
for text in fig.texts:
    text.set_fontsize(max(8.5,text.get_fontsize()))
fig.canvas.draw()
# 后三列绘图区尺寸一致，质心图保留东西与南北等比例。
boxes=[axis.get_window_extent() for axis in quantitative_axes]
assert np.ptp([box.width for box in boxes])<1e-6
assert np.ptp([box.height for box in boxes])<1e-6
figure(fig,'图1_四区真实演变');save(pd.DataFrame(example_rows),'四区图谱时相');save(pd.DataFrame(coordinates),'四区图谱位置面积序列')
if '--figure1-only' in sys.argv:
    sys.exit(0)

bins=['Near-stable area','Net gain','Net loss']

surface=select_analysis_observations(pd.read_csv(ROOT/'outputs/植被状态审校主表.csv'))
pairs=make_pairs(surface[surface.sensor=='sentinel2'],min_days=30,max_days=365,confirmed_only=False);pairs['region']=pairs.reef_id.map(geo)
save(pairs,'表面状态相邻观测对')
trans=pairs.groupby(['analysis_cover_state_3','next_state']).agg(n_pairs=('image_id','size'),n_cays=('sand_cay_id','nunique')).reset_index();save(trans,'六向状态转换与保持')
joint=surface.merge(primary[['sand_cay_id','sensor','reference_frame_id','net_area_bin','multimetric_class','region']],on=['sand_cay_id','sensor','reference_frame_id'],how='inner',validate='many_to_one');joint=joint[joint.analysis_vegetation_state.isin(ST)]
freq=joint.groupby(['sand_cay_id','net_area_bin','region','analysis_vegetation_state']).size().rename('n_observations').reset_index();freq['within_cay_fraction']=freq.n_observations/freq.groupby('sand_cay_id').n_observations.transform('sum');save(freq,'双轴联合沙洲级观测')
table=freq.groupby(['net_area_bin','analysis_vegetation_state']).agg(observations=('n_observations','sum'),cays_with_state=('sand_cay_id','nunique'),equal_cay_mass=('within_cay_fraction','sum')).reset_index();den=freq.groupby('net_area_bin').sand_cay_id.nunique();table['n_cays_in_bin']=table.net_area_bin.map(den);table['equal_cay_fraction']=table.equal_cay_mass/table.n_cays_in_bin;save(table,'双轴联合分布')
save(freq.groupby(['region','net_area_bin','analysis_vegetation_state']).agg(n_cays=('sand_cay_id','nunique'),n_observations=('n_observations','sum'),equal_cay_mass=('within_cay_fraction','sum')).reset_index(),'双轴联合分布区域支持')
# 同沙洲具有六方向中两向的记录优先，按记录长度确定，不按视觉变化幅度挑选。
changes=pairs[pairs.analysis_cover_state_3!=pairs.next_state];chosen=changes.groupby('sand_cay_id').size().sort_values(ascending=False).index[0]
sd=surface[(surface.sand_cay_id==chosen)&(surface.sensor=='sentinel2')&surface.analysis_cover_state_3.notna()].sort_values('date')
state_order=['low_cover','partial_cover','dominant_cover'];vis=[]
for state in state_order:
    sub=sd[(sd.analysis_cover_state_3==state)&(sd.cover_state_status_3=='confirmed')];q=sub.sort_values(['color_state_margin','date'],ascending=[False,True]).iloc[0]
    v=obs[obs.image_id==q.image_id].iloc[0].copy();v['analysis_cover_state_3']=state;v['vegetation_fraction']=q.vegetation_fraction;vis.append(v)
vis=sorted(vis,key=lambda q:q.date);box=crop([read_mask(q.resolved_mask_path) for q in vis])
fig=plt.figure(figsize=(7.8,7.8));gs=fig.add_gridspec(3,6,height_ratios=[1.4,1,1.1],wspace=.75,hspace=.7)
for j,q in enumerate(vis):
    ax=fig.add_subplot(gs[0,j*2:(j+1)*2]);image(ax,q,box);ax.set_xlabel(f'{q.analysis_cover_state_3.replace("_cover", "")}\nGreen fraction {q.vegetation_fraction:.1%}',fontsize=8)
    if j==0:tag(ax,'a')
ax=fig.add_subplot(gs[1,:3]);pos={'low_cover':(.1,.55),'partial_cover':(.9,.55),'dominant_cover':(.5,.05)}
for state,(x,y) in pos.items():
    ax.scatter(x,y,s=1250,color=STCOL[state_order.index(state)+1],zorder=3);n=int((pairs.analysis_cover_state_3==state).sum());selfn=int(((pairs.analysis_cover_state_3==state)&(pairs.next_state==state)).sum());ax.text(x,y,f'{state.replace("_cover", "").title()}\n{selfn}/{n} stay',ha='center',va='center',fontsize=7,color='white',zorder=4)
for row in trans.itertuples():
    if row.analysis_cover_state_3==row.next_state:continue
    x,y=pos[row.analysis_cover_state_3];xx,yy=pos[row.next_state];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,connectionstyle='arc3,rad=.25',shrinkA=27,shrinkB=27,lw=1+row.n_pairs/50,color='#435C70',zorder=5))
    vx,vy=xx-x,yy-y;ax.text((x+xx)/2+vy*.15,(y+yy)/2-vx*.15,str(row.n_pairs),ha='center',va='center',fontsize=9,zorder=6,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
ax.set(xlim=(-.12,1.12),ylim=(-.18,.95));ax.axis('off');tag(ax,'b')
ax=fig.add_subplot(gs[1,3:]);sd['date']=pd.to_datetime(sd.date);valid_sd=sd[sd.analysis_cover_state_3.isin(state_order)]
ax.scatter(valid_sd.date,valid_sd.analysis_cover_state_3.map({s:i for i,s in enumerate(state_order)}),c=valid_sd.analysis_cover_state_3.map({s:STCOL[i+1] for i,s in enumerate(state_order)}),s=18);ax.plot(valid_sd.date,valid_sd.analysis_cover_state_3.map({s:i for i,s in enumerate(state_order)}),lw=.6,color='#AAA');ax.set(yticks=range(3),yticklabels=['Low','Partial','Dominant'],ylim=(-.3,2.3),xlabel='Observation year');ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2));ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'));ax.set_title(chosen,fontsize=8);tag(ax,'c')
ax=fig.add_subplot(gs[2,:]);pivot=table.pivot(index='net_area_bin',columns='analysis_vegetation_state',values='equal_cay_fraction').reindex(index=bins,columns=ST).fillna(0);bottom=np.zeros(3)
for k,state in enumerate(ST):
    ax.barh(range(3),pivot[state],left=bottom,color=STCOL[k],label=state.title());bottom+=pivot[state].to_numpy()
ax.set(yticks=range(3),yticklabels=[f'{v}; n={int(den[v])}' for v in bins],xlabel='Mean within-cay observed state fraction',xlim=(0,1));ax.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.55,-.3));tag(ax,'d')
fig.subplots_adjust(left=.12,right=.98,bottom=.1,top=.92)
figure(fig,'图3_表面状态双向转换');save(pd.DataFrame(vis),'同沙洲状态图像');save(sd,'同沙洲覆盖状态时序')
confirmed=make_pairs(surface[surface.sensor=='sentinel2'],min_days=30,max_days=365,confirmed_only=True)
save(confirmed,'严格标签一致状态对');save(confirmed.groupby(['analysis_cover_state_3','next_state']).size().rename('n_pairs').reset_index(),'严格标签一致转换敏感性')
summary_confirmed={'pairs':len(confirmed),'cays':confirmed.sand_cay_id.nunique(),'changes':int((confirmed.analysis_cover_state_3!=confirmed.next_state).sum()),'changing_cays':confirmed.loc[confirmed.analysis_cover_state_3!=confirmed.next_state,'sand_cay_id'].nunique()}
summary={'primary_sequences':len(primary),'endpoint_bins':primary.net_area_bin.value_counts().to_dict(),'original_reviewed_classes':primary.multimetric_class.value_counts().to_dict(),'joint_cays':joint.sand_cay_id.nunique(),'joint_observations':len(joint),'state_pairs':len(pairs),'state_cays':pairs.sand_cay_id.nunique(),'state_changes':len(changes),'changing_cays':changes.sand_cay_id.nunique(),'state_example':chosen,'examples':[dict(cay_id=p.sand_cay_id,region=p.region,n=int(p.n_observations),span=float(p.span_years),delta=float(p.relative_change)) for p in cases]}
summary['strict_label_sensitivity']=summary_confirmed
(SRC/'重构观测核查.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
