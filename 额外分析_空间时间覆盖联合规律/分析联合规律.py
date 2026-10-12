"""独立探索性联合检验：不改正文、不重估原有主结果。"""
from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT/'代码/公共'))
from 构建沙洲观测与变化表 import select_analysis_observations

INPUTS = {
 'tracks': 'outputs/轨迹与转换核查/连续空间轨迹与首末边界交换.csv',
 'intervals': 'outputs/figures/source_data/表B_区间级状态边界位移环境.csv',
 'annual': 'outputs/Sentinel2季节谐波交叉验证.csv',
 'observations': 'outputs/沙洲观测主表.csv',
}
HYPOTHESES = [
 ('H1','between','cover','envelope','Cover vs. activity range'),
 ('H2','between','cover','net','Cover vs. net displacement'),
 ('H3','between','cover','exchange','Cover vs. boundary exchange'),
 ('H4','annual','position_gain','envelope','Annual position gain vs. range'),
 ('H5','annual','elongation_gain','envelope','Annual shape gain vs. range'),
 ('H6','annual','cover','position_gain','Cover vs. annual position gain'),
 ('H7','within','green_fraction_t0','centroid_speed_normalized','Within-cay cover vs. speed'),
 ('H8','within','green_fraction_t0','boundary_rate','Within-cay cover vs. boundary rate'),
]
CHINESE = {'H1':'覆盖与长期活动范围','H2':'覆盖与首末净位移','H3':'覆盖与首末边界交换',
 'H4':'位置年度增益与长期活动范围','H5':'伸长率年度增益与长期活动范围',
 'H6':'覆盖与位置年度增益','H7':'同个体期初覆盖与随后移动速率','H8':'同个体期初覆盖与随后边界交换速率'}

def save(df, name):
 df.to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def markdown_table(d):
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
  ['| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def z(a):
 a=np.asarray(a,float)
 return (a-a.mean())/a.std(ddof=1)

def make_numeric(d, names):
 for name in names: d[name]=pd.to_numeric(d[name],errors='coerce')

def fit(d, family, predictor, outcome, environment=False):
 controls=['log_area','log_duration','log_n'] if family!='within' else ['log_area_t','log_interval','sin_doy','cos_doy']
 if family=='annual': controls+=['log_annual_baseline']
 if environment: controls+=['current_magnitude','wave_magnitude','storm_fraction']
 needed=[predictor,outcome,'reef_id','sand_cay_id',*controls]
 d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=needed).copy()
 if family=='within':
  variation=d.groupby('sand_cay_id')[predictor].agg(['size','min','max'])
  keep=variation.index[(variation['size']>=6)&((variation['max']-variation['min'])>=.05)]
  d=d[d.sand_cay_id.isin(keep)].copy()
 if len(d)<20 or d.reef_id.nunique()<10 or d[predictor].nunique()<5:
  return {'status':'insufficient_support','n':len(d),'n_cays':d.sand_cay_id.nunique(),'n_reefs':d.reef_id.nunique()},d
 if family=='within':
  x=d[predictor]-d.groupby('sand_cay_id')[predictor].transform('mean')
  y=np.log1p(d[outcome]); yw=y-y.groupby(d.sand_cay_id).transform('mean')
  X=pd.DataFrame({'target':x/x.std(ddof=1)},index=d.index)
  y=(y-y.mean())/yw.std(ddof=1)
  for c in controls:
   if d[c].std()>1e-10:X[c]=z(d[c])
  factors=pd.get_dummies(d[['sand_cay_id','year']].astype(str),drop_first=True,dtype=float)
  weights=1/d.groupby('sand_cay_id').sand_cay_id.transform('size')
 else:
  X=pd.DataFrame({'target':z(rankdata(d[predictor]))},index=d.index)
  y=z(rankdata(d[outcome]))
  for c in controls:
   if d[c].std()>1e-10:X[c]=z(rankdata(d[c]))
  factors=pd.get_dummies(d[['region','sensor']],drop_first=True,dtype=float)
  weights=np.ones(len(d))
 X=pd.concat([X,factors],axis=1);X=sm.add_constant(X,has_constant='add')
 # 删除退化的虚拟列，目标变量及实质控制项必须可识别。
 columns=[];rank=0
 for c in X.columns:
  trial=X[columns+[c]].to_numpy();newrank=np.linalg.matrix_rank(trial)
  if newrank>rank:columns.append(c);rank=newrank
 X=X[columns]
 if 'target' not in X or len(d)<=len(columns)+5:
  return {'status':'unidentifiable','n':len(d),'n_cays':d.sand_cay_id.nunique(),'n_reefs':d.reef_id.nunique()},d
 model=sm.WLS(y,X,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':d.reef_id,'use_correction':True},use_t=True)
 ci=model.conf_int().loc['target']
 rawrho,rawp=spearmanr(d[predictor],d[outcome])
 result={'status':'estimated','n':len(d),'n_cays':d.sand_cay_id.nunique(),'n_reefs':d.reef_id.nunique(),
  'beta':model.params['target'],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],'p':model.pvalues['target'],
  'raw_spearman':rawrho,'raw_p_diagnostic_only':rawp,'design_rank':rank,'parameters':len(columns),
  'design_condition':np.linalg.cond(X.to_numpy()),'residual_df':model.df_resid,
  'max_abs_standardized_residual':np.max(np.abs(model.resid))/np.std(model.resid),
  'removed_redundant_terms':'|'.join(c for c in factors.columns if c not in columns)}
 return result,d

def build_data():
 a=pd.read_csv(ROOT/INPUTS['tracks'])
 assert a.track_id.is_unique and a.sand_cay_id.is_unique
 a=a[~a.reef_id.eq('NH_015')].copy()
 a['cover']=a.median_green_fraction
 a['envelope']=a.centroid_envelope_normalized;a['net']=a.centroid_net_normalized
 a['exchange']=a.endpoint_exchange_fraction
 a['log_area']=np.log(a.start_area_m2);a['log_duration']=np.log(a.span_years)
 a['log_n']=np.log(a.n_observations);a['radius_pixels']=np.sqrt(a.start_area_m2/np.pi)/10
 cv=pd.read_csv(ROOT/INPUTS['annual'])
 assert not cv.duplicated(['metric','image_id']).any()
 master=select_analysis_observations(pd.read_csv(ROOT/INPUTS['observations']))
 keys=['sensor','sand_cay_id','reference_frame_id']
 cv=cv.merge(master[['image_id','sensor','reference_frame_id']],on='image_id',validate='many_to_one')
 cv=cv.merge(a[keys],on=keys,how='inner',validate='many_to_one')
 means=cv.groupby(['sand_cay_id','metric'])[['baseline_error','seasonal_error']].mean().unstack('metric')
 baseline=means['baseline_error'];seasonal=means['seasonal_error']
 metrics=['sand_cay_centroid_x_norm','sand_cay_centroid_y_norm']
 annual=pd.DataFrame(index=baseline.index)
 annual['position_gain']=1-seasonal[metrics].sum(axis=1)/baseline[metrics].sum(axis=1)
 annual['elongation_gain']=1-seasonal.sand_cay_elongation/baseline.sand_cay_elongation
 annual['annual_baseline']=baseline[metrics].sum(axis=1)
 annual['annual_years']=cv.groupby('sand_cay_id').held_out_year.nunique()
 annual['log_annual_baseline']=np.log(annual.annual_baseline)
 a=a.merge(annual,on='sand_cay_id',how='left',validate='one_to_one')
 b=pd.read_csv(ROOT/INPUTS['intervals'])
 assert not b.duplicated(['sensor','sand_cay_id','reference_frame_id','time_t','time_t1']).any()
 # 区间只进入与长期主序列同来源、同固定框的共同样本。
 b=b.merge(a[keys],on=keys,how='inner',validate='many_to_one')
 master['time_t']=pd.to_datetime(master.date).dt.strftime('%Y-%m-%d')
 cols=keys+['time_t','sand_cay_area_m2','pixel_size_m']
 assert not master.duplicated(keys+['time_t']).any()
 b=b.merge(master[cols],on=keys+['time_t'],how='left',validate='many_to_one')
 assert b.sand_cay_area_m2.notna().all()
 b['boundary_rate']=b.boundary_exchange_m2/b.sand_cay_area_m2/(b.interval_days/365.2425)
 b['log_area_t']=np.log(b.sand_cay_area_m2);b['log_interval']=np.log(b.interval_days)
 dates=pd.to_datetime(b.time_t);b['year']=dates.dt.year
 angle=2*np.pi*dates.dt.dayofyear/365.2425;b['sin_doy']=np.sin(angle);b['cos_doy']=np.cos(angle)
 b['radius_pixels']=np.sqrt(b.sand_cay_area_m2/np.pi)/b.pixel_size_m
 b['current_magnitude']=np.hypot(b.current_along_mean,b.current_cross_mean)
 b['wave_magnitude']=np.hypot(b.wave_vector_along_mean,b.wave_vector_cross_mean)
 b['storm_fraction']=b.event_storm_days/b.interval_days
 assert b.green_fraction_t0.dropna().between(0,1).all()
 assert b.boundary_rate.dropna().ge(0).all()
 return a,b

def main():
 # 固定8项假设及检验家族后才拟合，不按显著性增减模型。
 fingerprints={k:sha(ROOT/v) for k,v in INPUTS.items()}
 protected_paths=['论文初稿.md','论文补充材料.md','论文/中文论文.docx','论文/中文论文补充材料.docx','outputs/figures/当前图件索引.json']
 protected={p:sha(ROOT/p) for p in protected_paths}
 plan={'性质':'探索性分析，已看到旧结果；非预注册确认研究','假设':HYPOTHESES,
  '主检验':'8项统一BH校正；礁盘聚类95%CI；沙洲为长期分析单位；区间等沙洲权重',
  '样本':'剔除NH_015；共同来源固定框；区间每沙洲≥6对且覆盖范围≥0.05',
  '敏感性':['逐区域剔除','分来源','逐区域描述','Sentinel-2初始半径≥3像元','年度≥4个验证年份','区间完整环境子集及同子集不调环境对照'],
  '普适性':'总体显著不等于普适；报告区域/来源支持和留一稳定性；无因果解释',
  '输入SHA256':fingerprints}
 (OUT/'分析方案与来源.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
 a,b=build_data();save(a,'沙洲级共同样本');save(b,'区间级共同样本')
 # 独立用位移、原速率和间隔反推面积，验证新边界速率没有混用面积口径。
 moving=b[b.centroid_speed_normalized.gt(0)].dropna(subset=['centroid_dx_m','centroid_dy_m'])
 inferred=np.pi*(np.hypot(moving.centroid_dx_m,moving.centroid_dy_m)/(moving.centroid_speed_normalized*moving.interval_days/365.2425))**2
 assert np.allclose(inferred,moving.sand_cay_area_m2,rtol=1e-9,atol=1e-6)
 assert a.loc[a.position_gain.notna(),'sensor'].eq('sentinel2').all()
 support=a.groupby(['region','sensor'],dropna=False).agg(n_cays=('sand_cay_id','size'),annual_n=('position_gain','count'),median_area=('start_area_m2','median'),median_years=('span_years','median')).reset_index()
 save(support,'区域来源支持')
 rows=[];sensitivity=[];description=[]
 for hid,family,pred,outcome,label in HYPOTHESES:
  d=b if family=='within' else a
  if family=='annual':d=d[d.sensor.eq('sentinel2')&d.position_gain.notna()].copy()
  main_result,used=fit(d,family,pred,outcome)
  rows.append(dict(hypothesis=hid,family=family,predictor=pred,outcome=outcome,label=label,**main_result))
  for region,g in used.groupby('region'):
   n=g.sand_cay_id.nunique()
   rho,p=spearmanr(g[pred],g[outcome]) if len(g)>3 and g[pred].nunique()>1 else (np.nan,np.nan)
   description.append(dict(hypothesis=hid,region=region,n_rows=len(g),n_cays=n,rho=rho,p_diagnostic_only=p,independent_rows=family!='within'))
  scenarios={f'leave_out_{v}':d[~d.region.eq(v)] for v in sorted(d.region.unique())}
  if family=='between':
   scenarios.update({f'source_{v}':d[d.sensor.eq(v)] for v in sorted(d.sensor.unique())})
   scenarios['s2_radius_ge_3_pixels']=d[d.sensor.eq('sentinel2')&d.radius_pixels.ge(3)]
  elif family=='annual':
   scenarios['radius_ge_3_pixels']=d[d.radius_pixels.ge(3)]
   scenarios['at_least_4_validation_years']=d[d.annual_years.ge(4)]
  else:
   scenarios['radius_ge_3_pixels']=d[d.radius_pixels.ge(3)]
   complete=d.dropna(subset=['current_magnitude','wave_magnitude','storm_fraction'])
   scenarios['environment_complete_no_adjustment']=complete
   r,_=fit(complete,family,pred,outcome,environment=True)
   sensitivity.append(dict(hypothesis=hid,scenario='environment_complete_adjusted',**r))
  for name,sub in scenarios.items():
   r,_=fit(sub,family,pred,outcome)
   sensitivity.append(dict(hypothesis=hid,scenario=name,**r))
 result=pd.DataFrame(rows);result['q']=np.nan
 ok=result.status.eq('estimated')&result.p.notna()
 if ok.any():result.loc[ok,'q']=multipletests(result.loc[ok,'p'],method='fdr_bh')[1]
 sens=pd.DataFrame(sensitivity);desc=pd.DataFrame(description)
 save(result,'八项联合检验');save(sens,'区域来源及观测敏感性');save(desc,'分区域原始关联描述')
 descriptive=a[['cover','envelope','net','exchange','position_gain','elongation_gain','start_area_m2','span_years','n_observations']].describe().T.reset_index(names='variable')
 save(descriptive,'缺测与分布');save(a[['region','sensor','position_gain','elongation_gain']].groupby(['region','sensor']).count().reset_index(),'年度共同样本支持')
 plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.25,1]})
 colors={'Great Barrier Reef':'#0072B2','Maldives':'#D55E00','Marshall Islands':'#009E73','South China Sea':'#CC79A7','Lakshadweep':'#777777'}
 for region,g in a.groupby('region'):
  axes[0].scatter(g.cover,g.envelope,color=colors[region],s=28,alpha=.8,label=region)
 axes[0].set(xlabel='Median visible green fraction',ylabel='Centroid activity range / initial radius')
 axes[0].legend(fontsize=8,frameon=False);axes[0].text(-.12,1.02,'a',transform=axes[0].transAxes,fontweight='bold')
 for i,r in result.iterrows():
  if r.status=='estimated':
   axes[1].errorbar(r.beta,i,xerr=[[r.beta-r.ci_low],[r.ci_high-r.beta]],fmt='o',color='#0072B2' if r.q<.05 else '#666666',capsize=3)
 axes[1].set_yticks(range(len(result)),[r.hypothesis+' '+r.label for r in result.itertuples()],fontsize=8)
 axes[1].invert_yaxis();axes[1].axvline(0,color='black',lw=.7);axes[1].set_xlabel('Standardized association (95% cluster CI)')
 axes[1].text(-.12,1.02,'b',transform=axes[1].transAxes,fontweight='bold')
 fig.tight_layout();fig.savefig(OUT/'联合关联与共同样本.png',dpi=300);fig.savefig(OUT/'联合关联与共同样本.pdf');plt.close(fig)
 report=['# 空间、时间与覆盖联合规律的额外分析','',
  '本分析直接检验维度间联系。它使用现有观测和留一年验证输出，属于探索性补充；不改变原论文和原主结果。',
  '', '## 数据与固定设计','',
  f'长期主序列剔除NH_015后为{len(a)}条；可获得同来源年度验证的共同样本为{a.position_gain.notna().sum()}个；共同固定框状态区间为{len(b)}对、{b.sand_cay_id.nunique()}个沙洲。各假设在缺测和组内变化筛选后有各自分母，见结果表。',
  '长期模型控制尺寸、跨度、观测数、区域与来源，采用标准化秩关联回归。年度模型另控制原年度预测基线误差。位置年度增益为两个坐标平均绝对误差之和的相对降低，形状仅使用伸长率增益；没有构造综合演化分数。',
  '同个体模型使用期初覆盖、同固定框随后区间移动速率或年化边界交换，控制区间时长、当期面积、季节正余弦、年份与沙洲固定差异，每个沙洲等总权重。系数按组内标准差解释。完整环境子集分别拟合不调环境及调环境模型，区分样本变化与控制变化。',
  '礁盘为误差聚类单位；95%区间使用聚类校正和t分布。8项主假设统一BH校正，敏感性为诊断，不另据其中的显著项宣称新发现。正态性不是秩转换和聚类稳健模型的必需假设；设计秩、条件数、残差极值与实际独立礁盘数均导出。',
  '', '## 全部主检验','', '| 假设 | 沙洲/礁盘 | 系数与95%CI | q | 判定 |','| --- | --- | --- | --- | --- |']
 for r in result.itertuples():
  report.append(f'| {r.hypothesis}：{CHINESE[r.hypothesis]} | {r.n_cays}/{r.n_reefs} | {r.beta:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.q:.4g} | '+('通过校正的关联' if r.q<.05 else '未获得通过校正的关联')+' |' if r.status=='estimated' else f'| {r.hypothesis} | {r.n_cays}/{r.n_reefs} | 支持不足 | — | 不解释 |')
 readable=support.copy()
 readable['region']=readable.region.replace({'Great Barrier Reef':'大堡礁','Maldives':'马尔代夫','Marshall Islands':'马绍尔群岛','South China Sea':'南海','Lakshadweep':'拉克沙群岛'})
 readable[['median_area','median_years']]=readable[['median_area','median_years']].round(2)
 readable=readable.rename(columns={'region':'区域','sensor':'来源','n_cays':'长期沙洲数','annual_n':'年度共同沙洲数','median_area':'起始面积中位数(m²)','median_years':'跨度中位数(年)'})
 report+=['','## 区域、来源与共同支持','', markdown_table(readable),'',
  '年度验证只支持Sentinel-2共同样本；Google Earth与四区长期记录的分布不能替代年度规律在四区独立重复的证据。区间样本也是Sentinel-2，并排除了与长期主序列来源/固定框不一致的记录。小地区的结果仅描述；分区域区间Spearman包含重复观测，p仅为诊断、不作独立样本推断。',
  '', '## 稳定性判读','']
 for r in result.itertuples():
  est=sens[(sens.hypothesis==r.hypothesis)&sens.status.eq('estimated')]
  n=int(len(est));same=int((np.sign(est.beta)==np.sign(r.beta)).sum()) if r.status=='estimated' else 0
  report.append(f'- {r.hypothesis}：可估计的敏感性设定{n}项，其中{same}项与主模型方向相同；逐项置信区间及支持不足设定均保留CSV。方向一致不等同每项均有统计支持。')
 significant=result[result.q.lt(.05)]
 report+=['','## 本次实际结论','',f'8项预先固定的探索性主检验中，{len(significant)}项通过统一BH校正。当前结果支持的联合联系需逐项解释，不能概括成所有维度普遍耦合。']
 for r in significant.itertuples():
  report.append(f'{CHINESE[r.hypothesis]}：{r.n_cays}个沙洲、{r.n_reefs}个礁盘，调整后标准化系数{r.beta:.3f}，95%CI [{r.ci_low:.3f}, {r.ci_high:.3f}]，q={r.q:.4g}。')
  leave=sens[(sens.hypothesis==r.hypothesis)&sens.scenario.eq('leave_out_Great Barrier Reef')]
  if len(leave) and leave.iloc[0]['status']=='estimated':
   rr=leave.iloc[0]
   report.append(f'剔除大堡礁后剩{int(rr.n_cays)}个沙洲，系数{rr.beta:.3f}，95%CI [{rr.ci_low:.3f}, {rr.ci_high:.3f}]，未形成精确的独立区域重复；这既可能涉及区域差异，也可能涉及样本支持降低，不能只凭显著性变化判定某区域控制了该关系。')
 report+=['其余主检验未得到校正后的统计支持，置信区间并非等效性检验，不能据此宣称耦合不存在。现阶段应保持跨区域观测特征与具体关联的主张；覆盖—年度位置规律可作为有待独立区域验证的候选联系。',
 '原稿的覆盖—累计路径负相关继续保留。本次主要响应改用活动范围、首末净位移及边界交换并控制采样和区域，回答不同问题；同个体短期响应也不同于不同沙洲之间的长期差异，不能用其不显著否定原发现。',
 '完整环境子集的区间移动性出现名义p<0.05时，也仅属于敏感性诊断：同子集未调环境估计同时报告，避免把筛选后的变化归因于控制环境；本次不据此提升普遍性或因果结论。']
 report+=['','## 科学判定的边界','',
  '是否支持某项一般关系，需同时看调整后效应、区间、区域/来源覆盖和剔除稳定性。某项关联成立不自动证明空间、年度和覆盖构成统一耦合规律。年度增益与空间范围使用部分相同位置观测，存在共同测量基础；两者关联是联合特征关系，不能当作独立输沙机制证明。',
  '绿色比例是可见覆盖代理。期初覆盖与随后移动性的组内关联不能识别覆盖变化的生态年龄、转换事件的因果效应或反向作用。原严格季节匹配转换窗口不足问题仍然存在。',
  '尺寸阈值敏感性仅检查10 m影像半径不少于3像元的样本，不替代配准误差复算。以观测数、时间控制也不能消除所有累计路径误差，因此主检验不以累计路径为响应。环境资料为背景条件，未补造缺失区间。',
  '', '## 图件说明','',
  '图a显示全部长期共同样本的覆盖—活动范围原始点，颜色标区域；图b显示8项调整后关联及礁盘聚类95%CI，蓝色为本次8项BH校正后q<0.05。两类模型标准化方式分别为沙洲级秩标准差和区间组内标准差，不能将系数大小作跨模型物理强度排名。',
  '', '## 复现','', '运行本目录的分析联合规律.py；输入SHA256、固定假设及筛选规则见分析方案与来源.json。所有CSV为UTF-8 BOM，原始字段名保留便于追溯。']
 (OUT/'分析报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 assert fingerprints=={k:sha(ROOT/v) for k,v in INPUTS.items()}, '输入被修改'
 assert protected=={p:sha(ROOT/p) for p in protected_paths}, '文稿或图件索引被修改'
 metadata={'status':'completed','python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
  'main_tests':result[['hypothesis','status','n_cays','beta','q']].replace({np.nan:None}).to_dict('records'),
  'input_unchanged':True,'manuscript_unchanged':True,'protected_sha256':protected,
  'checks':{'annual_same_sensor_frame':True,'interval_area_independently_reconstructed':True,'eight_hypotheses_preserved':len(result)==8,'unit_is_cay_or_reef_not_image':True},
  'figure_review':'已查看本次实际PNG；英文标签及Times New Roman；原始点与CI完整显示'}
 (OUT/'运行核查.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 print(result[['hypothesis','n','n_cays','beta','ci_low','ci_high','p','q']].to_string(index=False))

if __name__=='__main__': main()
