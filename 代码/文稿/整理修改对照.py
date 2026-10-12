import zipfile
"""登记逐面板处置及当前稿对原稿的差异，不重新生成正文。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'公共'))
from 项目路径 import ROOT
import pandas as pd
import difflib,json,hashlib
BASE=ROOT.parent/'归档/重构前'
rows=[
('1a','主图1a','研究区位置是四区真实图谱的必要入口'),
('1b','补图S1a','来源尺寸支持可检测性判断；不充当独立演化差异证据'),
('1c','主图2a–c重绘','连续活动范围、净位移、累计路径与边界交换独立展示'),
('1d、1e','主图2d、e；补充S1区域表','区域事实前移；独立中位数不组成收支'),
('2a、2b','正文3.2；补充S2分类定义','净面积分箱保留为结果分层；主图2改为连续空间轨迹'),
('2c、2d','补图S2a–d重绘','阈值及采样控制保留，并加入NH_015配准筛查'),
('3a、3b、3c','主图4b、c、a','真实轮廓先行，统计验证随后'),
('4a、4b','方法及补充S4定义','保留坐标约定，停止重复示意图'),
('4c','主图5a','四方向项完整保留；剔除混合网格后横轴海流关联通过校正，说明样本敏感性'),
('4d','补充S4长轴探索性表及原源表','长轴长度与质心响应区分，完整五项按来源保留，不作主结果'),
('4e','补图S4a–c重绘','窗口、像元扰动与沙洲剔除独立验证'),
('5a、5b','主图6a、b','沙洲级及同影像覆盖关联保留'),
('5c','主图3b、c','补齐六方向及同个体时序，旧不完整网络停用'),
('5d','补充S5模型表及补图S5b','模型历时及过预测校准保留，不当发育年龄'),
('5e','主图3a','改为同个体真实状态影像'),
('6a','补充S7拟合参数表及原源表','年度拟合路径不代替实测轮廓'),
('6b、6c','主图6c、5b','状态分布与高波浪暴露分别归位'),
('6d','补充S7时序统计表','保留地区月份检验，取消重复柱图'),
('7a','补图S6a、b','尺寸几何比例和可检测性控制'),
('7b','补充S1覆盖分层表','可靠数值保留，避免重复主图6'),
('7c','主图2d–g','区域事实前移，不推区域机制分型'),
('7d','补充S7趋势摘要及原结果表','未通过FDR，不作中心结果图'),
('原补图S1、S3','补图S7b、a及补充S7','匹配事件与风场阴性验证保留'),
('原补图S2','补充S7 ENSO、DHW表','气候检验保留，取消独立等权图'),
('原补图S4、S5','补图S1b、主图2d–g','校准控制和区域事实各归其位'),
('原补图S6','补充S7趋势摘要及源表','候选筛查不支持普适稳定化'),
('新增形状审查','补图S8及补充S8表；正文3.2短段','早晚伸长率差异及季度对照；C统一主体Crofton，年度原口径对照见S3，不新增分类')]
d=pd.DataFrame(rows,columns=['旧图或面板','当前位置','取舍原因']);d['执行状态']='已落实；原件及源表保留'
def table(d):return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in x)+' |' for x in d.itertuples(index=False,name=None))
d.to_csv(ROOT/'图表去留审计表.csv',index=False,encoding='utf-8-sig')
(ROOT/'图表去留审计表.md').write_text('# 图表逐面板处置\n\n当前六幅主图、八幅补图。S8独立承接伸长率补充，不删除主图有效面板。原件和可靠结果保留。\n\n'+table(d)+'\n',encoding='utf-8')
for name,target in [('论文初稿.md','修改前后对照.diff'),('论文补充材料.md','补充材料修改前后对照.diff')]:
 old=(BASE/name).read_text(encoding='utf-8');new=(ROOT/name).read_text(encoding='utf-8')
 (ROOT/target).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='原稿/'+name,tofile='当前/'+name)),encoding='utf-8')
assert (BASE/'论文初稿.md').read_text(encoding='utf-8').split('## 参考文献',1)[1]==(ROOT/'论文初稿.md').read_text(encoding='utf-8').split('## 参考文献',1)[1]
baseline=json.loads((BASE/'第一阶段审计/复核结果.json').read_text(encoding='utf-8'))['original_sha256']
assert all(hashlib.sha256((BASE/p).read_bytes()).hexdigest()==v for p,v in baseline.items())
# 人工审校的变更解释为当前权威，本工具只生成可复核差异。
review_backup=ROOT.parent/'归档/定向完善前.zip'
if not review_backup.exists():
    review_backup=ROOT.parent/'归档/讨论图件修改前.zip'
with zipfile.ZipFile(review_backup) as current_backup:
    for name,target in [('论文初稿.md','修改前后对照.diff'),('论文补充材料.md','补充材料修改前后对照.diff')]:
        prior=current_backup.read(name).decode('utf-8').replace('\r\n','\n');now=(ROOT/name).read_text(encoding='utf-8')
        (ROOT/target).write_text(''.join(difflib.unified_diff(prior.splitlines(True),now.splitlines(True),fromfile=review_backup.stem+'/'+name,tofile='当前/'+name)),encoding='utf-8')
print('逐面板审计及正文/补充差异完成；原件和参考文献一致')
