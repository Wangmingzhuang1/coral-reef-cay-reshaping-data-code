"""打包本轮文稿、代码、派生表及图源，不包含第三方影像或掩膜。"""
import csv
import hashlib
import importlib.metadata
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '公共'))
from 项目路径 import ROOT


def main():
    files = set(ROOT.glob('代码/**/*.py'))
    files.update(ROOT.glob('tests/**/*.py'))
    files.update(ROOT.glob('outputs/**/*.csv'))
    files.update(ROOT.glob('outputs/**/*.json'))
    files.update(p for p in (ROOT/'outputs/figures').rglob('*')
                 if p.is_file() and p.suffix.lower() in {'.png', '.pdf', '.svg', '.tiff'})
    joint = ROOT/'额外分析_空间时间覆盖联合规律'
    files.update(p for p in joint.glob('*') if p.suffix in {'.py', '.csv', '.json'})
    files.update(ROOT/p for p in ['论文初稿.md', '论文补充材料.md', '运行论文分析.py',
                                 '分析排除清单.csv', '修改前后对照.md',
                                 '修改前后对照.diff', '补充材料修改前后对照.diff'])
    files.update(p for p in (ROOT/'论文').glob('*.docx') if not p.name.startswith('~$'))
    prefixes = [(str(ROOT)+'\\', ''), (str(ROOT).replace('\\', '/')+'/', ''),
                (str(ROOT.parent)+'\\', '../'),
                (str(ROOT.parent).replace('\\', '/')+'/', '../')]
    payload = {}
    sanitized = set()
    for path in sorted(files):
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        if path.suffix in {'.csv', '.json', '.md', '.diff', '.py'}:
            content = data.decode('utf-8-sig')
            for prefix, replacement in prefixes:
                content = content.replace(prefix, replacement)
                content = content.replace(prefix.replace('\\', '\\\\'), replacement)
            # 清理历史机器用户名路径；保留文件名与目录尾段作为来源提示。
            import re
            content = re.sub(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n",]+[\\/]+',
                             '{USER_HOME}/', content)
            encoded = content.encode('utf-8-sig' if path.suffix=='.csv' else 'utf-8')
            if encoded != data:
                sanitized.add(relative)
            data = encoded
        payload[relative] = data
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'statsmodels', 'rasterio', 'opencv-python',
                 'scikit-image', 'matplotlib', 'python-docx', 'Pillow', 'PyMuPDF']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed under this package name'
    payload['依赖版本.json'] = json.dumps({'python':sys.version,'packages':versions},
                                         ensure_ascii=False,indent=2).encode('utf-8')
    payload['运行说明.txt'] = '''本轮审稿复核数据与代码（本地完整交付包；公开数据版本见归档记录）

解压后，以包含“代码”和“outputs”的目录为工作目录。
安装依赖版本.json所列科学计算库，然后执行：
python 代码/控制验证/复核审稿主要问题.py --check-derived

上述入口无需原始影像，核对前向预测的训练时间隔离、两组七指标BH校正、
1490个状态对的比例恒等式、90条网格一致位置记录，以及随包清单的所有SHA-256。
全部模型项、阴性结果、逐景预测和图源保留于outputs；文稿为当前两份Markdown及Word。

重新计算光谱、掩膜或原始主观测表需要未打包的第三方原始影像、逐期掩膜、
标注元数据与环境原始记录；完整本地交付检查还依赖工作区的历史冻结归档。
因此，不把派生表复核称为不依赖原始数据的全流程重建。
现有代码以本目录为论文重构根，原始影像数据放在其父目录data/source_dataset。
获取与许可：Sentinel-2为Copernicus Data Space（https://dataspace.copernicus.eu），
Google Earth历史截图须遵守其使用条款；作者掩膜及标注需随论文另行归档。
ERA5由Copernicus Climate Data Store提供；GLORYS12V1/WAVERYS由Copernicus Marine提供；
IBTrACS由NOAA NCEI提供。代码使用MIT，作者派生表使用CC-BY-4.0。

仅包内副本的个人绝对路径改为相对路径或{USER_HOME}，权威工作区输入不改。
文件清单SHA-256对应包内实际字节；审稿复核结果.json中master_sha256对应
工作区原始未脱敏主表，两者用途不同。清单的packaged_copy_changed标识副本字节变化。
本包不包含原始第三方影像和逐期掩膜，不包含独立控制点或尚未进行的盲重标结果。
'''.encode('utf-8')
    publication_path = ROOT/'outputs/公开归档核验.json'
    if publication_path.exists():
        publication = json.loads(publication_path.read_text(encoding='utf-8'))
        public_link = '公开代码与派生数据：'+publication['release_url']+'\n本地包另含文稿和图件，不能等同于公开数据包。\n\n'
        payload['运行说明.txt'] = public_link.encode('utf-8') + payload['运行说明.txt']
    stream=io.StringIO(newline='')
    writer=csv.writer(stream)
    writer.writerow(['relative_path','size_bytes','sha256','packaged_copy_changed'])
    for name,data in sorted(payload.items()):
        writer.writerow([name,len(data),hashlib.sha256(data).hexdigest(),name in sanitized])
    payload['文件清单.csv']=stream.getvalue().encode('utf-8-sig')
    archive=ROOT/'审稿复核数据代码.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for name,data in sorted(payload.items()):
            bundle.writestr(name,data)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for row in csv.DictReader(io.StringIO(bundle.read('文件清单.csv').decode('utf-8-sig'))):
            data=bundle.read(row['relative_path'])
            assert len(data)==int(row['size_bytes'])
            assert hashlib.sha256(data).hexdigest()==row['sha256']
    print(json.dumps({'archive':str(archive),'files':len(payload),
                      'size_mb':round(archive.stat().st_size/1024**2,2),
                      'all_manifest_hashes_verified':True},ensure_ascii=False))


if __name__=='__main__':
    main()
