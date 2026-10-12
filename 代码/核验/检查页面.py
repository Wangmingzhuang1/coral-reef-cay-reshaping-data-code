
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / "公共"))
from 项目路径 import ROOT as PROJECT_ROOT, configure_imports
configure_imports()
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
ROOT = PROJECT_ROOT / "排版核验"
for folder in [ROOT/'正文',ROOT/'补充材料']:
    doc=fitz.open(folder/'文档.pdf');thumbs=[]
    for stale in folder.glob('page-*.png'):
        if int(stale.stem.split('-')[-1]) > len(doc):
            stale.unlink()
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.save(folder/f'page-{i+1}.png')
        im.thumbnail((280,395));tile=Image.new('RGB',(300,425),'#E8E8E8');tile.paste(im,((300-im.width)//2,22));ImageDraw.Draw(tile).text((10,6),str(i+1),fill='black');thumbs.append(tile)
        if '图 1｜' in page.get_text() or '图 3｜' in page.get_text():print(folder.name,i+1,page.get_text()[:50])
    for k in range(0,len(thumbs),12):
        sheet=Image.new('RGB',(1200,1275),'white')
        for j,tile in enumerate(thumbs[k:k+12]):sheet.paste(tile,((j%4)*300,(j//4)*425))
        sheet.save(folder/f'预览-{k//12+1}.png')
    print(folder.name,len(doc))
