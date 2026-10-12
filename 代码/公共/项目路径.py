from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
def configure_imports():
    for folder in (ROOT / "代码").iterdir():
        if folder.is_dir() and folder.name != "__pycache__" and str(folder) not in sys.path:
            sys.path.insert(0,str(folder))
configure_imports()
