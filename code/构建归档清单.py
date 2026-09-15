"""为 outputs 快照生成不可变归档清单（合同哈希 + 脚本哈希 + 文件指纹）。

用法：python 构建归档清单.py --archive-id <id>
快照目录应为 archive/<id>/，由 Copy-Item 从 outputs/ 复制得到。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path


SCRIPTS = [
    "分析范围.py",
    "分析已成洲沙洲形态响应.py",
    "分析台风事件前后沙洲响应.py",
    "分析植被形态台风关系.py",
    "对齐流场波浪与沙洲变化.py",
    "构建沙洲观测与变化表.py",
    "潮位敏感性.py",
    "绘制当前结果图.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-id", required=True)
    args = parser.parse_args()
    dest = root / "archive" / args.archive_id
    if not dest.is_dir():
        raise SystemExit(f"archive directory missing: {dest}")
    contract_path = root / "outputs" / "分析合同核查.json"
    contract = (
        json.loads(contract_path.read_text(encoding="utf-8"))["contract_sha256"]
        if contract_path.is_file()
        else ""
    )
    manifest = {
        "archive_id": args.archive_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "contract_sha256": contract,
        "scripts": {name: sha256(root / name) for name in SCRIPTS if (root / name).is_file()},
        "files": {
            str(path.relative_to(dest)): sha256(path)
            for path in sorted(dest.rglob("*"))
            if path.is_file() and path.name != "archive_manifest.json"
        },
        "notes": (
            "Immutable snapshot of outputs/. Future reruns write to outputs/; "
            "compare against this manifest to detect what changed."
        ),
    }
    (dest / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"archive_manifest: {dest / 'archive_manifest.json'}")
    print(f"files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
