"""等待 Python 下载 worker 完成后自动执行潮位后处理链。

解压→配置→潮位敏感性→重跑统一模型与事件分析，全部退出码写入日志。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = Path(__file__).with_name("fes_postprocess_py.log")


def main(expected_workers: int) -> None:
    download_log = Path(__file__).with_name("fes_download_py.log")
    LOG.write_text("waiting\n", encoding="utf-8")
    while True:
        done = 0
        if download_log.is_file():
            done = sum(
                1
                for line in download_log.read_text(encoding="utf-8").splitlines()
                if line.startswith("WORKER_DONE")
            )
        if done >= expected_workers:
            break
        time.sleep(30)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write("downloads_complete\n")
    for script in [
        "构建潮位配置.py",
        "潮位敏感性.py",
        "分析已成洲沙洲形态响应.py",
        "分析台风事件前后沙洲响应.py",
    ]:
        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{script} exit={result.returncode}\n")
            if result.returncode != 0:
                handle.write(result.stdout[-2000:] + "\n" + result.stderr[-2000:] + "\n")
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write("POST_DONE\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
