"""测试包。

`tests/__init__.py` 存在的理由不只是"让 unittest 认出这是个包"，
还要让下面这条命令在任何工作目录下都能跑通：

    python -m unittest discover -s tests -t .

（`-t .` 把顶层目录钉在项目根，`replayprobe` 包才 import 得到。）

顺手也把项目根塞进 `sys.path`，这样即使有人漏了 `-t .`、
或者直接用 pytest 跑，也不会死在 `ModuleNotFoundError` 上 ——
**一个跑不起来的测试套件，等于没有测试套件。**
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
