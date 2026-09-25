"""让 `python -m replayprobe ...` 能直接跑。

不装任何依赖的前提下，这是最省事的上手方式 ——
`git clone` 之后立刻就能 `python -m replayprobe check`，
不用先 `pip install -e .`。**上手成本每高一档，愿意跑它的人就少一半。**
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
