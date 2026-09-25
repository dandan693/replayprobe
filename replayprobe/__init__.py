"""replayprobe —— Agent 轨迹回放与分叉诊断。

一次运行只能说明「这次它做了什么」。
**录下来、改一个变量、再放一遍，才能说明「换掉什么会让它走上另一条路」。**

对照 faultprobe / evalguard 的分工：

    evalguard    答得对不对（质量）
    faultprobe   坏了会怎样（鲁棒性）
    replayprobe  改动之后走的还是同一条路吗（行为回归）

三者共用同一份领域数据（`retail_truth.db`），所以三张成绩单可以并排看。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
