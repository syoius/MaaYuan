# 星石本地 OCR

正式入口是「百宝箱 → 星石背包整理」。简繁资源调用同一个 `StarInventorySync`，
通过 `FullShadowSession` 扫描主星、辅星和经验星曜。识别复用当前 Maa Context
及其 OCR 模型，几何、字段解析、品质判定和经验数量汇总在 Python 中完成。

主辅星按物理行列累计实例，同属性的独立实例分别保留。普通前进使用
600 ms 滑动和 600 ms 末端保持，识别前等待画面稳定。尾页以位置证据合并，
歧义时使用有界恢复，确认到底后才完成分区。

## 移植来源

几何、网格、品质和经验页后处理移植自 YuanStar 结构化 OCR
`e86a05aa2508e14b9a1e6907de19359cfdb78bfb` 的 `web/src/structured/` 实现，
包括 `profiles.ts`、`main-grid.ts`、`card-completeness.ts`、
`quality-postprocess.ts`、`experience-geometry.ts` 与 `experience-postprocess.ts`。
移植保留原有的采样、舍入和分类边界。在 720×1280 画面的固定内容区中，
只有完整圆盘和等级框、且名称框仅越过底边 1 px 的候选才允许裁去该像素，
原始名称框与修正证据随几何结果保留。

公开回归使用局部背包裁图、合成画布和固定 golden 数据。旧 F1 的位置去重、
同属性实例保留及冲突保护由 `test_star_four_row_inventory.py` 的公开尾页样本覆盖。

运行资源、传输条件和 YuanHub 兼容限制见
[星石运行说明](../agent/runtime/star_ocr/README.md)。
