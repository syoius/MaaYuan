# 诸葛亮智能圈

圈的语义由攻略说明确定。吕布圈是专属形态按钮；诸葛亮DOT圈是从普攻、下拉、技能中选一项的策略。通用编译器的原始圈token仍是吕布语义，必须先分流到专用适配器。

仓库根目录下的实现：

- `agent/custom/action/dragon_smart.py`：注册 `DragonSmartAction`，连续稳定采样后选择作业分支。
- `agent/custom/action/dragon_smart_vision.py`：背景、龙气图标和技能能量识别。
- `assets/resource/base/dragon_smart.json`：坐标和识别阈值；`dragon_smart_*.png` 模板及原图变体位于同资源的 `image/copilot/`。
- `tools/copilot/dragon_smart/build_test_job.py`：10回合观察作业适配示例，每回合1号位圈、其他四人普攻。
- `tools/copilot/dragon_smart/test_dragon_smart.py`：离线截图、合成状态、连续帧分支和原生引擎加载测试，不连接游戏。

优先级：龙脉外先看技能能量（足够上滑，不足普攻），不依赖龙气；龙脉内才按0下拉、1普攻、至少2下拉。无法确定所需状态时停止，不以一次未匹配推定相反状态。明确指定下拉的机制回合执行固定下拉，不应启用圈。

`DragonSmartAction` 参数需包含当前节点名 `node`、站位 `position`，以及 `branches` 中 `normal`、`defense`、`ultimate` 三个实际存在的动作节点名。条件节点只派发，不点击；每次调用重新选择分支，不能沿用上次状态。失败应接终止提示节点。

识别当前按720×1280、诸葛亮4点技能能量校准。背景和能量模板来自用户两张朱然截图；部分能量状态仅有合成测试，不能据此宣称跨敌人或实战准确率。能量足够不包含沉默、封技等额外技能限制。原 `DragonRestart` 仅检查图标命中，日志“2龙气”不能当作准确计数或龙脉状态证明。

部署需一起安装两个Python模块、`action/__init__.py`导入注册、JSON配置、全部模板（含变体）和新版作业。只刷新作业不能安装CustomAction；只重启旧安装也不会更新源码。优先把完整作业放到实际使用客户端的 `resource/copilot-cache/`，由用户刷新并激活；基础pipeline内只能有一个展开作业，不覆盖 `copilot_config.json`。

模板应从ADB直接抓取的720×1280游戏截图提取；电脑桌面截图即使缩放到相同尺寸，裁边仍会导致坐标错位。v2.1已使用原始截图校准，并以另截的一帧检查龙脉、龙气和能量识别；旧裁图模板仅作为兼容变体，变体不能重复计为不同区域的证据。

先核对实际客户端位置；停止并授权部署后再备份更新。2026-11-02用户反馈：v2.4.1的10回合观察测试无问题；不代表所有关卡和状态都验证完毕。

回合数支持当前/上限格式，例如1/10取1。新代码和观察作业说明见仓库tools/copilot/dragon_smart/README.md。
