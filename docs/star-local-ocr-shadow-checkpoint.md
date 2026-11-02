# 星石四行本地 OCR：开发阶段检查点

> 历史阶段记录，以下“尚未接入”“原上传保持”等描述仅对应当时检查点。当前正式入口、结构化同步及包内运行要求见 `agent/runtime/star_ocr/README.md`；不再使用旧截图上传流程。

当前交付是 opt-in Shadow 开发成果，尚未发布。没有切换线上星石导入或云端上传；经验星曜仅截图。

## 当前执行链

- `StarFullLocalOcrShadow`：统一导航 → 主星四行采集 → 尾部位置合并及到底确认 → 辅星同流程 → 经验截图。必须显式传入 `shadow: true`。
- 导航复用已有 `进入界面-观星`、OCR/TM 入口、`已在界面-观星`、返回节点和主辅星/经验 Tab 节点。根据游戏确定规则，观星重进背包即主星顶部；没有反向列表复位。
- 原生 Swipe 为700ms＋end_hold250ms，等待1000ms截图。几何/行距驱动每次≤30px的自适应，单Context串行ROI OCR。
- 普通页按四行顺序追加，不做逐颗跨图匹配或全局同名去重。仅尾部按像素小窗位置与行列映射去重，保留新增及不满四颗的末行；歧义/字段冲突安全停止。
- `StarLocalOcrShadow`、`StarFourRowShadow`和endhold工具是此前阶段的独立诊断入口。历史评分/稳定性分析不属于新的整包普通扫描链，不能据此启动旧单次试验替代新流程。
- Probe只新增显式 `swipe_stability_probe` 模式；原默认模式、正式orchestration和上传调用保持不变。debug-crop默认仍开启，整包Shadow单独关闭。

已保存的2026-10-31实机结果为主69、辅105、共174；186条观察尾部去重12条，两个分区都确认到底。普通翻页约3.41s，总动作约56.40s。这些是该次运行结果，业务代码没有将这些数量用作库存上限。

## 尚未发布就绪的依赖

1. `geometry_runtime`仍由宿主传入；其几何JS、catalog与aliases快照在本地审查目录`_runtime/`，本检查点不提交这些生成运行资源。Node桥接工具已在Node v24.19.0使用，预处理会使用`stripTypeScriptTypes`。
2. 专用整包宿主仍读取`.ux/audits/star-swipe-endhold-20261030/controller-config.json`，使用本机MuMu ADB安装路径及`127.0.0.1:16384`，输入/截图掩码8/64、720×1280。Action和工具保留D盘输出约束。这些是实验环境设置，不是可独立分发配置。
3. 早期单次宿主还保留固定审查路径、参考图和手动授权门禁。新增动作虽然已注册，但没有接入正常界面的正式星石本地OCR任务入口。
4. 真实图像回归测试读取未提交的`.ux/audits/star-four-row-finite-20261031/`、`star-full-local-ocr-shadow-20261031/`和早期F1夹具。缺少这些夹具或MaaFramework/OpenCV/NumPy环境时，不能把环境缺失视为算法失败，也不能声称真实图回归通过。没有提交截图或模型来强行满足测试。

## 本地检查

使用包含MaaFramework、OpenCV、NumPy的Python，在D盘项目目录运行，TEMP/TMP也须指向D盘。以下仅运行离线/模拟定向检查，不连接设备：

```text
python -B agent/tests/test_star_four_row_inventory.py -v
python -B agent/tests/test_star_full_local_ocr_shadow.py -v
python -B agent/tests/test_star_swipe_calibration.py -v
python -B agent/tests/test_star_four_row_shadow.py -v
python -B agent/tests/test_star_local_ocr_shadow.py -v
```

整包离线回放读取已保存逐页JSON与原图，调用当前`PositionInventory`，核对最终69/105、观察186及去重12。它不重新OCR、不启动游戏、不改写原run。`tools/star_local_ocr_offline.py`为早期F1专用一次性脚本，本检查点不提交，也不被当前整包执行链引用。

## 下一轮最小范围（本轮仅记录）

A. 经验星曜三个数量的MaaYuan端内结构化OCR。

B. 正式任务入口和Pipeline整合。

C. 将geometry/catalog运行资源迁入可正常打包的位置。

D. 验证任务能从MaaYuan正常界面启动，无需专用实验宿主。

E. 保持Shadow默认隔离，不启用云端上传。

结构化数据接收另行对接YuanHub；本阶段不改Backend或YuanHub，不把本地顺序/物理行列当成游戏持久实例ID。
