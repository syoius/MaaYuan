# 星石本地识别运行资源

这些 JS 和 JSON 是 YuanStar 纯函数的固定快照。来源、源文件和生成后哈希见
`provenance.json`。更新时使用开发工具的 `prepare` 子命令，运行任务时不编译
TypeScript，不访问源工作区，也不访问审查目录。

`star-postprocess.js`、`experience-geometry.js`、`experience-postprocess.js`
在上游快照上保留 MaaYuan 的局部兼容补丁，记录于
`provenance.json` 的 `localChanges`；重新 `prepare` 后须保留这些补丁并更新哈希。
名称仅通过现有 `ocr_aliases.json` 精确匹配到统一名称，未知名称仍停止确认；
名称清洗和等级判断同时接受“级/級”，负等级同时拒绝“负/負”。
复用当前 Context 中的默认 OCR 模型（简繁资源均默认配置 PP-OCRv6/small），
字段识别无需额外切换语言，不做全局繁简转换。
经验页沿用网页端按图标颜色分类和完整结果缺项补零的聚合规则；图标位置
只用于几何定位，不决定类型。只有选项卡、完整排列及内容覆盖均确认后，
才允许缺项补零（含明确空页）；有图标但数量失败、未知或重复类型仍为异常。

`install.py` 和 `install4release.py` 会复制整个 agent 资源目录，并用
`tools/ci/bundle_star_ocr_node.py` 携带 Node v24.19.0、许可证及校验清单。
支持 Windows、Linux、macOS 的 x64/arm64；发布矩阵明确传入目标架构。
正常任务只启动包内 `runtimes/star-ocr-node/node[.exe]`，缺少时在导航前失败，
不会偷偷改用系统 Node。Android 包目前不携带 Node，本功能尚不支持 Android。

正式入口为 `👆 百宝箱 → 星石背包整理`，调用已验收的本地完整OCR及结构化自动同步。
旧截图上传任务及其专属开关已移除；请使用具有 `star:sync:write` 权限的账号连接码。
需要 720×1280 模拟器画面、支持真实 end_hold 的当前 Controller，以及已加载的
Maa OCR Resource。任务复用当前 Context，不创建第二个 Resource 或 Tasker。
PlayCover、其他分辨率和跨平台设备行为尚未实机验收。

结果位于当前 MaaYuan 目录的 `debug/star-local-ocr/<run_id>/`；
`latest.json` 仅指向最近运行。schemaVersion=2 保留 v1 的主星/辅星实例字段，
经验部分采用 YuanStar 的 `orangeCount`、`purpleCount`、`whiteCount`、
`complete`、`warnings`。整体 `complete` 需要三个分区全部完成、容量识别可信且游戏当前数量与最终实例总数一致；v1 的 complete
只表示主星和辅星，消费者不可将其解释为经验数量已完成。失败数量为 null，
仅显式识别到 0，或完整确认经验页后缺失的种类，才输出 0。完整结果转换为精简 `maayuan-star-inventory` JSON 后自动发送到
`/open-api/star/inventory-sync`，不发送截图或诊断字段。同步失败保留完整结果与原
`scan_id`，下次任务先重试原批次；识别不完整不会上传。

同步成功会显示实际主辅星数量及背包容量。符合条件的密探佩戴自动保留，
旧养成计划仍由用户在 YuanHub 选择继承。连接码和传输目标沿用通用连接配置。

普通运行只导航当前入口一次；三入口覆盖验证是独立诊断参数。
逐页不可变检查点保存原始字段及来源截图；日常关闭重复 Context 全量 JSONL
和 debug crops。发生错误保留本次目录、有效实例和 error.json。
