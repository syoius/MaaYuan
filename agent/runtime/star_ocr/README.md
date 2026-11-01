# 星石本地识别运行资源

这些 JS 和 JSON 是 YuanStar 纯函数的固定快照。来源、源文件和生成后哈希见
`provenance.json`。更新时使用开发工具的 `prepare` 子命令，运行任务时不编译
TypeScript，不访问源工作区，也不访问审查目录。

`install.py` 和 `install4release.py` 会复制整个 agent 资源目录，并用
`tools/ci/bundle_star_ocr_node.py` 携带 Node v24.19.0、许可证及校验清单。
支持 Windows、Linux、macOS 的 x64/arm64；发布矩阵明确传入目标架构。
正常任务只启动包内 `runtimes/star-ocr-node/node[.exe]`，缺少时在导航前失败，
不会偷偷改用系统 Node。Android 包目前不携带 Node，本功能尚不支持 Android。

任务为 `星石背包整理（本地识别）`，默认不勾选，固定本地 Shadow。
需要 720×1280 模拟器画面、支持真实 end_hold 的当前 Controller，以及已加载的
Maa OCR Resource。任务复用当前 Context，不创建第二个 Resource 或 Tasker。
PlayCover、其他分辨率和跨平台设备行为尚未实机验收。

结果位于当前 MaaYuan 目录的 `debug/star-local-ocr/<run_id>/`；
`latest.json` 仅指向最近运行。schemaVersion=2 保留 v1 的主星/辅星实例字段，
经验部分采用 YuanStar 的 `orangeCount`、`purpleCount`、`whiteCount`、
`complete`、`warnings`。整体 `complete` 需要三个分区全部完成；v1 的 complete
只表示主星和辅星，消费者不可将其解释为经验数量已完成。失败数量为 null，
只有真实识别到的 0 才输出 0。此结果尚未接入任何云端导入契约。

普通运行只导航当前入口一次；三入口覆盖验证是独立诊断参数。
逐页不可变检查点保存原始字段及来源截图；日常关闭重复 Context 全量 JSONL
和 debug crops。发生错误保留本次目录、有效实例和 error.json。
