# 星石本地识别运行资源

正式识别使用 `agent/utils/` 下的 Python 实现。此目录只提供 `StarFieldResolver`
读取的 `star_catalog.json` 和 `ocr_aliases.json`；目录和别名保留原有 YuanStar
映射。Python 算法及这两份资源随 agent 一同打包，测试夹具不进入安装包。
Python 移植来源和关键兼容规则见 [星石 OCR 说明](../../../docs/star-local-ocr-shadow-checkpoint.md)。

名称仅按目录和别名精确匹配到统一名称，未知名称仍停止确认。名称清洗和等级
判断同时接受“级/級”，负等级同时拒绝“负/負”。字段识别复用现有 Context
的默认 OCR 模型（简繁资源均配置 PP-OCRv6/small），不做全局繁简转换。
经验页按图标颜色分类；选项卡、完整排列和内容覆盖均确认后，缺项才可补零
（含明确空页）。有图标但数量未知或种类重复时仍停止确认。

正式入口为 `👆 百宝箱 → 星石背包整理`，调用已验收的本地完整OCR及结构化自动同步。
旧截图上传任务及其专属开关已移除；请使用具有 `star:sync:write` 权限的账号连接码。
需要 720×1280 模拟器画面、支持真实 end_hold 的当前 Controller，以及已加载的
Maa OCR Resource。任务复用当前 Context，不创建第二个 Resource 或 Tasker。
PlayCover、其他分辨率和跨平台设备行为尚未实机验收。

结果位于当前 MaaYuan 目录的 `debug/star-local-ocr/<run_id>/`；
`latest.json` 仅指向最近运行。schemaVersion=2 保留 v1 的主星/辅星实例字段，
经验部分采用 YuanStar 的 `orangeCount`、`purpleCount`、`whiteCount`、
`complete`、`warnings`。整体 `complete` 要求主星和辅星完整、背包容量识别可信，
且游戏当前数量与最终实例总数一致。经验部分单独记录完成状态，待补全的品质
数量为 `null`，详细证据见 `experience-results.json`。仅显式识别到 0，或完整确认
经验页后缺失的种类，才输出 0。
满足传输条件的结果转换为精简 `maayuan-star-inventory` JSON 后自动发送到
`/open-api/star/inventory-sync`。请求始终带有 `experience.orange`、`experience.purple`
和 `experience.white`，各自保留确认的整数或显式 `null`，三项均为 `null` 时也会发送。
截图和诊断字段保留在本地。同步失败保留完整结果与原 `scan_id`，下次任务先重试原批次。

当前 YuanHub 接收端尚未支持经验数量 `null`，带 `null` 的请求可能返回
`422 star_sync_invalid`。需要接收端完成请求解析、服务端校验和持久化逻辑的适配，
才能同步含未知经验数量的批次。

同步成功会显示实际主辅星数量及背包容量。符合条件的密探佩戴自动保留，
旧养成计划仍由用户在 YuanHub 选择继承。连接码和传输目标沿用通用连接配置。

普通运行只导航当前入口一次；三入口覆盖验证是独立诊断参数。
逐页不可变检查点保存原始字段及来源截图；日常关闭重复 Context 全量 JSONL
和 debug crops。发生错误保留本次目录、有效实例和 error.json。
