# 百宝箱诊断原因映射

契约 v2 增量（保留 v1），`schema_version` 仍为 `1`。`issues[].reason` 为可选字符串，匹配 `[a-z][a-z0-9_]{0,63}`，仅使用下表枚举；未知时省略，不输出 `null`。不得包含异常消息、路径或连接码。

契约 v4：`data_review`、`identity_unresolved` 有密探 ID 时按密探拆分；图片档仅缓存最近 200 位密探每人最多 3 张 JPEG，结束时按问题先后每人优先取 1 张相关裁图（命盘问题优先命盘格，其余优先姓名），再补余图、合计最多 3 张，无归属图片仅在没有其他候选时使用，manifest 字段与上传协议不变。

图片档每套命盘仅裁一张覆盖全部 `DISC_CELLS` 的整盘图（命盘问题优先第一套）；失败时优先发送 `role=last_frame` 的最后整屏画面，ROI 与图片尺寸一致，JPEG 逐步降质至单张不超过 512 KiB、不缩放，总数仍最多 3 张；成功或仅待核对时不发送整屏画面。

| reason | 典型 code / stage | 中文说明 | 用户/开发排查提示 |
|---|---|---|---|
| page_timeout | navigation_failed / collector_exception，stage=main/detail/huaji/disc | 等待目标页面超时（stage 即页面） | 资源语言与游戏界面不一致、弹窗遮挡、卡顿 |
| resource_language_mismatch | navigation_failed / collector_exception，stage 为页面 | 资源语言与游戏界面不一致 | 用户所选资源（简中/繁中）与游戏画面文字不符，请让用户切换资源 |
| enter_operator_page_failed | navigation_failed，stage=enter | 无法进入密探界面 | 起始界面不在主界面/导航失败 |
| disc_plan_unknown | collector_exception，stage=disc | 无法确认当前命盘方案 | 命盘页标签 OCR 失败 |
| disc_tip_close_failed | collector_exception，stage=disc | 无法关闭命盘单套提示 | 提示弹窗样式变化 |
| disc_switch_unconfirmed | collector_exception，stage=disc | 命盘方案切换未确认 | 点击未生效/动画过慢 |
| disc_restore_failed | collector_exception，stage=disc | 命盘原配置恢复失败 | 采集后未能切回原命盘 |
| identity_changed_on_reread | collector_exception，stage=main | 数值补读时密探身份不符 | 补读期间页面被切换 |
| page_arrow_failed | navigation_failed，stage=navigation | 密探翻页箭头识别未执行 | 翻页按钮识别失败 |
| list_boundary_unverified | navigation_failed，stage=navigation | 无法复核密探列表边界 | 列表首尾判断失败 |
| page_turn_unconfirmed | navigation_failed，stage=navigation | 密探翻页未确认 | 翻页后身份未变化 |
| return_first_limit | navigation_failed，stage=navigation | 回到首位密探超过安全上限 | 列表过长或回滚失败 |
| current_operator_failed | collector_exception | 无法采集当前密探 | 见同次其他 issue |
| form_identity_unconfirmed | identity_unresolved / collector_exception，stage=identity | 切换形态后身份未确认 | SP/普通形态切换 |
| form_switch_failed | collector_exception | 形态切换或恢复失败 | 陈登/史子眇等双形态 |
| form_tip_close_failed | collector_exception | 无法关闭未解锁形态提示 | 未解锁形态弹窗 |
| name_and_disc_unresolved | identity_unresolved，stage=identity | 姓名 OCR 与命盘均无法确认密探 | 新密探未录入目录、OCR 失败 |
| name_fuzzy_disc_unconfirmed | identity_unresolved，stage=identity | 姓名仅模糊匹配且命盘未能唯一确认 | 新密探/命盘别名缺失（可在 YuanHub 补 aliases） |
| unmatched_in_report | identity_unresolved，stage=checkpoint | 报告中存在未匹配密探 | 同上 |
| section_needs_review | data_review，stage=checkpoint | 部分字段待核对（见 section_status） | 数值/命盘识别不完整 |
| server_marked_review | data_review，stage=preview/commit | 服务端标记部分接受或待核对 | 查看 YuanHub 待核对 |
| screenshot_failed | screenshot_or_ocr_failed，stage=screenshot | 截图失败 | 模拟器/ADB 异常 |
| ocr_failed | screenshot_or_ocr_failed，stage=ocr | OCR 识别异常 | OCR 模型/资源问题 |
| catalog_load_failed | catalog_unavailable，stage=catalog | 本地与远程密探目录均不可用 | 网络 + 内置目录损坏 |
| account_lookup_failed | account_lookup_failed，stage=account | 连接码绑定账号查询失败 | 连接码失效/网络 |
| annotations_failed | annotations_failed，stage=annotations | 养成状态读取失败 | 连接码无读取权限 |
| upload_network_error | upload_network_failed | 上报网络错误 | 网络 |
| upload_http_error | upload_http_failed | 上报 HTTP 错误 | 服务端状态码 |
| upload_rejected_by_server | upload_rejected | 服务端拒收（见 blocking_errors） | 数据不合规 |
| upload_response_invalid | upload_response_unverified | 上报响应无法验证 | 响应格式异常 |
| upload_auth_missing | collector_exception | 在线上传认证缺少连接码或地址 | 配置问题 |
| active_only_requires_sync | collector_exception | “仅更新养成中”未开启同步/连接码 | 配置问题 |
| unknown_scan_mode | collector_exception | 未知扫描模式配置 | 配置问题 |
| unexpected_exception | collector_exception | 未分类异常（看 error_type/stack） | 开发排查 |
| task_failed_no_detail | collector_exception，stage=finalize | 任务失败但未记录具体原因 | 开发排查 |

## 客户端调用点

采集器异常携带的 `reason` 由捕获处传入 `DiagnosticSession.issue()`；未分类异常由该方法兜底。

| reason | 调用点 |
|---|---|
| page_timeout | `_wait_for_page()`、`_require_page()` |
| resource_language_mismatch | `_wait_for_page()` 超时调用 `_diagnose_ui_script()`；语言不一致时替代 page_timeout，`_require_page()` 异常保留该原因 |
| enter_operator_page_failed | `_AgentInfoReader.run()` 进入界面异常捕获，无具体异常 reason 时 |
| disc_plan_unknown、disc_tip_close_failed、disc_switch_unconfirmed | `_toggle_disc()` |
| disc_restore_failed | `_collect_discs()` 的原命盘恢复检查 |
| identity_changed_on_reread | `collect_current_from_main()` 数值补读身份检查 |
| page_arrow_failed | `_operator_arrow_visible()` |
| list_boundary_unverified、page_turn_unconfirmed | `_move_operator()` |
| return_first_limit | `_seek_first_operator()` |
| current_operator_failed | `_collect_and_publish()` |
| form_identity_unconfirmed | `_wait_for_operator_form()` |
| form_switch_failed、form_tip_close_failed | `_collect_other_form()` |
| name_and_disc_unresolved | `_log_unconfirmed_name()` 无候选、`_AgentInfoReader.run()` 姓名与命盘均未确认分支 |
| name_fuzzy_disc_unconfirmed | `_log_unconfirmed_name()` 有模糊候选、`_prepare_growth_filter()` 命盘未能确认分支 |
| unmatched_in_report、section_needs_review | `DiagnosticSession.checkpoint()` 的 unmatched、section_status 分支 |
| server_marked_review | `DiagnosticSession.upload_response()` 的 partial/review > 0 分支 |
| screenshot_failed、ocr_failed | `screenshot()`、`ocr()` 的空结果及异常分支 |
| catalog_load_failed | `_load_operators()` |
| account_lookup_failed | `AgentInfoCollector.run()` 的 account 阶段异常捕获 |
| annotations_failed | `AgentInfoCollector.run()` 的养成状态读取异常捕获 |
| upload_network_error、upload_http_error、upload_rejected_by_server | `_upload_v3_if_enabled()` 的 ExchangeApiError code 映射 |
| upload_response_invalid | `_upload_v3_if_enabled()` 的未提交及响应异常分支、`DiagnosticSession.upload_response()` 的响应校验 |
| upload_auth_missing | `_upload_v3_if_enabled()` 认证配置检查 |
| active_only_requires_sync | `AgentInfoCollector.run()` 养成筛选配置检查 |
| unknown_scan_mode | `_AgentInfoReader.__init__()` |
| unexpected_exception | `DiagnosticSession.issue()` 中异常没有有效白名单 reason 的兜底 |
| task_failed_no_detail | `DiagnosticSession.finish()` 中失败且没有 issue 的兜底 |

## v2 可选字段与隐私边界

新增字段不输出 `null`，旧 manifest 可继续处理。

| 字段 | 约束与取值来源 |
|---|---|
| `client.ui_script` | `simplified` / `traditional` / `mixed` / `unknown`；仅页面等待超时时截图并整屏 OCR，每次运行缓存一次，只发送枚举，不记录或发送该次识别文字 |
| `issues[].candidate_operator_ids` | 最多 5 个不重复的 `char_[A-Za-z0-9_]{1,80}`；复用 `_log_unconfirmed_name()` 的 candidates、主界面 `_operator_candidates`，以及报告 unmatched 的 candidates，保持既有相似度降序，仍经过 `_PRIVATE` 与 secrets 过滤 |
| 身份类 issue 的 `ocr_samples` | 不要求命中目录，但只允许 `[㐀-鿿]{1,8}`，最多 4 条；来源为 `name_raw`、`_name_ocr_fallback` 或报告 unmatched 的 raw_name，仍经过 `_PRIVATE` 与 secrets 过滤；摘要档也发送。`data_review` 的 `section_needs_review` 同样允许报告 `diagnostics.disc_name_review` 中的原文，优先于目录内样本且合计最多 4 条；其他 issue 仍只允许目录内样本 |
| `images[].role=name_area` | 图片档裁剪 `roi_config["main_name"]`，替代攻击数值裁图；命盘格保持，合计最多 3 张；保留旧 role 校验以兼容待传包 |

简繁判定取整屏 OCR 的 CJK 字符：`convert(c, 'zh-hans') != c` 为繁体特征，`convert(c, 'zh-hant') != c` 为简体特征；无特征为 unknown，仅一方有特征时取该方，两者都有且少数方占比至少 20% 时为 mixed，否则取多数方。

繁中资源配简体画面、简中资源配繁体画面时，超时 reason 为 `resource_language_mismatch`，且本地提示不依赖诊断开关：

- 检测到游戏界面为简体中文，但当前资源为繁中，请在 MaaYuan 资源设置中选择『简中（游戏界面）』后重试
- 检测到游戏界面为繁体中文，但当前资源为简中，请在 MaaYuan 资源设置中选择『繁中（游戏界面）』后重试

## 识别轨迹与失败画面

识别轨迹与失败画面字段沿用 `schema_version=1`，均为可选增量；旧待传包没有这些字段时仍可投递。服务端严格拒绝未知字段，必须先发布 YuanHub 后端，再发布 MaaYuan 客户端。

- 摘要与图片档均附 `trace`：仅活动诊断会话收集识别/动作完成事件，锁保护内存缓冲；相邻相同 `(kind, node, stage, hit)` 折叠，最多 64 项，超出丢最旧。`t` 为会话开始后的单调时钟毫秒数，`dt` 为折叠持续时间，`n` 最多 10000。
- 节点名只允许首次读取并缓存的当前资源 `node_list` 中的名称，仍按字符白名单和 `_PRIVATE`/secrets 过滤，否则为 `(direct)`。识别细节直接读取通知内联的 `reco_details`，不查询识别详情或拉取原图、标注图；分数保留三位小数，框必须位于 `client.resolution` 内。
- 轨迹 OCR 仅发送 `[㐀-鿿]{1,8}` 或 `[0-9+%.]{1,12}`，复用 `_PRIVATE`/secrets 过滤；简繁判定那次整屏 OCR 关闭轨迹文字采集，仍只发送 `ui_script` 枚举。`issues[].node` 在记录问题时取轨迹中同阶段最近的未命中识别节点，无则省略；不同节点的问题分别计数。
- 仅图片档且 `outcome=failed` 生成 `last_frame`，保持第 0 张、总图数最多 3 张及既有 JPEG 降质逻辑。优先使用最近未命中识别时引用的 `last_frame`；`trace_index` 按最终缓冲重新定位，避免淘汰导致错位。被引用步骤已淘汰时仍保留失败画面，但不附关联字段；没有失败截图时回退普通最后画面。仅在收尾时读取一次仍在轨迹中的失败节点配置，`recognition.param.roi` 为分辨率范围内四个整数时附 `node_roi`，不推算其他格式。

## 图片上传方式

投递优先使用国内 COS 直传，manifest 的字段与 `schema_version=1` 不变：

1. worker 的 `_send()` 先经 `encode_manifest()` 校验隐私与密钥过滤，再向生产域名的 `/open-api/maayuan/diagnostics/intents` 发送 JSON；每张图附带字节数和 base64 MD5。JSON 总长最多 64 KiB。
2. intents 返回 accepted 且有图片时，按 index 使用响应 URL 与 headers 逐张 PUT 原图片字节；不附加业务 Bearer token，不跟随重定向。URL 必须为 HTTPS、默认或显式 443 端口、主机名以 `.myqcloud.com` 结尾且无 userinfo。每张图只尝试一次，失败继续下一张。
3. 全部图片尝试后，向 `/open-api/maayuan/diagnostics/{event_id}/complete` 发送 `{}`，只尝试一次；失败也不进入重试队列。duplicate、accepted_without_images 直接视为投递完成，不 PUT。
4. intents 仅在 404、405、409 时回退旧 multipart `deliver()`。其他响应复用原规则：仅网络错误、408、429、5xx 可重试，遵守 Retry-After；intents 与 multipart 各自最多两次、总预算 8 秒、单次最多 5 秒。收到已接受回执后，图片及 complete 不再重试。

每次 HTTP 请求均在可终止子进程内执行，使用 `subprocess.run(timeout=...)` 限制 DNS、TLS 和读取总时长；COS PUT 单张最多 15 秒，complete 最多 5 秒。子进程通过内存管道接收 URL、headers、图片或 token，stderr 丢弃，不将凭据放入命令行。

本地待传包仍只保留脱敏 manifest、图片字节及既有重试元数据，不保存预签名 URL 或签名；重试重新申请 intents。日志仅记录「诊断图片直传：成功 n 张 / 失败 m 张」计数，不记录 URL、签名、token 或服务器错误正文。


## 背包运行日志附件

背包格已采用数量且唯一原因是 `threshold_disagree` 时，各次读数和采用值仅记入扫描日志。数量未确定或同一格还有跨页冲突、底部边缘等原因时，照常上报该格的完整原因。同次扫描内逐格处理；成功扫描没有其他问题时，沿用 `bag_scan_log/completed`。

背包日志记录 `回顶确认`、`页面对应`、`格子检测重试` 和 `扫描覆盖`：包括分区、尝试次数、前后行中心、匹配行映射、实际位移、支持格数、身份锚点数、冲突数、静止确认次数、未确认身份格数与 `allow_absent_zero`。结束原因为 `bottom_confirmed` 或 `target_boundary_confirmed` 时仍需结合 `allow_absent_zero` 判断覆盖是否完整。`top_unconfirmed` 不提交；`position_unconfirmed`、`boundary_unconfirmed`、`coverage_gap`、`page_limit` 允许保存此前独立确认的条目，但禁止缺项补零。对页失败截图记录 `image_index`；开启本地背包调试时另存 `alignment-XXX.png`，成功恢复后的 `page-XX.png` 与读数对应同一帧。

背包诊断沿用 YuanHub 私有日志附件（gzip），关联同一 event_id/run_id。每次分类扫描结束收集本次扫描开始后的 debug/custom 日志；正常无可疑项时提交 bag_scan_log/completed，异常退出时提交 bag_scan_log/failed。背包日志不再截短单行或只保留末尾 200KiB，仍脱敏连接码、凭据、个人目录和邮箱。上限为压缩 4MiB、解压 16MiB；超过上限明确上传收集失败摘要，不冒充完整日志。不收集整天历史或 maa.log。

金锁日志记录 `金锁重新定位` 的原圆心、新圆心、整页位移与支持格数，以及 `详情定位重试`、最终点击位置、详情名称对应的实际物品与数量。定位失败附截图和空数量证据；后续页成功的详情允许修复先前失败，但冲突数量仍不写入。诊断图片优先保留最终无法确认的格子，仍最多 12 张。日志使用现有私有访问权限和服务端保留期，沿用诊断开关及配额。

背包页上传图在原 ROI 内绘制检测格子的边框和行列号（从 `R1C1` 开始），顶部统计检测格数。绿色表示本页处理结束时数量已确认（包括“详情确认”），橙色表示待确认，灰色表示已识别为其他分区对象。每格列出名称/数量或前两名候选、模板或元素匹配分数，以及低分、分差不足、底部数量裁切等原因。分数是匹配相似度，不是成功概率；标注是本页状态，跨页冲突后的最终提交结果以日志和 `bag_cells` 为准。只标实际检测到的格子，不把未检测区域伪装成识别成功。

标注只修改诊断 JPEG 副本，不改变识别输入、截图尺寸、ROI 或读数坐标，不添加 OCR 原文及账号信息。名称来自物品/密探目录；优先使用系统中文字体，无中文字体时降级为英文状态与稳定 ID。标签优先放在图标下方，底边空间不足时向内收；完整原图仍可通过既有本地背包调试开关保存。最多12张的配额内先预留最多4张格子总览（待确认页优先），其余保留数量证据和更多总览；正常扫描也可上传总览。标注失败保留已有图片，不中断库存扫描，不改变现有诊断开关与上传权限。

开启本地“背包调试截图”后，同一渲染器还会在原图旁保存 `page-XX-annotated.jpg`；本页详情处理后更新同名图，翻页对应失败另存 `alignment-XXX-annotated.jpg`。本地生成不依赖上传诊断、连接码或同步模式，“仅保存到本地”也可查看。关闭背包调试截图时不额外落盘，本地标注失败不阻断识别或诊断上传。

部署顺序：先部署支持 bag_scan_log 与新日志上限的 YuanHub 后端，再同步客户端或发布补丁。旧后端不能完整支持本次协议，禁止先发客户端。日志上传目前仍沿用现有直传流程，直传网络故障时需在 YuanHub 检查附件状态；诊断摘要被接收不等于附件完整到达。
