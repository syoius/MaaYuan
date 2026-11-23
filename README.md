<!-- markdownlint-disable MD033 MD041 -->

<div align="center">
  <h1>
  <img src="./logo.png" alt="戳一下！" style="vertical-align: middle; margin-right: 10px;">
  MaaYuan
</h1>
  <img alt="license" src="https://img.shields.io/github/license/syoius/MaaYuan">
  <img alt=".NET" src="https://img.shields.io/badge/.NET-≥%2010-512BD4?logo=csharp">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-blueviolet">
  <img alt="commit" src="https://img.shields.io/github/commit-activity/m/syoius/MaaYuan">
  <img alt="stars" src="https://img.shields.io/github/stars/syoius/MaaYuan?style=social">
  <a href="https://mirrorchyan.com/zh/projects" target="_blank"><img alt="mirrorc" src="https://img.shields.io/badge/Mirror%E9%85%B1-%239af3f6?logo=countingworkspro&logoColor=4f46e5"></a>
</div>

---

<div align="center">
  <strong><a href="#中文介绍">简体中文</a> · <a href="#english-summary">English</a></strong>
</div>

> **Development status:** MaaYuan is actively maintained. Current development takes place on the [`v5`](https://github.com/syoius/MaaYuan/tree/v5) branch.

## English Summary

**MaaYuan** is an open-source, cross-platform automation client for **如鸢 / 代号鸢**, built on [MaaFramework](https://github.com/MaaXYZ/MaaFramework). It combines image recognition with automated input to handle repetitive in-game tasks. Desktop builds use [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia) on Windows, macOS, and Linux, while Android is supported through [MaaFwApp](https://github.com/syoius/MaaFwApp/tree/maayuan).

### Open-source ecosystem

MaaYuan also serves as the core client of a broader open-source ecosystem:

- **[MyShare](https://share.maayuan.com/)** — a community platform for publishing, sharing, and importing reusable automation workflows directly into MaaYuan.  
  Frontend: [syoius/MaaYuan-Share-frontend](https://github.com/syoius/MaaYuan-Share-frontend)  
  Backend: [MrSnake0208/MaaYuan-Share-Backend](https://github.com/MrSnake0208/MaaYuan-Share-Backend)

- **YuanHub (Beta)** — an open platform developing a standardized data protocol and OpenAPI for structured, user-owned game account-state data extracted through MaaYuan workflows, such as character ownership, progression, and other account-state information. The data can be used by user-authorized applications and community tools.  
  Frontend: [MrSnake0208/YuanHub](https://github.com/MrSnake0208/YuanHub)  
  Backend: [MrSnake0208/BackEndV3-Share](https://github.com/MrSnake0208/BackEndV3-Share)

All components are open source and maintained in public GitHub repositories.

## 中文介绍

基于 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 的代号鸢/如鸢小助手。图像技术 + 模拟控制，解放双手！

图形界面基于 **[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)**，支持 `Win` / `Mac` /`Linux`。

Android 端基于 **[MaaFwApp](https://github.com/syoius/MaaFwApp/tree/maayuan)**，支持 Android 9 及以上的 arm64 设备，与桌面端使用相同发布版本号。构建、签名和更新渠道见 [Android 发布说明](android/README.md)。

建议将模拟器分辨率配置设置为 `1280x720, 320dpi` 。目前开发组仅基于进行此分辨率进行测试和调整，其他分辨率暂缺人手、不保证使用效果。欢迎多多反馈、加入测试开发！

船新版本 **[（自动战斗）作业站](https://share.maayuan.com/)** 锐意测试 ing~ 访问作业站可获得 `神秘代码`，启动 MaaYuan 后，进入 🎮 抄作业 专属界面 使用 `神秘代码` 即可一键导入对应作业文件。

- 作业编辑器 V2：新增类表格视图（类似辟雍学府-操作记录仪的编辑模式），编写作业更方便。

- 支持录入密探、星盘、攻击力/血量 等信息，还可只看包含/排除指定密探的作业，帮您快速筛选最合适的作业

✨ 如果喜欢 MaaYuan，欢迎在项目右上角点亮 Star 支持 ✨

## 功能介绍

### 密探姓名识别

密探信息采集先使用内置 OCR。姓名无法精确匹配名册时，会分别按两字、三字切分竖排姓名，将字块横向拼接后重试；检测到右侧红色觉醒徽标时，改截取左侧姓名字块，避免将徽标拼入姓名。只有唯一的精确名册候选才会被接受。横排重试仍无法确认时，先读取命盘专属名称确认身份，再继续采集其他详情，不直接按相似度或姓名子串选人。即使 OCR 只读到“觉醒”、清理后姓名为空，也会尝试命盘确认；没有姓名候选时，需要同一密探的两条不同专属命盘名称，证据不足则报错停止。原始 OCR 文本和横排重试的两次输出记录在 `collection_debug`；未匹配时，业务日志也会打印横排输出。

“采集密探信息”会显式关闭姓名 OCR 对比模式；普通采集中的横排重试只在原始姓名未匹配名册时触发，不会生成对比报告。

当前切分针对 106×226 姓名区域（其他截图尺寸先归一化）。78 张人工标注裁图的离线实验由 71/78 提升至 78/78；这是开发集结果，不代表不同立绘、SP 布局或新角色均已验证。无需安装额外 OCR 引擎或更换模型。

命盘 OCR 别名在 YuanHub 对应密探的 `discs` 项中维护，`ot_name` 与公共图鉴保持一致，例如：

```json
{"ot_name": "初始能量+3", "abbreviation": "初始+3", "aliases": ["初是能量+3", "初是+3"]}
```

`aliases` 可省略，或设置为非空字符串组成的数组；格式错误或空白的别名不参与匹配。采集和导出均识别这些别名，导出仍只写 `ot_name`；别名不能唯一映射时保留待核对，不按先后顺序选取。候选名称末尾为汉字时，还允许 OCR 恰好多出 `0`、`00` 或 `0O`；原有单个英文字母尾缀容错保留，以数字结尾的候选不适用新增规则。

描述别名 `desp_aliases` 为可选字符串数组，在 YuanHub 维护，只用于未解锁命盘按描述反查，上传仍只写 `ot_name`。

每次扫描前从 YuanHub 公开接口 `https://api-hub.maayuan.com/open-api/operator/catalog` 检查目录更新，无需连接码；自动编队复用同一同步逻辑，并保留原有检查间隔。接口直接返回含 `OPERATORS`、`PROFESSIONS`、`catalog_schema_version`、`catalog_version` 的目录，不使用 `data` 包装。完整远程快照校验后原子写入 `agent/operators.remote.json`，不与内置目录按 ID 合并，也不覆盖安装包内的 `agent/operators.json`。优先读取有效远程缓存，否则回退内置目录；网络失败、接口响应无效时继续使用可用本地目录，两者均不可用才停止。背包扫描仍读取内置目录。

下载上限为 5 MiB，保留 HTTPS 证书校验，最终响应地址须为 `api-hub.maayuan.com` 的 HTTPS 地址。目录最多含 10,000 条密探记录，每位最多 10,000 条命盘，密探及命盘的直接字符串字段（不含 `aliases`、`desp_aliases`）最长 4,096 字符。缓存验证器绑定源 URL 和快照内容，仅在快照成功落盘后保存新验证器；远程缓存缺失或损坏时不发送条件请求头。更新到支持此接口的客户端后，沿用现有界面、OCR 和养成规则的新密探及命盘别名可在 YuanHub 维护，用户在下一次扫描成功同步后即可使用，无需再次更新 MaaYuan；新 SP 形态、OCR 不支持的字或界面变化仍可能需要客户端更新。

### 密探形态、觉醒与上报

全量扫描默认先返回名单首位；开启“从当前密探开始”后，从当前打开的密探依次向后采集到名单末位，适合中断后接着扫描。开启“仅扫描一位”时此项无效。

- 🔄 全量采集陈登、史子眇时，通过“切换形态”分别读取普通版与 SP 版；确认身份并恢复原形态后才进入下一位。两个形态分别遵循养成筛选；“只采集当前一位”只读取当前形态。
- ⭐ 化极页面显示“已觉醒”时，即使姓名 OCR 漏掉觉醒徽标，也会导出觉醒星级 `31`；“待觉醒”不算已觉醒。
- 蓝卡（隐密）显示“最高等级”时，按公共图鉴记录为 4 星满阶、未觉醒，导出星级 `19`，不再误报为五星已觉醒。历史误报仍需在 YuanHub 复核，新采集使用修正后的规则。
- ☁️ 每位密探先保存本地报告，再排入后台队列按顺序预览和提交，与下一位采集并行。采集正常结束会等待队列完成后汇总同步结果；停止时跳过排队任务，等待当前请求返回，并保留本地报告。服务器返回拒收时会打印原因，即使 HTTP 状态为 200 也不会视为成功。

请在主界面控制器栏下方的『YuanHub 连接码』里填写。百宝箱和据点日常共用同一实例的连接码，切换简中、繁中资源无需重填。多开时，每个实例填写各自账号的连接码。请勿分享或泄露连接码。单实例留空时，优先使用配置中唯一的连接码，配置里没有连接码时再使用已记住的连接码。多开、配置中有多个不同的连接码或配置无法读取时，请在当前实例的主界面填写。

“帮助改进百宝箱”提供关闭、发送脱敏诊断（默认）、发送脱敏诊断和图片三档，目前仅用于“采集密探信息”。开启同步至 YuanHub 且已填写连接码时，采集异常、数据待核对、上传失败或无法验证上传响应会在扫描收尾后异步发送一次诊断。完整上传成功、主动停止、仅本地保存和没有连接码时不发送本次诊断。业务同步开关不代表同意发送图片；图片需单独选择。

诊断只含白名单错误码、阶段、计数、密探 ID、资源/框架/目录版本、字段状态、项目相对栈及识别轨迹。轨迹保留最近最多 64 步的节点名、阶段、命中与否、分数、识别框及经过滤的短 OCR 文本；连续相同步骤折叠计数，也记录动作成功与否。节点名仅来自当前资源，其他名称以 `(direct)` 代替；OCR 文本仅允许 1～8 个汉字或 1～12 位数字及 `+%.`，并过滤已知账号名、连接码等敏感值。问题可附同阶段最近未命中的节点名，供按节点分组。身份待核对时还可发送最多 4 条纯汉字姓名 OCR 原文（每条 1～8 字）及最多 5 个模糊候选密探 ID；上述文字信息在摘要档也适用。不发送账号名、连接码、绝对路径、异常消息、完整日志或整份业务报告。图片最多三张 JPEG，可包含密探姓名区、命盘格及任务失败时的整屏失败画面；失败画面优先对应最近一次未命中识别，并附轨迹索引和节点识别区域标注（节点配置有效时）。不新增点击；整屏画面可能包含游戏信息，两游戏实际隐私区域需实机核对。

页面等待超时时，每次运行仅截图并执行一次整屏 OCR 判定简繁，诊断只发送 `ui_script` 枚举，不发送或记录该次识别文字。资源语言与画面不一致时，诊断原因记为 `resource_language_mismatch`，并在本地提示切换到对应简中或繁中资源后重试；关闭诊断时也会提示。

诊断 issue 可附带固定白名单 `reason`，用于区分具体失败原因，中文说明见 [诊断原因映射](docs/toolbox-diagnostics-reasons.md)。本次没有已识别密探 entries 时只保存本地报告并跳过业务上报，未匹配密探仍记录为身份待核对。

诊断复用现有连接码，通过 HTTPS multipart POST 到 `https://api-hub.maayuan.com/open-api/maayuan/diagnostics`，无匿名入口、无新增 scope。`manifest` 为不超过 64 KiB 的 JSON，`files` 为 0～3 张、每张不超过 512 KiB 的 JPEG，请求不超过 2 MiB。200 响应必须回显 event_id，且 status 为 accepted、duplicate 或 accepted_without_images；其他响应按 HTTP 状态处理。仅网络错误、408、429、5xx 重试，单次最多 5 秒、每次投递总预算 8 秒、最多两次，并遵守 Retry-After。HTTP 在可终止的 Python 子进程内执行，凭据只经内存管道传入；后台只有一个有界队列 daemon worker，扫描不等待诊断网络结果。

离线包位于 `debug/toolbox_diagnostics/pending`，只存脱敏 manifest、可选诊断图片及本地重试元数据，不存连接码；通过不可逆凭据摘要限制为同一连接码补传。最多 20 包、20 MiB、7 天，下次采集尝试补传，超限丢弃并计数。关闭开关会在下次采集读取配置后删除未发送包；改为仅摘要会移除待传图。队列满、落盘/编码/传输失败只影响诊断，不改变原任务返回值。强杀、离线和限额可能造成缺报；本仓库不验证服务端部署状态。轨迹及图片关联字段沿用 `schema_version=1`，必须先发布支持这些字段的 YuanHub 后端，再发布客户端。

### 1. 支持的游戏任务:

- 🌿 日常一条龙

  > 所有任务遇到体力不足或次数不足时会自动停止，并尝试进行后续任务。
  >
  > 开启无限循环和自动吃鸟食时，也**不会**消耗白金币购买鸟食。

  - 🚀 模拟器、游戏自启动 (使用指定账号登录)
  - 📜 鸢报四合一（清鸟食）
  - 💰 领月卡福利
  - 🍚 领取每日进膳体力
  - 🏯 据点日常（✔ 派遣 ✔ 据点情报）
  - 🛍️ 据点商人购物
  - 🔮 观星（✔ 观一次 ✔ 一键观星（需月卡） ✔ 六步观星法）
  - 🏮 历练（✔ 自定义关卡 + 等级 + 次数）
  - 🗡 白鹄行动扫荡（✔ 每日扫荡 + 领奖）
  - 🗡 地宫扫荡（✔ 每日扫荡 ✔ 雀部解密）
  - 🔨 心纸营建自动派遣
  - 🔶 其他日常（✔ 每日分享 ✔ 相见互动 ✔ 家具互动 ✔ 首位密探升级 ✔ 每日送礼）
  - 🔍 楼主查岗（可选 ✔ 自动购买天机符传 ✔ 存 100 体力）

- 🛠 其他护肝功能

  - 🔄 刷 6-24 / 7-15
    - 随时随地一键开刷，可在日常结束后自动进入，刷光剩余体力
    - 可选`刷到体力耗尽`（不会消耗白金币回体力）或 `指定循环次数`
  - 🌺 自动爬兰台
    - 自动切换难度及期数
  - 🔄【宇宙匣】三千世界
    - 不耗体力刷共同出战类成就
  - 🌥️ 云梦巫乡挖煤
    - 自动导航、续关接管
    - 自定义识别速度及精度
  - 🔄 自动刷洞窟
    - 支持 `游戏内置自动战斗` 及 `抄作业` 自动刷满洞窟 30 层
  - 🔥 心纸营建自动走历险（首通/手动刷佛脚用，非自动派遣）
  - 🔥 船新版本 **[（自动战斗）作业站](https://share.maayuan.com/)** 锐意测试 ing~ 访问作业站可获得 `神秘代码`，启动 MaaYuan 后，进入 🎮 抄作业 专属界面 使用 `神秘代码` 即可一键导入对应作业文件。

### 2. MaaYuan 功能

- 预设切换
  - 官方提供多种预设，可直接载入使用
    - `代号鸢日常模板`
    - `如鸢日常模板`
    - `特色功能合集` （活动期间有刷取专项预设！）
    - `新版全部功能`
  - 可根据需求在任务列表中 添加 / 删除 / 切换 任务，为不同账号创建不同预设方案，一键长草~

## 使用说明

### Windows

- 对于绝大部分用户，请下载 `MaaYuan-win-x86_64-vXXX.zip`
- 若确定自己的电脑是 arm 架构，请下载 `MaaYuan-win-aarch64-vXXX.zip`

请注意！Windows 的电脑几乎全都是 x86_64 的，可能占 99.999%，除非你非常确定自己是 arm，否则别下这个！

解压后运行 `MaaYuan.exe` 即可。

如果遇到报错，可能是没有安装`.NET 桌面运行时 10`。可以找到文件夹中的`install-deps-win.bat`，**双击运行**，等待安装完成， **重新启动计算机** 后再次运行 `MaaYuan.exe`。

### MacOS

- 若使用 Intel 处理器，请下载 `MaaYuan-macos-x86_64-vXXX.zip`
- 若使用 M1, M2 等 arm 处理器，请下载 `MaaYuan-macos-aarch64-vXXX.zip`
- 解压缩之后，右击文件夹，点击“新建位于文件夹位置的终端窗口”
- 在终端窗口内逐行输入以下指令

  ```
  chmod a+x MaaYuan
  ./MaaYuan
  ```

  > 注：需要安装.NET 运行库（使用上面的命令启动失败时会直接返回下载地址）。也可以找到文件夹中的`install-deps-mac.sh`，**双击运行**，等待安装完成， **重新启动计算机** 后再次运行 `MaaYuan`。

- Mac 可能会提示：因为 Apple 无法检查其是否包含恶意软件

此时进入选择   设置  -  隐私与安全性，下方出现“已阻止…”点击   仍要打开。
多重复以上步骤，因为有很多文件会被检查。

~~我抄隔壁的我没 mac，欢迎测试反馈~~

### Linux

~~差不多同上吧大概，我用 wsl 的，欢迎测试反馈~~

## 从零开始学会写作业

0. [MaaYuan零基础开发指南](./docs/0.0-MaaYuan零基础开发指南.md)
  
   0.1 [开发工具介绍](./docs/0.1-开发工具介绍.md)

1. [MaaYuan开发规范](./docs/1.0-MaaYuan开发规范.md)

    1.1 [节点信息输出](./docs/1.1-节点信息输出.md)

    1.2 [interface字段配置](./docs/1.2-interface字段配置.md)

2. 编写你的第一份日常行动作业
3. 编写你的第一份战斗关卡作业
4. 利用通用模块，快速实现常规功能

   4.1 [日常行动通用模块](./docs/4.1-日常行动通用模块.md)

   4.2 战斗行动通用模块

   4.3 [其它通用模块](./docs/4.3-其它通用模块.md)

5. 把你的作业加入作业列表中
6. 向本项目贡献代码

## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！桌面图形界面由 **[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)** 提供。

感谢以下开发者对本项目作出的贡献:

<a href="https://github.com/syoius/MaaYuan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=syoius/MaaYuan&max=1000&columns=15&anon=1" />
</a>
