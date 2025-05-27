<!-- markdownlint-disable MD033 MD041 -->

<div align="center">
  <h1>
  <img src="./logo.png" alt="戳一下！" style="vertical-align: middle; margin-right: 10px;">
  MaaYuan
</h1>
  <img alt="license" src="https://img.shields.io/github/license/syoius/MaaYuan">
  <img alt=".NET" src="https://img.shields.io/badge/.NET-≥%208-512BD4?logo=csharp">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blueviolet">
  <img alt="commit" src="https://img.shields.io/github/commit-activity/m/syoius/MaaYuan">
  <img alt="stars" src="https://img.shields.io/github/stars/syoius/MaaYuan?style=social">
  <a href="https://mirrorchyan.com/zh/projects" target="_blank"><img alt="mirrorc" src="https://img.shields.io/badge/Mirror%E9%85%B1-%239af3f6?logo=countingworkspro&logoColor=4f46e5"></a>
</div>

---

基于 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 的代号鸢/如鸢小助手。图像技术 + 模拟控制，解放双手！

图形界面基于 **[MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia)**。

建议将模拟器分辨率配置固定为 `16:9` 或 `9:16` 。

作业生成器 `Maa鸢/司命` 开发测试中，[点我下载作业生成器](https://github.com/syoius/MaaYuan-SiMing/releases)，[试用网页版作业生成器](https://siming.cruisingcat.top)，[点我进入临时作业站](https://td3nqr3477.feishu.cn/base/BtwRb6yvxaRAsis3uVCccLVKnWg?table=tbl6DolUZQaGOMIb&view=vew54wfNwm)。

✨ 如果喜欢 MaaYuan，欢迎在项目右上角点亮 Star 支持 ✨

## 功能介绍

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
  - 🔥 【测试版】自动抄作业 （[点我下载作业生成器](https://github.com/syoius/MaaYuan-SiMing/releases)，[试用网页版作业生成器](https://siming.cruisingcat.top)，[点我进入临时作业站](https://td3nqr3477.feishu.cn/base/BtwRb6yvxaRAsis3uVCccLVKnWg?table=tbl6DolUZQaGOMIb&view=vew54wfNwm)）

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

如果遇到报错，可能是没有安装`.NET 桌面运行时 8`。可以找到文件夹中的`install-deps.ps1`，**右键**选择“使用 PowerShell 运行”，等待安装完成， **重新启动计算机** 后再次运行 `MaaYuan.exe`。

> MAA 本家提供的解决方案 ⬇️
>
> 请安装 [`Visual C++ 可再发行程序包`](https://aka.ms/vs/17/release/vc_redist.x64.exe) 和 [`.NET 桌面运行时 8`](https://dotnet.microsoft.com/en-us/download/dotnet/8.0#:~:text=Binaries-,Windows,-x64) 并 **重新启动计算机** 。
>
> 推荐使用 Windows 10 或 11 的用户使用 winget 工具进行安装，只需在终端中运行以下命令。
>
> ```
> winget install Microsoft.VCRedist.2015+.x64 Microsoft.DotNet.DesktopRuntime.8
> ```

### MacOS

- 若使用 Intel 处理器，请下载 `MaaYuan-macos-x86_64-vXXX.zip`
- 若使用 M1, M2 等 arm 处理器，请下载 `MaaYuan-macos-aarch64-vXXX.zip`
- 解压缩之后，右击文件夹，点击“新建位于文件夹位置的终端窗口”
- 在终端窗口内逐行输入以下指令

  ```
  chmod a+x MaaYuan
  ./MaaYuan
  ```

  > 注：需要安装.NET 运行库（使用上面的命令启动失败时会直接返回下载地址）

- Mac 可能会提示：因为 Apple 无法检查其是否包含恶意软件

此时进入选择   设置  -  隐私与安全性，下方出现“已阻止…”点击   仍要打开。
多重复以上步骤，因为有很多文件会被检查。

~~我抄隔壁的我没 mac，欢迎测试反馈~~

### Linux

~~差不多同上吧大概，我用 wsl 的，欢迎测试反馈~~

## 从零开始学会写作业

1. [准备工作](./docs/1.1-准备工作.md)
2. 编写你的第一份日常行动作业
3. 编写你的第一份战斗关卡作业
4. 利用通用模块，快速实现常规功能

   4.1 [日常行动通用模块](./docs/4.1-日常行动通用模块.md)

   4.2 战斗行动通用模块

5. 把你的作业加入作业列表中
6. 向本项目贡献代码

## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！图形界面由 **[MFAWPF](https://github.com/SweetSmellFox/MFAWPF)** 提供。

感谢以下开发者对本项目作出的贡献:

<a href="https://github.com/syoius/MaaYuan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=syoius/MaaYuan&max=1000&columns=15&anon=1" />
</a>
