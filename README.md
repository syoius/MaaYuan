<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

# MaaYuan

</div>

基于 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 的代号鸢/如鸢小助手。图像技术 + 模拟控制，解放双手！

Windows端图形界面使用 **[MFAWPF](https://github.com/SweetSmellFox/MFAWPF)**。

建议将模拟器分辨率配置固定为`16:9`或`9：16`。

## 功能介绍

- 刷6-24直到体力耗尽
  - [x] 选关界面自动进入
  - [x] 关卡内自动接管
  - [x] 不消耗白金币回体力
- 待办公务
  - [x] 优先清3级物资箱的物资支援
  - [x] 自动请求出战
  - [ ] 自选决策事件处理方式（测试中）
    - [ ] 不做决策事件/奖励选择五铢钱/奖励选择经验书
  - [x] 自动吃鸟食
## 使用说明

### Windows（✅有可视化界面）
 - 对于绝大部分用户，请下载 `MaaYuan-win-x86_64-vXXX.zip`
 - 若确定自己的电脑是 arm 架构，请下载 `MaaYuan-win-aarch64-vXXX.zip`
   
请注意！Windows 的电脑几乎全都是 x86_64 的，可能占 99.999%，除非你非常确定自己是 arm，否则别下这个！

解压后运行 `MaaYuan.exe` 即可

### MacOS（❌无可视化界面）
 - 若使用 Intel 处理器，请下载 `MaaYuan-macos-x86_64-vXXX.zip`
 - 若使用 M1, M2 等 arm 处理器，请下载 `MaaYuan-macos-aarch64-vXXX.zip`
 - 使用方式：
   ```
   chmod a+x MaaPiCli
   ./MaaPiCli
   ```
~~我抄隔壁的我没mac~~

### Linux （❌无可视化界面）
~~差不多同上吧大概，我用wsl的~~


## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！图形界面由 **[MFAWPF](https://github.com/SweetSmellFox/MFAWPF)** 提供。

感谢以下开发者对本项目作出的贡献:


<a href="https://github.com/syoius/MaaY/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=syoius/MaaY&max=1000" />
</a>
<a href="https://github.com/SweetSmellFox/MFAWPF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SweetSmellFox/MFAWPF&max=1000" />
</a>
<a href="https://github.com/MaaXYZ/MaaFramework/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MaaXYZ/MaaFramework&max=1000" />
</a>
