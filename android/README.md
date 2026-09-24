# MaaYuan Android 发布

Android 使用 [syoius/MaaFwApp](https://github.com/syoius/MaaFwApp/tree/maayuan) 的 `maayuan` 发行分支。
`build-config.json` 固定完整源码提交、MaaFramework、Agent Core、OpenCV、Python 开发头文件和 NDK；
CI 不跟随浮动的 `main` 或 latest 构建。

## 统一版本

`install.yml` 的 `meta.tag` 是所有平台共同的版本来源，Android 原样保留 `v` 前缀：

| 项目 | 示例 |
| --- | --- |
| Release tag / 桌面资源版本 | `v2.2.1-beta.1` |
| APK versionName / Android 内置资源版本 | `v2.2.1-beta.1` |
| APK 文件名 | `MaaYuan-android-arm64-v2.2.1-beta.1.apk` |
| Android versionCode | `github.run_number * 100 + github.run_attempt` |

versionCode 仅用于 Android 覆盖安装和资源重新解包，与用户看到的跨平台版本名分开。
同一 workflow 后续运行会递增，重新运行也会递增。迁移或重建 workflow 时应保留其递增关系。
只有工作流引用为合法 `v` 前缀 SemVer tag 时才发布；普通分支即使 HEAD 恰好指向已打 tag 的提交，也仍构建 CI 预览版。
带 SemVer 预发布部分的标签（包括 `alpha`、`beta`、`rc`）发布为 GitHub prerelease，仍使用正式应用身份。

| 触发引用 | Android 产物 | GitHub Release |
| --- | --- | --- |
| `refs/tags/v2.2.1` | 正式包，永久签名 | 正式发布 |
| `refs/tags/v2.2.1-beta.1` | 正式包，永久签名 | 预发布 |
| 普通分支 push / PR | 独立 `.ci` 预览包，debug 签名 | 不发布，仅上传 Actions artifact |
| 手动运行 | 选 SemVer tag 时同发布规则；选分支时为预览包 | 由所选引用决定 |

预览版本带 `-ci.<提交距离>-g<短 SHA>` 后缀，各平台继续共用同一版本值。
非法版本 tag（例如 `vnext`、`v2.2`）在版本判定阶段报错，不进入构建和发布。

## 安装身份与签名

正式应用名为「Maa鸢」，包名由 `applicationIdSuffix` 配置，默认 `com.aliothmoon.maafw.maayuan`。
这与本地 `.preview` 测试包及旧 MFAAvalonia Android 的包名不同，会独立安装。

在 MaaYuan 仓库配置以下 Secrets，签名材料不提交到源码：

- `ANDROID_KEYSTORE_BASE64`：发布 keystore 的 Base64 内容。
- `ANDROID_KEYSTORE_PASSWORD`：keystore 密码。
- `ANDROID_KEY_ALIAS`：发布密钥别名。
- `ANDROID_KEY_PASSWORD`：密钥密码。

另配置公开仓库变量 `ANDROID_SIGNING_CERT_SHA256`，内容为签名证书 SHA-256。
CI 在打包后检查证书、实际 versionName/versionCode、包名和 ABI。
密钥及密码需要长期备份；后续正式包保持同一包名和签名。

合法 SemVer tag 构建使用 release 构建和上述永久签名；缺少任一签名配置会失败。
普通分支 push 和 PR 始终使用独立 `.ci` 包名、「Maa鸢 · CI」名称和 debug 签名，
即使仓库已配置发布密钥，也不会使用正式包身份或读取发布密钥。
CI 预览包与正式包可并存；debug 签名由构建环境生成，不保证不同 CI 构建之间可覆盖安装。

## 构建内容

1. 从 MaaYuan 资源副本生成 Android PI；桌面 `assets/interface.json` 不被改写。
2. 补齐简繁资源的中文及英文 OCR 模型，规范 Android StartApp 启动参数。
3. 使用固定的 MaaAgentCoreAndroid，补充项目依赖；以 Android NDK 编译 CPython 3.13 的 cv2。
4. 用 MaaFwApp profile 指定应用名称、图标、Agent 环境和 APK 载荷。
5. Gradle 运行单元测试并构建 APK；解包验证资源与 Agent，生成 SHA-256 和验证报告。

Agent 缓存在依赖配置或构建脚本变化时更新。OpenCV 的 Android Python 模块开关在构建目录中开启，
修改逻辑保存在 `build_runtime.py`；相关开源许可证随运行时进入 APK。
头文件使用同一 CPython 3.13 ABI 的 Chaquopy 3.13.9 开发包，扩展实际链接 Agent Core 的 3.13.15 运行时。

本地预处理和测试：

```powershell
python -m unittest discover -s android/tests -v
python android/prepare.py --version v2.2.1-beta.1 --version-code 10101
```

`android/.build/profile.yaml` 指向该目录的 `payload` 和 `agent-dist`；设置 `PI_PROFILE` 后运行 fork 的 Gradle 构建。
`android-build.json` 随资源打包，记录版本及 MaaFwApp 提交，便于复现。

## 更新与维护

更新源仍为 `syoius/MaaYuan` Releases，只匹配 `MaaYuan-android-` 前缀和设备 ABI；
预览版与旧 MFAAvalonia APK 的文件名不匹配该前缀。MirrorChyan 尚未配置 Android 渠道时保持 `mirrorchyanRid: null`。

同步上游后，先在 fork 的 `maayuan` 分支验证，再更新 `sourceRevision`。
每次发布固定到通过测试的提交。资源、发布配置在 MaaYuan 管理；Android 外壳代码在 fork 管理。
