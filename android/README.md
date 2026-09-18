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
`alpha`、`beta` 和 `rc` 标签均发布为 GitHub prerelease。

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

非 PR 构建配置好上述签名后使用 release 构建；未配置时，普通 CI 构建使用独立 `.ci` 包名、
「Maa鸢 · CI」名称和 debug 签名。PR 不接收发布签名。tag 发布缺少签名会失败，
不会把临时 debug 签名包作为正式版本分发。

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
