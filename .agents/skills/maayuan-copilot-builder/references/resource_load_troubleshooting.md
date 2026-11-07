> 历史排查记录：版本、路径和激活方式以当时事件为准。当前默认导出仅写缓存，旧文中“三处自动同步/清旧作业”的操作不适用于当前默认导出。请先读 SKILL.md。

# 排查「资源加载失败」的方法论

> 本文不是答案清单，而是**推导方法**。`SKILL.md` 的速查表只覆盖 v2.0.0-alpha.1 实测到的现象；
> 客户端一升级，报错可能完全变样。那时按本文的步骤重新推一遍，比翻速查表可靠。

## 0. 先建立正确的心智模型

MaaFramework 是**单资源包**架构：客户端启动时加载整个 bundle（`resource/base`），
其中 `base/pipeline/` 下**每一个** `.json` 都会被当作 pipeline 定义解析。
**任何一处解析不了，整个资源包拒绝加载**，客户端弹「资源加载失败」。

推论：作业文件不是"独立资源"，它的节点最终要混进这个包里。任何命名冲突、
格式污染都会连带炸掉全部功能——包括与作业毫不相关的日常任务。

## 1. 拿到真实报错（不要猜）

客户端 UI 只说「资源加载失败」，真正的原因在两处：

```bash
# 客户端日志
D:/MaaYuan-win-x86_64-v2.0.0-alpha.1/logs/log-YYYYMMDD.log
```

MaaFramework 自己的 `[ERR]` 行会给出精确原因，例如：

```
key already exists [key=抄作业-进入编队界面]
value is not object [key=difficulty]
Invalid next node name [node.name=检测回合1]
```

**但客户端日志里看不到这些**——它只记 .NET 层的 `MaaJobStatusException`。
要拿到引擎级报错，必须自己跑一次加载：

```bash
cd D:/MaaYuan-win-x86_64-v2.0.0-alpha.1
python/python.exe -c "
import maa.resource as res
from pathlib import Path
h = res.Resource()
j = h.post_bundle(Path(r'D:\MaaYuan-win-x86_64-v2.0.0-alpha.1\resource\base'))
j.wait()
print('RESULT loaded:', h.loaded)
"
```

三个坑：
- **必须用 `python/python.exe`**。MaaYuan 自带 python 才有 `maa` 模块，系统 python 没有。
- **报错走 stdout**，不是 stderr。管道给 `grep` 前先 `2>&1` 合并。
- **输出带 ANSI 颜色码**，先 `sed 's/\x1b\[[0-9;]*m//g'` 去掉，否则 grep 匹配不到。
- 中文在 Git Bash 里常显示为乱码，但**关键字（`key already exists` 等）是 ASCII**，照 grep 得到的信息足够定位。

## 2. 定位污染源：二分隔离法

报错说"某个 key 重名"，但**重名可能来自几十个文件**。不要逐个猜，做二分：

```bash
# 1. 先把所有作业文件移走，只留 copilot_config.json
mv resource/base/pipeline/copilot/xxx.json /tmp/bak/

# 2. 加载测试
# 3. 加载成功 → 污染源就在移走的文件里，恢复一个再测
```

**关键判据：`copilot_config.json` 自己引用了只有作业才有的节点**
（如「替代-检测回合1」的 next 是「检测回合1」）。
所以 `base/pipeline` 里**必须**有且只有一个作业文件，纯 config 状态必然报
`Invalid next node name [检测回合1]`。这不是 bug，是设计。

## 3. 确认客户端到底怎么读作业（关键，别猜）

**不要假设客户端有"专门的作业加载器"。** 要用证据。三个证据来源：

### 证据 A：看客户端自己写的文件

客户端选中的作业会被它自己写进 `base/pipeline/copilot/`。
**判据是时间戳**：文件 mtime 与客户端启动时刻吻合，就说明是客户端写的，不是你放的。

```bash
ls -la --time-style=full-iso resource/base/pipeline/copilot/
```

**如果你删掉一个文件，启动客户端后它又回来了——那就证明是客户端写的。**
（2026-10-06踩过：误删了 `录制作业-20260924-010031.json`，客户端启动后写回，
它和我的作业撞 19 个同名节点，直接把整个资源包搞挂，且此后再也加载不成功。）

> [!TIP]
> 这个"删了又回来"的信号非常有价值——它一次性证明了
> **客户端会主动写这个目录，因此该目录的内容不受你控制**。
> 排查冲突时，第一件事就是确认哪些文件是客户端放的。

### 证据 B：读客户端自己产出的同类文件

客户端生成的作业文件揭示了**它写出来的节点长什么样**。对比三种格式：

```bash
python/python.exe -c "
import json
for f in ['resource/copilot-hold/录制作业-20260922-115407.json',    # 客户端产出：裸节点 map
          'resource/recordings/录制作业-20260924-010031.json',       # 录制原始：带 actions
          'resource/copilot-cache/金乌枝1-吕布队.json']:              # 作业本体：带 version/difficulty
    d = json.load(open(f, encoding='utf-8'))
    print(f.split('/')[-1][:30], '->', list(d.keys())[:4])
"
```

关键观察：同一个作业在三个目录里有三种形态。
**这直接证明了「同一份作业要按位置变换形态」**——2026-10-06 就是靠这个发现
"引擎要的是裸节点 map，UI 要的是带 version 的完整文档"。

> [!NOTE]
> 注意各目录的职责分工是**实测推断**出来的，不是从代码读到的。
> 换版本时按第 1、2 节重新验证，别直接沿用。

### 证据 C：扫客户端 dll 里的路径字符串

```bash
python/python.exe -c "
import re
d = open('libs/MFAAvalonia.Core.dll','rb').read()
for pat in ['copilot-cache','copilot-hold','job.json','pipeline_metadata','observer_metadata']:
    key = pat.encode('utf-16-le')   # .NET 字符串是 UTF-16LE
    print(pat, len(re.findall(re.escape(key), d)))
"
```

**注意**：
- 主程序集 `MFAAvalonia.dll` 只有 36KB，是壳。真正的逻辑在 `libs/MFAAvalonia.Core.dll`（2.8MB）。
- .NET 字符串是 **UTF-16LE**，用 `strings` 工具（默认按 ASCII）**搜不到中文和路径**，
  必须按 UTF-16LE 编码去 `re.finditer`。
- 别对整个 2.8MB 文件做逐字节 Python 循环，会跑到超时。用 `re.finditer` 只在命中点附近解码。

扫到的字符串配合上下文能还原客户端的目录策略。实测（v2.0.0-alpha.1）：

| 字符串 | 命中 | 含义 |
|---|---|---|
| `copilot-cache` | 3 | 作业列表来源（UI 读这里） |
| `job.json` | 1 | 客户端会把选中作业写成此名 |
| `删除旧作业失败: ` | 1 | 客户端会清理旧作业——所以同目录只能有一个 |
| `自动抄作业` | 4 | 默认任务的入口节点名 |
| `recordings` | 2 | 录制作业存放处 |
| `已迁移 copilot-cache 至资源根目录` | — | 目录策略会随版本变，别写死 |

> [!WARNING]
> **命中 0 次 ≠ 目录不存在。** `copilot-hold` 和 `pipeline_metadata` 在 dll 里
> 搜不到，但它们作为**磁盘目录**真实存在。所以：
> - dll 字符串只能用来**证实**客户端会主动操作的路径；
> - 想了解目录的实际用途，要去**看目录里的文件**，而不是搜代码。

## 4. 确认结论：构造最小验证

**任何假设都要用实测否证，不能只靠推理。** 2026-10-06 犯过的错：
"把 `金乌枝1-吕布队.json` 放进 base 加载失败" → 误以为"作业不能进 base"。
实际上是它带着 `difficulty` 标量字段。**剥成裸节点 map 后放进去完全正常。**

反例的价值在于：它证明了旧假设的**真正**失效原因。

标准验证动作：

```bash
# 剥离标量顶层字段，只留节点，再测
python/python.exe -c "
import json
d = json.load(open('作业.json', encoding='utf-8'))
json.dump(d['actions'], open('摊平.json','w',encoding='utf-8'), ensure_ascii=False)
"
# 加载测试 -> 对比结果
```

## 5. 硬性约束（v2.0.0-alpha.1 实测）

以下是**当前版本**的结论，升级后需按第 3 节重新验证。

### 5.1 作业必须同时落三处

| 位置 | 放什么 | 格式 |
|---|---|---|
| `resource/copilot-cache/` | 完整作业文档，供 UI 列出 | 带 `version`/`doc`/`actions` |
| `resource/base/pipeline/copilot/` | 引擎真正加载的节点 | **裸节点 map**（只含 actions） |
| `resource/base/pipeline_metadata/copilot/` | 观察者元数据 | `*.observer_metadata.json` |

`export_copilot_json()` 已自动完成三处同步。

### 5.2 两条不可违反的约束

**A. 同目录只能有一个作业文件。** 两个作业会撞约 19 个同名节点
（`检测回合N`、`回合N行动M`、`作业信息`、`抄作业*`）。客户端会写自己的作业进来，
编译新作业必须清旧的——`export_copilot_json` 已自动做。

**B. 不能覆盖 `copilot_config.json` 已有节点。** 引擎对重复 key 直接拒绝加载：

```
key already exists [key=抄作业-进入编队界面]
```

**特别注意：写 `{"enabled": false}` 想"禁用"它也不行。** 重名就是重名，
`enabled` 语义在解析层之前就失败了。2026-10-06 正是这么埋雷的。

正确做法：不覆盖。若确实要改变基础节点行为，去改 `copilot_config.json` 本身。

### 5.3 报错速查

| 引擎报错 | 根因 | 正解 |
|---|---|---|
| `key already exists [key=xxx]` | 覆盖了 config 已有节点 | 删掉该节点（禁用也不行） |
| `value is not object [key=difficulty]` | 完整作业文档进了 base | 那里只放裸节点 map |
| `check_all_validity failed`（无明细） | 两作业并存互相覆盖 | 该目录只留一个 |
| `Invalid next node name [检测回合1]` | 缺作业节点 | 正常；纯 config 状态必然报 |
| `not callable` on `node_list`/`failed` | 误当方法调 | 它们是 property |
| `No module named 'maa'` | python 环境不对 | 用 `python/python.exe` |
| 验证脚本报「未找到安装目录」 | 有 `resource/base` 空壳残留 | 删空壳；探测须校验 `copilot_config.json` 存在 |

## 6. 环境陷阱

- **测试文件不要用相对路径写死**。`"resource/base/pipeline/xxx.json"` 会在
  当前工作目录留下空壳目录，还会让安装根探测被骗过去。用 `tempfile.TemporaryDirectory()`。
- **安装根探测要严格**：只判断 `resource/base` 会被空壳骗到，
  必须校验 `resource/base/pipeline/copilot/copilot_config.json` 存在。
- **Git Bash + Python 路径**：`/d/xxx` 在 Python 里不通，要用 `D:\xxx`。

## 7. 交付前

```bash
python/python.exe scripts/verify_resource_load.py   # 退出码 0 才算可交付
```

它做三件事：静态扫全包重名 key → 检查 cache 卫生 → 实测 `post_bundle(base)`。

**不要跳过它。** 这个故障的表现是「整个客户端都用不了」，
而根因往往在看似无关的目录里，手工排查一轮要半小时。