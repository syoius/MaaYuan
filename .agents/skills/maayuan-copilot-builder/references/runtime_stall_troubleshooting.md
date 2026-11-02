> 历史排查记录：版本、路径和激活方式以当时事件为准。当前默认导出仅写缓存，旧文中“三处自动同步/清旧作业”的操作不适用于当前默认导出。请先读 SKILL.md。

# 排查「战斗中途中停下」的方法论

> 与 `resource_load_troubleshooting.md` 是**两个不同的故障**：
> 那份处理「客户端整个都用不了」，本份处理「客户端能进，但作业跑到某一步停住」。
>
> 本文记录**如何推导**，不是答案清单。换版本/换设备后按第 1~3 节重新推。

## 0. 心智模型：什么会卡，什么不会卡

这是排查的**第一步**，方向搞错会白费很多时间。

作业节点分两类：

| 类型 | 动作 | 会卡住吗 |
|---|---|---|
| **操作节点**（Click / Swipe / DirectHit） | 直接按下就走，不看画面 | **不会**。执行完必定进入 next |
| **识别节点**（TemplateMatch / OCR / Custom） | 要先在画面里找到目标 | **会**。找不到就一直重试，直到 `timeout`耗尽 |

**所以作业停在某处，问题一定在那一步或紧邻的识别节点上，其他节点不用怀疑。**

用户看到的是"停下了"，实际发生的是：识别节点反复重试 → 20 秒左右超时 →
`Task timeout` → **整个任务被判失败结束**（不是跳过这一步）。

关键日志特征（`debug/maa.log`）：

```
TemplateMatcher::analyze  <节点名> [all_results_=[{"box":[...],"score":0.68...}]]
Recognizer::recognize reco [result={... "score":0.68 ...}]
Node.Recognition.Failed
PipelineTask::run_next Task timeout [pretask.name=<上一个节点>]
```

`pretask.name` 指的是**上一个已完成的节点** —— 真正卡住的是它的 next 指向的那个识别节点。

## 1. 定位停在哪个节点（3 种途径）

### 途径 A：看作业面板的 focus（最快）

作业详情里每步都有 `focus`，形如：

```
第2回合[第2回合] 第2动(总第2动): 4号位切形态
```

**最后显示的那一条 focus 就是卡住的位置。** 让用户念/截图给你，比自己猜快得多。

### 途径 B：搜引擎日志（最准）

```bash
cd D:/MaaYuan-win-x86_64-v2.0.0-alpha.1
python/python.exe -c "
import re
d=open('debug/maa.log','rb').read().decode('utf-8','replace')
lines=d.splitlines()
for i,l in enumerate(lines):
    if 'Task timeout' in l:
        print('--- 超时上下文 ---')
        for j in range(max(0,i-14), i+3):
            if 'all_results_' in lines[j]:
                print('   ', re.sub(r'\[[^\]]*\.cpp\]\[L\d+\]','',lines[j])[:150])
            elif 'Recognition.Failed' in lines[j] or 'timeout' in lines[j]:
                print('   ', re.sub(r'\[[^\]]*\.cpp\]\[L\d+\]','',lines[j])[:120])
"
```

### 途径 C：看失败截图

客户端会在 `debug/on_error/` 自动存下失败时刻的整屏截图，文件名带**节点名**和时间戳：

```bash
ls -lt --time-style=+%H:%M debug/on_error/ | head
# 回合2行动1_2026.10.06-19.47.26.374.png
```

**这是最有价值的物证** —— 能直接看到那一刻画面上有什么、按钮在哪、是什么状态。

> [!TIP]
> 文件名里的节点名是**卡住节点的上一个**（即超时后`pretask.name` 那个），
> 因为作业在该节点的 next 里出错。往前推一格就是真正的卡点。

## 2. 读懂匹配分数（本次实战的核心）

从日志或截图算出分数后，**分数的含义决定下一步动作**：

| 分数特征 | 含义 | 怎么办 |
|---|---|---|
| 稳定在阈值附近（如 0.66~0.74 vs 阈值 0.8） | **模板和实际画面不像**，但位置对 | 换模板 / 降阈值 |
| 分数接近 0 | 模板根本没出现 | 目标不存在或状态不对（按钮灰显？UI 变了？） |
| 分数很高但动作没生效 | 识别没问题，是动作逻辑错 | 查点击坐标、动作类型 |

**关键判断：分数是"稳定偏低"还是"忽高忽低"。**

- **稳定偏低**（每次重试都在同一区间）→ 模板外观问题，调阈值或换模板。
- **忽高忽低**（有时 0.9 有时 0.3）→ 画面有动画/光效干扰，要等稳定或用区域截图。

本次实战就是"稳定偏低"：12 次重试分数在 0.66~0.74 之间，
证明**引擎定位是准的**（命中框正好落在按钮上），纯粹是外观相似度不够。

### 离线复算分数（可代替反复试跑）

`debug/on_error/` 有失败截图，用 PIL 就能复算 MaaFramework 的匹配分数，
不必反复实跑战斗：

```python
from PIL import Image
import math

def ncc(a, b):                      # 归一化互相关，与引擎算法一致
    pa = list(a.getdata()); pb = list(b.getdata())
    ma = sum(sum(p) for p in pa) / (len(pa) * 3)
    mb = sum(sum(p) for p in pb) / (len(pb) * 3)
    num = da = db = 0
    for x, y in zip(pa, pb):
        for c in range(3):
            d1, d2 = x[c] - ma, y[c] - mb
            num += d1 * d2; da += d1 * d1; db += d2 * d2
    return num / math.sqrt(da * db + 1e-9)

tpl = Image.open('resource/base/image/copilot/lb_l2h.png').convert('RGB')
shot = Image.open('debug/on_error/<失败截图>.png').convert('RGB')
actual = shot.crop((470, 1089, 470 + 59, 1089 + 51))   # 用日志里的命中框
print(ncc(tpl, actual))# 与引擎实测分数应高度吻合
```

2026-10-06 实测：离线算 0.7429，引擎实测 0.70~0.74 —— **吻合，可放心用于校准阈值**。

全 ROI 搜索最佳位置还可以确认引擎有没有找偏：

```python
best = max(((ncc(tpl, shot.crop((x, y, x+59, y+51))), (x, y))
            for x in range(440, 540, 4) for y in range(1070, 1140, 4)))
```

## 3. 模板图跟着站位走（最容易踩的坑）

**模板图里带着背景，换站位就可能失效。**

2026-10-06 的实际案例：

- 参照作业 `金乌枝1-吕布队.json` 里，吕布站 **1 号位**，模板 `lb_l2h.png` 是那时截的。
- 本次作业吕布站 **4 号位**，我直接沿用了那份配置（模板/ROI/阈值逐字相同）。
- 结果分数只有 0.66~0.74，**永远过不了 0.8**。
- 引擎定位完全正确（命中框就在按钮上），差异纯粹来自背景/光效。

**排查时必做**：从失败截图上裁出实际按钮，和旧模板并排看。

```python
btn = shot.crop((470, 1089, 470+59, 1089+51))
btn.resize((btn.width*5, btn.height*5), Image.NEAREST).save('debug/_cmp/actual.png')
old = Image.open('resource/base/image/copilot/lb_l2h.png').convert('RGB')
old.resize((old.width*5, old.height*5), Image.NEAREST).save('debug/_cmp/tpl.png')
```

裁新模板时**以引擎的命中框为准**（不要自己估位置）：

```python
btn.save('resource/base/image/copilot/lb_form_s4.png')   # 命名带站位号
```

> [!WARNING]
> 新模板若裁自**失败瞬间**，画面可能带异常状态（强光、灰显）。
> 稳妥做法是**双保险**：新模板 + 旧模板一起列进 `template` 数组，
> 再加一个坐标兜底分支（见第 4 节）。

## 4. 双保险：识别 + 坐标兜底（工程要求）

**任何识别节点都不该让整个任务因它失败。** 坐标操作（Click/Swipe）本来就是
确定位置，没理由绕一圈去识别。

推荐结构（编译器已实现，见 `special_operations.switch_form`）：

```json
"回合2行动2": {
  "recognition": {
    "type": "TemplateMatch",
    "param": {
      "template": ["copilot/lb_form_s4.png", "copilot/lb_l2h.png", "copilot/lb_h2l.png"],
      "roi": [459, 1072, 80, 95],
      "threshold": 0.65
    }
  },
  "action": {"type": "Click", "param": {}},
  "post_delay": 1500,
  "next": ["回合2行动2兜底", "史子眇sp", "回合2行动3"]
},
"回合2行动2兜底": {
  "recognition": "DirectHit",
  "action": "Click",
  "target": [499, 1114, 5, 5],
  "post_delay": 1500,
  "next": ["史子眇sp", "回合2行动3"]
}
```

三层保障：
1. **站位专属模板放首位** —— 命中率高；
2. **旧模板兜底** —— 两种形态状态都能认；
3. **ROI 按站位收窄 + 坐标兜底分支** —— 识别失败也照常执行。

兜底节点的 `next` 必须**指向原节点本该去的地方**，否则会走死路。

## 5. 编译期就该拦住的问题

**资源加载时不会校验模板图是否存在。** 缺图不会报「资源加载失败」，
而是等战斗中跑到那一步才发现匹配永远失败 —— 排查成本极高。

所以加了 `scripts/validate_templates.py`，编译后立刻检查：

```bash
python/python.exe scripts/validate_templates.py          # 校验全部作业
python/python.exe scripts/validate_templates.py作业.json  # 只校验指定作业
```

它会递归收集所有 `TemplateMatch.template` 引用，逐个确认
`<MAA根>/resource/base/image/` 下存在，缺失则退出码 1。

## 6. 排查清单（按顺序执行）

1. 问用户最后一条 `focus` 是什么 → 定位卡点（第 1 节途径 A）
2. 搜 `debug/maa.log` 的 `Task timeout` + `all_results_` → 确认分数与命中框（第 2 节）
3. 看 `debug/on_error/` 失败截图 → 确认按钮是否在、画面什么状态（第 1 节途径 C）
4. 分数稳定偏低 → 怀疑模板；并排比对新旧模板（第 3 节）
5. 用命中框裁新模板，按站位命名，列进 `template` 数组首位
6. ROI 收窄到本站位，阈值降到实测分数之下
7. **加坐标兜底分支**（第 4 节）—— 这条是硬性工程要求
8. 改档位配置 JSON（坐标/ROI/阈值/兜底的**唯一真源**），不要在编译器里硬编码
9. 重新编译 → `validate_templates.py` → `verify_resource_load.py` → 跑测试

## 7. 与「资源加载失败」的区别

别混淆，两者排查路径完全不同：

| 症状 | 属于 | 看哪里 |
|---|---|---|
| 客户端进不去，任何功能都用不了 | `resource_load_troubleshooting.md` | `logs/`、`base/pipeline` |
| 能进，但作业跑到某步停住 | 本文档 | `debug/maa.log`、`debug/on_error/` |