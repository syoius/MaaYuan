# 角色物品与数量识别使用说明

本文说明如何准备 `*-bag.png` 角色模板、生成 `.npz` 索引，以及在 Maa Pipeline 中调用 `AgentItemRecognition` 批量识别角色和数量。识别器可同时用于寿春额外奖励和原始背包界面。

所有命令均假定当前目录为项目根目录 `.\MaaYuan`。

## 文件说明

- `extract_bag_portraits.py`：从全屏背包截图中定位名称包含“心纸”的条目，并裁出 `70×58` 角色模板。
- `extract_bag_items.py`：按截图顺序提取普通道具模板，遇到第一个“心纸”时停止。
- `crop_bag_icons.py`：将独立的圆形角色图标（解包素材）转换成统一的 `70×58` 模板。
- `build_shouchun_agent_index.py`：从模板生成运行时使用的压缩 NPZ 索引。
- `build_bag_item_index.py`：根据 `agent/items.json` 递归校验并生成普通道具索引。
- `build_agent_item_digit_index.py`：从已确认截图生成轻量的 `0–9` 数字字形索引。
- `bag-templates/`：当前汇总的 `*-bag.png` 模板。
- `../../agent/custom/reco/agent_item.py`：运行时 Custom Recognition。
- `../../assets/resource/base/pipeline/shouchun_reward.json`：single 和 batch 调用示例。

## 环境与依赖

先安装项目已有依赖：

```powershell
python -m pip install -r requirements.txt
```

这些脚本和识别器只使用项目已有的 `maafw`、`opencv-python` 和 NumPy，不需要新增第三方库。NumPy 会随 OpenCV 安装。

模板提取 OCR 默认使用项目配置的 `ppocr_v6/small`。角色物品的运行时数量识别默认使用数字字形索引，不调用 OCR；`ppocr_v6/small` 仅用于模板提取和可选的旧 OCR 模式。源码模型位于：

```text
assets/MaaCommonAssets/OCR/ppocr_v6/small
```

发布后的默认模型位于：

```text
resource/base/model/ocr/
```

## 一、从全屏背包截图提取模板

脚本只保留 OCR 文本中含“心纸”（同时兼容“心紙”）的条目。未包含该标记的物品、被遮挡的不完整条目和无法映射到 `operators.json` 的条目不会作为有效模板写入。

处理指定截图：

```powershell
python tools\analytics\extract_bag_portraits.py `
  tools\analytics\bag-template-1.png `
  tools\analytics\bag-template-2.png `
  -o tools\analytics\bag-templates-new `
  --debug
```

处理 glob 或整个目录：

```powershell
python tools\analytics\extract_bag_portraits.py `
  "tools\analytics\bag-template-*.png" `
  -o tools\analytics\bag-templates-new `
  --overwrite `
  --debug
```

常用参数：

- `-o/--output`：输出目录，必填。
- `--overwrite`：允许覆盖已有同名模板。
- `--recursive`：输入目录时递归扫描。
- `--debug`：在输出目录的 `debug/` 下保存带 OCR 框和裁切框的图片。
- `--model`：覆盖离线 OCR 模型目录。
- `--ocr-threshold`：调整名称 OCR 的最低置信度。
- `--safe-top`、`--safe-bottom`：调整用于排除顶部或底部遮挡条目的安全范围。

输出内容包括：

- `<agent-id>-bag.png`：统一的 `70×58` BGR 模板。
- `bag-extraction-report.json`、`bag-extraction-report.csv`：每个候选的识别、映射和裁切状态。
- `missing-operators.csv`：`operators.json` 中尚未获得模板的角色。

建议先查看调试图和报告，再将确认无误的模板合并到 `tools/analytics/bag-templates/`。

### 提取“心纸”之前的普通道具

`extract_bag_items.py` 使用 OCR 原文命名道具模板，按文件名自然顺序、从上到下、从左到右处理；遇到第一个含“心纸”或“心紙”的名称时，不导出该项并停止处理后续内容：

```powershell
python tools\analytics\extract_bag_items.py `
  "tools\analytics\bag-items-sample-*.png" `
  -o tools\analytics\bag-items-extracted `
  --debug
```

默认每张截图只接受顶部四个完整的四列行，以避开底部装饰截断的第五行。第 1–3 张截图底部项目会在下一张顶部完整出现，因此不会遗漏或重复；当前四张样例会导出 51 个模板，并在 `bag-items-sample-4.png` 第一行第四列的“心纸【司马徽】”处停止。

输出文件默认为 `<OCR名称>-bag.png`，尺寸为 `70×58`。传入 `--output-suffix ""` 可改为 `<OCR名称>.png`。`bag-items-extraction-report.json` 和 `.csv` 会记录 OCR 原文、置信度、来源、行列及裁切框；使用 `--debug` 时还会生成定位图。运行后应先人工检查 OCR 文件名，尤其是生僻道具名。

相关参数：

- `--max-full-rows`：每张截图接受的完整行数，默认 `4`。
- `--label-to-crop-top`：名称框顶边到模板顶边的距离，默认 `96`。
- `--crop-width`、`--crop-height`：模板尺寸，默认 `70×58`。
- `--ocr-threshold`、`--model`：OCR 阈值和模型目录。
- `--overwrite`：允许覆盖运行前已有的同名文件；同一次运行内的重名仍会自动添加 `-2`、`-3`。

## 二、转换独立角色图标

如果输入是独立圆形图标（解包素材），而不是全屏背包截图，可以使用：

```powershell
python tools\analytics\crop_bag_icons.py `
  tools\analytics\standalone-icons `
  -o tools\analytics\bag-templates-new `
  --recursive
```

输出仍为 `<原文件名>-bag.png`，并生成 `crop-bag-icons-report.json`。合并前请确认文件名中的角色名能映射到 `agent/operators.json`。

## 三、生成 NPZ 索引

### 寿春派遣奖励界面：0.9 比例

默认命令生成 `0.89/0.90/0.91` 三组模板，并写入 `agent/shouchun-agent-index.npz`：

```powershell
python tools\analytics\build_shouchun_agent_index.py
```

三组相邻比例用于容忍寿春派遣奖励界面的轻微缩放差异。

### 原始背包界面：1.0 比例

`*-bag.png` 本身就是从背包界面原尺寸截取的，因此只需要一组 `1.00` 模板：

```powershell
python tools\analytics\build_shouchun_agent_index.py `
  --scales 1.0 `
  --output agent\bag-agent-index.npz
```

生成的索引只包含 `templates_100`，比三比例索引更小，匹配时也只执行一次完整模板验证。

### 普通道具索引与 items.json

普通道具使用 `agent/items.json` 维护稳定的拼音 ID、中文名、分类和模板路径。运行时所需信息会直接写入 NPZ，因此发布识别不需要再次读取 JSON；JSON 用于生成期校验和后续维护：

```powershell
python tools\analytics\build_bag_item_index.py
```

默认递归扫描 `tools/analytics/bag-items/` 和 `tools/analytics/bag-items-more/`，并输出 `agent/bag-item-index.npz`。所有递归发现的 `*-bag.png` 必须在 `items.json` 中恰好登记一次，文件名必须为 `<拼音-id>-bag.png`；有遗漏、重复、中文文件名、尺寸不是 `70×58` 或 ID 与文件名不一致时会直接报错。

`items.json` 中的 `refine_groups` 会随索引写入。当前六种金锁共用 `element-jinsuo` 复核组：第一次完整匹配命中任一金锁后，识别器自动扩展六个候选，并对模板右下角 `[43,29,27,29]` 做彩色 NCC 复核。

### 自定义输入与输出

```powershell
python tools\analytics\build_shouchun_agent_index.py `
  --templates tools\analytics\bag-templates `
  --operators agent\operators.json `
  --scales 1.0 `
  --feature-scale 1.0 `
  --output agent\bag-agent-index.npz
```

说明：

- `--scales` 接受一个或多个比例，最多保留两位小数。
- `--feature-scale` 必须是 `--scales` 中的某一项；省略时取中间项。
- 模板必须统一为 `70×58`，文件名必须以 `-bag.png` 结尾。
- 每个模板 ID 都必须能映射到 `agent/operators.json`。
- 生成结束时会进行 Top-1 自检；不是全部通过时脚本会报错退出。

发布时将索引放在发布目录的 `agent/` 下。项目的安装脚本会复制整个 `agent` 目录。

## 四、在 Pipeline 中调用识别器

### 自适应布局模式（推荐）

当物品起始高度、行距或行数会变化时，使用 `layout_mode: "auto"`。识别器会在 ROI 内检测圆形物品底板，自动聚类成行并按从左到右、从上到下识别，不需要传入 `grid`：

```json
{
  "recognition": {
    "type": "Custom",
    "param": {
      "custom_recognition": "AgentItemRecognition",
      "custom_recognition_param": {
        "index_path": "agent/shouchun-agent-index.npz",
        "layout_mode": "auto",
        "roi": [33, 400, 655, 650],
        "top_k": 5,
        "match_threshold": 0.9,
        "recognize_count": true,
        "count_mode": "digit_template",
        "count_required": true,
        "count_max_digits": 2
      }
    }
  }
}
```

自动模式下，ROI 只负责排除标题、基础奖励和其他圆形 UI，不再负责计算格子中心。`grid` 可以省略；也可以传列数（例如 `4`）或 `[列数, 行数]` 作为期望上限。检测结果超过该上限时会拒绝本次布局，便于发现 ROI 混入了其他圆形控件。

圆形半径默认根据所加载索引的模板比例推断。其他尺寸的界面可以用 `auto_circle_radius: [最小半径, 最大半径]` 覆盖默认范围。

自动模式默认使用 `top_k: 20`、`feature_y_offsets: [-4, 0, 4]` 和稍宽的局部模板搜索框，以吸收圆检测约数个像素的误差；参数显式传入时以传入值为准。寿春/派遣可以显式使用 `top_k: 5` 降低开销。

### 可滚动列表的分页识别 Action

当奖励列表需要滑动才能看全时，使用 `PagedItemRecognition` Custom Action。它会在内部重复执行自动布局识别、滑动列表、对齐相邻页面的重叠行，并在确认到达底部后一次性写入去重报告：

```json
{
  "派遣奖励分页记录": {
    "action": {
      "type": "Custom",
      "param": {
        "custom_action": "PagedItemRecognition",
        "custom_action_param": {
          "index_path": "agent/shouchun-agent-index.npz",
          "layout_mode": "auto",
          "roi": [33, 400, 655, 650],
          "top_k": 5,
          "match_threshold": 0.9,
          "recognize_count": true,
          "count_mode": "digit_template",
          "count_required": true,
          "count_max_digits": 2,
          "acquisition_channel": "派遣",
          "max_pages": 10,
          "swipe_wait_ms": 700,
          "overlap_threshold": 0.9
        }
      }
    }
  }
}
```

不要在这个节点的 `recognition` 中再次配置 `AgentItemRecognition`；分页 Action 会直接调用相同的底层识别函数，并且只在整个列表扫描完成后写一次报告。

分页参数：

- `acquisition_channel`：可选的获取渠道，会写入报告的同名列，并供后续在线上报使用。推荐使用稳定值 `背包`、`据点情报`、`派遣`；省略时为空字符串。
- `swipe`：可选的 `[x1, y1, x2, y2, duration_ms]`。省略时每页会根据自动布局结果动态计算：X 取最靠近 ROI 中央的两列间隙，Y 从末行中心滑到首行中心，避免手势起止点落在物品上。只有一行时，Y 才回退到 ROI 高度的 `82% → 37%`。
- `swipe_duration`：省略 `swipe` 时或 `swipe` 只有四项时使用，默认 `500 ms`。
- `swipe_wait_ms`：滑动后的界面稳定等待，默认 `700 ms`。
- `max_pages`：最多截图页数，默认 `10`。达到上限但仍未确认底部时 Action 失败且不写入不完整报告。
- `overlap_threshold`：相邻页面逐行图像相似度阈值，默认 `0.90`。

去重依据是“上一页末尾行与下一页开头行的实际物品图像”，不是简单的 `item_id + count`。因此同一角色在不同物理行真实重复出现时会保留，而滑动造成的同一格重复展示只写一次。再次滑动后整页内容不再变化时，Action 判定到达列表底部。报告中的 `mode` 为 `paged`，`row` 和 `slot` 是去重后的全局位置。Action 完成时列表会停留在底部。

### 固定 Batch 模式

```json
{
  "recognition": {
    "type": "Custom",
    "param": {
      "custom_recognition": "AgentItemRecognition",
      "custom_recognition_param": {
        "index_path": "agent/shouchun-agent-index.npz",
        "roi": [33, 816, 655, 112],
        "grid": [5, 1],
        "top_k": 5,
        "match_threshold": 0.9,
        "recognize_count": true
      }
    }
  }
}
```

固定模式是默认模式。`grid` 的格式为 `[列数, 行数]`，ROI 会被等分成对应网格，识别顺序为从左到右、从上到下。它适用于起始位置和行距都稳定的布局。

### Single 模式

```json
{
  "recognition": {
    "type": "Custom",
    "param": {
      "custom_recognition": "AgentItemRecognition",
      "custom_recognition_param": {
        "roi": [33, 816, 131, 112],
        "grid": 1,
        "recognize_count": true
      }
    }
  }
}
```

### 背包 1.0 索引

在当前 `debug` 背包截图上，可直接使用下面这组统一参数。纵向搜索偏移用来兼容滚动位置约 12 px 的变化：

```json
{
  "index_path": "agent/bag-agent-index.npz",
  "roi": [34, 245, 672, 940],
  "grid": [4, 5],
  "feature_y_offsets": [-2, 0, 2, 4, 6, 8, 10, 12],
  "match_search_box": [-41, -43, 82, 90],
  "recognize_count": true,
  "count_required": true,
  "count_mode": "digit_template",
  "count_max_digits": 6,
  "count_min_roi_bottom_distance": 150
}
```

### 数量字形索引

寿春和背包共用同一个 `0–9` 字形索引：

```powershell
python tools\analytics\build_agent_item_digit_index.py
```

默认输出为 `agent/agent-item-digit-index.npz`。运行时逐位分割和匹配数字，因此两位、三位或上万的数量使用同一套字形；只需通过 `count_max_digits` 设置允许的最大位数。

## 五、识别参数

路径参数：

- `index_path`：NPZ 路径；也兼容 `npz_path` 和旧拼写 `npx_path`。
- `record_path`：报告路径；也兼容 `output_path`。
- 相对路径均从项目根目录解析。
- 默认索引为 `agent/shouchun-agent-index.npz`。
- 默认报告为项目根目录下的 `AgentItemReport.csv`。
- 报告路径没有扩展名时自动补 `.csv`。
- `.csv` 使用带表头的追加写入；`.txt` 和 `.jsonl` 使用每行一个 JSON 对象的追加写入。

匹配参数：

- `roi`：必填，格式为 `[x, y, width, height]`。
- `layout_mode`：`fixed`（默认）或 `auto`。`auto` 根据圆形物品底板定位实际中心。
- `grid`：固定模式下为 `1` 或 `[列数, 行数]`；自动模式下可省略，也可为列数或 `[列数, 行数]` 的期望上限。
- `auto_circle_radius`：自动模式的 `[最小半径, 最大半径]`；默认根据 NPZ 模板比例推断。
- `auto_circle_threshold`：圆检测累加阈值，默认 `35`；越高越严格。
- `auto_circle_min_distance`：两个圆心的最小距离；默认是推断最小半径的 `1.7` 倍。
- `auto_row_tolerance`、`auto_column_tolerance`：自动行列聚类容差；通常无需设置。
- `top_k`：粗筛后进行完整模板验证的候选数；固定模式默认 `5`，自动模式默认 `20`。
- `coarse_threshold`：粗筛最低分，默认 `0.0`。
- `match_threshold`：完整模板匹配最低分，默认 `0.90`。
- `recognize_count`：是否识别数量，默认 `true`。
- `count_required`：数量失败时是否整格无效，默认 `true`。
- `count_mode`：默认 `digit_template`；可设为 `ocr` 使用旧 OCR 路径。
- `digit_index_path`：默认 `agent/agent-item-digit-index.npz`。
- `count_max_digits`：允许的最大位数，默认 `6`；寿春建议 `2`。
- `count_digit_threshold`：单个数字最低匹配分，默认 `0.45`。
- `count_binary_threshold`：亮色数字分割阈值，默认从数字索引读取，当前为 `170`。
- `count_badge_threshold`：灰色数量徽标阈值，默认从数字索引读取，当前为 `190`。
- `count_min_value`：允许的最小数量，默认 `1`。
- `count_min_roi_bottom_distance`：角色实际中心到 ROI 底边的最小安全距离；默认 `0`（关闭）。距离不足时整格以 `count-too-close-to-roi-bottom` 排除。
- `feature_y_offsets`：粗筛框需要检查的纵向偏移数组；滚动背包建议使用下文参数。
- `enable_refine`：是否启用索引内定义的条件复核组，默认 `true`。
- `refine_threshold`、`refine_min_margin`：覆盖索引内的复核最低分和第一/第二名最小分差。
- `count_ocr_model`、`count_ocr_threshold`、`count_ocr_scale`：仅在 `count_mode: "ocr"` 时使用。

头像粗筛框和搜索框相对于网格单元中心：

```json
{
  "feature_box": [-24, -27, 48, 44],
  "match_search_box": [-41, -38, 82, 71]
}
```

完整模板命中后，识别器会从实际 `match_box` 反算头像中心；数量框相对于这个校正后的头像中心，而不是原始网格中心。数字索引当前默认数量框为 `[-35, 20, 95, 44]`。`match_search_box` 必须完整容纳目标模板；1.0 模板尺寸为 `70×58`。

### 寿春派遣奖励界面识别推荐参数

```json
{
  "index_path": "agent/shouchun-agent-index.npz",
  "layout_mode": "auto",
  "roi": [33, 816, 655, 112],
  "top_k": 5,
  "recognize_count": true,
  "count_mode": "digit_template",
  "count_required": true,
  "count_max_digits": 2
}
```

### 背包界面识别推荐参数

```json
{
  "index_path": "agent/bag-agent-index.npz",
  "layout_mode": "auto",
  "roi": [34, 245, 672, 940],
  "top_k": 20,
  "recognize_count": true,
  "count_mode": "digit_template",
  "count_required": true,
  "count_max_digits": 6,
  "count_min_roi_bottom_distance": 150
}
```

自动布局不依赖每次滚动停止位置。背包角色索引建议使用 `top_k: 20`，因为少数角色的灰度粗筛排名低于 12；完整模板验证仍能以接近 `1.0` 的分数区分。当前截图第五行角色中心到 ROI 底边只有 `85–94 px`，而第四行有 `271–282 px`，所以建议以 `150 px` 安全线排除被底部装饰截断的数量。

### 普通道具背包识别推荐参数

普通道具截图的首行纵向位置变化比角色截图更大，因此需要更宽的搜索框和 `8–54` 的偶数偏移：

```json
{
  "index_path": "agent/bag-item-index.npz",
  "layout_mode": "auto",
  "roi": [34, 245, 672, 940],
  "top_k": 8,
  "match_threshold": 0.9,
  "enable_refine": true,
  "recognize_count": true,
  "count_mode": "digit_template",
  "count_required": true,
  "count_max_digits": 6,
  "count_min_roi_bottom_distance": 150
}
```

自动布局会直接使用实际圆心，不再需要为不同首行高度列出大量 `feature_y_offsets`。该参数在背包样图上会自动推断 `4×5` 布局；`count_min_roi_bottom_distance` 仍负责排除数量被底边截断的最后一行。

## 六、报告内容

每个成功识别的格子写入一行，主要字段包括：

- `timestamp`：带时区的识别时间。
- `invocation_id`：同一次 single/batch 调用的唯一 ID。
- `mode`：固定布局为 `single` 或 `batch`，自动布局为 `auto`。
- `acquisition_channel`：`PagedItemRecognition` 配置的获取渠道；其他识别或未配置时为空。
- `slot`、`row`、`column`：网格位置。
- `entity_type`、`item_id`、`item_name`：通用条目类型、稳定 ID 和中文名。
- `agent_id`、`operator_id`、`operator_name`：向后兼容的角色字段；普通道具索引中与对应 item 字段一致。
- `count`、`count_score`、`count_raw`：数量及逐位模板匹配信息。
- `coarse_score`、`match_score`、`match_scale`：角色匹配诊断信息。
- `refined`、`refine_group`、`refine_score`、`refine_margin`：条件复核状态和分数。
- `cell_box`、`match_box`、`item_center`、`count_box`：实际使用的坐标和校正后头像中心。
- `index_path`：本次识别使用的索引。

Custom Recognition 返回的 `detail.layout` 还会包含自动模式的 `detected_count`、`inferred_grid`、行列中心以及每个检测圆心，便于定位 ROI 或圆检测参数问题。CSV 中自动布局调用的 `mode` 为 `auto`。

只有成功通过角色匹配阈值的格子会写入报告。未识别格子的原因会包含在 Custom Recognition 返回的 `detail.rejected` 中。

## 七、常见问题

### 新界面完全识别不到角色

依次检查：

1. 是否使用了正确比例的索引：寿春界面用 0.9，原始背包界面用 1.0。
2. 自动模式检查 `detail.layout` 是否检测到正确数量的圆；固定模式检查 `roi` 和 `grid` 是否让每个格子的中心落在正确位置。
3. `feature_box` 是否位于头像内部。
4. `match_search_box` 是否完整覆盖头像且大于模板尺寸。
5. 临时降低 `match_threshold` 查看最佳候选及分数，再决定是否调整阈值。

### 角色正确但数量为空

确认发布包中存在 `agent/agent-item-digit-index.npz`，并检查 `count_box` 是否覆盖完整的灰色数量徽标。默认 `count_required: true`，因此没有可见数量、数字分割不完整或任一位低于阈值时，该格会以 `count-not-recognized` 进入 `detail.rejected`，不会写入报告。

### 新增模板后识别器没有更新

把模板放入 `bag-templates/` 后必须重新生成对应 NPZ，并确保 Pipeline 指向新文件。识别器会按文件修改时间和大小缓存索引；文件更新后会自动重新加载。
