# Design Spec — 俯视小地图距离可视化 (`viz_minimap.py`)

日期: 2026-06-17
作者: Frank (KNQ 实习 / SoccerMaster 足球 pipeline)

## 1. 目标

为 `pitch_distances.py` 算出的球场距离做一个**俯视小地图(bird's-eye)动态可视化**,
输出整段序列的 MP4 + 一份同目录的全量距离 CSV,方便看球员到边界距离随时间的变化、
并供后期取数。

核心需求(来自最初任务): 看清**球员到球场边界的物理距离(米)**,且能动态展示数字变化。

## 2. 范围与非目标

- **范围**: 一个独立脚本 `viz_minimap.py`,消费 `pitch_distances.py` 的输出,产出 MP4、CSV、可选单帧 PNG。
- **非目标**:
  - 不修改 `pitch_distances.py`(两段解耦)。
  - 不重新计算距离逻辑——边界距离已在 `_per_player.csv` 中。
  - 不做球检测(calib 模式下球轨迹由上游提供或缺失,见 §6 边界情况)。

## 3. 架构 / 数据流

```
pitch_distances.py ─► <seq>_per_player.csv      (已有, 全量, 含 role=="ball" 行)
        │
viz_minimap.py     ─► <seq>.mp4                   (转播式动态焦点动画)
                   ├─► <seq>_boundary_distances.csv  (同目录, 全量, 自包含)
                   └─► <seq>_frame_<id>.png          (可选, --image-id 抽检)
```

`viz_minimap.py` 只读 `_per_player.csv`,不依赖原始帧图像。输出全部落在
`--out-dir`(默认与输入 CSV 同目录),使每个序列的输出目录自包含。

## 4. 焦点逻辑(动态 / 持球者)

每帧:
1. 取该帧 `role=="ball"` 行的球场坐标 (x, y)。
2. 焦点 = 该帧中离球最近的 `player`/`goalkeeper`(球场米坐标欧氏距离最小者)。
3. 渲染:
   - 焦点高亮(大点 + 球衣号),其余球员淡化(alpha≈0.32)。
   - **黄线**: 焦点 → 最近边界,标实时米数(核心需求)。
   - **细白线**: 焦点 → 最近对手(不同 team 的 player/gk),标米数。
   - 球画成白点。
4. 焦点逐帧切换(转播跟拍持球者),HUD 数字随之跳变。

唯一稳定的球员标识是 `track_id`(球衣号不唯一: SNGS-060 左右队都有 #10)。

## 5. 输出规格

### 5.1 MP4 (`<seq>.mp4`)
- 整段俯视小地图动画。FIFA 标准标线(外框 105×68、中线、中圈 r=9.15、禁区
  16.5×40.32、球门区 5.5×18.32、点球点、球门),原点在中圈。
- 配色: 左队红、右队蓝、裁判橙、球白。
- 左上 HUD: 当前帧号 + 持球者球衣号 + 持球者到最近边界米数。
- Writer: ffmpeg(确认目标机器已装)。

### 5.2 距离 CSV (`<seq>_boundary_distances.csv`)
- 列: `image_id, track_id, role, team, jersey, x, y, to_goal_line, to_touch_line, to_nearest_boundary`
- **每帧每人一行,全量不漏**(不只是持球者)。
- 实现: 从 `_per_player.csv` 裁出上述列写到 MP4 同目录(目录自包含)。

### 5.3 单帧 PNG (可选)
- `--image-id <id>` 指定帧出 PNG 抽检,默认不出。

## 6. 参数 / CLI

| 参数 | 默认 | 说明 |
|---|---|---|
| `per_player_csv` (位置参数) | — | `pitch_distances.py` 输出的 `*_per_player.csv` |
| `--out-dir` | 输入 CSV 同目录 | 输出根目录 |
| `--fps` | 25 | MP4 帧率 |
| `--stride N` | 1 | 每 N 帧取一帧,控体积/速度 |
| `--focus <track_id>` | 无(动态) | 给定则固定跟该球员;不给走动态持球者 |
| `--image-id <id>` | 无 | 出单帧 PNG 抽检 |
| `--length` / `--width` | 105 / 68 | 球场尺寸 |

> 注: 设计选定动态焦点为默认。`--focus` 作为可选的固定模式覆盖,低成本保留。

## 7. 边界情况

- **某帧无球**: carry-forward,沿用上一帧持球者;若整段开头就无球,用第一帧任意
  player 作初值。
- **整段无球**(calib 模式 / 无球检测): 退回 `--focus <track_id>` 固定模式;
  未指定则不画焦点高亮,仅画所有点 + CSV 照常输出。
- **焦点无对手**(同队全在场?极少): 跳过白线,只画到边界的黄线。
- **球员在界外**(距离为负): 正常标负值(`is_off_pitch` 上游已处理),不崩。
- **空帧 / 缺列**: 跳过该帧,日志记一行,不中断整段渲染。

## 8. 验证

- 在 SNGS-060(GT,含球标注)上跑通,人工抽检:
  - 持球者高亮是否跟随球移动。
  - 黄线米数是否与 `_per_player.csv` 的 `to_nearest_boundary` 一致(同一来源,应完全相等)。
  - 坐标系约定(原点中圈、x 长轴 ±52.5、y 宽轴 ±34)与 `pitch_distances.py` 一致。
- 全量 CSV 行数 = 各帧球员/裁判/球的标注行数之和(与 `_per_player.csv` 一致)。

## 9. 依赖

- `matplotlib`(已确认 3.10.x)+ ffmpeg(已确认目标机器有)。
- 不引重库;纯标准库 + matplotlib。
