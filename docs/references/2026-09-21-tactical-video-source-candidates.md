# 战术视频候选来源（2026-09-21）

目标：补充与 `data/GSR_2025_videos` 相似的足球片段，优先约 30 秒、比赛摄像机或转播主视角、无回放，并覆盖破线传球、快速反击、长传冲吊。

## SoccerNet 之外：允许切分时的来源

允许从长录像取 30 秒窗口后，不必局限于现成的短片数据集。以下区分已确认的视频入口与尚未确认的下载/使用权限；本轮没有下载或逐段播放验证素材。

| 来源与已核实入口 | 适合怎样使用 | 限制 |
| --- | --- | --- |
| [NWSL+ 全场录像库](https://plus.nwslsoccer.com/)；[2025 决赛 Washington Spirit vs. Gotham FC](https://plus.nwslsoccer.com/video/887883) 的播放器显示时长 2:24:20 | 从职业比赛转播中筛连续实况窗口，适合补充破线、反击和长传候选 | 无原生战术标签；可能有导播切镜/即时回放。部分入口要求登录，地区及账号可用性需实测；已确认观看页面，未确认允许下载或再分发 |
| [FIFA 比赛档案](https://www.fifa.com/fifaplus/en/archive) | 官方档案包含男、女足世界杯比赛、片段和集锦；寻找完整比赛版本以保留战术前因后果 | 已确认档案入口，但本轮未验证具体全场视频的播放和下载权限；不要将集锦当作连续实况 |
| [TeamTrack 官方仓库](https://github.com/AtomScott/TeamTrack)；[Kaggle](https://www.kaggle.com/datasets/atomscott/teamtrack) / [作者提供的 Google Drive](https://drive.google.com/drive/u/1/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ) | 官方明确提供 4K 至 8K 全场地视野视频；选 `soccer_side`，再考虑 `soccer_top`；目录明确包含原始 MP4，可按序列长度切窗 | 全场地视野不等于完整 90 分钟比赛；视角与 GSR 的移动转播主镜头不同。只有跟踪标注，不应当成有战术真值；Kaggle 数据卡标 MIT，发布前仍需核对具体数据条款 |
| [SoccerTrack 作者数据页](https://www.kaggle.com/datasets/atomscott/soccertrack) | 明确提供大学年龄球员的 11 对 11 比赛录像，含 `wide_view` 和 `top_view`、30 fps、球员与球的跟踪标注以及 GNSS 数据；页面列出 Version 6，6.83 GB | 不是职业转播主镜头；不要使用 `viz_results` 中已叠加标注框的视频作为原始输入。数据卡标 LGPL 3.0；本轮未核实逐段时长及与 TeamTrack 足球部分的重合，不能直接相加计量 |
| [Hudl Wyscout](https://www.hudl.com/products/wyscout) | 官方提供完整比赛、片段和事件时间轴，并支持向 Hudl 传送 full matches 与事件 XML；适合有预算时按事件缩小检索范围 | 商业订阅/机构采购路线，不是开放数据集；本地导出、模型训练及公开 benchmark 使用权限需向供应商确认，不能由可观看或可传送推断 |
| [FIFA Training Centre Game Library](https://www.fifatrainingcentre.com/en/resources/game-library)；[攻守转换到进攻分类](https://www.fifatrainingcentre.com/en/resources/game-library/transition-to-attacking.php) | 按战术定向找片：分类页明确列出 `Transition to attack`、`Breaking lines` 等比赛片段；适合先定位案例，再查同场比赛的连续实况 | 是教学剪辑库，不是完整比赛；可能有定格、慢放和标线，未保证连续 30 秒。项目已有 [FIFA 素材处理代码](../../pipeline/fifa_game_library.py)，不能把整库算成全新来源 |

FIFA 的具体案例：[The technical mastery in Poland's transitions to attack](https://www.fifatrainingcentre.com/en/game/tournaments/fu20wwc/2026/technical-mastery-in-polands-transitions-to-attack.php)。官方说明中，Clip 1 为施压夺回球权后反击，Clip 2 为快速一脚出球并破线，Clip 3 为防守角球后的快速转换，Clip 4 为从本方深区发起并得分的转换。这些是来源文字标签，不是本轮逐帧确认的战术真值。

“Match replay / 全场回看”是整场比赛的点播录像，不等于其中每一段都是慢动作回放。目标是从中取正常速度、时间连续的 30 秒；同样，“无镜头切换”也不能单独证明片段不是回放。

建议先取有明确下载入口的 TeamTrack 做小样本；若更看重与 GSR 接近的职业转播画面，则从 NWSL+ 等官方全场录像定位比赛与时段，并确认研究下载权限。破线需保留出球前的防线，反击需保留球权转换，长传冲吊需保留起脚、落点与后续争抢。若镜头切换前不足 30 秒，应舍弃或另找窗口，不能重复帧、拼回放凑时长。

研究记录应保留来源 URL、比赛标识、半场、媒体时间起止点及画面上的比赛时钟；两种时间不能直接互换。同一比赛的不同机位和切片应分到同一数据划分中，避免训练与测试泄漏。

公开观看不等于获准批量下载、训练或再分发。[FIFA 服务条款](https://www.fifatrainingcentre.com/en/terms-of-service.php)对平台内容使用有明确限制；未确认授权的来源先记录链接和候选时段，不默认纳入可公开分发的视频集。

## 结论

| 优先级 | 来源 | 视频形态 | 无回放把握 | 战术筛选能力 | 获取方式 |
| --- | --- | --- | --- | --- | --- |
| 1 | [SportsMOT](https://github.com/MCG-NJU/SportsMOT) 足球子集 | 25 fps、720p；足球片段平均 673.9 帧，约 27 秒；足球视角含高位视角 | 官方保证片段内无镜头切换；是否实况、正常速度仍需看片确认 | 无原生战术标签，需人工或模型筛选 | 注册 [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/12424#participate)，在 `Participate / Get Data` 下载 |
| 2 | [SoccerNet Ball Action Spotting 2024](https://github.com/SoccerNet/sn-spotting) | 7 场完整、未裁剪的高低分辨率转播比赛 | 中：完整转播可能含回放，需按镜头标签或人工剔除 | 强：12 类动作含 `Pass`、`Drive`、`High Pass`、`Cross` | [SN-BAS-2024](https://huggingface.co/datasets/SoccerNet/SN-BAS-2024)，公开，约 19.3 GB |
| 3 | [SoccerNet Dense Video Captioning](https://github.com/SoccerNet/sn-caption) | 471 个足球转播视频，描述带时间戳；challenge 为 42 场 | 中：完整转播可能含回放 | 中强：可用 `through ball`、`breaks the line`、`counterattack`、`long ball` 等文本检索 | 标签/特征公开；原始 224p/720p 视频在 [SoccerNet_raw_HQ](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ) 申请 gated 访问 |
| 4 | [SoccerNet-v2 Action Spotting](https://github.com/SoccerNet/sn-spotting) | 500 场完整、未裁剪的转播比赛 | 中：含导播切镜与回放 | 弱：17 类是比赛事件，不是战术；适合扩大候选池 | 原始视频同上，需申请 Hugging Face 访问 |

## 不应计作新增来源

- 现有 `SNGS-001` 至 `SNGS-200` 对应 SoccerNet GSR 的 200 个标注片段；[官方仓库](https://github.com/SoccerNet/sn-gamestate)明确写明规模为 200 clips。
- [SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) 虽公开且总计约 35.1 GB，但已确认 `challenge.zip` 与 2024 版的大小和 LFS SHA-256 完全相同，不能视为新的比赛视频池。
- [SN-BAS-2025](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025) 与 2024 的 `challenge.zip` 完全相同；train/valid/test 仅有几百字节级压缩包差异。在核对内部视频指纹前，应按同源版本更新处理，而不是新增素材。

## 推荐执行顺序

1. 先下载 SportsMOT，只取 `splits_txt/football.txt` 中的序列；它最接近 30 秒且片段内无切镜，但仍需核实是否包含回放。
2. 下载 SN-BAS-2024，围绕 `High Pass` 向前 10 秒、向后 20 秒裁剪长传候选；围绕连续的 `Pass`/`Drive` 裁剪破线与反击候选。
3. 在 `Labels-caption.json` 中检索以下词组并扩成 30 秒窗口：`through ball`、`line-breaking pass`、`splits the defense`、`counterattack`、`counter attack`、`break`、`transition`、`long ball`、`long pass`、`direct ball`、`ball over the top`。
4. 对所有候选做两道过滤：镜头内不得出现 replay/logo transition；与现有 200 段计算视频感知哈希，排除同源或重叠窗口。
5. 最终人工确认战术语义。`High Pass` 只是长传候选，不自动等于长传冲吊；`Pass + Drive` 也不自动等于破线或快速反击。

## 许可注意

- SportsMOT 标注为 CC BY-NC 4.0，且官方条款禁止未经书面许可再分发数据集。
- SoccerNet 代码许可证不等于比赛转播视频版权许可；原始视频应按数据页访问条款使用，不应随项目公开再分发。
- UEFA.tv、FIFA+、YouTube 官方集锦虽然可人工找素材，但通常含回放、缺少结构化标签且版权边界更严，不建议作为可再分发 benchmark 的主来源。

## 一手来源

- SportsMOT 官方仓库：https://github.com/MCG-NJU/SportsMOT
- SoccerNet Action/Ball Action Spotting：https://github.com/SoccerNet/sn-spotting
- SoccerNet Dense Video Captioning：https://github.com/SoccerNet/sn-caption
- SoccerNet 数据下载总览：https://github.com/SoccerNet/SoccerNet
- SoccerNet GSR：https://github.com/SoccerNet/sn-gamestate