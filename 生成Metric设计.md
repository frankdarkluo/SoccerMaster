# 足球战术理解 - 足球战术报告生成Metric设计

> 导出自 ChatGPT，时间：9/21/2026, 12:34:16 PM

## 足球关系诊断更新（2026-09-22，候选）

当前 [data/pilot_study.csv](data/pilot_study.csv) 共 261 行、259 条非空 AI 润色，覆盖全部 12 类：GSR 104 条、AA 155 条。`AA-train-2163` 视频缺失，`GSR-001` 为独立开发样例，两行继续留空。这里的覆盖是草稿覆盖，不是 259 条已审核 GT。

本轮在 [参考协议](benchmark/pilot_reference_protocol.md) 增加 12 类统一操作定义候选，从现有数据中按原标签每类选两条，共 24 条进行 AI 诊断试标。实际查看的证据图、逐项判断与来源快照见 [试标底稿](benchmark/pilot_diagnostic_trial.json)。试标覆盖类别，不保证每类两个确认正例，也没有补齐所有子型；两条 `overlap-underlap` 都是外侧套边，内切套上仍缺。

| 维度 | 候选细化 | 暂不做的推断 |
| --- | --- | --- |
| Where | 执行队相对半场 + 可见场地锚点 + 绑定主体/参照物/时刻的空间关系 | 不强制 15 区，不用屏幕左右代替进攻方向，不把局部守者身后升级成最后防线身后 |
| Evidence | 每类三个候选核心判据，分开动作事实、主体对应和时序衔接 | 不从战术标签填真值，不把配合前缀写成完整结构，不把射门/进球设成所有战术的必要条件 |
| When | 目标窗与上下文分开，记录关键动作是否落在窗口内 | 不悄悄前移窗口，不用窗外触球补足窗内证据；暂不发布新的时间指标结果 |
| Faithfulness | 已支持/矛盾/未知的逐断言审核，另查球权、接应、结果和意图是否被夸大 | 未知不算已证实幻觉，定义规则检查不算视频事实，不对同一事实重复计分 |

试标先解决能否观察、如何表述与如何人工确认，不直接把新增项叠加到原四项等权总分。原半场指标保留作基础诊断；细化空间关系与动作证据的正式清单、去重、分母及裁决结果须在人审后冻结。

交付为 [Excel 审核表](outputs/pilot_diagnostic_review/pilot_diagnostic_review.xlsx) 和 [逐条可读稿](outputs/pilot_diagnostic_review/pilot_diagnostic_review.md)。人工在“逐项审核”选择“同意 / 修改 / 看不清”，再在“样例总览”裁决标签或窗口；“同意”表示同意 AI 的判断，也可以是同意某项为未知或矛盾，不表示所有命题都为真。现有 CSV、原标签和历史账本不自动回写。

以下 2026-09-21 半场方案和算例保留为历史基础；涉及 104 条/11 类的范围仅指当时 GSR 子集。新候选不声称已经冻结或完成人工一致率检验。

## 当前 Pilot 的 Where 决定（2026-09-21）

先把粗区域标稳。下方对话中的 `3 thirds x 5 channels = 15 zones` 保留为早期讨论，不作为当前 pilot 的必填标注或评分标准。

- 当前使用执行队相对半场：`own_half` / `opponent_half`；无法可靠赋单一值的适用字段为 `null`，同时注明中线边界、跨半场、窗口缺口或实例未确认等原因。
- 传跑类标起点和目标；逼抢、犯规、后场倒脚只标发生区域。参照队及动作位置的取法预先固定，同窗的传球者与跑者起点可以不同。
- 未知与不适用分开。仅在人工确认且适用的字段上计算粗半场准确率，并报告有效标注覆盖；不能以模型自己弃答来缩小分母。起点/目标型与单区域型先各自记录内平均，再汇总记录。
- 下底、向侧后方回送、进入防守者身后等关键空间关系保留为战术证据，不能仅由半场值推断；本轮不强制细网格，也不自动确认 AI 草稿为 GT。

日常直接阅读 [data/pilot_study.csv](data/pilot_study.csv) 的 `判断依据(AI润色版)`：已确认的粗区域信息已自然融入段落，未新增列，也不要求另查空间表。104 条记录的结构化空间证据保留在 [benchmark/pilot_reference_gsr.json](benchmark/pilot_reference_gsr.json)，[benchmark/pilot_spatial_coarse.csv](benchmark/pilot_spatial_coarse.csv) 仅作为辅助核验表。参照规则、位置取法及未决项详见 [benchmark/pilot_reference_protocol.md](benchmark/pilot_reference_protocol.md#粗区域标注-v1)。以下历史对话未改写，涉及 15 区的旧示例不应直接作为现行 GT。

## 判断依据与生成评测总结（2026-09-21）

### 1. 当前判断依据怎么写

以人工原始依据为事件主线，结合视频采样帧和关键连续原帧进行核验，再扩写有证据支持的细节。不是从战术名称套模板，也不是重新编写一段理想化的战术过程。

每段通常包含四部分，自然融在文字中，不要求机械分成四句话：

1. 场景与位置：执行球队、起点或发生区域，以及必要的半场、禁区、中线等位置。
2. 可见动作：传球、带球、跑动、争抢、接触及窗口内的可见结果。
3. 动作关系：传球与前插、回送与接应、压迫与出球，以及防守者的反应。
4. 判别依据与边界：哪些观察支持战术判断，实际发展到尝试、完成还是成功，哪些动作没有被窗口覆盖。

原始 `判断依据` 保留；补充和更正在 `判断依据(AI润色版)` 中表达。区域直接融入正文，已明确的内容不重复添加，不确定的位置不强分半场。参与争抢不等于赢得球权，前插不等于有效接球，也不凭动作先后推断主观意图。

完整参考由 `clip + tactic + interval + 判断依据(AI润色版)` 共同构成。标签和时间窗已有单独字段，不必再次从段落猜测；段落主要提供动作、空间和关系证据。

截至 2026-09-21，共 261 行，其中 GSR 的 104 条具有 AI 润色参考，覆盖 95 个视频、11 类战术；当时完整 pilot 的标签范围为 12 类，但这 104 条中没有 `overlap-underlap`。2026-09-22 AA 补充后已有 259 条润色、覆盖 12 类，详见本页开头。空值不能作为无战术负例。AI 扩写仍为 `machine_draft`，原标注经过人工核验不代表 AI 新增内容已自动获得相同审核状态。

### 2. 智能体可以直接生成文字报告

可以继续使用自然语言维护参考，也可以让智能体生成自然语言分析。评测时临时抽取可计算字段，不要求人工重写一套重复标注，也不强制智能体输出 JSON。

```text
视频 -> 智能体生成报告 -> 独立抽取 Tactic / When / Where / Evidence
固定的参考答案       -> 同一视频内对齐战术实例 -> 分项计算
```

生成报告必须明确指出战术、执行球队、起止时间、区域及支持事实。若未给出时间，评测器不能根据描述替模型猜一个时间段。

参考端可借助现有字段和文字预整理，人工确认后冻结版本。预测端使用固定的抽取规则，只整理模型实际说出的内容，并保留每个抽取结果对应的原句。预测抽取阶段不读取 GT 来修补答案；同义表达归一化不等于补充缺失事实。

### 3. 四项指标如何映射和计算

| 指标 | 参考来源 | 预测映射 | 主要计算方式 |
| --- | --- | --- | --- |
| Tactic | CSV 的 `tactic`，争议标签先裁决 | 将报告中的战术名称归一到固定标签 | 分类或多标签 Macro-F1 |
| When | CSV 的 `interval`，正式评测前核对窗口定义 | 抽取报告明确给出的起止秒数 | 实例时间交并比 tIoU，汇总为 mIoU |
| Where | 正文里的区域及已有粗区域核验记录 | 统一执行队参照，抽取本方或对方半场 | 每条适用且 GT 已知字段的准确率，再按记录平均 |
| Evidence | 从词条定义和参考中固定的可观察证据项 | 判断报告是否明确表达相应事实或关系 | Precision / Recall / F1 |

**Tactic：** 片段级识别允许多个标签，同一视频内同类战术出现多次时，类别只计一次，发生次数另由实例匹配处理。设评测类别集合为 $C$，则：

$$
S_{tactic}=\frac{1}{|C|}\sum_{c\in C}\frac{2TP_c}{2TP_c+FP_c+FN_c}
$$

评测类别集合及无样本类别的处理应预先固定，并报告支持数量。单条预测正确的 1 分不能称为数据集 Macro-F1。

**When：** 对已经对齐的预测区间 $P$ 和参考区间 $G$：

$$
tIoU=\frac{|P\cap G|}{|P\cup G|},\qquad S_{when}=\operatorname{mean}(tIoU)
$$

可同时报告达到 0.3、0.5、0.7 阈值的比例。当前 `interval` 可用于试算，但尚不能宣称每条都是最小战术证据窗口。集成生成中的漏检、无效时间和未匹配实例如何进入统计，仍需提前固定，不能只汇报成功匹配项而隐去漏检。

**Where：** 传跑类比较起点和目标，逼抢、犯规和后场控球比较发生区域。先确认是同一执行队、同一动作位置，再比较半场；不能让两支不同球队各自的“本方半场”被误判为相同位置。

对第 $i$ 条记录，令 $K_i$ 为适用且 GT 已确认的字段集合：

$$
S_{where,i}=\frac{\sum_{k\in K_i}\mathbf{1}[\hat z_{ik}=z_{ik}]}{|K_i|},\qquad
S_{where}=\operatorname{mean}_{i:|K_i|>0}(S_{where,i})
$$

GT 未知与字段不适用都不强行评分，但须分别报告原因和覆盖率；GT 已知而模型未答或弃答，不能因此缩小分母。当前粗区域草稿的 190 个适用字段中有 180 个赋值、10 个未决，这不是模型得分。

**Evidence：** 每类固定少量可观察证据项及必要的混淆项，不建立庞大的通用事件本体。逐样本区分真、假、未知，并在 GT 已知项上比较预测的正证据集合：

$$
Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN},\qquad
F1=\frac{2TP}{2TP+FP+FN}
$$

漏掉已确认的正证据计 FN，声称已确认不成立的证据计 FP。参考没有提到的内容不能直接标假；空集合、零分母及 micro/macro 汇总口径应统一约定。

证据判断采用语义和关系，而非关键词。例如撞墙配合需要“同一球员先传球、再前插、队友向其跑动路线回送”；仅出现“传球”“跑动”两个词不算满足该关系。

### 4. GSR-003 的完整算例

以下暂接受当前参考作为 GT，模型预测是虚构示例，不是实际模型实验结果。本例四个证据项用于说明算法，不表示 12 类的证据清单已经全部冻结。

参考标签为 `long-ball`，时间为 `[6,12]` 秒，执行队为蓝队，区域为本方半场到对方半场。润色依据节选：

> 蓝队门将在本方禁区内助跑起高球，直接越过中线送向白队半场、中圈前方的接应区域。蓝队球员与白队球员迎球起跳争抢，附近蓝队队友随即接应后续落球并继续向前处理。

假设预先确认本例的四项正证据：

| 证据 | GT | 假设报告是否表达 |
| --- | --- | --- |
| 直接向前长传 | 真 | 是 |
| 跳过中间短传组织，直接联系前方 | 真 | 是 |
| 争抢第一落点 | 真 | 是 |
| 队友跟进后续落球 | 真 | 否 |

假设智能体生成：

> 7-12 秒，蓝队采用长传冲吊。门将在本方半场直接起长球，不经中场短传送到白队半场，前方队友与白队争抢第一落点。

| 项目 | 计算 | 本例结果 |
| --- | --- | --- |
| 战术 | 预测 `long-ball` 与参考一致 | 单条正确，不是 Macro-F1 |
| 时间 | 交集 `[7,12]` 长 5 秒，并集 `[6,12]` 长 6 秒 | $tIoU=5/6\approx0.833$ |
| 区域 | 蓝队本方半场起球、对方半场接应均正确 | $(1+1)/2=1$ |
| 证据 | $TP=3,FP=0,FN=1$ | $P=1,R=0.75,F1=6/7\approx0.857$ |

模型措辞不必与参考相同。这里时间少覆盖 1 秒、漏掉后续落球接应，但战术和粗区域正确，各项指标分别呈现这些差异。

### 5. 汇总与正式使用边界

主结果表优先报告四个分项。如果必须提供总分，可沿用文档提出的等权平均：

$$
S_{report}=\frac{S_{tactic}+S_{when}+S_{where}+S_{evidence}}{4}
$$

这里的四项是同一固定评测协议下的数据集级结果，均归一到 $[0,1]$，不是把单条命中率冒充 Macro-F1。分项使用的有效样本数和覆盖率必须一并报告。

正式使用前还需完成以下事项：

1. 冻结证据项：每类判据、每条真/假/未知值、关系判断和汇总口径。当前没有完整的 evidence checklist 或自动评分器。
2. 冻结实例匹配：同视频多标签、同标签多次发生需要一对一对齐，规定漏检、多报及重复报告的处理。pilot 是选定实例，不是全片穷尽标注，不能把未收录事件自动算作模型误报。
3. 校准抽取与判定：用人工裁定样本检查抽取器是否准确理解执行队、否定、尝试与成功、时间和空间关系。固定提示与版本，保存原句证据，不能仅凭规则存在就声称评分可靠。
4. 核查正文额外断言：例如四项都对但额外编造进球，证据清单未必能捕获。需另核对报告事实忠实性；参考没写的真实事实不自动算错，不以 BLEU 等文本相似度替代事实判断。
5. 固定输入与数据范围：综合生成只给视频和统一任务说明，不泄露参考、正确标签及答案时间窗；已给定的字段不能再解释为自主识别能力。若仅使用 104 条 AI 参考，须说明只覆盖 11 类，不能宣称已完成 12 类完整比较。

结论：文字参考和文字生成都可以保留，评分层负责固定的语义映射与公式计算。当前优先完成证据清单、实例对齐规则和新增内容审核，不需要继续加长依据，也不需要恢复 15 区强制标注。

---

# 消息 1

## 用户

其实我稍微想了一下，ordered event set不是不能做，主要是GT这个太难弄，再就是GT event type很难搞；因为足球里面的事件错综复杂，我们得找人定义这些event set才可以，或者我们通过现有的词条表csv来收集对应的event set GT label或许可以，但成本比较高，你如果有反对意见也可以提出来

## 助手

我基本同意你的判断。**真正昂贵的不是让模型生成 event sequence，而是为了评测它，先人为构造一个覆盖足球复杂事件空间的统一 `event type ontology + ordered GT`。** 这一步很容易把项目拖成“足球事件本体构建”，偏离你真正想做的 tactic understanding。

你现在的词条表其实已经给了一个更便宜的替代方案：**不要定义全局 event type，只定义 tactic-specific evidence slots。** 录音里你对长传冲吊本来就已经描述成“获得球权 → 起纵向长球 → 争第一点 → 整体前压争第二点”，而且这些内容本身就是战术定义的一部分。

比如长传冲吊不需要先发明一个通用的足球事件体系，可以直接写：

```
Long-ball evidence:
E1: direct forward long ball
E2: first-ball contest
E3: attacking team pushes forward
E4: second-ball contest
```

然后每个视频只标：

```
{
  "E1": true,
  "E2": true,
  "E3": true,
  "E4": false
}
```

这时 `What happened` 就可以变成：

> **Which observable pieces of evidence supporting the long-ball tactic are present in this clip? Select all that apply.**

Metric 直接就是 **Precision / Recall / F1**。你不需要让标注员为几千条视频重新写 event sequence，也不需要先定义一个覆盖“传球、跑位、压迫、争顶、回撤、拉边、套边……”所有足球行为的统一 taxonomy。

我甚至建议你进一步收缩成下面这套结构：

$$
\text{Tactic Report}
=
\{\text{Tactic},\text{When},\text{Where},\text{Evidence}\}
$$

其中：

- **Tactic**：战术类别；
- **When**：战术发生区间；
- **Where**：origin / target / progression zone；
- **Evidence**：该战术词条定义中哪些关键 observable criteria 被视频满足。

这样你的 `What happened` 实际上就变成 **Evidence Recognition**，而不是 Event Sequence Reconstruction。

### 顺序信息怎么办？

我不建议完全扔掉，但可以把它降级成**词条级知识，而不是每条视频都重新标**。

例如长传冲吊的 canonical pattern：

$$
E_1 \rightarrow E_2 \rightarrow E_3
$$

这是 tactic ontology 中定义一次即可。视频标注员主要确认这些 evidence 是否发生；只有当某个战术**顺序本身是区分战术的关键条件**时，再额外标：

```
Does E1 occur before E2? Yes / No
```

这样就从“生成完整 ordered event set”变成一两个 **pairwise temporal constraints**，成本会低很多。

### 我唯一的反对意见

我不建议因为 GT 贵，就彻底把 `What happened` 删除。因为如果最终只有：

> tactic + timestamp + zone

审稿人很容易质疑：**模型究竟依据什么认为这是这个战术？**

所以最好的折中不是完整 Event Sequence GT，而是：

> **Tactic-specific evidence checklist**

它既保留了“战术是由哪些可观测证据组成”的核心思想，又不会迫使你构造一套庞大的通用足球 event ontology。

我现在会优先把你现有 CSV 的每个战术词条自动整理成大约 **2–5 个 observable evidence predicates**，然后让足球标注员只负责**核验和修改这些 criteria**，而不是从零设计 event set。这和你们之前讨论的“让模型先预生成、人做核验以减少 human effort”也是完全一致的。


<div align="center">◆ ◆ ◆</div>

# 消息 2

## 用户

ok，根据我们上面的讨论，我们再中期总结一下，给定一个视频，就用我们上面的SNGS-001.mp4，我们的文字prompt是什么, 模型输出是什么？以及每道题的题目设计是什么（包括tactic）？以及如何算分？

## 助手

可以。按照我们现在收缩后的版本，我建议把整个 benchmark 固定成 **4 个 diagnostic tasks + 1 个 integrated structured report generation task**。前三个核心理解维度是 `when / where / evidence`，再加一个 `tactic recognition`；最终 report 把四者合并。这个也符合你和王博士讨论的“structured open-ended generation + field-level evaluation”路线。

## 1. 给模型的统一文字 Prompt

下面这个可以作为 **Integrated Tactical Report Generation** 的 v1 prompt。实际 benchmark 做四道子题时，再从中抽取对应部分。

You are given a broadcast soccer video and a predefined soccer tactic ontology.

Analyze the video and generate a grounded structured tactical report.

For each tactical pattern that is visually supported by the video, identify:

1. **Tactic** — which tactic occurs.
2. **When** — the smallest temporal interval containing the visual evidence required to establish the tactic.
3. **Where** — where the tactic starts and develops on the pitch, using only the predefined pitch-zone vocabulary.
4. **Evidence** — which tactic-specific observable evidence criteria are satisfied in the video.

### Rules

- Use only tactic labels defined in the provided tactic ontology.
- Base all judgments on observable video evidence.
- Do not infer player intention, tactical purpose, or causal explanation.
- Do not explain why the tactic succeeds or fails.
- Do not include subsequent outcomes that are not necessary to establish the tactic.
- If a field cannot be reliably determined from the broadcast view, return `null`.
- Multiple tactics may be reported if multiple tactically valid occurrences are present.
- Use the smallest time interval that contains the evidence required to identify the tactic.
- Use only the provided controlled pitch-zone vocabulary.
- For the evidence field, evaluate only the predefined evidence criteria associated with the predicted tactic.

### Output

Return valid JSON only:

```
{
  "tactics": [
    {
      "tactic": "<tactic label>",

      "temporal_grounding": {
        "start_sec": 0.0,
        "end_sec": 0.0
      },

      "spatial_grounding": {
        "origin_zone": "<zone or null>",
        "target_zone": "<zone or null>"
      },

      "evidence": [
        {
          "evidence_id": "<criterion id>",
          "present": true
        }
      ],

      "tactical_report": "<short factual description based only on the structured fields above>"
    }
  ]
}
```

Do not output hidden reasoning. Output only the final structured result.

---

# 2. SNGS-001 的 GT 可以长这样

这里我们已经知道目标战术是 **长传冲吊 / long-ball**。按照前面的人工观察，暂时可以定义：

```
{
  "tactics": [
    {
      "tactic": "long-ball",

      "temporal_grounding": {
        "start_sec": 16.5,
        "end_sec": 22.5
      },

      "spatial_grounding": {
        "origin_zone": "middle_third_center",
        "target_zone": "attacking_third_center"
      },

      "evidence": [
        {
          "evidence_id": "LB-E1",
          "present": true
        },
        {
          "evidence_id": "LB-E2",
          "present": true
        },
        {
          "evidence_id": "LB-E3",
          "present": true
        },
        {
          "evidence_id": "LB-E4",
          "present": true
        },
        {
          "evidence_id": "LB-N1",
          "present": false
        },
        {
          "evidence_id": "LB-N2",
          "present": false
        }
      ],

      "tactical_report": "The attacking team plays a direct long ball from the middle third into the attacking third. A forward contests the first ball and supporting players push forward around the second-ball area."
    }
  ]
}
```

其中这里的 evidence ontology **不是全足球统一 event taxonomy**，而是这个 tactic 自己的 criteria：

| ID | Evidence criterion |
| --- | --- |
| LB-E1 | A direct long forward ball is played |
| LB-E2 | The ball bypasses normal midfield build-up |
| LB-E3 | An attacker contests the first ball |
| LB-E4 | Supporting players push forward around the second-ball area |
| LB-N1 | The team progresses mainly through consecutive short passes |
| LB-N2 | The ball is deliberately recycled backward to maintain possession |

这里 `N1/N2` 是 negative/confounder evidence，主要防止模型所有选项全选。

---

# 3. 四道题具体怎么设计

我建议 **前三个 diagnostic task 在问题里给定 tactic label**，这样测的是单独能力；只有第一题自己识别 tactic。

## Q1. Tactic — What tactic is this?

**Question**

> Which tactical pattern from the predefined tactic ontology is visually supported by this video?

输出：

```
long-ball
```

SNGS-001 GT：

```
long-ball
```

### Metric

如果一个 clip 只考一个主要 tactic：

$$
\text{Accuracy}
$$

整个数据集同时报告：

$$
\text{Macro-F1}
$$

因为不同 tactic 数量肯定不平衡，**Macro-F1 比 overall accuracy 更重要**。

如果以后允许一个视频出现多个 tactic，则变成 multi-label：

$$
Precision,\ Recall,\ F1
$$

---

# 4. Q2. When — Temporal Grounding

这里把 tactic 告诉模型：

> The video contains a **long-ball tactic**.
>
> Identify the smallest temporal interval containing the observable evidence required to establish this tactic.
>
> Return `[start_sec, end_sec]`.

GT：

```
[16.5, 22.5]
```

注意这里你的理解已经确定了：

> **不是一直标到后面的射门。**

长传 → 第一落点 → 前压争第二落点已经完成了长传冲吊的 tactical evidence chain，后面的射门属于 outcome。

### Metric

主指标：

$$
\boxed{
tIoU=
\frac{|P\cap G|}
{|P\cup G|}
}
$$

同时可以报告：

- `mIoU`
- `R@0.3`
- `R@0.5`
- `R@0.7`

例如预测 `[17,22]`，GT `[16.5,22.5]`，那么分数仍然很高，而不是简单算错。

---

# 5. Q3. Where — Spatial Grounding

同样给出 tactic：

> The video contains a **long-ball tactic**.
>
> Where does this tactic originate and where is the ball directed?
>
> Use only the predefined pitch-zone vocabulary.

输出：

```
{
  "origin_zone": "middle_third_center",
  "target_zone": "attacking_third_center"
}
```

我建议 v1 **暂时只保留 origin + target**。

之前说的完整 progression：

```
origin → development → target
```

可以以后再加。因为 benchmark 第一版没有必要把 spatial annotation 做得太重。

### Zone vocabulary

统一做：

$$
3\text{ thirds}\times5\text{ channels}=15\text{ zones}
$$

比如：

```
defensive_third_left_wing
defensive_third_left_half_space
defensive_third_center
...
middle_third_center
...
attacking_third_right_wing
```

并且全部按照**进攻方的进攻方向归一化**。

### Metric

最简单：

$$
Acc_{origin}
$$

和：

$$
Acc_{target}
$$

总体：

$$
\boxed{
S_{where}
=
\frac{
Acc_{origin}+Acc_{target}
}{2}
}
$$

以后如果加入 progression，再扩展即可。

---

# 6. Q4. What happened → Evidence Recognition

这里我觉得是我们这轮讨论里最重要的收缩。

**不再要求模型生成完整 event sequence。**

问题改成：

> The video contains a **long-ball tactic**.
>
> Which of the following observable evidence criteria are present during this tactical occurrence?
>
> Select all that apply.

例如：

**A.** A direct long forward ball is played.

**B.** The ball bypasses normal midfield build-up.

**C.** An attacker contests the first ball.

**D.** Supporting players push forward around the second-ball area.

**E.** The team progresses mainly through consecutive short passes.

**F.** The ball is deliberately recycled backward to maintain possession.

SNGS-001：

```
A, B, C, D
```

也就是：

```
[1,1,1,1,0,0]
```

### Metric

这是标准 multi-label classification。

$$
Precision
=
\frac{TP}{TP+FP}
$$

$$
Recall
=
\frac{TP}{TP+FN}
$$

$$
\boxed{
F1_{evidence}
=
\frac{2PR}{P+R}
}
$$

这样正好同时处理：

- **Hallucination** → 乱选了 E/F → Precision ↓
- **Omission** → 漏掉了 C/D → Recall ↓

因此暂时都不需要另外设计 hallucination / omission metric。

---

# 7. 最终 Integrated Tactical Report Generation

最终真正有特色的任务是：

$$
Video
\rightarrow
\{
Tactic,\ When,\ Where,\ Evidence
\}
$$

也就是：

```
{
  "tactic": "long-ball",
  "when": [16.5, 22.5],
  "where": {
    "origin": "middle_third_center",
    "target": "attacking_third_center"
  },
  "evidence": ["LB-E1", "LB-E2", "LB-E3", "LB-E4"],
  "report": "..."
}
```

这样 **QA 和 generation 就不是两个互不相关的东西**：

> 四道 diagnostic task 用来测模型到底哪一步不行；
>
> Tactical Report Generation 测模型能否把四种能力组合起来。

---

# 8. 最终怎么算总分？

我现阶段**不建议先拍脑袋搞权重**。

论文主表直接报告四列：

| Model | Tactic ↑ | When ↑ | Where ↑ | Evidence ↑ |
| --- | ---: | ---: | ---: | ---: |
| GPT-X | Macro-F1 | mIoU | Spatial Acc. | Evidence F1 |
| Gemini-X | … | … | … | … |
| Qwen-VL | … | … | … | … |

其中：

$$
\boxed{
\begin{aligned}
S_{tactic}&=\text{Macro-F1}\\
S_{when}&=\text{mIoU}\\
S_{where}&=\frac{Acc_{origin}+Acc_{target}}{2}\\
S_{evidence}&=\text{Multi-label F1}
\end{aligned}
}
$$

如果 leaderboard **必须要有一个 overall number**，v1 最简单可以：

$$
\boxed{
TacticReportScore
=
\frac{
S_{tactic}
+
S_{when}
+
S_{where}
+
S_{evidence}
}{4}
}
$$

所有分数 normalize 到 `[0,1]`。

但论文分析应该始终**优先报告四个 subscore，而不是只报 overall score**。以后如果有足够专家评分，再像 RadCliQ 那样拟合权重。

---

## 当前版本一句话总结

我们现在其实已经从最初比较复杂的：

> `Tactic + When + Where + Ordered Event Sequence + Why`

收缩成了一个比较可落地的：

$$
\boxed{
\text{Tactical Understanding}
=
\text{Tactic}
+
\text{Temporal Grounding}
+
\text{Spatial Grounding}
+
\text{Tactic-specific Evidence Recognition}
}
$$

最终再把这四部分 verbalize 成一个 **structured tactical report**。

我认为这个版本已经比较适合作为你接下来 **20–30 个视频 pilot annotation** 的正式 schema 了。


<div align="center">◆ ◆ ◆</div>

# 消息 3

## 用户

假如我是小白，请你用人话通过通俗易懂的方式再把我们要做的tactic, temporal grounding, spatial grounding 以及tactic evidece recognition以及最后的structured tactical report给解释一下，包括怎么想到的这些，以及metric具体是什么？怎么计算，都告诉我一下

## 助手

可以。把它想成：**我们不是只问模型“这是什么战术”，而是要求它像一个足球分析员一样，把判断拆成几件可以检查、可以打分的事情。**你和王博士原本讨论的核心也是这样：单纯 QA 容易有 bias，因此更希望最终生成 structured report，而 report 内部再分别评价 grounding、evidence 等字段。

## 先用一句人话概括整个任务

给模型一个 30 秒足球视频，我们希望它回答四件事：

> **① 这是什么战术？
>
> ② 这个战术什么时候发生？
>
> ③ 它在球场什么位置发生？
>
> ④ 我凭什么相信这是这个战术？**

然后把四个答案合起来，就是：

> **Structured Tactical Report（结构化战术报告）**

例如我们刚才的 SNGS-001：

```
战术：长传冲吊
时间：16.5s–22.5s
空间：中场中路 → 进攻三区中路
证据：
✓ 直接向前长传
✓ 越过中场组织
✓ 前锋争第一点
✓ 队友整体前压争第二点
```

这已经是一份最简单的 structured tactical report。

---

# 一、为什么要拆成这四件事？

最初你的问题其实非常简单：

> 给视频 → 问模型：这是什么战术？

比如：

> Is this a counter-attack or a rest-defense recovery?

但很快会遇到一个问题：

**如果模型答错了，我们不知道它到底错在哪里。**

可能是：

- 它连战术发生的时间都没找对；
- 时间找对了，但是没看懂球场空间；
- 时间和空间都看对了，但不知道长传冲吊的关键特征；
- 什么都看懂了，只是足球知识不够，最后 label 选错。

而你自己的词条表，本来就已经把战术拆成了类似：

> 时间窗 + 空间约束 + 事件/证据 + 战术定义

录音里你举长传冲吊时也是这样解释的：从获得球权、发动纵向长球，到争第一点、整体前压争第二点；并且强调战术只发生在一个特定时间范围内，而不是整个 30 秒视频。

所以自然就得到了：

$$
\boxed{
\text{Tactic Understanding}
=
\text{Tactic}
+
\text{When}
+
\text{Where}
+
\text{Evidence}
}
$$

这不是为了“凑四个任务”，而是在问：

> **一个模型真的理解战术，需要掌握哪些最基本的信息？**

---

# 二、Task 1：Tactic——“这是什么战术？”

这是最直观的。

## 人话理解

就像给一个足球教练看视频，然后问：

> **你觉得刚才这段踢的是什么战术？**

例如 SNGS-001：

```
GT: 长传冲吊
```

模型回答：

```
Prediction: 长传冲吊
```

就是对。

如果回答：

```
快速反击
```

就是错。

---

## 题目可以怎么问？

> **Which tactical pattern is visually supported by this video?**

然后告诉模型只能从我们的 tactic ontology 里面选择，比如：

```
long-ball
counter-attack
line-breaking-pass
cutback
run-in-behind
...
```

---

# 三、Tactic 怎么打分？

最简单就是 **Accuracy**。

假如有 100 个视频：

```
模型答对 70 个
```

那么：

$$
Accuracy=\frac{70}{100}=70\%
$$

很简单。

---

## 为什么还需要 Macro-F1？

因为数据一定不平衡。

例如：

```
counter-attack: 500 条
long-ball: 200 条
cutback: 50 条
one-two: 20 条
```

如果模型只把大类做好，小类全部做错，overall accuracy 可能仍然很好看。

所以我们更关心：

> **每个战术是不是都做得好？**

因此论文里建议主报：

$$
\boxed{\text{Macro-F1}}
$$

它基本等价于：

> 每个 tactic 单独算 F1，然后所有 tactic 平均。

这样不会让 counter-attack 这种大类“淹没”其他战术。

---

# 四、Task 2：Temporal Grounding——“它什么时候发生？”

这个名字听起来复杂，其实特别简单。

## 人话理解

30 秒视频里并不是 30 秒都在做长传冲吊。

我们真正想知道：

> **长传冲吊从第几秒开始，到第几秒结束？**

SNGS-001，我们之前大致确定：

```
GT:
16.5s → 22.5s
```

为什么不是到后面的射门结束？

因为：

```
长传
→ 第一落点
→ 前压争第二点
```

这些已经足够证明“长传冲吊”发生了。

后面的射门属于：

> **这个战术之后产生的结果**

而不是定义这个战术本身必须包含的部分。

---

# 五、Temporal Grounding 怎么算分？

这里最经典的是：

$$
\boxed{\text{Temporal IoU}}
$$

IoU 就是：

> **模型预测的时间段和人工 GT 时间段重叠了多少。**

例如：

```
GT:         16.5 -------- 22.5
Prediction:      17 ----- 22
```

GT 长度：

$$
22.5-16.5=6
$$

预测区间完全包含在 GT 里面，交集长度：

$$
22-17=5
$$

并集长度：

$$
6
$$

所以：

$$
tIoU=\frac{5}{6}=0.833
$$

也就是：

> **83.3%**

这个比要求：

```
必须精确预测 16.5 和 22.5
```

合理得多。

因为人工自己标时间，也可能差个 0.5–1 秒。

---

## 整个数据集报告什么？

可以报告：

$$
\boxed{\text{mIoU}}
$$

也就是所有视频 IoU 的平均值。

例如：

```
Video 1: 0.83
Video 2: 0.70
Video 3: 0.90
```

平均：

$$
mIoU=0.81
$$

还可以报告：

```
R@IoU=0.3
R@IoU=0.5
R@IoU=0.7
```

意思就是：

> 有多少视频的 temporal IoU 超过 0.3 / 0.5 / 0.7。

---

# 六、Task 3：Spatial Grounding——“它在哪里发生？”

同样，名字听起来复杂，但其实就是：

> **这个战术从球场哪里开始，又把球送到了哪里？**

---

## 为什么需要它？

因为足球战术很依赖空间。

例如：

> 中后场一脚球直接打到前场

和：

> 在前场禁区附近横传

即使都是“长距离传球”，战术意义完全不同。

所以模型不能只看：

> “有人踢了一脚长球。”

还要知道：

> **球场上的空间关系是什么。**

---

# 七、怎么定义球场的位置？

不要让模型报：

```
x = 0.437
y = 0.626
```

太复杂，而且广播视频存在摄像机移动、缩放、单应性问题。

第一版最简单：

## 纵向分成三块

```
Defensive third
Middle third
Attacking third
```

## 横向分成五条

```
Left wing
Left half-space
Center
Right half-space
Right wing
```

所以一共：

$$
3\times5=15
$$

个区域。

例如：

```
middle_third_center
attacking_third_right_half_space
```

---

# 八、SNGS-001 的 Where 是什么？

大致可以写：

```
{
  "origin_zone": "middle_third_center",
  "target_zone": "attacking_third_center"
}
```

人话就是：

> **长传从中场中路附近发动，打到进攻三区中路附近。**

第一版其实做到这里就够了。

不需要一上来就：

```
origin
→ development zone 1
→ development zone 2
→ target
```

否则人工成本又会上升。

---

# 九、Spatial Grounding 怎么打分？

非常简单，分别看两个位置：

### Origin 对不对？

```
GT: middle_third_center
Pred: middle_third_center
```

对：

$$
Acc_{origin}=1
$$

### Target 对不对？

```
GT: attacking_third_center
Pred: attacking_third_center
```

对：

$$
Acc_{target}=1
$$

那么这个视频：

$$
S_{where}
=
\frac{1+1}{2}
=1
$$

也就是满分。

---

如果预测成：

```
origin: middle_third_center     ✓
target: attacking_third_left_wing   ✗
```

那么：

$$
S_{where}
=
\frac{1+0}{2}
=0.5
$$

整个数据集再平均即可。

所以：

$$
\boxed{
S_{where}
=
\frac{Acc_{origin}+Acc_{target}}{2}
}
$$

第一版足够用了。

以后可以再做更复杂的“相邻区域扣少一点分”，但不是现在必须做。

---

# 十、Task 4：Tactic Evidence Recognition——“你凭什么说这是这个战术？”

这是目前最容易被误解，但其实**非常重要**的一项。

我们一开始想的是：

> Event Sequence Reconstruction

例如要求模型生成：

```
获得球权
→ 长传
→ 争第一点
→ 队友前压
→ 争第二点
```

但后来发现一个现实问题：

> **GT 太贵了。**

因为你必须定义：

- 什么算一个 event？
- 全足球到底有多少 event type？
- “整体前压”算 event 吗？
- “吸引防守”算 event 吗？
- 不同战术需要什么 event？

很快就会变成另一个大工程。

所以我们把它收缩成：

$$
\boxed{\text{Tactic-specific Evidence Recognition}}
$$

---

# 十一、Evidence Recognition 到底是什么意思？

不是让模型自己发明 event。

而是我们提前告诉它：

> **判断长传冲吊需要看这几件事。**

例如词条表整理出来：

```
A. 是否出现直接向前的长球？
B. 是否越过正常中场短传组织？
C. 前锋是否争抢第一落点？
D. 队友是否整体前压进入第二落点区域？
E. 是否主要通过连续短传推进？
F. 是否只是为了保有球权而回传？
```

其中：

```
A/B/C/D = 支持长传冲吊
E/F = 反例/混淆证据
```

---

## 给模型的问题

> **Which of the following observable evidence criteria are present during this long-ball tactic? Select all that apply.**

然后：

```
A
B
C
D
E
F
```

SNGS-001 GT：

```
A, B, C, D
```

或者写成：

```
[1,1,1,1,0,0]
```

---

# 十二、Evidence 怎么打分？

这是标准的：

$$
Precision,\ Recall,\ F1
$$

假设 GT：

```
A B C D
```

模型预测：

```
A B D E
```

那么：

### TP：选对的

```
A B D
```

所以：

$$
TP=3
$$

### FP：乱选的

```
E
```

所以：

$$
FP=1
$$

### FN：漏掉的

```
C
```

所以：

$$
FN=1
$$

因此：

$$
Precision
=
\frac{3}{3+1}
=
0.75
$$

意思是：

> 模型说出来的证据里，75%是真的。

Recall：

$$
Recall
=
\frac{3}{3+1}
=
0.75
$$

意思是：

> 真正应该发现的证据，它发现了75%。

最后：

$$
F1
=
\frac{2PR}{P+R}
=
0.75
$$

---

## 为什么这个 metric 很漂亮？

因为它天然对应两个问题：

### 模型乱说

比如它选了：

```
E
```

视频里其实没有。

这就是：

> **hallucination**

Precision 会下降。

### 模型漏看

比如没选：

```
C = 第一落点争抢
```

这就是：

> **omission**

Recall 会下降。

所以：

$$
\boxed{\text{Evidence F1}}
$$

已经能够同时衡量：

> 有没有胡说 + 有没有漏掉。

不需要第一版再创造很多复杂 metric。

---

# 十三、所以现在四个任务到底是什么？

用最直白的话总结：

| Task | 人话 | 输出 | Metric |
| --- | --- | --- | --- |
| **Tactic** | 这是什么战术？ | `long-ball` | Macro-F1 |
| **Temporal grounding** | 它什么时候发生？ | `[16.5,22.5]` | Temporal IoU |
| **Spatial grounding** | 它从哪里打到哪里？ | `middle-center → attacking-center` | Zone Accuracy |
| **Evidence recognition** | 有哪些事实证明它是这个战术？ | `A,B,C,D` | Precision / Recall / F1 |

实际上非常直观：

$$
\boxed{
What
+
When
+
Where
+
Evidence
}
$$

---

# 十四、那 Structured Tactical Report 又是什么？

最重要的一点：

> **Structured Tactical Report 不是第五个完全不同的任务。**

它其实就是：

> **把前面四个问题一次性全部回答。**

例如模型最终输出：

```
{
  "tactic": "long-ball",

  "temporal_grounding": {
    "start_sec": 16.5,
    "end_sec": 22.5
  },

  "spatial_grounding": {
    "origin_zone": "middle_third_center",
    "target_zone": "attacking_third_center"
  },

  "evidence": [
    "direct_long_forward_ball",
    "bypass_midfield_build_up",
    "first_ball_contest",
    "supporting_players_push_forward"
  ],

  "report": "The team plays a direct long ball from midfield into the attacking third. The forward contests the first ball while supporting players push forward around the second-ball area."
}
```

于是你得到：

$$
Video
\rightarrow
\boxed{
Tactic
+
When
+
Where
+
Evidence
}
$$

最后再把这些字段组织成人能看的文字。

这就是所谓：

> **Structured Tactical Report Generation**

---

# 十五、为什么最终要做 Report，而不是只做四道 QA？

这个其实来自你和王博士讨论的一个很重要的问题。

如果只有：

```
Q1
Q2
Q3
Q4
```

审稿人可能觉得：

> 又一个 football VideoQA benchmark。

而且 QA 有选择题 bias。

王博士提出的思路是：

> **让模型生成一份规范的 structured tactical report。**

这样最终应用就变成：

```
足球视频
       ↓
AI理解
       ↓
战术分析报告
       ↓
战术解说 / 教练分析 / 视频检索
```

也就是说：

### QA 的作用

是：

> **diagnosis**

告诉我们模型到底哪里不行。

### Report 的作用

是：

> **integration**

看模型能不能把所有能力组合起来。

录音中你们实际上已经讨论到了这一点：QA 可以保留提供容易理解的数字，而 open-ended structured report 可以作为更主要的生成任务，再针对内部 schema 分项评价。

---

# 十六、Structured Report 怎么打分？

关键就在这里：

> **不要主要用 BLEU。**

假设 GT：

> The team plays a direct long ball from midfield into the attacking third.

模型：

> A long forward pass is launched from the middle third toward the opposition's defensive line.

两句话文字差很多。

但是足球事实其实一样。

所以 BLEU 可能给低分：

```
BLEU = low
```

但模型其实答得完全正确。

---

因此我们直接评价 report 里面的四个字段：

$$
S_{tactic}
$$

$$
S_{when}
$$

$$
S_{where}
$$

$$
S_{evidence}
$$

分别就是：

$$
S_{tactic}=\text{Macro-F1}
$$

$$
S_{when}=\text{Temporal IoU}
$$

$$
S_{where}=\text{Zone Accuracy}
$$

$$
S_{evidence}=\text{Evidence F1}
$$

---

# 十七、最后需要一个总分吗？

第一版论文里，我反而建议：

> **不要过度强调一个 overall score。**

主结果表：

| Model | Tactic | When | Where | Evidence |
| --- | ---: | ---: | ---: | ---: |
| GPT | 0.xx | 0.xx | 0.xx | 0.xx |
| Gemini | 0.xx | 0.xx | 0.xx | 0.xx |
| Qwen-VL | 0.xx | 0.xx | 0.xx | 0.xx |

这比：

```
Overall = 62.7
```

有信息得多。

因为你真正想知道：

> GPT 是足球知识不行？

还是：

> temporal grounding 不行？

还是：

> spatial reasoning 不行？

还是：

> evidence recognition 不行？

---

如果 leaderboard 必须有 overall，可以非常简单地：

$$
\boxed{
TacticReportScore
=
\frac{
S_{tactic}
+
S_{when}
+
S_{where}
+
S_{evidence}
}{4}
}
$$

例如：

```
Tactic = 0.70
When = 0.80
Where = 0.60
Evidence = 0.70
```

那么：

$$
Overall
=
\frac{0.7+0.8+0.6+0.7}{4}
=
0.70
$$

也就是：

```
70
```

但这个 overall 只是方便排名。

**科学分析仍然看四个分项。**

---

# 十八、整个课题现在其实可以画成这一张图

```
               Soccer Broadcast Video
                       │
                       ▼
         ┌─────────────────────────┐
         │   Tactical Understanding │
         └─────────────────────────┘
                       │
     ┌─────────────────┼──────────────────┐
     │                 │                  │
     ▼                 ▼                  ▼
 What tactic?        When?              Where?
  战术是什么       什么时间发生       在哪里发生
     │                 │                  │
     │                 │                  │
     └────────────┐    │    ┌─────────────┘
                  ▼    ▼    ▼
               Evidence?
            凭什么这么判断？
                  │
                  ▼
     Structured Tactical Report
                  │
                  ▼
AI tactical commentary / coaching / retrieval
```

而最核心的 benchmark 定义就是：

$$
\boxed{
\text{Soccer Tactical Understanding}
=
\text{Recognition}
+
\text{Temporal Grounding}
+
\text{Spatial Grounding}
+
\text{Evidence Recognition}
}
$$

我认为现在这个版本比最开始的 **Temporal + Observation + Judgement + Tactic Reasoning** 已经清晰很多：四项之间没有太严重的概念重叠，每一项都有明确 GT，也都有比较客观、容易解释的 metric。

