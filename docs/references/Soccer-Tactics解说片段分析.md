# 解说片段分析

> 导出自 ChatGPT，时间：8/2/2026, 11:40:48 AM

# 消息 1

## 用户

请你首先认真阅读这几篇文章，然后回答一下我的问题,我目前最想了解到的一点是我想通过LLM inference来生成一些基本的战术解读，那么我最关键的几个问题如下: 1）我们目前SoccerMaster代码 (https://github.com/frankdarkluo/SoccerMaster/tree/main)的输入有什么可以帮的上忙的吗? 或者有更多利用价值的吗？ 2) 这三篇文章里你觉得哪篇最值得精读的? 目前公司可能不太有战术解读相关的数据，如果需要人标的话就比较费精力了，不过我也可以问一下KNQ公司。 3）如果想仅凭LLM 推理（这种多模态超智能模型，比如ChatGPT codex， Claude 或者豆包）去进行一定程度的战术解读的话，那么我们根据这几篇文章，我们最好需要哪些输入信息？这些信息最好能够通过规则引擎去获得。不然的话，就可能得找这些作者要数据了。4）这些信息是否需要更新进入当前的specs或者plans? 如果可以直接写完整相信plan的话也可以。一些我和其他人交流的内容如下，也供你参考: GenTac是一个基于扩散模型的生成框架，将足球战术建模为连续多人轨迹和离散语义事件上的随机过程，通过从历史跟踪数据学习玩家运动的基础分布来生成多样化、可信的长期未来轨迹，从而突破传统确定性预测的局限。该框架支持丰富的上下文条件（包括对手行为、特定球队或联赛风格以及战术目标），同时将连续空间动态映射到15类战术事件空间中，实现了轨迹预测、事件识别和反事实模拟的统一。在TacBench基准上的广泛评估显示，GenTac在几何精度、集体结构一致性、风格模拟和可控反事实模拟方面表现出色，并成功泛化至篮球、美式足球和冰球等其他团队运动。

关键要点
• GenTac采用基于扩散模型的生成框架来预测足球战术，将其概念化为多人轨迹和离散语义事件的随机过程，而非传统的确定性轨迹预测方法。

• 该框架通过因果滑动窗口策略自回归生成长期未来轨迹，在1秒至5秒预测范围内保持稳定，平均位移误差范围为0.62-4.55米，最终位移误差为1.22-10.80米。

• GenTac支持五种不同的条件约束机制：无条件、对手条件、球队条件、联赛条件和目标条件，使模型能够模拟特定球队风格、联赛特征和战术目标。

• 框架通过集体结构一致性指标（如拉伸指数、表面积、团队宽度和长度等）成功保持球队的集体结构完整性，避免生成物理不合理的配置。

• GenTac将连续轨迹预测与离散战术事件识别相结合，在事件类型分类中达到71.2%的准确率，在15个细粒度子类型中达到53.7%的准确率。

• 通过目标条件约束，该框架能够进行可控的反事实战术模拟，在进攻指导下增加威胁指标，在防守指导下增强空间控制能力。

• GenTac成功泛化至多个团队运动领域，包括篮球、美式足球和冰球，在5秒预测地平线上实现最小平均位移误差分别为0.32米、1.06米和1.04米。

相关问题
如何将GenTac框架中的集体结构一致性约束扩展到非标准化或不规则场地的运动中？ 

GenTac在真实比赛中的实时应用是否需要进一步的模型优化或架构调整？ 

该框架是否可以整合诸如球员疲劳、伤病状况或教练战术意图等隐性因素来改进预测准确性？ 

深度阅读工具
高亮摘录
澄清关键概念
识别假设
了解作者
查找相关新闻
查看相关论文
观看相关视频
展示更多
这篇文章有没有介绍他们的tactical annotation是怎么获取的呢？
GPT-5.6 Terra


已列出1个资源
已读取1个资源
已读取1个资源


有的，文中在 **TacBench 的“事件（tactical event）数据集构建”**部分和 补充的人工标注/校验部分都写到他们的 tactical annotation（战术事件标注）是怎么来的。

1) TacBench-Event 的标注从哪里来？
TacBench 的战术事件标注由三个公开赛事实数据源提供，并且这些数据是与多玩家轨迹同步的：

SkillCorner（Australian A-League）
Sportec DFL（German Bundesliga & 2. Bundesliga）
Metrica Sports（Metrica Sports）
文中说明：他们把这三者的事件数据做了一个统一的分类体系（把 15 个细粒度子类型合并到 5 个宏类：Build / Transition / Threat / Set Piece / Interruption）。

2) 这些标注是“原始标签直接用”还是“他们自己再标”？
是两者结合：

大多数事件：直接使用各数据集原本提供的 labels。
在一些复杂战术阶段（主要在 Build / Transition / Threat 这几个类别里更容易出现标注不清的情况）：他们会在自动流程上**增加人工标注（manual labels）**来保证准确性。
也就是说，tactical annotation 不是完全凭原始数据“照搬”，对容易混淆的复杂阶段做了人工校正/补标。

3) 另外：他们还做了“轨迹质量的人工筛选”（不是事件标注，但和数据标注流程相关）
在 broadcast-derived 的轨迹（来自 SoccerFactory pipeline）上，他们还做了一个人工标注与验证协议：由专家标注者系统性审查轨迹片段，并剔除出现“物理不合理跳变”的片段；他们提到标注界面里包含视频回放、精确定位起止帧、以及一致性检查功能。

这部分主要是为了保证后续事件推断/事件识别的输入轨迹可靠，而不是直接把“战术事件类型”从零人工逐条标出来。

如果你希望我进一步精确到文中对应的小节标题/原句位置（例如是“12.1.4 Event Curation and Split”下哪一句开始讲“manual labels”），我也可以继续帮你把相关段落逐条抽出来。@深度研究 

2604.18210v1.pdf
PDF
总结
GPT-5.6 Terra
摘要
本文提出 TacticGen，这是一个用于“生成足球战术”而非仅做“轨迹预测”的生成式框架：它将战术建模为在比赛语境条件下、由多名球员与球共同参与的多智能体运动序列，并使用多智能体扩散 Transformer 来刻画球员之间的合作与对抗互动。模型在包含超过 330 万事件与近 1 亿追踪帧的大规模顶级联赛数据上训练，能够在轨迹精度与真实感方面达到领先水平；更重要的是，TacticGen 通过扩散采样时的分类器引导（classifier guidance）机制，使同一个预训练模型在推理阶段可按规则、自然语言或学习到的价值模型来对齐不同战术目标，从而具备适应性。作者还系统验证了其随模型规模、训练步数与数据量增长的可扩展性，并通过足球专家的盲评案例研究证明生成战术在现实战术规划中具有实际效用。

关键要点
提出任务转变：从“预测可能发生什么”推进到“生成应当发生什么”的战术生成决策支持。
多智能体扩散 Transformer 核心架构：将球员与球的联合轨迹建模为扩散过程的可生成分布，并显式表征多智能体交互。
针对足球特性改造注意力方式：在 MADiT 中更强调跨智能体的关系建模，而非单纯时间维依赖，从而捕捉团队协同/对抗。
语境与事件双条件编码：用上下文编码（含球的两种设置：预测式或条件式）与事件编码（动作类型、全局特征、时间戳等）共同条件生成。
采样阶段的分类器引导实现多目标适应：无需为不同意图重训，通过引导项把生成轨迹“拉向”用户指定的战术标准。
引导目标的三种来源：支持可微规则函数、由大语言模型生成可执行可微函数，以及基于强化学习训练的价值模型引导。
可扩展性遵循缩放规律：实验表明模型规模更大、训练更充分、数据更多时误差持续下降，且存在可进一步提升的空间。
专家评估证明真实感与战术效用：盲评结果显示生成轨迹往往与真实轨迹难以区分，并且在战术质量偏好上显著优于真实样本（80% 场景胜出）。
相关问题
如何将 TacticGen 的“价值模型/奖励信号”与更细粒度的战术评价体系（如对抗强度、对位关系、风险-收益）结合以提升可解释性？ 
在更真实的对抗条件下（例如双方同时受控或更长时间尺度），TacticGen 的稳定性与长期一致性表现如何？ 
能否将场边视频、球员属性（身高/体重/疲劳）与更细语义角色（门将/边后卫/前锋等）融入该框架，从而扩展到跨联赛与跨运动场景泛化？ 
深度阅读工具
高亮摘录
澄清关键概念
识别假设
了解作者
查找相关新闻
查看相关论文
观看相关视频
展示更多

Preview
这个图对我们而言有帮助吗？我想通过LLM inference来生成一些基本的战术解读
GPT-5.6 Terra
有帮助，而且对你“用 LLM inference 生成基本战术解读”很对口——这张图把 TacticGen 的“如何被引导（steering）” 和 **“在推理时的两种工作模式”**讲得很清楚。你可以把它当成一个“把语言变成战术意图/输出形式”的设计蓝图，而不必照搬它的全部模型细节。

图里哪些点对你最有用？
1) Three ways to steer tactical generation（用来定义“LLM 输出什么”）
图给了三类引导来源，你做 LLM inference 时可以映射成三种目标形式：

Language-based guidance（语言引导）：
LLM 直接生成教练式指令/高层意图（例如“扩大边路宽度”“寻找半空间接应”“提高压迫强度”），再由下游模块把它转成可执行的条件（如果你有战术生成器/价值模型就接上，没有的话也能用于“解读模板”的生成）。
Rule-based guidance（规则引导）：
如果你只想做“基本战术解读”，这一支更轻量：LLM 生成的不是复杂策略，而是对可解释战术概念的文字说明/打分理由（support width、zone occupation、pitch control 等），再结合你已有的轨迹特征做解释。
Value-based guidance（价值引导）：
如果你能得到任何“收益/威胁/风险”的指标（哪怕是粗粒度的），LLM 可以围绕“长期收益”给解读；但若你没有价值模型，这一支你可能只能做概念化表达。
对你最直接可落地的是：让 LLM 先产出“意图/规则层面的解释结构”，再把解释和你手头的观察量绑定。

2) Two operating modes（用来决定“解读哪一种任务类型”）
图里两种推理模式意味着两种解读风格：

Short-context trajectory prediction（短上下文预测）：
适合做“接下来可能怎么走”的叙述式解读（更像预测+解释）。
Ball-conditioned tactical generation（基于球的条件生成）：
适合做“如果球走到某条路径，我们的队形/跑位如何配合”的what-if 场景解读（更像战术推演）。
如果你目前只有比赛当前时刻/短片段信息，优先用第一种解读范式；如果你能定义球的去向（哪怕是人工指定的路径/目标区域），第二种会更有“战术味”。

结论：这图对你“做基础战术解读”是有帮助的
你可以用它来把 LLM 的输出定义为：

高层战术目标/意图（language-based）
可解释的战术概念（rule-based）
（可选）与长期质量相关的理由（value-based）
并且选择是做“短预测解读”还是“基于球路径的推演解读”。
我想确认一个关键点：你现在的输入数据是什么——是 (A) 轨迹/球路+队形特征，还是 (B) 只有文本/事件？（这会决定你更适合用图里的哪种模式来做解读。）

(A) 轨迹/球路+队形特征
GPT-5.6 Terra


很好，你的输入是 (A) 轨迹/球路+队形特征，这正好匹配图里的 Short-context trajectory prediction（短上下文预测）那种“预测式解读”工作流：你让 LLM 把“刚发生/接下来几秒的空间变化”翻译成可解释的战术语言。

下面给你一个可直接落地的做法（偏“rule-based / language-based”的轻量版本），尤其适合做基本战术解读。

1) 你可以让 LLM 产出的“解读类型”（推荐顺序）
结合你的数据形态，我建议按这个顺序设计输出：

战术意图（意图层）：例如“压迫强度提升/边路纵深/中路渗透/拉开宽度/压缩回防”等
空间组织（结构层）：例如 support width（支撑宽度）、占据区域（zone occupation）、队形形态（横向拉开/纵向堆叠）
对位/协防解释（互动层）：从轨迹上读出“谁在盯谁、谁在补位、谁在制造空档”
接下来 1-3 秒的预测性叙述：结合短上下文轨迹变化（例如“前插后会被封堵/边后卫补横/中卫后撤形成封锁带”）
这样即使你没有价值模型，依然可以做到“可解释 + 战术味”。

2) 把你的数值/几何特征“喂”给 LLM 的方式
你可以把原始轨迹/球路特征做成一个紧凑摘要（不要把每帧轨迹全塞给 LLM）：

时间窗：例如过去 
t
0
→
t
n
o
w
t 
0
​
 →t 
now
​
  和未来预测 
t
n
o
w
→
t
n
o
w
+
3
s
t 
now
​
 →t 
now
​
 +3s
球路：球当前位置、速度方向、是否在向边路/肋部推进
队形几何：支撑宽度、团队长度、攻防队形中心（大致即可）
关键球员轨迹摘要：3 个最关键球员的“相对位移/速度变化/接近球距离变化”
区域标签（可选）：把场地粗分为若干 zone（例如 9宫格或肋部/半空间/边路），然后统计占位变化
然后 LLM 的任务就是：根据这些摘要生成语言化战术解读。

3) 推荐的提示词输出格式（让结果更像“解读报告”）
你可以要求 LLM 每次输出固定结构，便于后续评估/检索：

Summary（1-2句）：这段战术在做什么
Tactical Intent（意图）：1-3条要点
Space Organization（空间组织）：引用你给的几何特征（如宽度/长度变化、球在哪个区域）
Player Interaction（关键互动）：用“玩家角色/相对位置变化”解释对位与协防
What’s next（下一步推断）：1-3条“接下来几秒更可能发生什么”，并说明理由（对应轨迹证据）
4) 你可以直接用的“prompt骨架”（简短版）
你把下面字段替换成你的实际输入即可：

System/指令：你是一名足球战术分析助手……输出必须包含 Summary / Intent / Space / Interaction / Next。

User内容（示例）：

Match context: [比赛阶段/队伍A队伍B/阵型(可选)]
Time window: observed [t0tnow], forecast [tnowtnext]
Ball: pos=[...], velocity_dir=[...], target_zone=[...]
Team geometry: support_width=[...]->[...], team_length=[...]->[...]
Key players (top3):
P1 id=?, role=?, relative_motion=[...], dist_to_ball=[...]->[...]
P2 ...
P3 ...
Zone occupation change: [边路/半空间/中路占位变化简述]
Your task: Generate a basic tactical interpretation (no more than 150-200 Chinese characters per section).

## 助手

# 用 LLM inference 做基础战术解读的可行路径

## 结论

如果你的目标是**不先训练一个专门的战术模型，而是先用 LLM inference 生成“基本但可信”的战术解读**，那么最现实的路线不是照着 TacticGen / GenTac / TacEleven 去做生成模型，而是走一条更轻的路：

**SoccerMaster 现有 GSR / calibration / tracking / jersey / role / team / pitch-coordinate 管线提供“可计算的战术状态”；规则引擎先把这些状态压成少量可验证的战术事实；LLM 只负责把事实组织成中文战术解说。**

这条路和几篇论文的共同结论是一致的：它们都不是直接让模型“凭视频感觉说战术”，而是依赖**结构化上下文、球员/球队状态、事件语义、球路或目标条件**来驱动推理或生成。TacticGen 明确把条件输入写成过去轨迹、球、事件类型、时间戳和全局事件特征，并区分了“短上下文预测”和“给定球路的条件生成”；GenTac 则把战术建模为轨迹 + 离散事件，并用一组集体结构指标和攻防目标指标来评估；TacEleven 用的是“历史轨迹 + 高层指令 + 事件级文本-轨迹对 + critic”；GameSight 进一步说明，好的解说不是端到端 caption，而是**实体对齐 + 知识增强**两阶段。

所以对你来说，最重要的不是先想“哪个大模型最聪明”，而是先定义：

**LLM 到底看什么格式的战术状态，哪些断言可以直接说，哪些只能保守说。**

## SoccerMaster 现有输入和代码里最有价值的部分

基于我对你上传的 SoccerMaster repo 的直接检查，**对“战术解读”最有价值的不是 SoccerMaster 的 caption / classification 头，而是它接入的 sn-gamestate 游戏状态重建链路**。当前你已经有或几乎已经有以下几类信息：

- 单目转播视频上的**检测、跟踪、分割 refinement**
- **camera calibration / pitch localization**
- 每个检测实例的 **track_id**
- **team / role / jersey number**
- **bbox_pitch** 对应的球场米制落点
- 以及你自己 superpowers 里已经开始做的 **formation / topology / relations** 这层后处理

这意味着你手里已经有了做基础战术解说最关键的“几何底座”：

**谁在场上、谁和谁是一队、他们在球场上的相对位置、谁在往哪里移动、结构是变紧凑还是被拉开。**

这比单纯的 event caption 更重要，因为“为什么这样跑”“为什么这里形成优势”“为什么这是阵地战而不是纯快攻”，核心都来自这些几何与关系量，而不是来自一句 end-to-end 文本标签。

更具体地说，SoccerMaster 现在最值得榨干的输入价值有四层。

第一层是**身份与坐标**。这层已经足够支撑大部分“观察型”战术句子：
“右侧边路形成二打一”“后防线明显收得更低”“弱侧球员站位更宽”“中路有人从第二线前插”。
如果没有统一到 pitch coordinates，上面这些话几乎都不可靠。

第二层是**track-level continuity**。`track_id` 比球衣号更重要，因为球衣号会丢、会错、会被遮挡，但 track 可以让你在 2–5 秒窗口里可靠地说“这个人是从身后超到持球人外侧的”，这正是 overlap / run-in-behind 这种概念的前提。

第三层是**team structure**。你现有 formation-topology 设计里已经有 block height、depth、width、hull area、lane/band counts、side overload、inter-line gaps 这类量；它们和 GenTac 用来评估 collective structure consistency 的指标高度同构。GenTac 明确把 surface area、stretch index、team width、team length、centroid displacement、Frobenius distance、sync order parameter 当成集体结构评估指标；同时还用 off-ball expected threat、width threat、depth threat、defensive shape disruption、defensive dominant region 来刻画攻防目标。也就是说，你现在的 topology/relations 方向不是旁支，而正好踩在文献正中央。

第四层是**coarse semantic priors**。SoccerMaster 里的 caption classification / event-related模块不是没用，但它们更适合充当：
- 当前片段是 **build-up / transition / threat / interruption / set piece** 的粗先验
- 或者“这段更像 clearance / corner / ball out of play / goal / shot off target”之类的可见事件先验

它们**不应该直接承担“战术解释器”角色**。更合理的用法是：
**event prior + relations/topology + LLM narrative**。

一句话说，SoccerMaster 现在对战术解读最有帮助的部分不是“它能不能直接生成一句高水平战术话”，而是：

> 它已经把“可验证的场上状态”这件事做到了七八成。

## 三篇里面最值得精读的是哪篇

如果你现在只能深读一篇，我建议你先读 **GenTac**。
如果排优先级，我会排成：

**GenTac > TacticGen > TacEleven**

原因很直接。

### GenTac 最值得先读

GenTac 最贴你当前问题，因为它最像一篇“**如何把战术数字化**”的论文，而不是单纯“如何训练一个更强的生成模型”。

它对你最有用的地方有三点。

第一，它明确给了一个**可落地的任务拆法**：
把开放比赛战术理解拆成
- 多球员轨迹预测
- tactical event grounding / forecasting
- 条件化的风格/目标模拟
而不是一上来就要求模型能自由生成长篇高质量 commentary。

第二，它给了你一个非常实用的**中间语义层**：
5 个宏类、15 个细粒度 tactical events，加上结构指标和攻防目标指标。
这对你目前最重要，因为你完全可以不学它的扩散模型，只学它的“表示层”。也就是把你的 tactical state 组织成：
- 宏观阶段：build / transition / threat / interruption / set piece
- 微观观察：switch / overload / compactness / line-break / progression / clearance / defended 等
- 结构指标：width / length / stretch / centroid / overload
- 攻防指标：depth threat / width threat / local control / defensive compactness
这已经足够支撑 LLM 生成“基础战术解读”。

第三，GenTac 说明了你**不一定要从零造一个大战术标注集**。它的 TacBench 把轨迹和 tactical event 绑在一起，本质上告诉你：战术理解并不一定非得从主观长文本开始，你可以先从**结构化事件空间 + 规则可测指标 + 少量人工校验**做起。

### TacticGen 第二重要

TacticGen 更适合你拿来回答一个系统设计问题：

> 如果以后我想让系统不仅“解释刚才发生了什么”，还想“按某种战术目标去推演或重写未来几秒”，接口应该怎样设计？

它真正有价值的是**control interface**，不是 backbone 本身。
它把 inference-time steering 很清楚地分成：
- rule-based guidance
- language-based guidance
- value-based guidance
并且明确说条件输入包括过去轨迹、事件类型、goal difference / event outcome / timestamp 等全局事件特征，还把实际工作模式分成 predictive ball modeling 和 conditional ball modeling。

这对你当前的启发是：

你不用做扩散模型，但你应该照它的思路，把上层接口设计成两层：
- **state facts**：当前战术状态是什么
- **guidance / objective tags**：你想让 LLM从什么角度解释它
比如“强调宽度利用”“强调压迫效果”“强调中路人数优势”“强调边中转换”

这会让你的 LLM 输出稳定很多。

### TacEleven 第三

TacEleven 很强，但离你当前资源条件最远。
它需要：
- 超过 100 万 text-to-trajectory pairs 的 curated dataset
- 一个 language-controlled tactical generator
- 一个 multimodal tactical critic
- 用 xG / xT / pitch control 去筛选方案
这对“做开场可用的战术讲解 demo”来说太重了。

它最值得借鉴的不是训练范式，而是两个想法：

- **战术可以拆成事件级 text-to-trajectory pairs**
- **critic 要评估 proposal，而不是让 generator 一步到位**

对于你的系统，这可以被轻量化成：

- 规则引擎先提出若干 tactical facts / candidates
- LLM 再从中选择值得讲的那几个，并给理由

这很像你现在 hybrid commentary 的方向，但不需要 TacEleven 那种大规模数据依赖。

### 还有一篇你其实也该认真看

虽然你问的是“三篇”，但对你的**解说系统设计**来说，**GameSight**几乎是必须读的辅助材料。

因为 GameSight 直接回答了一个你眼前的问题：

> 做 commentary 的时候，哪些信息不能只靠视频帧直觉，必须额外补？

它发现 human 的 player/entity 对齐在 84.5% 的正确案例里都不只依赖 long-view tracking，而是结合了 line-up、历史事件、close-up、face/jersey/team cues；同时它把 commentary 设计成两阶段：先 entity alignment，再 knowledge refinement。更重要的是，它指出 live televised commentary 明显比 live text commentary 更频繁地引入内外部知识。

这对你直接对应的结论是：

**战术解读也应该两阶段：先事实对齐，再语言增强。**

## 如果只靠 LLM inference，要给它哪些输入信息

最关键的不是“给 LLM 更多东西”，而是“给 LLM **更少但更对** 的东西”。

你现在最适合的输入不是原始视频，也不是每帧全量坐标 dump，而是一份**战术状态摘要**。这份摘要最好由规则引擎从 SoccerMaster/GSR 输出自动生成。

我建议分成四层。

### 观察层

这是 LLM 可以**用确定语气说**的部分。

必须包含：

- 时间窗口与攻击方向
- 双方可见球员和球的 pitch coordinates 摘要
- track_id、team、role、可用时再加 jersey
- 当前或最近 2–5 秒的球路
- 持球方 / 持球人候选
- 双方结构指标
  - block height
  - block depth
  - block width
  - hull area
  - line gaps
  - lane/band counts
  - side overload
- 局部关系指标
  - 相对 carrier 的 longitudinal / lateral offset
  - nearest defender distance
  - relative depth vs second-last defender
  - local numerical superiority near ball
  - moved-past-carrier / outside-lane / central-lane 等布尔或分数

这些量基本都能通过现有 GSR + topology + relations 层得到，或者只需要很小扩展。

### 阶段层

这是 LLM 用来决定“这段话总体属于什么情境”的。

建议至少有：

- macro phase：`build-up / transition / threat / interruption / set piece`
- possession stability
- ball progression direction
- danger zone / target zone
- 是否完成明显的 line break 或 side switch
- 是否进入 final third / box-adjacent area

这层非常重要，因为没有 phase，LLM 很容易把同样的几何形态解释错语境。GenTac 把 tactical event grounding / forecasting 明确做成一个单独层，就是因为“几何正确”不等于“语义正确”。

### 解释层

这是允许 LLM 说“为什么值得注意”的，但必须受规则约束。

这里不要直接给结论长句，而是给：

- `candidate_concepts`
- 每个 concept 的 evidence refs
- 置信度
- 允许的 claim level

例如：

```json
{
  "concept_id": "switch_of_play",
  "evidence": ["ball.lateral_displacement>22m", "weak_side_occupancy_increase", "defensive_width_shift_delay"],
  "confidence": 0.84,
  "claim_level": "effect"
}
```

然后在 prompt 里硬性规定：

- `observation` 可以肯定说
- `effect` 只有 evidence 足够时可以肯定说
- `intention` 只能用“可能/看起来是在/似乎想要”这类保守措辞

这和你之前修订过的 claim-level 思路一致，也和 GameSight 的“先对齐再增强”一致。

### 约束层

这是避免 LLM 胡说的关键。

至少加入：

- visible-region / coverage
- calibration confidence
- tracking continuity
- identity confidence
- possession confidence
- unsupported regions / not observed regions

如果没有这层，LLM 天然会把“没看清”补成“推理到了”。

## 最好由规则引擎先产出的输入格式

你现在最应该新增的不是更多 prompt，而是一个中间文件，比如：

`tactical_state.json`

推荐结构如下。

```json
{
  "clip_id": "SNGS-xxxx",
  "window": {"t0": 12.0, "t1": 16.0},
  "attack_direction": "left_to_right",
  "phase": {
    "macro": "build-up",
    "possession_team": "left",
    "possession_confidence": 0.83,
    "transition_flag": false
  },
  "ball": {
    "x": 8.4,
    "y": -12.1,
    "speed": 7.3,
    "dx": 5.8,
    "dy": -1.5,
    "target_zone": "right_halfspace"
  },
  "team_states": {
    "left": {
      "block_height_m": 6.1,
      "block_width_m": 39.2,
      "block_depth_m": 24.5,
      "hull_area_m2": 512.0,
      "side_overload": 0.27,
      "line_count": 3,
      "inter_line_gaps_m": [8.9, 11.7]
    },
    "right": {
      "block_height_m": -11.8,
      "block_width_m": 31.4,
      "block_depth_m": 17.2,
      "hull_area_m2": 388.0,
      "side_overload": -0.03,
      "line_count": 2,
      "inter_line_gaps_m": [9.8]
    }
  },
  "local_relations": [
    {
      "type": "overlap_candidate",
      "runner_track_id": 27,
      "carrier_track_id": 12,
      "evidence": ["rel_x:-2.4->1.8", "outside_lane:true", "runner_disp:+7.1m"],
      "confidence": 0.81,
      "claim_level": "observation"
    }
  ],
  "semantic_priors": {
    "event_like": ["progression", "threat"],
    "goal_side_pressure": "medium"
  },
  "visibility": {
    "left_coverage": 0.89,
    "right_coverage": 0.84,
    "not_observed": ["far_left_touchline_high"]
  }
}
```

这个文件就是你给 LLM 的主输入。
然后 LLM 只需要完成两件事：

- 从 `team_states + local_relations + semantic_priors` 中挑最值得讲的点
- 按 claim level 生成中文解说

这比“直接把 minimap 图片 + 原始轨迹 + glossary 全扔进去”更稳。

## 这些信息里哪些已经能从 SoccerMaster 现有管线得到

先说最实用的判断：

**你现在已经拥有做 observation-level tactical commentary 的 70% 输入。**

已经具备或很容易导出的：

- 球员与球场位置
- 跟踪连续性
- 球队归属
- 角色
- 球衣号
- 单目相机到 pitch coordinates 的映射
- 结构级统计的实现起点
- 事件粗先验

真正还缺的，主要是下面这几类：

- **稳定的 possession / carrier 状态**
- **ball destination / intended side** 这类球路摘要
- **local tactical relations**
  比如 runner 相对 carrier 的 from→to 变化
- **line-breaking / switch / overload / compactness** 的规则判定
- **visibility / confidence metadata**
- 更强的 **phase segmentation**
- 如果你想讲“为什么这样站位更好”，还缺
  - pitch control / defensive dominant region
  - simple pass-lane openness / receiver availability
  - local time-to-intercept 近似量

这里要强调一个很重要的现实判断：

> 你现在**不需要**作者们的训练数据，才能做出第一版基础战术解读。

你真正需要的是他们论文里已经公开的**表示方式和指标思想**，而不是他们的大模型本体。
只有当你未来要做：
- 真实 tactical benchmark
- 学习式 tactical recognizer
- counterfactual simulator / value model
时，才有必要去问作者要更细的数据或注释接口。

## 需要更新到当前 specs 和 plans 吗

需要，而且我建议是**直接更新**，不是只加一段备注。

你当前的 KB 扩充 spec 已经做对了一半：它把战术概念从“写死在代码里”改成“带测试的数据”，这条方向没问题。

但结合这几篇文章后，我认为要补的不是更多概念，而是要在 KB 前面再插一层：

**从“战术概念库”升级成“战术状态表示层 + 概念层 + 解说层”。**

### 该更新的核心点

#### 在 KB 之前新增 `tactical_state` 层

现在 spec 更偏向“如何写 concept recipe”，还不够强调“LLM 和 rules 应先共享什么中间状态”。

应该新增一个设计原则：

> 任何 tactical concept 都先从统一的 tactical_state.json 上解析，而不是直接对原始 relations 做概念级推断。

这样有三个好处：

- concepts.yaml 不会越来越像隐式 DSL
- LLM prompt 更稳定
- benchmark / evaluator / commentary 都能吃同一种状态表示

#### 吸收 GenTac 的结构指标和 phase 设计

建议把以下量正式写进 representation spec：

- surface area / hull area
- stretch / compactness proxy
- team width / length / centroid displacement
- offensive metrics：off-ball threat / width threat / depth threat 的简化近似
- defensive metrics：shape disruption / dominant region 的可计算近似
- macro phase：build / transition / threat / interruption / set piece

这些变量不一定一次都做完，但应该先进入 spec，作为 planned fields。GenTac 的价值就在于它已经证明这组量足够支撑“战术”这一层的表述。

#### 吸收 TacticGen 的输入接口思想

建议在 plan 里显式加入两种模式：

- **observed-state commentary mode**
  当前主模式，只解释已发生/正在发生的窗口
- **ball-conditioned what-if mode**
  后续扩展模式，当你有人工指定或规则预测的 ball path 时，生成“如果球走向弱侧/肋部，会造成什么结构变化”的推演性解说

这不要求你训练 TacticGen，但要把接口留出来。TacticGen 把 predictive 和 conditional-ball 分开，是非常值得沿用的系统设计。

#### 吸收 GameSight 的两阶段范式

建议把当前 commentary 生成 spec 明确改成：

- Stage A：**tactical fact selection**
- Stage B：**commentary realization + optional knowledge refinement**

并且在 schema 里加入：

- `fact_refs`
- `claim_level`
- `external_knowledge_used`
- `internal_match_context_used`

因为 GameSight 说明了：
只做“看到什么”不够；只做“润色”会幻觉；必须先把事实对齐，再增强表达。

## 一个我认为最适合你当前资源条件的完整计划

## 近期目标

先做一个**不用新训练数据、对大模型依赖最小、但能稳定产出基本战术解读**的 v1。

### 核心产物

- `tactical_state.json`
- `tactical_facts.json`
- `commentary_tactical.json`
- 一个小型 benchmark add-on

### 可解释范围

只先做三类：

- **phase / state**
- **structure**
- **local movement pattern**

先不做高主观度的意图和反事实。

## 表示层

### 新增 `tactical_state.json`

由现有 GSR + topology + relations 导出，不直接让 LLM碰原始轨迹。

字段分为：

- clip/window metadata
- ball state
- possession / phase
- team structure metrics
- local player-player / player-ball relations
- visibility/confidence
- semantic priors

### 必做的新规则量

你之前讨论过 query grammar 只扩四个原语，我认为这里继续成立。
近期只补会显著改变表达能力的量：

- `track_id` 绑定
- `abs`
- `from -> to transition`
- `duration`

同时新增几个派生量：

- `ball_progress_x`
- `ball_switch_y`
- `runner_passed_carrier`
- `depth_vs_last_line`
- `local_num_adv_near_ball`
- `weak_side_occupancy_delta`

## 概念层

### v1 只做 8–12 个高可观测概念

建议第一批：

- build_up_from_back
- transition_attack
- threat_entry
- compact_block
- side_overload
- switch_of_play
- overlap_run
- run_in_behind
- halfspace_occupation
- line_breaking_progression
- local_numerical_superiority
- width_depth_stretch

建议延期：

- decoy_run
- press_resistance
- second_ball_protection
- third_man_intention
- pinning_defender

原因不是它们不重要，而是目前没有足够可靠的反事实或 defender-response 表示。

## 解说层

### 统一 claim contract

每条 tactical fact 必须包含：

- `concept_id`
- `evidence_refs`
- `confidence`
- `claim_level`
- `visibility_ok`

生成时严格限制：

- observation：可 assertive
- effect：仅 evidence 足够时 assertive
- intention：只能 hedged

### prompt 只做一件事

要求 LLM 从候选 facts 中选择 1–3 个最值得说的点，并按固定格式输出：

- Summary
- Tactical Reading
- Why it matters
- What may happen next

如果是短片段，只输出一段；如果是完整攻势，可以两段。

## 评测层

### 没有大标注集时的最小可行 benchmark

先不要追求长 free-form gold。先做：

- 50–100 个 windows
- 每个 window
  - macro phase
  - 1–2 个 core concept
  - 1 个 hard negative
  - claim level 边界
- match-disjoint split
- 输出指标：
  - concept precision
  - false tactical insertions per minute
  - overclaim rate
  - human preference on usefulness / trust

这比一开始就走 TacEleven/GenTac 那种大数据路线更符合你现在的资源。

### 如果能问 KNQ，要问什么

不是先问“能不能给我一个大战术标注集”，而是问：

- 能否提供 50–100 个典型片段
- 每个片段只标
  - phase
  - 最值得讲的一个 tactical point
  - 一句自然中文说明
- 如果可能，再给 1 个相似但不该触发的 negative

这会比要求他们做 full annotation 更容易落地。

## 最后的判断

我对你这个方向的判断是：

**完全可做，而且当前最优策略不是训练一个战术大模型，而是把 SoccerMaster/GSR 现有几何输入最大化利用起来，先做“事实化战术状态 → LLM 叙述”的架构。**

如果只选一篇最值得精读：**GenTac**。
如果只选一个对你系统设计最有启发的额外参考：**GameSight**。
如果只选一个当前代码里最该深挖的资产：**sn-gamestate / bbox_pitch / track_id / topology 这一整层，而不是 caption generation 头。**

更直接一点说：

> 你现在离“基础战术解读”并不远，缺的不是更大的 LLM，而是一个更干净的 `tactical_state` 表示层。它一旦建好，LLM 才真的有东西可推理。


<div align="center">◆ ◆ ◆</div>

# 消息 2

## 用户

你是否同意采用下面的混合方案？
代码确定性地产生版本化 tactical_state；Doubao只根据公开概念目录提出 concept_id + window + track_id actors；代码用私有固定 recipe 验证；最后 Doubao只把通过验证的 observation/effect 写成自然中文。

## 助手

我同意。这是当前最稳妥的 **hybrid architecture**：既保留 LLM 对开放视频片段的候选发现能力，又把最终战术断言控制在可复现、可审计的规则证据上。它也符合几篇论文共同体现的思路：战术语义建立在轨迹、球队结构和事件上下文上，而 commentary 应先完成事实对齐，再进行语言增强。

建议明确写成以下五段流水线：

```text
GSR / tracking / calibration / topology
        ↓
versioned tactical_state
        ↓
Doubao proposal
{concept_id, window, actor_track_ids}
        ↓
private versioned recipe verification
        ↓
verified_tactical_facts
        ↓
Doubao Chinese realization
```

## 我认可的责任边界

### 1. `tactical_state`：代码负责事实

它应完全确定性地产生，并记录：

```json
{
  "schema_version": "1.0",
  "extractor_version": "git-sha",
  "window": [6.2, 9.4],
  "attack_direction": "left_to_right",
  "tracks": [],
  "ball_state": {},
  "possession": {},
  "team_structure": {},
  "relations": {},
  "visibility": {},
  "confidence": {}
}
```

这层不出现“套边”“牵制”“拉开空间”之类战术结论，只放可测量状态。GenTac 使用多人轨迹、球队宽度/长度、表面积、stretch index、centroid displacement 等结构量来连接几何运动和战术语义，证明这个中间层是合理的。

### 2. Doubao proposal：只负责提出假设

模型只允许输出：

```json
{
  "concept_id": "overlap_run",
  "window": {"t0": 6.2, "t1": 9.4},
  "actors": {
    "carrier_track_id": 12,
    "runner_track_id": 27
  }
}
```

公开概念目录可以提供：

- 名称和定义；
- aliases；
- inclusion / exclusion criteria；
- 典型语言例子；
- 可支持的 claim levels。

但**不要提供 recipe、阈值、checker 实现和测试样本**。否则模型会根据验证规则构造容易通过的 proposal，重新引入 self-verification 的变体。

### 3. verifier：代码负责裁决

验证器只读取：

```text
proposal
+tactical_state
+private recipe version
```

输出应包含：

```json
{
  "fact_id": "fact-0017",
  "concept_id": "overlap_run",
  "status": "verified",
  "actors": {...},
  "window": {...},
  "evidence_refs": [
    "relation.runner_27_vs_carrier_12.longitudinal_transition",
    "relation.runner_27.outside_lane_duration",
    "track.27.forward_displacement"
  ],
  "verified_claim_levels": ["observation"],
  "confidence": 0.83,
  "state_version": "1.0",
  "recipe_version": "overlap_run@2"
}
```

这里最重要的一点是：

> **验证 observation 不等于验证 effect。**

例如验证出“27 号从持球人外侧套上”，并不能自动允许说“制造了二打一”或“拉走了防守人”。后两者需要分别验证局部人数差、防守人响应、空间变化等 effect recipe。

### 4. 中文生成：只看已验证事实

最后一次 Doubao 调用最好只接收：

- `verified_tactical_facts`；
- 少量基础比赛上下文；
- 可使用的球衣号/队名映射；
- 输出风格和长度要求。

**不要再次把完整、未验证的 `tactical_state` 交给它自由推理**，否则模型可能绕过验证器，额外补充没有通过的战术结论。

例如输入：

```json
{
  "verified_facts": [
    {
      "concept_id": "overlap_run",
      "actors": {"runner_jersey": 7},
      "window": [6.2, 9.4],
      "claim_level": "observation"
    }
  ]
}
```

允许输出：

> 7号从持球人外侧快速套上，为边路进攻增加了一个接应点。

但如果 `effect` 未验证，不允许输出：

> 这次套边成功带走边后卫，在肋部制造了空当。

## 两个必要补丁

### 强制 abstention

没有通过验证时，系统应返回：

```json
{
  "tactical_commentary": null,
  "reason": "no_verified_tactical_fact"
}
```

然后继续普通事件解说。**不应为了每个窗口都插入战术话术而降低阈值。**

### v1 暂停 intention

建议 v1 只允许：

```text
observation
effect
```

暂不允许：

```text
intention
```

除非仅使用明确的保守表达，并且有对应证据：

> 这次跑动可能是在尝试牵制边路防守。

TacticGen 本身也把战术表示建立在球员、球、事件和上下文条件上；它能够进行目标引导，不意味着从一次真实跑动中可以唯一恢复球员真实意图。

## 最终判断

这套方案应当成为当前 spec 的核心架构，而不是一个可选实现：

```text
LLM proposes hypotheses.
Code owns evidence and verification.
LLM realizes verified facts.
```

它同时解决了三个主要问题：

1. 避免 LLM 自拟条件、自我验证；
2. 不需要先建设大规模战术文本标注集；
3. 能对每句战术解说追溯到 `state_version + recipe_version + evidence_refs`。

唯一需要严格坚持的是：**Doubao 可以提案和写作，但不能定义事实、阈值或验证标准。**


<div align="center">◆ ◆ ◆</div>

# 消息 3

## 用户

第三问 [Q:STRUCTURE]：
build_up / transition / threat / interruption / set_piece 这类 macro phase，是否应当只属于 tactical_state.derived_context，而不再与 overlap_run 等战术概念混在同一个 KB 层级？
我的推荐答案：是。
建议明确分为：
macro_phase：比赛正处于什么阶段；
tactical_concept：这个阶段中发生了什么可验证结构或跑位；
claim_level：我们可以说到 observation、effect 还是 intention。
例如：
macro_phase: transition
verified concept: switch_of_play
claim: “球队在转换阶段迅速把球转到弱侧。”
这意味着现有 positional_attack、counter_attack 应迁移为 phase/context 定义；KB v2 则专注于 overlap_run、compact_block、side_overload、switch_of_play 等可验证概念，避免 phase 与 tactic 重复计数。
你是否同意这个 taxonomy 分层？

## 助手

同意，而且这层分离应当写进 schema，作为强约束。GenTac 本身也把 trajectory/event context 与更细的 tactical event grounding 分开处理；你这里进一步拆成 `macro_phase → verified concept → claim_level`，会比把所有标签塞进同一 KB 更清晰。

建议结构固定为：

```yaml
derived_context:
  macro_phase:
    label: transition
    confidence: 0.82
    source: rule_v1
    window: [12.0, 16.5]

verified_concepts:
  - concept_id: switch_of_play
    actors: [...]
    window: [13.2, 15.1]
    verified_claim_levels:
      - observation
      - effect
```

三层职责应严格区分：

- `macro_phase`：当前比赛处境，回答“现在处在哪种阶段”。
- `tactical_concept`：阶段内发生的可验证结构、跑位或配合，回答“具体发生了什么”。
- `claim_level`：证据允许系统说到多深，回答“这句话可以有多强”。

因此建议迁移：

```text
positional_attack → macro_phase / phase subtype
counter_attack    → macro_phase / transition subtype
build_up          → macro_phase
threat            → macro_phase 或 danger_state
interruption      → macro_phase
set_piece         → macro_phase
```

KB v2 只保留可独立验证的模式：

```text
overlap_run
run_in_behind
switch_of_play
side_overload
compact_block
line_break
halfspace_occupation
local_numerical_superiority
width_depth_stretch
```

唯一需要注意的是，`threat` 更像一个**语义状态或危险等级**，不一定总是严格的比赛阶段。可以暂时放在：

```yaml
derived_context:
  macro_phase: build_up
  danger_state: threat
```

而不是强行要求 `macro_phase` 只能取一个互斥值。最终解说由两者组合：

> 球队在由守转攻阶段迅速把球转向弱侧，利用对方防线尚未完成横向移动的空当。

其中：

```text
transition            = context
switch_of_play        = verified concept
防线未完成横移         = verified effect
“为了利用空当”         = intention，需谨慎或暂不输出
```

这能避免 phase 与 concept 重复计数，也能让 benchmark 分别评估 phase accuracy、concept precision 和 overclaim rate。

