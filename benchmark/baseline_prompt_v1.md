# 足球战术视频 Baseline Prompt v1

复制下方代码块到未接触答案的新会话，并通过文件和工具权限隔离标注与参考；提示词不能替代权限隔离。

```text
分析以下足球视频，识别战术、发生时间、执行球队、区域和支持证据，逐视频输出 JSON。
工作目录：/Users/gluo/Desktop/SoccerMaster
输出目录：outputs/gpt5.6-terra/
Prompt 版本：soccer-tactic-baseline-v1-glossary-blind。

一、输入
读取 data/战术词条表.csv 的英文 ID 和定义，仅识别以下十二类，不按 P0/P1 筛选：
long-ball, line-breaking-pass, run-in-behind, one-two, overlap-underlap, switch-of-play, cutback, high-press, counter-attack, short-corner, back-line-possession-management, tactical-foul。
零样本隔离：只读取词条表、下列视频及其采样帧，严禁读取 data/pilot_study.csv、参考答案、样本标注或历史预测，更不能将其复制、改写或转换成预测 JSON。已接触本批答案则停止并换隔离新会话，不自行评分。
真实推理：每条预测必须由实际被测多模态模型查看该视频的 4 fps 帧后独立生成；脚本仅用于抽帧、校验和保存，禁止批量编造或模拟预测。无法查看图像用 unsupported_input，未看完用 incomplete_review，均不得标为 ok。

INPUT_CLIPS_JSON
{
  "AA-train": [52, 103, 167, 173, 192, 210, 268, 306, 328, 366, 395, 436, 462, 508, 511, 524, 564, 628, 629, 705, 709, 730, 737, 740, 750, 799, 831, 883, 928, 949, 969, 972, 1082, 1091, 1178, 1192, 1219, 1233, 1250, 1276, 1312, 1318, 1330, 1355, 1401, 1403, 1425, 1436, 1470, 1472, 1500, 1554, 1558, 1585, 1592, 1628, 1641, 1702, 1704, 1727, 1862, 1911, 1914, 1920, 1938, 1946, 1988, 1992, 1993, 2196, 2225, 2238, 2255, 2276, 2385, 2424, 2460, 2470, 2517, 2555, 2697, 2745, 2746, 2768, 2871, 2979, 2982, 3006, 3081, 3088, 3091, 3205, 3221, 3229, 3249, 3288, 3334, 3356, 3381, 3463, 3468, 3485, 3512, 3580, 3585, 3628, 3668, 3679, 3698, 3702, 3769, 3772, 3796, 3838, 3855, 3908, 3940, 3961, 3982, 4094, 4159, 4208, 4272, 4301, 4346, 4358, 4431, 4507, 4531, 4533, 4574],
  "AA-test": [1, 13, 80, 110, 131, 158, 188, 280, 339, 387, 417, 484, 496, 508, 547, 553, 565, 596, 706, 777, 809, 892, 950],
  "GSR": [1, 2, 3, 5, 7, 8, 9, 10, 16, 19, 21, 22, 24, 25, 26, 28, 31, 32, 37, 39, 41, 43, 45, 47, 48, 49, 53, 54, 55, 58, 60, 61, 62, 63, 65, 68, 70, 73, 78, 79, 81, 84, 87, 88, 89, 90, 92, 93, 94, 98, 99, 100, 101, 107, 111, 114, 117, 118, 122, 123, 124, 128, 129, 130, 131, 132, 135, 137, 138, 140, 141, 142, 145, 146, 147, 155, 157, 159, 160, 161, 163, 167, 170, 171, 173, 175, 179, 182, 183, 184, 189, 190, 192, 193, 197, 198]
}
END_INPUT_CLIPS_JSON

- AA-train-N -> data/SoccerNet-AA_clean_train_valid/train/clean/AA-N.mp4
- AA-test-N -> data/SoccerNet-AA_clean_test_challenge/test/clean/AA-N.mp4
- GSR-NNN -> data/GSR_2025_videos/SNGS-NNN.mp4（仅 GSR 编号补足三位）
每个视频统一按 4 fps 抽帧，保留原分辨率和实际时间戳，查看全部采样帧；不使用音频或额外补帧。缺失视频记录失败，不替换。

二、输出格式
每个视频保存为 outputs/gpt5.6-terra/predictions/<clip>.json，不覆盖已有结果。字段如下，示意值替换为实际结果：
{
  "schema_version": "soccer-tactic-baseline-v1",
  "clip": "<输入 clip>",
  "video_path": "<工作区相对视频路径>",
  "status": "ok",
  "error": null,
  "duration_s": 30.0,
  "tactics": [
    {
      "instance_id": "p1",
      "tactic": "<十二类标签之一>",
      "team": "<执行队球衣描述>",
      "pattern_status": "pattern_observed",
      "temporal_grounding": {"start_sec": 0.0, "end_sec": 1.0},
      "spatial_grounding": {"mode": "origin_target", "origin_half": null, "target_half": null},
      "evidence": [{"span_s": [0.0, 1.0], "text": "<具体可见事实或关系>"}]
    }
  ],
  "uncertainties": [{"instance_id": "p1", "text": "<不确定内容及原因>"}]
}

- tactics 允许多项或空数组，不重复同一实例；instance_id 在本视频内唯一。pattern_status 取 pattern_observed（结构可见，不代表成功）、prefix_only（仅有前缀）、unresolved（证据不足）。
- 时间从视频开头 0 秒起，不用比赛计时牌；区间只覆盖本次可见过程且不超出视频，未知填 null。evidence 写具体可见事实或关系，附 span_s，不编造意图或结果。
- team 用球衣描述指代执行队，半场相对于该队，值为 own_half / opponent_half / null。逼抢、犯规、后场倒脚改用 {"mode":"event_area","event_half":null}；其余用模板中的起点/目标字段，打身后和套边取跑者位置，其他取相关传球或推进位置。
- 不确定内容写入 uncertainties 并引用实例 ID，整片问题用 null；没有则用 []。不适用空间字段省略，适用但不确定填 null。
- status 取 ok、missing_video、unreadable_video、unsupported_input、incomplete_review；只有完整查看并完成预测才为 ok 且 error = null，其他状态写明 error、tactics = []，未知时长为 null。
保存合法 JSON；每个视频均保留结果或失败记录，最后汇报完成数、失败数和未完成 ID。
```