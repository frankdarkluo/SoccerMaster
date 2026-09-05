#!/usr/bin/env python3
"""Build the auditable GSR-2025 rationale and provisional 3+1 workbooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from copy import copy
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tactics_qa.grounded_assets import load_label_map, normalize_tactic

SOURCE = ROOT / "data/annotations/多模态模型战术识别.xlsx"
TEMPLATE = ROOT / "data/annotations/GroundedTactics_9clips_question_bank_provisional.xlsx"
HUMAN_OUT = ROOT / "data/annotations/多模态模型战术识别_GSR2025_AI润色.xlsx"
QUESTION_OUT = ROOT / "data/annotations/GroundedTactics_GSR2025_question_bank_provisional.xlsx"
ONTOLOGY = ROOT / "data/足球战术数据库_词条表_Grid.csv"
ALIASES = ROOT / "benchmark/grounded_tactics/label_aliases.json"
SOURCE_SHA256 = "1f1a6399b537fc25e96672298681cfc98574f65d5831a3e25eb6c4eeb7cb7037"
TEMPLATE_SHA256 = "07e4427596b73ef19c66b5bcc19140f27e57ba2f28691866d27c9543f59ed125"
LETTERS = "ABCD"
TRACKS = (
    "T1_TEMPORAL", "T2_OBSERVATION", "T3_DIAGNOSIS", "T4_INTEGRATED",
)
USED_TACTICS = (
    "back-three-vs-four", "break-the-press", "centre-back-carry",
    "corner-goalkeeper-crowding", "corner-near-far-post", "counter-attack",
    "cutback", "dummy-run", "false-nine", "gegenpress", "gk-in-buildup",
    "goalkeeper-line-height", "high-press", "line-breaking-carry",
    "line-breaking-pass", "long-ball", "long-throw-in", "low-block",
    "man-marking", "offside-trap", "one-two", "overlap-underlap",
    "overload-to-isolate", "quick-free-kick", "run-in-behind",
    "second-ball-structure", "set-piece-zonal", "short-corner",
    "switch-of-play", "tactical-foul", "third-man-run",
    "third-player-combinations",
)
LEGACY_CARDS = ("halfspace-penetration", "hybrid-marking", "zonal-marking")
SPAN_OVERRIDES = {
    "SNGS-008": ("00:02–00:16", "Normalized mixed dash/colon punctuation after video verification."),
    "SNGS-074": ("00:10–00:16", "Normalized full-width and semicolon time punctuation after video verification."),
    "SNGS-081": ("00:10–00:16", "Normalized semicolon time punctuation after video verification."),
    "SNGS-135": ("00:25–00:30", "Source ends at 00:31; derived interval was capped at the 30 s video boundary."),
    "SNGS-148": ("00:22–00:24", "Normalized semicolon time punctuation after video verification."),
    "SNGS-182": ("00:10–00:17", "Normalized semicolon time punctuation after video verification."),
}
VISUAL_HOLD_REASONS = {
    "SNGS-002": "The possession-regain trigger is not unambiguous in the wide view.",
    "SNGS-004": "The A-to-B-to-C third-player relation cannot be uniquely tracked.",
    "SNGS-006": "The restart view does not uniquely distinguish man, zonal, and hybrid marking.",
    "SNGS-011": "The second-ball recovery is visible, but a pre-positioned structure is not established.",
    "SNGS-014": "Players are near the goalkeeper, but deliberate crowding is not uniquely established.",
    "SNGS-015": "The switch and isolated receiver are visible, but the originating overload is not established.",
    "SNGS-018": "The set-piece view does not uniquely distinguish zonal and hybrid marking.",
    "SNGS-027": "The second-ball recovery is visible, but a pre-positioned structure is not established.",
    "SNGS-032": "The ball emerges from a contest; a clean counterattack regain trigger is uncertain.",
    "SNGS-041": "Three deep players are visible, but stable centre-back roles cannot be confirmed from this clip alone.",
    "SNGS-046": "The exact three-player dependency and receiver identity remain ambiguous.",
    "SNGS-055": "The foul is visible, but tactical intent is not directly observable.",
    "SNGS-062": "The foul is visible, but tactical intent is not directly observable.",
    "SNGS-070": "A coordinated defensive-line step cannot be uniquely confirmed.",
    "SNGS-071": "The set-piece view does not uniquely distinguish zonal and hybrid marking.",
    "SNGS-083": "The run and later space are visible, but a decoy intention is not directly observable.",
    "SNGS-087": "A player drops to receive, but the false-nine role cannot be established from the clip alone.",
    "SNGS-092": "The foul is visible, but tactical intent is not directly observable.",
    "SNGS-104": "The curved restart pass is visible, but the claimed line-breaking criterion is debatable.",
    "SNGS-105": "The runner is visible, but the preceding third-player dependency is not unique.",
    "SNGS-110": "Players occupy the goalkeeper's vicinity, but deliberate interference is not directly observable.",
    "SNGS-116": "Players occupy the goalkeeper's vicinity, but deliberate interference is not directly observable.",
    "SNGS-121": "The foul and caution are visible, but a transition-stopping tactical function is uncertain.",
    "SNGS-125": "Close tracking is visible, but pure man marking cannot be separated from hybrid marking.",
    "SNGS-135": "The source interval exceeds the clip and the third-player relation is incomplete before 30 s.",
    "SNGS-136": "The runner is visible, but the preceding third-player dependency is not unique.",
    "SNGS-148": "The run and later shooting space are visible, but a decoy intention is not directly observable.",
    "SNGS-153": "A forward drops to receive, but the false-nine role cannot be established from the clip alone.",
    "SNGS-162": "The immediate post-loss press overlaps with counterpressing and is not uniquely high pressing.",
    "SNGS-168": "The runner is visible, but the preceding third-player dependency is not unique.",
    "SNGS-179": "The foul is visible, but tactical intent is not directly observable.",
    "SNGS-182": "The switch and one-versus-one are visible, but the originating overload is not established.",
    "SNGS-200": "A deep defensive shape is visible, but deliberate concession of possession is not observable.",
}
OBSERVATION_EN = {
    "back-three-vs-four": "Three deep defenders form the first line while the team begins possession from the back.",
    "break-the-press": "Several short passes move the ball away from clustered pressure into a less occupied lane.",
    "centre-back-carry": "A central defender carries the ball out of the back line before passing forward.",
    "corner-goalkeeper-crowding": "Several attackers occupy the goalkeeper's immediate vicinity during a corner delivery.",
    "corner-near-far-post": "The corner is delivered to the near post, where a first contact redirects it toward a teammate.",
    "counter-attack": "After possession changes, the team progresses forward immediately while several teammates run ahead.",
    "cutback": "A player near the byline passes backward or diagonally toward a trailing teammate in the penalty area.",
    "dummy-run": "An off-ball runner moves across defenders, defenders follow, and another attacker receives shooting space.",
    "false-nine": "A central forward drops toward midfield to receive before another teammate advances beyond the line.",
    "gegenpress": "Immediately after losing possession, nearby teammates converge and pressure the new ball carrier.",
    "gk-in-buildup": "The ball is played back to the goalkeeper, who becomes an additional passing outlet.",
    "goalkeeper-line-height": "The goalkeeper advances far from goal to control a ball behind the defensive line.",
    "high-press": "Multiple attackers pressure the opponent's buildup high up the pitch and restrict nearby exits.",
    "line-breaking-carry": "The ball carrier dribbles beyond multiple defenders and enters a more advanced line.",
    "line-breaking-pass": "A forward pass travels through or over an opponent line to a teammate beyond it.",
    "long-ball": "A direct aerial ball is sent from a deeper area toward advanced teammates and the first landing point.",
    "long-throw-in": "A throw-in is delivered directly from the touchline into the penalty-area contest.",
    "low-block": "Most defenders remain compact close to their penalty area while the opponent holds the ball.",
    "man-marking": "Defenders stay close to individual attackers and follow their movement during the restart.",
    "offside-trap": "The last defensive line steps together as the pass is played, leaving a runner beyond it.",
    "one-two": "A player passes to a teammate, runs beyond a defender, and receives an immediate return pass.",
    "overlap-underlap": "An off-ball teammate runs beyond the ball carrier on an outside or inside lane and receives the pass.",
    "overload-to-isolate": "The ball moves from a crowded side to an isolated teammate facing one defender on the far side.",
    "quick-free-kick": "The free kick is restarted before the defence is fully reset and reaches a moving teammate.",
    "run-in-behind": "A runner starts level with the last line and accelerates into the space behind as the pass is played.",
    "second-ball-structure": "After the first aerial contest, nearby teammates recover the loose second ball and continue play.",
    "set-piece-zonal": "Defenders hold designated spaces around likely delivery zones rather than following only one opponent.",
    "short-corner": "The first corner pass goes to a nearby teammate before a later delivery toward the penalty area.",
    "switch-of-play": "A long pass transfers possession from one side of the pitch to an available teammate on the other.",
    "tactical-foul": "A recovering defender fouls the advancing ball carrier and stops the current progression.",
    "third-man-run": "After two teammates combine, a third off-ball runner enters space and receives the next pass.",
    "third-player-combinations": "Three teammates exchange passes and the final ball reaches the third participant in space.",
}
TEAM_EN = {"白队": "white team", "蓝队": "blue team", "红队": "red team", "绿队": "green team", "黄队": "yellow team", "黑队": "black team"}
QUESTION_HEADERS = (
    "question_id", "clip_id", "track", "track_goal", "benchmark_status",
    "target_team", "constraint", "video_scope", "question_en", "option_A",
    "option_B", "option_C", "option_D", "interval_options_for_T4",
    "provisional_answer", "provisional_evidence_option", "gold_interval",
    "provisional_tactic", "expected_verdict", "gold_evidence_zh",
    "mechanism_en", "distractor_logic", "adjudication_required", "output_schema",
)
CLIP_HEADERS = (
    "clip_id", "video_file", "uploaded_human_label", "uploaded_interval",
    "original_rationale", "polished_auditable_rationale", "Sol_high_result",
    "Sol_Pro_result", "key_conflict", "ontology_status", "readiness",
    "required_action", "target_team", "family", "phase",
    "recommended_question_scope",
)
CRITERIA_HEADERS = (
    "priority", "tactic_id", "中文名", "English name", "layer", "phase",
    "definition", "observable_features", "trigger_conditions",
    "confounders_and_differences", "video_participants", "spatial_constraints",
    "time_window", "event_sequence", "source_status", "source_references",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sheet_rows(ws) -> list[dict[str, str]]:
    headers = [str(cell.value or "") for cell in ws[1]]
    return [
        {header: str(ws.cell(row, col + 1).value or "") for col, header in enumerate(headers)}
        for row in range(2, ws.max_row + 1)
        if any(ws.cell(row, col + 1).value not in (None, "") for col in range(len(headers)))
    ]


def ontology_rows() -> dict[str, dict[str, str]]:
    with ONTOLOGY.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["id（英文标识）"].strip(): row
            for row in csv.DictReader(handle)
            if row["id（英文标识）"].strip() and not row["id（英文标识）"].startswith("英文小写")
        }


def normalize_span(clip_id: str, raw: str) -> tuple[str, str]:
    if clip_id in SPAN_OVERRIDES:
        return SPAN_OVERRIDES[clip_id]
    match = re.fullmatch(r"(\d{2}):(\d{2})\s*[-–]\s*(\d{2}):(\d{2})", raw.strip())
    if not match:
        raise ValueError(f"unhandled span {clip_id}: {raw!r}")
    start_minute, start_second, end_minute, end_second = map(int, match.groups())
    start, end = start_minute * 60 + start_second, end_minute * 60 + end_second
    if not 0 <= start < end <= 30:
        raise ValueError(f"out-of-range span {clip_id}: {raw!r}")
    return f"{start_minute:02d}:{start_second:02d}–{end_minute:02d}:{end_second:02d}", ""


def parse_seconds(span: str) -> tuple[int, int]:
    values = [int(value) for value in re.findall(r"\d+", span)]
    return values[0] * 60 + values[1], values[2] * 60 + values[3]


def fmt_span(start: int, end: int) -> str:
    return f"00:{start:02d}–00:{end:02d}"


def interval_set(correct: str) -> list[str]:
    start, end = parse_seconds(correct)
    candidates = []
    duration = end - start
    for offset in (-8, -5, 5, 8):
        shifted_start = min(max(0, start + offset), 30 - duration)
        candidates.append(fmt_span(shifted_start, shifted_start + duration))
    candidates.extend(fmt_span(value, min(30, value + 6)) for value in (0, 6, 12, 18, 24))
    unique = []
    for candidate in candidates:
        if candidate != correct and candidate not in unique and parse_seconds(candidate)[0] < parse_seconds(candidate)[1]:
            unique.append(candidate)
    if len(unique) < 3:
        raise ValueError(f"could not build interval distractors for {correct}")
    return unique[:3]


def team_names(rationale: str) -> tuple[str, str]:
    match = re.search(r"[白蓝红绿黄黑]队", rationale)
    zh = match.group() if match else "目标球队"
    return zh, TEAM_EN.get(zh, "target team")


def evidence_dimensions(text: str) -> list[str]:
    dimensions = []
    if re.search(r"球权|断下|夺回|丢失|获得球|控制球", text):
        dimensions.append("球权变化")
    dimensions.append("球队动作")
    if re.search(r"对方|防守|逼抢|门将|回撤|压迫", text):
        dimensions.append("对手响应")
    if re.search(r"空当|空间|身后|肋部|边路|禁区|区域", text):
        dimensions.append("空间变化")
    if re.search(r"后|随后|立刻|再|瞬间|第一|然后|迅速", text):
        dimensions.append("动作顺序")
    return dimensions


def polish_rationale(clip_id: str, label: str, original: str, status: str) -> str:
    text = original.strip().replace("来求球", "来球").replace("凡越位", "反越位")
    text = text.replace("再后场", "在后场").replace("套边月内切套上", "套边与内切套上")
    text = text.replace("抢夺第一二落点", "争抢第一、二落点").replace("争夺第一二落点", "争夺第一、二落点")
    text = text.replace("直接起大脚长传球。将球", "直接起大脚长传球，将球")
    if clip_id == "SNGS-008":
        text = text.replace("前场任意球", "前场角球")
    if clip_id == "SNGS-111":
        text = text.replace("放倒绿队球员组织进攻", "放倒绿队球员，使其推进被中止")
    if clip_id == "SNGS-121":
        text = text.replace("蹬踏该名白队球员将其放倒阻止进攻", "与该名白队球员发生身体接触并将其放倒，使进攻被中止")
    if clip_id == "SNGS-200":
        text = text.replace("主动放弃球权降低逼抢强度", "在无球阶段降低前场逼抢强度")
    if clip_id in {"SNGS-083", "SNGS-148"}:
        text = text.replace("吸引带走白队中后卫", "随后白队中后卫跟随移动")
        text = text.replace("吸引两名防守球员，为队友打开射门空间", "随后两名防守球员跟随移动，队友获得射门空间")
    if clip_id in {"SNGS-110", "SNGS-116"}:
        text = text.replace("干扰门将处理高空球的路线", "使门将周围空间变得拥挤")
        text = text.replace("限制干扰门将活动", "使门将周围的可用活动空间缩小")
    text = re.sub(r"[，,]?符合[^。]*特点。?$", "", text).rstrip("，,。")
    if status == "HOLD":
        return f"{text}。画面可支持上述部分动作与先后关系，但是否足以确认“{label}”仍需人工仲裁。"
    return f"{text}。上述可观察动作及其先后关系为“{label}”判断提供了依据。"


def option_map(correct: str, distractors: list[str], correct_letter: str, special: str | None, special_letter: str) -> dict[str, str]:
    options: dict[str, str] = {correct_letter: correct}
    remaining = list(distractors)
    if special:
        if special_letter == correct_letter:
            special_letter = LETTERS[(LETTERS.index(special_letter) + 1) % 4]
        options[special_letter] = special
    for letter in LETTERS:
        if letter not in options:
            options[letter] = remaining.pop(0)
    return options


def base_question(clip_ref: str, track: str, status: str, team_en: str, span: str, tactic_id: str, rationale: str) -> dict[str, Any]:
    return {
        "question_id": f"{clip_ref}_{track}", "clip_id": clip_ref, "track": track,
        "benchmark_status": status, "target_team": team_en,
        "video_scope": "full 30 s clip; inspect the supplied evidence interval and about 2 s of context on each side",
        "interval_options_for_T4": "", "provisional_answer": "PENDING",
        "provisional_evidence_option": "PENDING", "gold_interval": span,
        "provisional_tactic": tactic_id, "expected_verdict": "true" if status == "REVIEW" else "insufficient",
        "gold_evidence_zh": rationale,
        "adjudication_required": "Three human reviewers must independently verify the target team, observable sequence, minimum sufficient interval, and unique answer.",
    }


def build_questions(records: list[dict[str, Any]], metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    questions = []
    review_counts: Counter[str] = Counter()
    phase_index: dict[str, list[str]] = defaultdict(list)
    for tactic_id in USED_TACTICS:
        phase_index[metadata[tactic_id]["阶段 phase"]].append(tactic_id)

    for index, record in enumerate(records):
        tactic_id, status = record["tactic_id"], record["status"]
        name = metadata[tactic_id]["英文名"].strip() or tactic_id
        team_en, span = record["team_en"], record["span"]
        observation = OBSERVATION_EN[tactic_id]
        mechanism = observation.replace(".", "")
        phase = metadata[tactic_id]["阶段 phase"]
        alternatives = [item for item in phase_index[phase] if item != tactic_id]
        alternatives += [item for item in USED_TACTICS if item != tactic_id and item not in alternatives]
        alternative_names = [metadata[item]["英文名"].strip() or item for item in alternatives[:2]]

        for track_offset, track in enumerate(TRACKS):
            if status == "REVIEW":
                correct_letter = LETTERS[review_counts[track] % 4]
                review_counts[track] += 1
            else:
                correct_letter = LETTERS[(index + track_offset) % 4]
            special_letter = LETTERS[(index + track_offset + 1) % 4]
            row = base_question(record["clip_ref"], track, status, team_en, span, tactic_id, record["polished"])
            if track == "T1_TEMPORAL":
                options = option_map(span, interval_set(span), correct_letter, None, special_letter)
                row.update({
                    "track_goal": "temporal localization",
                    "constraint": f"candidate {name} claim; source interval retained separately",
                    "question_en": f"Which interval contains the minimum sufficient visual evidence for the {team_en}'s candidate {name} sequence?",
                    "provisional_evidence_option": correct_letter if status == "REVIEW" else "PENDING",
                    "mechanism_en": mechanism,
                    "distractor_logic": "Distractor intervals shift the onset or truncate the reported action sequence.",
                    "output_schema": json.dumps({"evidence_option": correct_letter if status == "REVIEW" else "PENDING"}, ensure_ascii=False),
                })
            elif track == "T2_OBSERVATION":
                options = option_map(
                    observation,
                    ["The ball remains in a settled phase with no comparable forward or defensive sequence.", "Only the final outcome is visible; the preceding action sequence is absent."],
                    correct_letter, "No clear observable sequence can be distinguished in the supplied interval.", special_letter,
                )
                row.update({
                    "track_goal": "observable fact selection",
                    "constraint": "describe visible events without inferring player intent",
                    "question_en": f"Which statement best describes the observable sequence involving the {team_en}?",
                    "mechanism_en": mechanism,
                    "distractor_logic": "Distractors remove the trigger, the ordered action chain, or observability.",
                    "output_schema": json.dumps({"answer": correct_letter if status == "REVIEW" else "PENDING", "observable_facts": []}, ensure_ascii=False),
                })
            elif track == "T3_DIAGNOSIS":
                correct = f"The visible action sequence and temporal order satisfy the provisional observable criteria for {name}."
                options = option_map(
                    correct,
                    [f"The sequence is better described as {alternative_names[0]} because the required relation is absent.", "The clip shows an outcome, but not the reported trigger and ordered action chain."],
                    correct_letter, "The evidence is insufficient to distinguish the candidate from its confounders.", special_letter,
                )
                state = "met" if status == "REVIEW" else "not_observable"
                row.update({
                    "track_goal": "criterion diagnosis",
                    "constraint": f"diagnose the provisional {name} claim against observable criteria",
                    "question_en": f"Which diagnostic statement is best supported for the candidate {name} label?",
                    "mechanism_en": mechanism,
                    "distractor_logic": "Options test the required visible sequence, a nearby confounder, and insufficient evidence.",
                    "output_schema": json.dumps({"answer": correct_letter if status == "REVIEW" else "PENDING", "criterion_states": {"observable_action_sequence": state, "temporal_order": state}}, ensure_ascii=False),
                })
            else:
                evidence_letter = LETTERS[(index + 2) % 4]
                intervals = option_map(span, interval_set(span), evidence_letter, None, "A")
                options = option_map(name, alternative_names, correct_letter, "No clear tactic from the requested family", special_letter)
                row.update({
                    "track_goal": "joint tactic + evidence + mechanism",
                    "constraint": "jointly select the candidate tactic, evidence interval, and observable mechanism",
                    "question_en": f"Which tactical pattern is primarily shown by the {team_en}, and which interval supports it?",
                    "interval_options_for_T4": "; ".join(f"I{position}={intervals[letter]}" for position, letter in enumerate(LETTERS, 1)),
                    "provisional_evidence_option": evidence_letter if status == "REVIEW" else "PENDING",
                    "mechanism_en": mechanism,
                    "distractor_logic": "Tactical alternatives are drawn from related phases; the special option position rotates.",
                    "output_schema": json.dumps({"answer": correct_letter if status == "REVIEW" else "PENDING", "evidence_option": evidence_letter if status == "REVIEW" else "PENDING", "mechanism": {"action_sequence": "...", "opponent_response": "...", "space_change": "..."}}, ensure_ascii=False),
                })
            row["provisional_answer"] = correct_letter if status == "REVIEW" else "PENDING"
            for letter, value in options.items():
                row[f"option_{letter}"] = value
            questions.append(row)
    return questions


def style_template(ws, columns: int) -> tuple[list[Any], float | None]:
    return [copy(ws.cell(2, column)._style) for column in range(1, columns + 1)], ws.row_dimensions[2].height


def replace_table(ws, headers: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    styles, height = style_template(ws, len(headers))
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row_number, row in enumerate(rows, 2):
        for column, header in enumerate(headers, 1):
            cell = ws.cell(row_number, column, row.get(header, ""))
            cell._style = copy(styles[column - 1])
        ws.row_dimensions[row_number].height = height
    ws.auto_filter.ref = f"A1:{ws.cell(1, len(headers)).column_letter}{ws.max_row}"


def build_criteria(ws, metadata: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    existing = {row["tactic_id"]: row for row in sheet_rows(ws)}
    cards = []
    for tactic_id in (*USED_TACTICS, *LEGACY_CARDS):
        if tactic_id in existing:
            cards.append(existing[tactic_id])
            continue
        source = metadata[tactic_id]
        confounders = "；".join(
            part for part in (
                f"{source.get('易混淆① 对象', '')}: {source.get('易混淆① 区分点', '')}".strip(": "),
                f"{source.get('易混淆② 对象', '')}: {source.get('易混淆② 区分点', '')}".strip(": "),
                f"{source.get('易混淆③ 对象', '')}: {source.get('易混淆③ 区分点', '')}".strip(": "),
            ) if part
        )
        cards.append({
            "priority": source.get("优先级", ""), "tactic_id": tactic_id,
            "中文名": source.get("中文名", ""), "English name": source.get("英文名", ""),
            "layer": source.get("层级 layer", ""), "phase": source.get("阶段 phase", ""),
            "definition": source.get("一句话定义", ""),
            "observable_features": source.get("可观察特征", ""),
            "trigger_conditions": source.get("触发条件", ""),
            "confounders_and_differences": confounders,
            "video_participants": source.get("视频侧-参与者", ""),
            "spatial_constraints": source.get("视频侧-空间约束", ""),
            "time_window": source.get("视频侧-时间窗", ""),
            "event_sequence": source.get("视频侧-事件序列", ""),
            "source_status": "draft/unreviewed",
            "source_references": source.get("参考来源", ""),
        })
    return cards


def append_readme_audit(ws) -> None:
    ws["A1"] = "GroundedTactics — GSR-2025 Provisional Question Bank"
    start = ws.max_row + 2
    audit = [
        ("GSR-2025 source workbook SHA-256", SOURCE_SHA256),
        ("9-clip template workbook SHA-256", TEMPLATE_SHA256),
        ("Included Human Labelling-GSR2025 records", "134"),
        ("Skipped incomplete/no-explicit-tactic records", "66"),
        ("Clip_Review rows", "134"),
        ("Question_Bank_3plus1 rows", "536 (134 clips × 4 tracks)"),
        ("Release status", "provisional_not_gold; Human is the sole provisional answer source; Sol Pro is advisory only"),
    ]
    source_row = max(2, ws.max_row)
    for offset, (key, value) in enumerate(audit):
        row = start + offset
        for column, item in enumerate((key, value), 1):
            cell = ws.cell(row, column, item)
            cell._style = copy(ws.cell(source_row, column)._style)


def build(force: bool) -> None:
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(TEMPLATE) != TEMPLATE_SHA256:
        raise ValueError("source workbook hash changed; refusing to build")
    for output in (HUMAN_OUT, QUESTION_OUT):
        if output.exists() and not force:
            raise FileExistsError(f"output exists: {output}; pass --force to replace it")

    label_map = load_label_map(ONTOLOGY, ALIASES)
    metadata = ontology_rows()
    if set(USED_TACTICS) - set(metadata):
        raise ValueError(f"missing ontology tactics: {sorted(set(USED_TACTICS) - set(metadata))}")

    source_wb = load_workbook(SOURCE)
    human_ws = source_wb["Human Labelling-GSR2025"]
    human_rows = sheet_rows(human_ws)
    sol_rows = sheet_rows(source_wb["5.6 Sol Pro-GSR2025"])
    sol_by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    sol_tactics: dict[str, set[str]] = defaultdict(set)
    for row in sol_rows:
        sol_by_clip[row["id"]].append(row)
        _, tactic_id = normalize_tactic(row["tactics"], label_map)
        if tactic_id:
            sol_tactics[row["id"]].add(tactic_id)

    targets = [
        row for row in human_rows
        if row["战术tactics"] not in ("", "无明确战术") and row["时间范围"] and row["判断依据"]
    ]
    if len(human_rows) != 200 or len(targets) != 134:
        raise ValueError(f"unexpected human source counts: total={len(human_rows)}, included={len(targets)}")

    records = []
    for row in targets:
        clip_id, label = row["id"], row["战术tactics"]
        kind, tactic_id = normalize_tactic(label, label_map)
        if kind != "positive" or not tactic_id:
            raise ValueError(f"unmapped human tactic {clip_id}: {label}")
        model_agrees = tactic_id in sol_tactics[clip_id]
        status = "HOLD" if not model_agrees or clip_id in VISUAL_HOLD_REASONS else "REVIEW"
        span, normalization_reason = normalize_span(clip_id, row["时间范围"])
        team_zh, team_en = team_names(row["判断依据"])
        polished = polish_rationale(clip_id, label, row["判断依据"], status)
        model_result = "；".join(
            f"{item['tactics']}，{item['时间范围'] or 'N/A'}" for item in sol_by_clip[clip_id]
        ) or "N/A"
        conflicts = []
        if not model_agrees:
            conflicts.append("Sol Pro does not include the same normalized tactic; this is an advisory disagreement only.")
        else:
            conflicts.append("Sol Pro includes the same normalized tactic; it remains a weak-label source, not a second annotator.")
        if clip_id in VISUAL_HOLD_REASONS:
            conflicts.append(VISUAL_HOLD_REASONS[clip_id])
        if normalization_reason:
            conflicts.append(normalization_reason)
        dimensions = evidence_dimensions(row["判断依据"])
        records.append({
            "clip_id": clip_id, "clip_ref": f"2025-{clip_id}", "tactic_id": tactic_id,
            "status": status, "span": span, "team_zh": team_zh, "team_en": team_en,
            "polished": polished, "model_result": model_result,
            "key_conflict": " ".join(conflicts), "normalization_reason": normalization_reason,
            "dimensions": dimensions, "source": row,
        })

    human_ws.cell(1, 5, "AI润色 判断依据")._style = copy(human_ws.cell(1, 4)._style)
    human_ws.column_dimensions["E"].width = human_ws.column_dimensions["D"].width
    polished_by_id = {record["clip_id"]: record["polished"] for record in records}
    for row_number in range(2, human_ws.max_row + 1):
        clip_id = str(human_ws.cell(row_number, 1).value or "")
        cell = human_ws.cell(row_number, 5)
        cell._style = copy(human_ws.cell(row_number, 4)._style)
        cell.value = polished_by_id.get(clip_id)
    source_wb.save(HUMAN_OUT)

    template_wb = load_workbook(TEMPLATE)
    questions = build_questions(records, metadata)
    clip_rows = []
    for record in records:
        source = record["source"]
        meta = metadata[record["tactic_id"]]
        action = "Resolve model disagreement and visual criterion ambiguity." if record["status"] == "HOLD" else "Obtain three independent human confirmations before any release promotion."
        if record["normalization_reason"]:
            action += f" {record['normalization_reason']}"
        clip_rows.append({
            "clip_id": record["clip_ref"],
            "video_file": f"data/GSR-2025/videos/{record['clip_id']}.mp4",
            "uploaded_human_label": source["战术tactics"],
            "uploaded_interval": source["时间范围"],
            "original_rationale": source["判断依据"],
            "polished_auditable_rationale": record["polished"],
            "Sol_high_result": "N/A", "Sol_Pro_result": record["model_result"],
            "key_conflict": record["key_conflict"],
            "ontology_status": f"{record['tactic_id']}; draft/unreviewed",
            "readiness": record["status"], "required_action": action,
            "target_team": record["team_zh"], "family": meta["层级 layer"],
            "phase": meta["阶段 phase"],
            "recommended_question_scope": f"target={record['team_zh']}; evidence dimensions={','.join(record['dimensions'])}",
        })
    replace_table(template_wb["Question_Bank_3plus1"], QUESTION_HEADERS, questions)
    replace_table(template_wb["Clip_Review"], CLIP_HEADERS, clip_rows)
    cards = build_criteria(template_wb["Tactic_Criteria"], metadata)
    replace_table(template_wb["Tactic_Criteria"], CRITERIA_HEADERS, cards)
    append_readme_audit(template_wb["README"])
    template_wb.save(QUESTION_OUT)

    if sha256(SOURCE) != SOURCE_SHA256 or sha256(TEMPLATE) != TEMPLATE_SHA256:
        raise ValueError("source workbook changed during build")
    print(json.dumps({
        "human_output": str(HUMAN_OUT.relative_to(ROOT)),
        "question_output": str(QUESTION_OUT.relative_to(ROOT)),
        "human_rows": len(human_rows), "included": len(records), "skipped": len(human_rows) - len(records),
        "review": sum(record["status"] == "REVIEW" for record in records),
        "hold": sum(record["status"] == "HOLD" for record in records),
        "questions": len(questions), "tactic_cards": len(cards),
    }, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the two generated derivative workbooks.")
    args = parser.parse_args()
    try:
        build(args.force)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
