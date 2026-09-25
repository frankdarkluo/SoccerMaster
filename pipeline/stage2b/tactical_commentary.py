"""Two-stage evidence-grounded tactical commentary generation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pipeline.topology.analysis import analyze_clip
from pipeline.utils.video import get_video_info
from pipeline.video_models import generate_json, model_name

PROMPT_VERSION = "tactical-commentary-two-stage-v2-topology"
ANALYSIS_FIELDS = (
    "fact_index", "evidence_spans", "state_before", "position_or_action",
    "observed_change", "space_origin", "space_evidence", "opponent_constraint",
    "space_change", "functional_effect", "optional_dilemma", "terminal_sequence",
    "mechanism_chain", "mechanism_result", "play_result", "confidence",
    "confidence_reasons", "evidence_gaps", "topology_fact_ids",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _locked_fact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "fact_index": 0,
        "tactic_id": row["tactic_id"],
        "tactic_zh": row["tactic_zh"],
        "window": row["window"],
        "evidence_text": row["evidence_text"],
    }


def load_fact(review_path: Path, clip_uid: str) -> dict[str, Any]:
    matches = [
        row for row in _load_jsonl(review_path)
        if row.get("clip_uid") == clip_uid and row.get("verdict") == "correct"
    ]
    if len(matches) != 1:
        raise ValueError(f"{clip_uid} must have exactly one correct locked fact")
    return matches[0]


def load_exemplar(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "deep-commentary-exemplar-v2":
        raise ValueError("unsupported commentary exemplar")
    return value


def analysis_schema() -> dict[str, Any]:
    nullable = {"type": ["string", "null"]}
    properties: dict[str, Any] = {
        "fact_index": {"type": "integer", "enum": [0]},
        "evidence_spans": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}, "observation": {"type": "string"}},
            "required": ["start_s", "end_s", "observation"],
            "additionalProperties": False,
        }},
        "state_before": nullable, "position_or_action": nullable,
        "observed_change": nullable, "space_origin": {"type": "string", "enum": ["created", "existing", "unclear"]},
        "space_evidence": {"type": "string"}, "opponent_constraint": nullable,
        "space_change": nullable, "functional_effect": nullable,
        "optional_dilemma": nullable, "terminal_sequence": {"type": "array", "items": {"type": "string"}},
        "mechanism_chain": {"type": "string"},
        "mechanism_result": {"type": "string", "enum": ["effective", "partial", "ineffective", "unclear"]},
        "play_result": {"type": "string", "enum": ["success", "failure", "mixed", "unclear"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100, "multipleOf": 10},
        "confidence_reasons": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        "topology_fact_ids": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
    }
    return {"type": "object", "properties": properties, "required": list(ANALYSIS_FIELDS), "additionalProperties": False}


WRITER_SCHEMA = {
    "type": "object",
    "properties": {"text_zh": {"type": "string"}},
    "required": ["text_zh"],
    "additionalProperties": False,
}

def topology_facts(topology: dict[str, Any], fact: dict[str, Any]) -> list[dict[str, Any]]:
    """Slice immutable visible-topology facts to the reviewed evidence window."""
    start_s = float(fact["window"]["start_s"])
    end_s = float(fact["window"]["end_s"])
    facts = []
    for window in topology.get("windows", []):
        if float(window["end_s"]) < start_s or float(window["start_s"]) > end_s:
            continue
        fact_id = f"topology:{window['window_id']}"
        facts.append({
            "fact_id": fact_id,
            "start_s": window["start_s"],
            "end_s": window["end_s"],
            "attacking_team": window["attacking_team"],
            "position_source": topology["position_source"],
            "lines": [
                {
                    "line_id": line["line_id"],
                    "team": line["team"],
                    "role": line["role"],
                    "status": line["status"],
                    "support_ratio": line["support_ratio"],
                    "member_track_ids": [
                        member["track_id"] for member in line["members"]
                    ],
                    "member_jerseys": [
                        member["jersey_number"] for member in line["members"]
                        if member["jersey_number"] is not None
                    ],
                    "evidence_gaps": line["evidence_gaps"],
                }
                for line in window["lines"]
            ],
            "zones": [
                {
                    "zone_id": zone["zone_id"],
                    "type": zone["type"],
                    "supporting_line_ids": zone["supporting_line_ids"],
                }
                for zone in window["zones"]
            ],
        })
    return facts


def validate_topology_citations(
    analysis: dict[str, Any],
    facts: list[dict[str, Any]],
) -> None:
    available = {fact["fact_id"] for fact in facts}
    cited = analysis.get("topology_fact_ids")
    if not isinstance(cited, list) or any(item not in available for item in cited):
        raise ValueError("analysis cites an unknown topology fact")


def build_analysis_prompt(
    fact: dict[str, Any],
    duration_s: float,
    exemplar: dict[str, Any],
    locked_topology: list[dict[str, Any]],
) -> str:
    locked = _locked_fact(fact)
    locked["window"] = {**locked["window"], "end_s": min(float(locked["window"]["end_s"]), duration_s)}
    target = {
        "clip_duration_s": duration_s,
        "locked_fact": locked,
        "locked_visible_topology": locked_topology,
        "topology_limit": "candidate lines are uncertain and cannot be promoted to visible facts",
    }
    return f"""你是中文足球战术文字直播分析员。完整原始比赛片段已作为视频输入。

任务：只解释锁定战术在当前回合中的可见机制，按“动作或站位 → 对手约束或空间变化 → 战术作用 → 本次结果”形成证据链。不得重新识别 Tag，不得添加副标签。

规则：
1. Tag、evidence_text 和 evidence window 是人工审核后锁定的事实，视频只能补充可见空间关系、对手反应和回合结果。
2. 不得编造球员身份、号码、完整阵型、教练意图或画外信息。
3. 依次记录动作前状态、关键动作、可见变化、空间来源、对手约束、战术作用和终结序列，再汇总 mechanism_chain。
4. evidence_spans 只写带秒数的直接可见事实；时间必须在 0 到 clip_duration_s 内。
5. 只有 space_origin=created 且 observed_change、space_evidence 明确时，才能声称牵制、拉开、打开或创造空间。
6. 证据不足时降低措辞确定性，不得补齐因果；终结未呈现时 play_result=unclear。
7. mechanism_result 与 play_result 分开判断，例如破线成功但射门被扑可为 effective + failure。
8. confidence 为 0–100 的 10 倍数；只返回匹配 schema 的 JSON。

人工方法示例（只学习方法，不复制球队、结果或措辞）：
{json.dumps(exemplar, ensure_ascii=False)}

目标片段：
{json.dumps(target, ensure_ascii=False)}
9. topology_fact_ids 只引用确实用于分析的 fact_id；candidate 只能作为信息缺口，不得写成已确认阵型。
"""


def build_writer_prompt(
    fact: dict[str, Any], analysis: dict[str, Any], locked_topology: list[dict[str, Any]],
) -> str:
    cited = set(analysis["topology_fact_ids"])
    source = {
        "locked_fact": _locked_fact(fact),
        "analysis": analysis,
        "cited_visible_topology": [
            item for item in locked_topology if item["fact_id"] in cited
        ],
    }
    return f"""你是中文足球战术文字直播编辑。你不看视频，只根据已经完成的视频观察链写完整文字。

只能使用 locked_fact 和 analysis 中已有事实；保留 mechanism_chain 的核心关系。space_origin 不是 created 时不得声称创造空间；终结描述必须服从 terminal_sequence。写 2–3 个完整中文句子、50–140 字，至少包含一个明确的“动作或站位 → 空间、约束或作用”关系。只返回 JSON：{{"text_zh":"..."}}。

待写材料：
{json.dumps(source, ensure_ascii=False)}
"""


def validate_text(value: Any) -> str:
    if not isinstance(value, str) or not 50 <= len(value.strip()) <= 140:
        raise ValueError("text_zh must contain 50..140 characters")
    text = value.strip()
    sentences = [part for part in re.split(r"[。！？!?]+", text) if part.strip()]
    if not 2 <= len(sentences) <= 3:
        raise ValueError("text_zh must contain 2..3 sentences")
    if not any(marker in text for marker in ("通过", "利用", "迫使", "拉开", "从而", "因此", "使得", "让", "由此", "形成", "创造")):
        raise ValueError("text_zh lacks a mechanism relation")
    return text


def run(
    review_path: Path,
    preprocessing_root: Path,
    output_root: Path,
    exemplar_path: Path,
    *,
    provider: str,
    clip_uids: Iterable[str],
    force: bool = False,
    repo_root: Path | None = None,
    topology_root: Path | None = None,
) -> list[dict[str, Any]]:
    exemplar = load_exemplar(exemplar_path)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    topology_root = (
        Path(topology_root) if topology_root
        else repo_root / "outputs/tactical_topology"
    )
    outcomes = []
    for clip_uid in clip_uids:
        clip = clip_uid.split(":", 1)[-1]
        target = output_root / provider / f"{clip}.json"
        if target.is_file() and not force:
            outcomes.append({"clip_uid": clip_uid, "status": "skipped"})
            continue
        fact, video = load_fact(review_path, clip_uid), preprocessing_root / clip / "clip.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        topology_path = topology_root / f"{clip}.json"
        topology = analyze_clip(
            repo_root,
            clip_uid,
            topology_path,
            fps=25.0,
            force=False,
            preprocessing_root=preprocessing_root,
        )
        locked_topology = topology_facts(topology, fact)
        analysis_prompt = build_analysis_prompt(
            fact, float(get_video_info(video)["duration_s"]), exemplar, locked_topology,
        )
        analysis, usage1, attempts1 = generate_json(
            provider, analysis_prompt, analysis_schema(), video_path=video,
        )
        validate_topology_citations(analysis, locked_topology)
        writer_prompt = build_writer_prompt(fact, analysis, locked_topology)
        written, usage2, attempts2 = generate_json(provider, writer_prompt, WRITER_SCHEMA)
        text = validate_text(written["text_zh"])
        record = {
            "schema_version": "tactical-commentary-two-stage-v2-topology",
            "prompt_version": PROMPT_VERSION,
            "clip_uid": clip_uid,
            "provider": provider,
            "model": model_name(provider),
            "status": "success",
            "locked_fact": _locked_fact(fact),
            "topology": {
                "position_source": topology["position_source"],
                "path": str(topology_path),
                "sha256": hashlib.sha256(topology_path.read_bytes()).hexdigest(),
                "locked_facts": locked_topology,
            },
            "analysis": analysis,
            "commentary_zh": text,
            "analysis_prompt_sha256": hashlib.sha256(analysis_prompt.encode()).hexdigest(),
            "writer_prompt_sha256": hashlib.sha256(writer_prompt.encode()).hexdigest(),
            "attempts": [attempts1, attempts2],
            "api_usage": [*usage1, *usage2],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outcomes.append({"clip_uid": clip_uid, "status": "success"})
    return outcomes
