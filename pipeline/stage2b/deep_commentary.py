"""Text-first evidence-grounded tactical commentary experiment."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from pipeline.atomic import atomic_copy, atomic_write_json
from pipeline.config import REPO_ROOT
from pipeline.stage2b.deep_commentary import (
    _locked_fact,
    _matching_segment_indexes,
    load_correct_facts,
)
from pipeline.stage2b.generate import DEFAULT_ARK_MODEL, ENERGIES, ark_chat
from pipeline.stage2b.video import video_duration_s


PROMPT_VERSION = "fifa-causal-text-first-v2"
TEMPERATURE = 0.2
REQUEST_TIMEOUT_S = 1800.0
DEFAULT_REVIEW_PATH = REPO_ROOT / "benchmark/tactical_prototypes/recognition_review.jsonl"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "outputs/human_review_commentary_demo"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/deep_commentary_text_first_v2"
DEFAULT_EXEMPLAR_PATH = (
    REPO_ROOT / "benchmark/tactical_prototypes/deep_commentary_exemplar_v2.json"
)
DEFAULT_CLIP_UIDS = (
    "event_clips:0006",
    "event_clips:0007",
    "soccernetgs:SNGS-116",
    "soccernetgs:SNGS-117",
    "soccernetgs:SNGS-148",
)
MECHANISM_RESULTS = {"effective", "partial", "ineffective", "unclear"}
PLAY_RESULTS = {"success", "failure", "mixed", "unclear"}
SPACE_ORIGINS = {"created", "existing", "unclear"}
WRITING_AB_EXPERIMENT_ID = "fifa-causal-text-first-writing-ab"
WRITING_AB_FILENAME = "deep_analysis_writing_ab.json"
MECHANISM_MARKERS = (
    "通过", "利用", "迫使", "牵制", "拉开", "打开", "压缩", "从而",
    "因此", "使得", "让", "由此", "导致", "形成", "创造",
)
FAILURE_MARKERS = ("失败", "未能", "没能", "越位", "被扑", "丢失", "被断", "偏出")
_RESULT_FIELDS = {
    "fact_index",
    "evidence_spans",
    "state_before",
    "position_or_action",
    "observed_change",
    "space_origin",
    "space_evidence",
    "opponent_constraint",
    "space_change",
    "functional_effect",
    "optional_dilemma",
    "terminal_sequence",
    "mechanism_chain",
    "mechanism_result",
    "play_result",
    "confidence",
    "confidence_reasons",
    "evidence_gaps",
    "text_zh",
    "energy",
}


def load_exemplar(path: Path = DEFAULT_EXEMPLAR_PATH) -> dict:
    exemplar = json.loads(Path(path).read_text(encoding="utf-8"))
    if exemplar.get("schema_version") != "deep-commentary-exemplar-v2":
        raise ValueError("unsupported text-first exemplar schema")
    if not isinstance(exemplar.get("locked_fact"), dict) or not isinstance(
        exemplar.get("response"), dict
    ):
        raise ValueError("exemplar requires locked_fact and response objects")
    return exemplar


def build_prompt(fact: dict, *, duration_s: float, exemplar: dict) -> str:
    locked_fact = _locked_fact(fact)
    locked_fact["window"] = {
        **locked_fact["window"],
        "end_s": min(float(locked_fact["window"]["end_s"]), duration_s),
    }
    target = {
        "clip_duration_s": duration_s,
        "locked_fact": locked_fact,
    }
    return f"""你是中文足球战术文字直播分析员。完整原始比赛片段已作为视频输入。

任务：只解释锁定战术在当前回合中的可见机制，按“动作或站位 → 对手约束或空间变化 → 战术作用 → 本次结果”写成完整文字直播稿。不得重新识别 Tag，不得添加副标签。

规则：
1. Tag、evidence_text 和 evidence window 是人工审核后锁定的事实。视频用于观察全片中的空间关系、对手反应和回合结果，不能改写锁定事实。
2. 只分析唯一锁定战术。不得编造球员身份、号码、完整阵型、赛前安排、教练意图或画外信息。“为什么”仅表示当前可见动作或站位产生的功能。
3. 在写 text_zh 前，按顺序完成内部观察：
   a. state_before：关键动作前的局部站位和空间；
   b. position_or_action：触发变化的可见动作或站位；
   c. observed_change：动作后防守者或空间实际发生的变化；
   d. space_origin、space_evidence：判断空间由当前动作创造，还是已经存在并被利用；
   e. opponent_constraint、space_change、functional_effect：说明实际防守约束、空间和受益者如何利用；
   f. optional_dilemma：只有画面明确呈现即时选择困境时才填写，否则为 null；
   g. terminal_sequence：按顺序核对射门者、射门区域、射门次数、门将处理，以及射门后的补射、庆祝或回收动作；
   h. mechanism_chain：汇总一条最重要、可回指画面的局部因果链。
4. evidence_spans 只记录带秒数的直接可见事实；推断写入后续字段，无法确认时填 null。时间必须在 0.0 到 clip_duration_s 之间，真实视频时长优先于人工窗口取整值。
5. space_origin 规则：created 表示画面明确显示附近防守者移动并使某处或某人获得更多空间；existing 表示空间在关键动作前已经存在，当前动作只是传入、跑入或继续利用；unclear 表示镜头、遮挡或片段长度不足以判断来源。
6. 只有 created 且 observed_change 与 space_evidence 具体时，text_zh 才能使用“牵制、拉开、打开、创造空间”等确定表达；其余情况使用“进入、利用、获得空间”。
7. 以实际发生的防守反应为主。可以填写明显的选择困境，但不得依赖镜头外球员、完整阵型或球员心理。简单回合允许得到简单但完整的机制，不得强行添加小配合。
8. 证据链不完整时只讲可见的局部机制，不得用确定语气补齐因果。终结信息不清楚时只降低结果措辞精度，不得删除前面证据充分的机制。
9. terminal_sequence 可以在片段未呈现终结时为空，此时 play_result 必须为 unclear；其他 play_result 必须有非空终结序列。
10. mechanism_result 判断该机制是否在画面中产生预期作用；play_result 判断本次回合最终成败。破线成功但射门被扑可为 effective + failure；跑位制造空间但越位也必须把机制与失败分开。
11. text_zh 必须为 2–3 个完整中文句子，硬范围 50–140 字，目标 60–120 字。第一句写可见动作或站位；随后解释约束、空间变化及作用；最后交代本次结果和必要的不确定性。
12. text_zh 至少包含一个明确机制连接词，例如“通过、利用、迫使、牵制、拉开、打开、压缩、从而、因此、使得、让、由此、导致、形成、创造”。不得只复述动作和终结结果。
13. confidence 取 0–100 且只能是 10 的倍数；confidence_reasons 非空。它只供审计，不控制文字深度。
14. 只返回一个 JSON 对象，不要 Markdown、解释、短音频稿或额外字段。

输出字段必须恰好为：
fact_index, evidence_spans, state_before, position_or_action, observed_change,
space_origin, space_evidence, opponent_constraint, space_change,
functional_effect, optional_dilemma, terminal_sequence, mechanism_chain,
mechanism_result, play_result, confidence, confidence_reasons, evidence_gaps,
text_zh, energy。
evidence_spans 每项必须恰好包含 start_s、end_s、observation。
space_origin 只能是 created、existing、unclear。
mechanism_result 只能是 effective、partial、ineffective、unclear。
play_result 只能是 success、failure、mixed、unclear。
energy 只能是 calm、engaged、excited、explosive。

以下示例由项目人员依据 SNGS-101 完整视频和锁定证据人工编写。只学习分析方法，不复制球队、结果或措辞：
{json.dumps(exemplar, ensure_ascii=False)}

现在分析目标片段：
{json.dumps(target, ensure_ascii=False)}
"""


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _text_list(value: object, field: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field} must be {'a non-empty' if nonempty else 'an'} array")
    return [_nonempty_text(item, f"{field} item") for item in value]


def _nullable_text(value: object, field: str) -> str | None:
    return None if value is None else _nonempty_text(value, field)


def _has_mechanism_relation(text: str) -> bool:
    return any(marker in text for marker in MECHANISM_MARKERS)


def _validate_written_text(text: object) -> str:
    text = _nonempty_text(text, "text_zh")
    if not 50 <= len(text) <= 140:
        raise ValueError("text_zh must contain 50..140 characters")
    if text[-1] not in "。！？!?":
        raise ValueError("text_zh must end with sentence punctuation")
    sentences = [part.strip() for part in re.split(r"[。！？!?]+", text) if part.strip()]
    if not 2 <= len(sentences) <= 3:
        raise ValueError("text_zh must contain 2..3 sentences")
    if not _has_mechanism_relation(text):
        raise ValueError("text_zh lacks an explicit tactical mechanism relation")
    return text


def parse_response(raw: str, *, duration_s: float) -> dict:
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid text-first JSON: {exc}") from exc
    if not isinstance(result, dict) or set(result) != _RESULT_FIELDS:
        raise ValueError("text-first response fields do not match the schema")
    if result["fact_index"] != 0 or isinstance(result["fact_index"], bool):
        raise ValueError("fact_index must be integer 0")

    spans = result["evidence_spans"]
    if not isinstance(spans, list) or not spans:
        raise ValueError("evidence_spans must be a non-empty array")
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != {
            "start_s", "end_s", "observation"
        }:
            raise ValueError(f"evidence_spans[{index}] has invalid fields")
        start, end = span["start_s"], span["end_s"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) < float(start)
            or float(end) > duration_s
        ):
            raise ValueError(f"evidence_spans[{index}] must be within clip bounds")
        span["observation"] = _nonempty_text(
            span["observation"], f"evidence_spans[{index}].observation"
        )

    for field in (
        "state_before", "position_or_action", "observed_change",
        "opponent_constraint", "space_change", "functional_effect",
        "optional_dilemma",
    ):
        result[field] = _nullable_text(result[field], field)
    if not any(
        result[field] is not None
        for field in ("opponent_constraint", "space_change", "functional_effect")
    ):
        raise ValueError("response must contain at least one supported tactical mechanism")
    if result["space_origin"] not in SPACE_ORIGINS:
        raise ValueError("space_origin is invalid")
    result["space_evidence"] = _nonempty_text(
        result["space_evidence"], "space_evidence"
    )
    if result["space_origin"] == "created" and result["observed_change"] is None:
        raise ValueError("created space requires observed_change")
    result["terminal_sequence"] = _text_list(
        result["terminal_sequence"], "terminal_sequence", nonempty=False
    )
    result["mechanism_chain"] = _nonempty_text(
        result["mechanism_chain"], "mechanism_chain"
    )
    if result["mechanism_result"] not in MECHANISM_RESULTS:
        raise ValueError("mechanism_result is invalid")
    if result["play_result"] not in PLAY_RESULTS:
        raise ValueError("play_result is invalid")
    if result["play_result"] != "unclear" and not result["terminal_sequence"]:
        raise ValueError("known play_result requires terminal_sequence")

    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int):
        raise ValueError("confidence must be an integer")
    if not 0 <= confidence <= 100 or confidence % 10:
        raise ValueError("confidence must be 0..100 and a multiple of 10")
    result["confidence_reasons"] = _text_list(
        result["confidence_reasons"], "confidence_reasons", nonempty=True
    )
    result["evidence_gaps"] = _text_list(
        result["evidence_gaps"], "evidence_gaps", nonempty=False
    )
    result["text_zh"] = _validate_written_text(result["text_zh"])
    if result["energy"] not in ENERGIES:
        raise ValueError("energy is invalid")
    return result


def build_writer_prompt(fact: dict, result: dict) -> str:
    analysis = {key: value for key, value in result.items() if key != "text_zh"}
    source = {"locked_fact": _locked_fact(fact), "analysis": analysis}
    return f"""你是中文足球战术文字直播编辑。你不看视频，只根据已经完成的视频观察链写完整文字。

规则：
1. 只能使用 locked_fact 和 analysis 中已有的事实与机制，不得新增 Tag、球员身份、号码、完整阵型、教练安排、主观心理或画外信息。
2. 必须保留 mechanism_chain 的核心关系，并优先解释实际发生的防守约束和空间利用。
3. space_origin 不是 created 时，不得声称动作牵制、拉开、打开或创造了空间，只能写进入、利用或获得空间。
4. 终结描述必须服从 terminal_sequence；序列为空时不得补写射门、扑救或进球。
5. 写 2–3 个完整中文句子，硬范围 50–140 字，目标 60–120 字。至少包含一个明确的“动作或站位 → 空间、约束或作用”关系。
6. 简单回合保持简单，不强行增加配合、阵型或反事实解释。只返回 JSON：{{"text_zh":"..."}}。

待写材料：
{json.dumps(source, ensure_ascii=False)}
"""


def parse_writer_response(raw: str) -> str:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid writer JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"text_zh"}:
        raise ValueError("writer response fields do not match the schema")
    return _validate_written_text(value["text_zh"])


def patch_commentary(commentary: list[dict], fact: dict, result: dict) -> list[dict]:
    if not isinstance(commentary, list):
        raise ValueError("baseline commentary must be a JSON array")
    matches = _matching_segment_indexes(commentary, fact)
    if len(matches) != 1:
        raise ValueError("locked fact must match exactly one baseline tactical segment")
    patched = copy.deepcopy(commentary)
    patched[matches[0]].update(
        text_zh=result["text_zh"],
        fallback_text_zh=result["text_zh"],
        energy=result["energy"],
    )
    return patched


def _failure(
    path: Path,
    *,
    clip_uid: str,
    stage: str,
    error: Exception,
) -> dict:
    record = {
        "schema_version": "deep-tactical-commentary-text-first-v2",
        "prompt_version": PROMPT_VERSION,
        "clip_uid": clip_uid,
        "written_status": "failed",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
    }
    atomic_write_json(path, record)
    return record


def run_written_experiment(
    *,
    review_path: Path = DEFAULT_REVIEW_PATH,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    clip_uids: Sequence[str] = DEFAULT_CLIP_UIDS,
    exemplar_path: Path = DEFAULT_EXEMPLAR_PATH,
    force: bool = False,
    call: Callable = ark_chat,
) -> list[dict]:
    """Run every multimodal call, recording per-clip failure without fallback."""
    review_path, source_root = Path(review_path), Path(source_root)
    output_root, exemplar_path = Path(output_root), Path(exemplar_path)
    facts = load_correct_facts(review_path, clip_uids)
    exemplar = load_exemplar(exemplar_path)
    outcomes = []

    for clip_uid in clip_uids:
        clip_id = clip_uid.split(":", 1)[-1]
        source = source_root / clip_id
        target = output_root / clip_id
        analysis_path = target / "comments" / "deep_analysis.json"
        if source.resolve() == target.resolve():
            raise ValueError("experiment output must differ from baseline source")
        if analysis_path.is_file() and not force:
            record = json.loads(analysis_path.read_text(encoding="utf-8"))
            outcomes.append({
                "clip_uid": clip_uid,
                "status": record.get("written_status", "failed"),
                "output": target,
            })
            continue

        clip = source / "clip.mp4"
        baseline_path = source / "comments" / "commentary.json"
        try:
            if not clip.is_file() or not baseline_path.is_file():
                raise FileNotFoundError(
                    f"missing baseline clip or commentary for {clip_uid}"
                )
            fact = facts[clip_uid]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            matches = _matching_segment_indexes(baseline, fact)
            if len(matches) != 1:
                raise ValueError(
                    f"{clip_uid} must match exactly one baseline tactical segment"
                )
            duration_s = video_duration_s(clip)
            prompt = build_prompt(fact, duration_s=duration_s, exemplar=exemplar)
        except Exception as exc:
            record = _failure(
                analysis_path, clip_uid=clip_uid, stage="input_validation", error=exc
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        try:
            raw = call(
                prompt,
                video_path=clip,
                temperature=TEMPERATURE,
                timeout_s=REQUEST_TIMEOUT_S,
                max_retries=0,
            )
        except Exception as exc:
            _failure(
                analysis_path, clip_uid=clip_uid,
                stage="multimodal_request", error=exc,
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        try:
            result = parse_response(raw, duration_s=duration_s)
            patched = patch_commentary(baseline, fact, result)
        except Exception as exc:
            _failure(
                analysis_path, clip_uid=clip_uid,
                stage="response_validation", error=exc,
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        atomic_copy(clip, target / "clip.mp4")
        atomic_write_json(target / "comments" / "commentary.json", patched)
        atomic_write_json(analysis_path, {
            "schema_version": "deep-tactical-commentary-text-first-v2",
            "prompt_version": PROMPT_VERSION,
            "clip_uid": clip_uid,
            "model": os.environ.get("ARK_RESPONSES_MODEL", DEFAULT_ARK_MODEL),
            "temperature": TEMPERATURE,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "max_retries": 0,
            "written_status": "success",
            "locked_fact": _locked_fact(fact),
            "response": result,
            "audio_adaptation": {"status": "pending"},
        })
        outcomes.append({"clip_uid": clip_uid, "status": "success", "output": target})
    return outcomes


def _error_summary(error: Exception) -> dict:
    return {
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
    }


def _write_ab_failure(
    path: Path, *, clip_uid: str, stage: str, error: Exception
) -> None:
    atomic_write_json(path, {
        "schema_version": "deep-tactical-commentary-writing-ab",
        "experiment_id": WRITING_AB_EXPERIMENT_ID,
        "prompt_version": PROMPT_VERSION,
        "clip_uid": clip_uid,
        "pair_status": "failed",
        "failure_stage": stage,
        **_error_summary(error),
    })


def run_writing_ab_experiment(
    *,
    review_path: Path = DEFAULT_REVIEW_PATH,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    clip_uids: Sequence[str] = DEFAULT_CLIP_UIDS,
    exemplar_path: Path = DEFAULT_EXEMPLAR_PATH,
    force: bool = False,
    call: Callable = ark_chat,
) -> list[dict]:
    """Compare direct writing with text-only writing from the same video analysis."""
    review_path, source_root = Path(review_path), Path(source_root)
    output_root, exemplar_path = Path(output_root), Path(exemplar_path)
    facts = load_correct_facts(review_path, clip_uids)
    exemplar = load_exemplar(exemplar_path)
    outcomes = []

    for clip_uid in clip_uids:
        clip_id = clip_uid.split(":", 1)[-1]
        source = source_root / clip_id
        target = output_root / clip_id
        analysis_path = target / "comments" / WRITING_AB_FILENAME
        if source.resolve() == target.resolve():
            raise ValueError("experiment output must differ from baseline source")
        if analysis_path.is_file() and not force:
            record = json.loads(analysis_path.read_text(encoding="utf-8"))
            outcomes.append({
                "clip_uid": clip_uid,
                "status": record.get("pair_status", "failed"),
                "output": target,
            })
            continue

        clip = source / "clip.mp4"
        baseline_path = source / "comments" / "commentary.json"
        try:
            if not clip.is_file() or not baseline_path.is_file():
                raise FileNotFoundError(
                    f"missing baseline clip or commentary for {clip_uid}"
                )
            fact = facts[clip_uid]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if len(_matching_segment_indexes(baseline, fact)) != 1:
                raise ValueError(
                    f"{clip_uid} must match exactly one baseline tactical segment"
                )
            duration_s = video_duration_s(clip)
            prompt = build_prompt(fact, duration_s=duration_s, exemplar=exemplar)
        except Exception as exc:
            _write_ab_failure(
                analysis_path, clip_uid=clip_uid,
                stage="input_validation", error=exc,
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        try:
            raw = call(
                prompt,
                video_path=clip,
                temperature=TEMPERATURE,
                timeout_s=REQUEST_TIMEOUT_S,
                max_retries=0,
            )
        except Exception as exc:
            _write_ab_failure(
                analysis_path, clip_uid=clip_uid,
                stage="multimodal_request", error=exc,
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        try:
            result = parse_response(raw, duration_s=duration_s)
        except Exception as exc:
            _write_ab_failure(
                analysis_path, clip_uid=clip_uid,
                stage="multimodal_validation", error=exc,
            )
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        record = {
            "schema_version": "deep-tactical-commentary-writing-ab",
            "experiment_id": WRITING_AB_EXPERIMENT_ID,
            "prompt_version": PROMPT_VERSION,
            "clip_uid": clip_uid,
            "model": os.environ.get("ARK_RESPONSES_MODEL", DEFAULT_ARK_MODEL),
            "temperature": TEMPERATURE,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "max_retries": 0,
            "clip_duration_s": duration_s,
            "locked_fact": _locked_fact(fact),
            "multimodal_response": result,
            "single_call": {"status": "success", "text_zh": result["text_zh"]},
            "two_stage": {"status": "pending"},
            "pair_status": "pending",
        }
        if not (target / "clip.mp4").is_file():
            atomic_copy(clip, target / "clip.mp4")

        writer_prompt = build_writer_prompt(fact, result)
        try:
            writer_raw = call(
                writer_prompt,
                temperature=TEMPERATURE,
                timeout_s=REQUEST_TIMEOUT_S,
                max_retries=0,
            )
        except Exception as exc:
            record["two_stage"] = {
                "status": "failed",
                "failure_stage": "writer_request",
                **_error_summary(exc),
            }
            record["pair_status"] = "failed"
            atomic_write_json(analysis_path, record)
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        try:
            writer_text = parse_writer_response(writer_raw)
        except Exception as exc:
            record["two_stage"] = {
                "status": "failed",
                "failure_stage": "writer_validation",
                **_error_summary(exc),
            }
            record["pair_status"] = "failed"
            atomic_write_json(analysis_path, record)
            outcomes.append({"clip_uid": clip_uid, "status": "failed", "output": target})
            continue

        record["two_stage"] = {"status": "success", "text_zh": writer_text}
        record["pair_status"] = "complete"
        atomic_write_json(analysis_path, record)
        outcomes.append({"clip_uid": clip_uid, "status": "complete", "output": target})
    return outcomes


def _blind_assignments(path: Path, clip_uids: Sequence[str]) -> dict:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        assignments = value.get("assignments")
        if not isinstance(assignments, dict) or not set(clip_uids) <= set(assignments):
            raise ValueError("blind key does not match requested clips")
        return {clip_uid: assignments[clip_uid] for clip_uid in clip_uids}

    paths = ["single_call"] * ((len(clip_uids) + 1) // 2)
    paths.extend(["two_stage"] * (len(clip_uids) // 2))
    secrets.SystemRandom().shuffle(paths)
    assignments = {
        clip_uid: {"甲": path_name, "乙": (
            "two_stage" if path_name == "single_call" else "single_call"
        )}
        for clip_uid, path_name in zip(clip_uids, paths)
    }
    atomic_write_json(path, {
        "experiment_id": WRITING_AB_EXPERIMENT_ID,
        "assignments": assignments,
    })
    return assignments


def write_blind_ab_review(output_root: Path, outcomes: Sequence[dict]) -> Path:
    output_root = Path(output_root)
    clip_uids = [outcome["clip_uid"] for outcome in outcomes]
    assignments = _blind_assignments(
        output_root / "blind_text_ab_key.json", clip_uids
    )
    lines = [
        "# FIFA 战术文字生成 A/B 盲评",
        "",
        "请先看完整视频，再分别阅读甲、乙两版。评审前不要打开 blind_text_ab_key.json。",
        "",
    ]
    for index, outcome in enumerate(outcomes, start=1):
        clip_uid = outcome["clip_uid"]
        clip_id = clip_uid.split(":", 1)[-1]
        record = json.loads(
            (Path(outcome["output"]) / "comments" / WRITING_AB_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        lines.extend([
            f"## {index}. {clip_uid}",
            "",
            f"- 视频：[打开 clip.mp4](./{clip_id}/clip.mp4)",
        ])
        locked_fact = record.get("locked_fact")
        prior_path = Path(outcome["output"]) / "comments" / "deep_analysis.json"
        if not locked_fact and prior_path.is_file():
            locked_fact = json.loads(
                prior_path.read_text(encoding="utf-8")
            ).get("locked_fact")
        if locked_fact:
            lines.append(f"- 锁定 Tag：{locked_fact['tactic_zh']}")
        lines.append("")
        if record.get("pair_status") != "complete":
            lines.extend([
                f"本条 A/B 不完整：{record.get('failure_stage') or record.get('two_stage', {}).get('failure_stage', 'unknown')}。",
                "",
            ])
            continue
        texts = {
            "single_call": record["single_call"]["text_zh"],
            "two_stage": record["two_stage"]["text_zh"],
        }
        for label in ("甲", "乙"):
            lines.extend([f"### 版本{label}", "", f"> {texts[assignments[clip_uid][label]]}", ""])
        lines.extend([
            "1. 哪一版更清楚地解释了机会形成的原因？有无遗漏或编造机制？",
            "",
            "   回答：",
            "",
            "2. 哪一版对传球、带球、射门、扑救或进球事实更准确？",
            "",
            "   回答：",
            "",
            "3. 甲更好、乙更好还是基本相同？最主要的理由是什么？",
            "",
            "   回答：",
            "",
        ])
    path = output_root / "blind_text_ab_review.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_writing_ab_report(output_root: Path, outcomes: Sequence[dict]) -> Path:
    clips = []
    complete = 0
    for outcome in outcomes:
        record = json.loads(
            (Path(outcome["output"]) / "comments" / WRITING_AB_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        complete += int(record.get("pair_status") == "complete")
        clips.append({
            "clip_uid": outcome["clip_uid"],
            "pair_status": record.get("pair_status", "failed"),
            "failure_stage": record.get("failure_stage") or record.get(
                "two_stage", {}
            ).get("failure_stage"),
        })
    report = {
        "experiment_id": WRITING_AB_EXPERIMENT_ID,
        "prompt_version": PROMPT_VERSION,
        "requested_clips": len(outcomes),
        "complete_pairs": complete,
        "failed_pairs": len(outcomes) - complete,
        "decision_ready": len(outcomes) == 5 and complete == 5,
        "clips": clips,
    }
    return atomic_write_json(Path(output_root) / "writing_ab_report.json", report)


def available_audio_s(commentary: list[dict], index: int, duration_s: float) -> float:
    start = float(commentary[index]["timestamp_s"])
    later = [
        float(segment["timestamp_s"])
        for position, segment in enumerate(commentary)
        if position != index and float(segment["timestamp_s"]) > start
    ]
    boundary = min(later) if later else duration_s
    available = boundary - start
    if available <= 0:
        raise ValueError("tactical segment has no available audio time")
    return available


def compression_target_chars(
    full_text: str, *, available_s: float, full_duration_s: float
) -> int:
    if available_s <= 0 or full_duration_s <= 0:
        raise ValueError("audio durations must be positive")
    return max(18, math.floor(len(full_text) * available_s / full_duration_s * 0.95))


def build_compression_prompt(
    analysis: dict,
    *,
    full_duration_s: float,
    available_s: float,
    target_chars: int,
    exemplar: dict,
) -> str:
    source = {
        "locked_fact": analysis["locked_fact"],
        "mechanism": {
            field: analysis["response"][field]
            for field in (
                "position_or_action", "opponent_constraint",
                "space_change", "functional_effect",
            )
        },
        "mechanism_result": analysis["response"]["mechanism_result"],
        "play_result": analysis["response"]["play_result"],
        "full_text_zh": analysis["response"]["text_zh"],
        "full_tts_duration_s": full_duration_s,
        "available_audio_s": available_s,
        "target_chars": target_chars,
    }
    return f"""你是战术文字的音频压缩编辑。只压缩给定完整稿，不看视频、不添加事实、不改变锁定 Tag。

要求：
1. 只返回 JSON：{{"audio_text_zh":"..."}}。
2. audio_text_zh 不超过 target_chars 个字符，中文标点计数。
3. 至少保留一个“动作或站位 → 空间、约束或作用”的明确关系，并使用“通过、利用、迫使、牵制、拉开、打开、压缩、从而、因此、使得、让、由此、导致、形成、创造”之一。
4. 必须保持 mechanism_result 与 play_result 的区别。若 play_result 为 failure 或 mixed，必须保留越位、被扑、丢失球权等原有失败结果；不得写成成功。
5. 只删减和改写，不补充球员身份、阵型、教练意图或新战术。

人工压缩示例：
{json.dumps(exemplar["audio_adaptation_example"], ensure_ascii=False)}

待压缩材料：
{json.dumps(source, ensure_ascii=False)}
"""


def parse_compression_response(
    raw: str, *, target_chars: int, play_result: str
) -> str:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid audio-compression JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"audio_text_zh"}:
        raise ValueError("audio-compression response fields do not match the schema")
    text = _nonempty_text(value["audio_text_zh"], "audio_text_zh")
    if len(text) > target_chars:
        raise ValueError("audio_text_zh exceeds target_chars")
    if not _has_mechanism_relation(text):
        raise ValueError("audio_text_zh lacks a tactical mechanism relation")
    if play_result in {"failure", "mixed"} and not any(
        marker in text for marker in FAILURE_MARKERS
    ):
        raise ValueError("audio_text_zh dropped the failed play result")
    return text


def prepare_audio_adaptation(
    output: Path,
    *,
    synthesizer,
    call: Callable = ark_chat,
    probe: Callable[[Path], float],
    voice: str = "default",
    prompt_wav: Path | None = None,
    prompt_text: str | None = None,
    exemplar_path: Path = DEFAULT_EXEMPLAR_PATH,
) -> dict:
    """Measure full text, optionally compress once, and persist the audio candidate."""
    output = Path(output)
    analysis_path = output / "comments" / "deep_analysis.json"
    commentary_path = output / "comments" / "commentary.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("written_status") != "success":
        raise ValueError("audio adaptation requires successful written commentary")
    commentary = json.loads(commentary_path.read_text(encoding="utf-8"))
    fact = analysis["locked_fact"]
    matches = _matching_segment_indexes(commentary, fact)
    if len(matches) != 1:
        raise ValueError("audio adaptation requires one locked tactical segment")
    index = matches[0]
    duration_s = video_duration_s(output / "clip.mp4")
    available_s = available_audio_s(commentary, index, duration_s)
    full_text = analysis["response"]["text_zh"]
    work_dir = output / "voice" / "adaptation"
    full_wav = work_dir / "full_text.wav"
    kwargs = {
        "voice": voice,
        "prompt_wav": prompt_wav,
        "prompt_text": prompt_text,
    }
    synthesizer.synthesize(full_text, full_wav, **kwargs)
    full_duration_s = probe(full_wav)

    audio_text = full_text
    compressed_duration_s = full_duration_s
    compression_calls = 0
    target_chars = len(full_text)
    adaptation = {
        "status": "running",
        "failure_stage": None,
        "full_tts_duration_s": full_duration_s,
        "available_audio_s": available_s,
        "target_chars": target_chars,
        "audio_text_zh": audio_text,
        "compression_calls": compression_calls,
        "compressed_tts_duration_s": compressed_duration_s,
        "final_tempo": None,
    }
    analysis["audio_adaptation"] = adaptation
    atomic_write_json(analysis_path, analysis)
    if full_duration_s > available_s:
        target_chars = compression_target_chars(
            full_text,
            available_s=available_s,
            full_duration_s=full_duration_s,
        )
        compression_calls = 1
        adaptation.update(
            target_chars=target_chars,
            compression_calls=compression_calls,
        )
        atomic_write_json(analysis_path, analysis)
        prompt = build_compression_prompt(
            analysis,
            full_duration_s=full_duration_s,
            available_s=available_s,
            target_chars=target_chars,
            exemplar=load_exemplar(exemplar_path),
        )
        raw = call(
            prompt,
            temperature=TEMPERATURE,
            timeout_s=REQUEST_TIMEOUT_S,
            max_retries=0,
        )
        audio_text = parse_compression_response(
            raw,
            target_chars=target_chars,
            play_result=analysis["response"]["play_result"],
        )
        adaptation["audio_text_zh"] = audio_text
        atomic_write_json(analysis_path, analysis)
        compressed_wav = work_dir / "compressed_text.wav"
        synthesizer.synthesize(audio_text, compressed_wav, **kwargs)
        compressed_duration_s = probe(compressed_wav)

    final_tempo = (
        1.0 if compressed_duration_s <= available_s
        else compressed_duration_s / available_s * 1.01
    )
    status = "ready" if final_tempo <= 1.5 else "failed"
    commentary[index]["fallback_text_zh"] = audio_text
    adaptation = {
        "status": status,
        "failure_stage": None if status == "ready" else "duration_fit",
        "full_tts_duration_s": full_duration_s,
        "available_audio_s": available_s,
        "target_chars": target_chars,
        "audio_text_zh": audio_text,
        "compression_calls": compression_calls,
        "compressed_tts_duration_s": compressed_duration_s,
        "final_tempo": final_tempo,
    }
    analysis["audio_adaptation"] = adaptation
    atomic_write_json(commentary_path, commentary)
    atomic_write_json(analysis_path, analysis)
    return adaptation


def record_audio_failure(output: Path, stage: str, error: Exception) -> None:
    path = Path(output) / "comments" / "deep_analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    current = analysis.get("audio_adaptation", {})
    analysis["audio_adaptation"] = {
        **current,
        "status": "failed",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
    }
    atomic_write_json(path, analysis)


def mark_audio_rendered(output: Path, final_video: Path) -> None:
    path = Path(output) / "comments" / "deep_analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))
    analysis["audio_adaptation"].update(
        status="success",
        final_video=str(final_video),
    )
    atomic_write_json(path, analysis)


def write_mentor_review(output_root: Path, outcomes: Sequence[dict]) -> Path:
    lines = ["# FIFA 战术文字直播 v2", ""]
    for outcome in outcomes:
        if outcome["status"] != "success":
            continue
        analysis = json.loads(
            (Path(outcome["output"]) / "comments" / "deep_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        lines.extend([
            f"## {outcome['clip_uid']}",
            "",
            analysis["response"]["text_zh"],
            "",
        ])
    path = Path(output_root) / "mentor_text_review.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def write_experiment_report(output_root: Path, outcomes: Sequence[dict]) -> Path:
    written_success = 0
    audio_success = 0
    audio_failed = 0
    compressed = 0
    mechanism_preserved = 0
    failure_candidates = 0
    failure_preserved = 0
    tempo_review = []
    clips = []

    for outcome in outcomes:
        item = {
            "clip_uid": outcome["clip_uid"],
            "written_status": outcome["status"],
            "audio_status": "not_run",
        }
        if outcome["status"] == "success":
            written_success += 1
            analysis = json.loads(
                (Path(outcome["output"]) / "comments" / "deep_analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            adaptation = analysis.get("audio_adaptation", {})
            item["audio_status"] = adaptation.get("status", "pending")
            item["final_tempo"] = adaptation.get("final_tempo")
            if item["audio_status"] == "success":
                audio_success += 1
            elif item["audio_status"] == "failed":
                audio_failed += 1
            if adaptation.get("compression_calls") == 1:
                compressed += 1
                audio_text = adaptation.get("audio_text_zh") or ""
                mechanism_preserved += int(_has_mechanism_relation(audio_text))
                if analysis["response"]["play_result"] in {"failure", "mixed"}:
                    failure_candidates += 1
                    failure_preserved += int(any(
                        marker in audio_text for marker in FAILURE_MARKERS
                    ))
            tempo = adaptation.get("final_tempo")
            if isinstance(tempo, (int, float)) and tempo > 1.0:
                tempo_review.append(outcome["clip_uid"])
        clips.append(item)

    report = {
        "prompt_version": PROMPT_VERSION,
        "requested_clips": len(outcomes),
        "written_successes": written_success,
        "written_failures": len(outcomes) - written_success,
        "prompt_evaluation_valid": written_success >= 3,
        "written_depth_gate_count": written_success,
        "audio_successes": audio_success,
        "audio_failures": audio_failed,
        "audio_success_rate": (
            audio_success / written_success if written_success else None
        ),
        "compressed_clips": compressed,
        "compressed_mechanism_preservation_rate": (
            mechanism_preserved / compressed if compressed else None
        ),
        "failed_result_preservation_rate": (
            failure_preserved / failure_candidates if failure_candidates else None
        ),
        "tempo_listening_review_required": tempo_review,
        "clips": clips,
    }
    return atomic_write_json(Path(output_root) / "experiment_report.json", report)
