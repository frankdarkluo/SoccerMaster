"""Open tactical-tag generation and scoring for the frozen 67-clip benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pipeline.topology.digest import render_digest, window_digest
from pipeline.video_models import generate_json, model_name, temporary_window_clip

PHASES = ("phase1_direct", "phase2_observation_first")
PROMPT_VERSIONS = {
    "phase1_direct": "p0-two-pass-screen-then-score-v3",
    "phase2_observation_first": "p0-two-pass-questionnaire-screen-then-score-v3",
}
SCHEMA_VERSION = "p0-two-pass-screen-then-score-v3"
EXPERIMENT_ID = "p0-two-pass-screening-and-scoring-v3"
CONFIDENCE_THRESHOLDS = (0, 50, 60, 70, 80, 90)
SCREEN_FPS = 1.0
SCORE_FPS = 6.0
# Ark rejects fps outside [0.2, 5.0]; Gemini uses SCORE_FPS as playback slowdown, not API fps.
DOUBAO_SCORE_FPS = 5.0
SCORE_WINDOW_MAX_S = 5.0
RESTART_TYPES = ("corner", "free_kick", "throw_in", "goal_kick", "kickoff", "open_play")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_lines(value: str | None) -> list[str]:
    return [line.strip(" -•\t") for line in (value or "").splitlines() if line.strip(" -•\t")]


def load_glossary(path: Path) -> dict[str, dict[str, Any]]:
    cards = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        tactic_id = (row.get("id（英文标识）") or "").strip()
        if not tactic_id or tactic_id.startswith("英文小写连字符"):
            continue
        confusing = []
        for index in ("①", "②", "③"):
            target = (row.get(f"易混淆{index} 对象") or "").strip()
            distinction = (row.get(f"易混淆{index} 区分点") or "").strip()
            if target or distinction:
                confusing.append({"object": target, "distinction": distinction})
        cards[tactic_id] = {
            "tactic_id": tactic_id,
            "name_zh": (row.get("中文名") or "").strip(),
            "name_en": (row.get("英文名") or "").strip(),
            "priority": (row.get("优先级") or "").strip(),
            "definition": (row.get("一句话定义") or "").strip(),
            "observable_cues": split_lines(row.get("可观察特征")),
            "triggers": split_lines(row.get("触发条件")),
            "confusing": confusing,
        }
    return cards


def compact(card: dict[str, Any]) -> dict[str, Any]:
    return {key: card[key] for key in ("tactic_id", "definition", "observable_cues", "triggers", "confusing")}


def response_schema(tactic_ids: list[str]) -> dict[str, Any]:
    evidence = {
        "type": "object",
        "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "visible_movement_zh": {"type": "string"}, "tactical_link_zh": {"type": "string"},
        },
        "required": ["start_s", "end_s", "visible_movement_zh", "tactical_link_zh"],
        "additionalProperties": False,
    }
    assessment = {
        "tactic_id": {"type": "string", "enum": tactic_ids},
        "verdict": {"type": "string", "enum": ["present", "absent"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason_zh": {"type": "string"},
        "matched_cues": {"type": "array", "items": {"type": "string"}},
        "evidence_spans": {"type": "array", "maxItems": 3, "items": evidence},
    }
    return {"type": "object", "properties": {
        "assessments": {"type": "array", "items": {
            "type": "object", "properties": assessment, "required": list(assessment), "additionalProperties": False,
        }},
    }, "required": ["assessments"], "additionalProperties": False}


def build_prompt(clip_alias: str, duration_s: float, cards: list[dict[str, Any]], topology: str | None = None, screen_facts: dict[str, Any] | None = None) -> str:
    count = len(cards)
    return f"""You are judging a short, automatically trimmed football clip against a list of concept cards.

Watch the complete clip and read ALL {count} concept cards. These cards were forwarded by a deliberately over-inclusive low-frame-rate filter that was told to include anything plausible rather than exclude it. Being forwarded is NOT evidence of presence: most forwarded cards are still absent, and the trimmed excerpt may contain no tactic at all. Tactics are NOT mutually exclusive, so several may hold at once, but absent is the default verdict unless this clip visibly demonstrates the card's defining mechanism.

LOW-FPS SCREEN FACTS:
{json.dumps(screen_facts or {}, ensure_ascii=False)}
Treat these as fallible observations to verify against the high-frame-rate clip, not as verdicts.

Apply restart type as the first causal gate. If this is a corner, free kick, throw-in, goal kick, or kickoff, only mark a card present when its definition explicitly covers that restart. Never relabel a set-piece delivery as cutback, long-ball, line-breaking-pass, run-in-behind, or counter-attack merely because the ball travels forward or enters the box. For open play, distinguish normal progression from a genuine possession transition; counter-attack requires a visible regain followed immediately by a short, direct threat sequence.

Hard negatives against prior over-firing: a forward pass plus a runner moving toward goal is ordinary progression, not automatic present for run-in-behind, line-breaking-pass, or long-ball. Mark those present only when that card's own last-line / distance / aerial-span cues are visibly satisfied in this clip. If the same single pass would make you mark two or more of those three present, prefer absent on the ones whose cues you cannot show. Prefer absent whenever confidence would rest on narrative plausibility rather than a visible cue match.

For every card, decide present or absent independently based on whether its defining mechanism is visibly supported. Rank confidence by the strength of that card's own evidence, not by how prominent the sequence is relative to other tactics. Before marking a card present, read its `confusing` entries: name the nearest confusable pattern and state the visible evidence that rules it out. If you cannot rule it out from this clip, mark the card absent. When present, copy at least one matched cue exactly from its card and cite 1 to 3 decisive time spans within this clip's bounds. When absent, a brief reason is enough and evidence_spans/matched_cues may be empty. Write reasons and evidence in Chinese. Do not infer from filenames, prior labels, commentary, shirt identity, coach intent, another model, or unstated facts. Return exactly one assessment per card, covering every tactic_id exactly once. Return JSON only.

Clip: {clip_alias}
Clip bounds: 0.000 to {duration_s:.3f} seconds.
QUANTITATIVE GEOMETRY AID:
{topology or "Unavailable for this clip."}
This is auxiliary measurement, not a verdict. position_source and gaps state its provenance and limitations. If it conflicts with the video, trust the video.
CONCEPT CARDS:
{json.dumps([compact(card) for card in cards], ensure_ascii=False)}

OUTPUT JSON SCHEMA:
{json.dumps(response_schema([card['tactic_id'] for card in cards]), ensure_ascii=False)}
"""


def validate_response(payload: Any, tactic_ids: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"assessments"}:
        raise ValueError("response fields are not strict")
    assessments = payload["assessments"]
    if not isinstance(assessments, list) or len(assessments) != len(tactic_ids):
        raise ValueError("assessments must cover every card exactly once")
    seen: set[str] = set()
    for assessment in assessments:
        expected = {"tactic_id", "verdict", "confidence", "reason_zh", "matched_cues", "evidence_spans"}
        if not isinstance(assessment, dict) or set(assessment) != expected:
            raise ValueError("assessment fields are not strict")
        tactic_id, verdict, confidence = assessment.get("tactic_id"), assessment.get("verdict"), assessment.get("confidence")
        if tactic_id not in tactic_ids or tactic_id in seen or verdict not in ("present", "absent"):
            raise ValueError("invalid or duplicate assessment")
        if isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100:
            raise ValueError("invalid assessment confidence")
        if not isinstance(assessment["reason_zh"], str) or not isinstance(assessment["matched_cues"], list) or any(not isinstance(cue, str) for cue in assessment["matched_cues"]):
            raise ValueError("invalid assessment text or cues")
        spans = assessment["evidence_spans"]
        if not isinstance(spans, list) or len(spans) > 3:
            raise ValueError("invalid evidence spans")
        for span in spans:
            if not isinstance(span, dict) or set(span) != {"start_s", "end_s", "visible_movement_zh", "tactical_link_zh"}:
                raise ValueError("evidence fields are not strict")
            if not isinstance(span["visible_movement_zh"], str) or not isinstance(span["tactical_link_zh"], str):
                raise ValueError("invalid evidence text")
        seen.add(tactic_id)
        if verdict == "present" and (not assessment["matched_cues"] or not spans):
            raise ValueError("present verdict requires matched cues and evidence")
    return payload


def screen_compact(card: dict[str, Any]) -> dict[str, Any]:
    return {"tactic_id": card["tactic_id"], "definition": card["definition"]}


def screen_response_schema(tactic_ids: list[str], phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    properties: dict[str, Any] = {
        "restart_type": {"type": "string", "enum": list(RESTART_TYPES)},
        "possession_transition_s": {"type": ["number", "null"]},
        "decisive_window": {
            "type": "object",
            "properties": {"start_s": {"type": "number"}, "end_s": {"type": "number"}},
            "required": ["start_s", "end_s"], "additionalProperties": False,
        },
        "candidate_tactic_ids": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": tactic_ids}},
    }
    required = ["restart_type", "possession_transition_s", "decisive_window", "candidate_tactic_ids"]
    if phase == PHASES[1]:
        properties.update({
            "receiver_relative_to_last_defender": {"type": "string", "enum": ["ahead", "level", "behind", "not_applicable"]},
            "defenders_goal_side_count": {"type": "integer", "minimum": 0},
            "ball_path": {"type": "string", "enum": ["ground", "air", "not_applicable"]},
            "terminal_location_zh": {"type": "string"},
        })
        required += ["receiver_relative_to_last_defender", "defenders_goal_side_count", "ball_path", "terminal_location_zh"]
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_screen_prompt(clip_alias: str, duration_s: float, cards: list[dict[str, Any]], phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    questionnaire = (
        ""
        if phase == PHASES[0] else
        " Also answer four purely observational questions about the decisive window: whether the receiver of the key pass was ahead of, level with, behind, or not applicable to the last defender at the moment of reception; how many defenders were goal-side of the ball; whether the ball path was on the ground or in the air; and the terminal location of the sequence, in Chinese. These are facts about player and ball positions only — do not name or imply any tactic."
    )
    return f"""You are screening a football clip before a closer, higher-frame-rate look, at low frame rate.

First identify how play restarted in this clip (corner, free_kick, throw_in, goal_kick, kickoff, or open_play meaning no set-piece restart) and, if possession changed hands, the timestamp of that transition (null if it never changes). Then find one decisive time window no longer than {SCORE_WINDOW_MAX_S:.1f} seconds that the closer look should focus on to judge which tactics are present.{questionnaire}

Finally, from the {len(cards)} tactic names and one-line definitions below, list only the tactic_ids (0 to 8) whose defining mechanism plausibly appears anywhere in the clip. Restart type is the first causal gate: for a set-piece restart, include only cards whose definitions explicitly cover that restart; never nominate cutback, long-ball, line-breaking-pass, run-in-behind, or counter-attack merely because the restart delivery travels forward or enters the box. For open play, counter-attack requires a visible regain immediately followed by direct threat, not ordinary progression. This is a coarse pre-filter, not a verdict — when unsure within the compatible family, include rather than exclude. An empty list means nothing plausibly present, and skips the closer look.

Clip: {clip_alias}
Clip bounds: 0.000 to {duration_s:.3f} seconds.
CANDIDATE TACTICS:
{json.dumps([screen_compact(card) for card in cards], ensure_ascii=False)}

OUTPUT JSON SCHEMA:
{json.dumps(screen_response_schema([card['tactic_id'] for card in cards], phase), ensure_ascii=False)}
"""


def validate_screen_response(payload: Any, tactic_ids: list[str], phase: str, duration_s: float) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    expected = {"restart_type", "possession_transition_s", "decisive_window", "candidate_tactic_ids"}
    if phase == PHASES[1]:
        expected |= {"receiver_relative_to_last_defender", "defenders_goal_side_count", "ball_path", "terminal_location_zh"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("screen response fields are not strict")
    if payload["restart_type"] not in RESTART_TYPES:
        raise ValueError("invalid restart_type")
    transition = payload["possession_transition_s"]
    if transition is not None and (isinstance(transition, bool) or not isinstance(transition, (int, float)) or not 0 <= transition <= duration_s + 0.001):
        raise ValueError("invalid possession_transition_s")
    window = payload["decisive_window"]
    if not isinstance(window, dict) or set(window) != {"start_s", "end_s"}:
        raise ValueError("invalid decisive_window")
    start_s, end_s = window.get("start_s"), window.get("end_s")
    if any(isinstance(value, bool) for value in (start_s, end_s)) or not isinstance(start_s, (int, float)) or not isinstance(end_s, (int, float)) or not 0 <= start_s < end_s <= duration_s + 0.001 or end_s - start_s > SCORE_WINDOW_MAX_S + 0.001:
        raise ValueError("decisive_window outside clip bounds")
    candidates = payload["candidate_tactic_ids"]
    if not isinstance(candidates, list) or len(candidates) > 8 or len(set(candidates)) != len(candidates) or any(tactic_id not in tactic_ids for tactic_id in candidates):
        raise ValueError("invalid candidate_tactic_ids")
    if phase == PHASES[1]:
        if payload["receiver_relative_to_last_defender"] not in ("ahead", "level", "behind", "not_applicable"):
            raise ValueError("invalid receiver_relative_to_last_defender")
        count = payload["defenders_goal_side_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid defenders_goal_side_count")
        if payload["ball_path"] not in ("ground", "air", "not_applicable"):
            raise ValueError("invalid ball_path")
        if not isinstance(payload["terminal_location_zh"], str):
            raise ValueError("invalid terminal_location_zh")
    return payload


def validate_evidence_bounds(payload: dict[str, Any], duration_s: float) -> None:
    for assessment in payload["assessments"]:
        for span in assessment["evidence_spans"]:
            try:
                start_s, end_s = float(span["start_s"]), float(span["end_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("evidence span outside query bounds") from exc
            if any(isinstance(span.get(key), bool) for key in ("start_s", "end_s")) or not 0 <= start_s < end_s <= duration_s + 0.001:
                raise ValueError("evidence span outside query bounds")


def aggregate_responses(payloads: list[dict[str, Any]], tactic_ids: list[str]) -> dict[str, Any]:
    """Reduce an odd number of score samples by independent per-card majority vote."""
    if not payloads or len(payloads) % 2 == 0:
        raise ValueError("score samples must be a positive odd number")
    for payload in payloads:
        validate_response(payload, tactic_ids)
    assessments = []
    for tactic_id in tactic_ids:
        samples = [next(item for item in payload["assessments"] if item["tactic_id"] == tactic_id) for payload in payloads]
        verdict = Counter(item["verdict"] for item in samples).most_common(1)[0][0]
        winners = [item for item in samples if item["verdict"] == verdict]
        confidence = round(sum(item["confidence"] for item in winners) / len(winners))
        representative = min(winners, key=lambda item: abs(item["confidence"] - confidence))
        assessments.append({**representative, "confidence": confidence})
    return {"assessments": assessments}


def score_screen_facts(payload: dict[str, Any], time_scale: float = 1.0) -> dict[str, Any]:
    """Keep factual screen observations and rebase transition time to the trimmed score clip."""
    window = payload["decisive_window"]
    transition = payload["possession_transition_s"]
    facts = {"restart_type": payload["restart_type"], "possession_transition_s": None}
    if transition is not None and window["start_s"] <= transition <= window["end_s"]:
        facts["possession_transition_s"] = round((transition - window["start_s"]) * time_scale, 3)
    for key in ("receiver_relative_to_last_defender", "defenders_goal_side_count", "ball_path", "terminal_location_zh"):
        if key in payload:
            facts[key] = payload[key]
    return facts


def offset_evidence(payload: dict[str, Any], offset_s: float, time_scale: float = 1.0) -> dict[str, Any]:
    """Translate score-window evidence spans back onto the source clip timeline."""
    return {"assessments": [
        {**assessment, "evidence_spans": [
            {**span, "start_s": round(float(span["start_s"]) / time_scale + offset_s, 3), "end_s": round(float(span["end_s"]) / time_scale + offset_s, 3)}
            for span in assessment["evidence_spans"]
        ]}
        for assessment in payload["assessments"]
    ]}


def duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def frozen_sources(repo_root: Path, clip_manifest: Path, ground_truth: Path) -> dict[str, dict[str, Any]]:
    index = {row["clip_uid"]: {**row, "absolute_video_path": (repo_root / row["video_path"]).resolve()} for row in load_jsonl(clip_manifest) if row.get("video_path")}
    clip_uids = sorted({claim["clip_uid"] for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary"})
    if len(clip_uids) != 67:
        raise ValueError(f"expected 67 frozen clips, got {len(clip_uids)}")
    if missing := set(clip_uids) - set(index):
        raise ValueError(f"clips missing from clip manifest: {sorted(missing)}")
    return {clip_uid: index[clip_uid] for clip_uid in clip_uids}


def scored_cards(glossary_path: Path, ground_truth: Path) -> dict[str, dict[str, Any]]:
    """P0 cards plus every tactic referenced by a human-reviewed primary claim."""
    glossary = load_glossary(glossary_path)
    scored_ids = {claim["tactic_id"] for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary"}
    required = {tactic_id for tactic_id, card in glossary.items() if card["priority"] == "P0"} | scored_ids
    if missing := required - set(glossary):
        raise ValueError(f"scored tactics missing from glossary: {sorted(missing)}")
    cards = {tactic_id: glossary[tactic_id] for tactic_id in required}
    if len(cards) != 23:
        raise ValueError(f"expected 23 scored cards, got {len(cards)}")
    return cards


def _load_topology(repo_root: Path, clip_uid: str) -> dict[str, Any] | None:
    topology_path = repo_root / "outputs/tactical_topology" / f"{clip_uid.split(':', 1)[-1]}.json"
    return json.loads(topology_path.read_text(encoding="utf-8")) if topology_path.is_file() else None


def run(output_root: Path, glossary_path: Path, clip_manifest: Path, ground_truth: Path, repo_root: Path, *, phase: str, provider: str, clip_uids: Iterable[str] | None = None, force: bool = False, retry_failed: bool = False, score_samples: int | None = None, temperature: float | None = None, use_topology: bool = True) -> dict[str, int | float | str | None]:
    """Two-pass per clip: a low-fps screen over the full video narrows to <=8 cards and a
    decisive window, then a high-fps score pass judges only that window against that subset."""
    score_samples = (3 if provider == "gemini" else 1) if score_samples is None else score_samples
    temperature = 0.0 if provider != "gemini" and temperature is None else temperature
    if score_samples < 1 or score_samples % 2 == 0:
        raise ValueError("score_samples must be a positive odd number")
    if provider != "gemini" and score_samples != 1:
        raise ValueError("self-consistency sampling is Gemini-only")
    if provider == "gemini" and temperature is not None:
        raise ValueError("Gemini 3.6 Flash does not support temperature")
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    cards_by_id = scored_cards(glossary_path, ground_truth)
    cards = sorted(cards_by_id.values(), key=lambda card: card["tactic_id"])
    tactic_ids = list(cards_by_id)
    sources = frozen_sources(repo_root, clip_manifest, ground_truth)
    selected = set(clip_uids or sources)
    if unknown := selected - set(sources):
        raise ValueError(f"unknown clips: {sorted(unknown)}")
    counts = Counter(clips=0, calls=0, success=0, failed=0, skipped=0)
    for clip_uid in sorted(selected):
        target = output_root / phase / provider / f"{clip_uid.split(':', 1)[-1]}.json"
        if target.is_file() and not force:
            current = json.loads(target.read_text(encoding="utf-8"))
            if not retry_failed or current.get("status") != "failed":
                counts.update(clips=1, skipped=1)
                continue
        video = sources[clip_uid]["absolute_video_path"]
        digest = sha256(video)
        alias = f"clip-{digest[:12]}.mp4"
        clip_duration = duration(video)
        topology = _load_topology(repo_root, clip_uid) if use_topology else None
        score_prompt: str | None = None
        score_playback_slowdown: float | None = None
        provider_calls = 0
        try:
            screen_prompt = build_screen_prompt(alias, clip_duration, cards, phase)
            provider_calls += 1
            screen_payload, screen_usage, screen_attempts = generate_json(
                provider, screen_prompt, screen_response_schema(tactic_ids, phase), video_path=video, fps=SCREEN_FPS if provider == "doubao" else None,
            )
            validate_screen_response(screen_payload, tactic_ids, phase, clip_duration)
            narrowed_ids = sorted(set(screen_payload["candidate_tactic_ids"]))
            if narrowed_ids:
                window = screen_payload["decisive_window"]
                narrowed_cards = [cards_by_id[tactic_id] for tactic_id in narrowed_ids]
                score_playback_slowdown = SCORE_FPS if provider == "gemini" else 1.0
                score_duration = (window["end_s"] - window["start_s"]) * score_playback_slowdown
                windowed_topology = window_digest(topology, window["start_s"], window["end_s"], time_scale=score_playback_slowdown) if topology else None
                topology_text = render_digest(windowed_topology) if windowed_topology else None
                with temporary_window_clip(video, window["start_s"], window["end_s"], playback_slowdown=score_playback_slowdown) as clip_window:
                    score_prompt = build_prompt(
                        alias, score_duration, narrowed_cards, topology_text, score_screen_facts(screen_payload, score_playback_slowdown),
                    )
                    sampled_payloads, score_usage, score_attempts = [], [], 0
                    for _ in range(score_samples):
                        provider_calls += 1
                        sample, sample_usage, sample_attempts = generate_json(
                            provider, score_prompt, response_schema(narrowed_ids), video_path=clip_window,
                            fps=DOUBAO_SCORE_FPS if provider == "doubao" else None, temperature=temperature,
                        )
                        validate_response(sample, narrowed_ids)
                        validate_evidence_bounds(sample, score_duration)
                        sampled_payloads.append(sample)
                        score_usage += sample_usage
                        score_attempts += sample_attempts
                    score_payload = offset_evidence(aggregate_responses(sampled_payloads, narrowed_ids), window["start_s"], score_playback_slowdown)
                    assessment_samples = [offset_evidence(sample, window["start_s"], score_playback_slowdown)["assessments"] for sample in sampled_payloads]
            else:
                score_payload, score_usage, score_attempts = {"assessments": []}, [], 0
                assessment_samples = []
            result = {"status": "success", "screen": screen_payload, **score_payload, "assessment_samples": assessment_samples}
            usage, attempts = screen_usage + score_usage, screen_attempts + score_attempts
        except Exception as exc:
            usage, attempts = [], getattr(exc, "attempts", 1)
            result = {"status": "failed", "failed_stage": "provider_request_or_validation", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        result.update({
            "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "phase": phase, "provider": provider,
            "model": model_name(provider), "clip_uid": clip_uid, "clip_sha256": digest,
            "prompt_version": PROMPT_VERSIONS[phase],
            "screen_prompt_sha256": hashlib.sha256(screen_prompt.encode()).hexdigest() if result["status"] == "success" else None,
            "screen_fps": SCREEN_FPS,
            "score_prompt_sha256": hashlib.sha256(score_prompt.encode()).hexdigest() if score_prompt else None,
            "score_fps": (DOUBAO_SCORE_FPS if provider == "doubao" else SCORE_FPS) if score_prompt else None,
            "score_playback_slowdown": score_playback_slowdown,
            "temperature": temperature, "score_samples": score_samples, "attempts": attempts, "clip_duration_s": clip_duration,
            "topology_enabled": use_topology, "topology_position_source": topology["position_source"] if topology else None,
            "api_usage": usage,
        })
        write_json(target, result)
        counts.update(clips=1, calls=provider_calls)
        counts[result["status"]] += 1
    return {"phase": phase, "provider": provider, **dict(counts)}


def claim_match(run: dict[str, Any], claim: dict[str, Any], confidence_threshold: int) -> bool:
    window = claim["window"]
    return any(
        assessment["tactic_id"] == claim["tactic_id"]
        and assessment["verdict"] == "present"
        and assessment["confidence"] >= confidence_threshold
        and any(min(float(span["end_s"]), float(window["end_s"])) > max(float(span["start_s"]), float(window["start_s"])) for span in assessment["evidence_spans"])
        for assessment in run["assessments"]
    )


def metrics(claims: list[dict[str, Any]], runs: dict[Any, dict[str, Any]], confidence_threshold: int) -> dict[str, Any]:
    rows = [(claim, "present" if claim_match(runs[claim["clip_uid"]], claim, confidence_threshold) else "absent") for claim in claims if claim["clip_uid"] in runs]
    correct = sum(predicted == claim["gt_verdict"] for claim, predicted in rows)
    positives = sum(claim["gt_verdict"] == "present" for claim, _ in rows)
    predicted_positive = sum(predicted == "present" for _, predicted in rows)
    true_positive = sum(predicted == "present" and claim["gt_verdict"] == "present" for claim, predicted in rows)
    precision = true_positive / predicted_positive if predicted_positive else None
    recall = true_positive / positives if positives else None
    f1 = None if not rows else 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "expected_claims": len(claims), "covered_claims": len(rows), "correct": correct,
        "accuracy": correct / len(claims) if len(rows) == len(claims) else None,
        "covered_accuracy": correct / len(rows) if rows else None,
        "precision": precision, "recall": recall, "f1": f1,
        "always_absent_baseline": sum(claim["gt_verdict"] == "absent" for claim, _ in rows) / len(rows) if rows else None,
    }


def report(output_root: Path, glossary_path: Path, ground_truth: Path) -> dict[str, Any]:
    cards = scored_cards(glossary_path, ground_truth)
    claims = [claim for claim in load_jsonl(ground_truth) if claim.get("score_set") == "primary"]
    if len(claims) != 73:
        raise ValueError(f"expected 73 primary claims, got {len(claims)}")
    if missing := {claim["tactic_id"] for claim in claims} - set(cards):
        raise ValueError(f"primary claims reference tactics missing from scored cards: {sorted(missing)}")
    conditions: dict[str, Any] = {}
    for phase in PHASES:
        conditions[phase] = {}
        for provider in ("gemini", "doubao"):
            attempted, valid = {}, {}
            for path in sorted((output_root / phase / provider).glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                attempted[payload["clip_uid"]] = payload
                if payload.get("status") != "success":
                    continue
                try:
                    validate_screen_response(payload["screen"], list(cards), phase, payload["clip_duration_s"])
                    queried_ids = sorted(set(payload["screen"]["candidate_tactic_ids"]))
                    validate_response({"assessments": payload["assessments"]}, queried_ids)
                    validate_evidence_bounds({"assessments": payload["assessments"]}, payload["clip_duration_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                valid[payload["clip_uid"]] = payload
            execution = {"expected_clips": 67, "attempted_clips": len(attempted), "valid_clips": len(valid), "invalid_or_failed_clips": len(attempted) - len(valid), "missing_clips": 67 - len(attempted), "status": "complete" if len(valid) == 67 else "partial"}
            pr_curve = {str(threshold): metrics(claims, valid, threshold) for threshold in CONFIDENCE_THRESHOLDS}
            conditions[phase][provider] = {"execution": execution, "pr_curve": pr_curve}
    value = {
        "experiment_id": EXPERIMENT_ID,
        "metric_scope": "73 primary claims across 23 scored cards (P0 ∪ tactics referenced by human-reviewed claims); confidence-threshold PR curve, not exhaustive 67-video accuracy",
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "conditions": conditions,
    }
    evaluation = output_root / "evaluation"
    write_json(evaluation / "summary.json", value)
    headline = str(CONFIDENCE_THRESHOLDS[0])
    lines = ["# P0 开放战术筛查报告（逐卡片判定）", "", "主指标为 73 条 primary claims 在 23 张已计分卡片上的 claim-level precision/recall/F1；表格取置信度阈值 0（模型自报的 present/absent），完整 PR 曲线见 summary.json 的 pr_curve。不得表述为 67 个视频的穷尽 accuracy。", "", "| phase | provider | attempted | valid | failed | precision | recall | f1 | always-absent baseline |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase in PHASES:
        for provider in ("gemini", "doubao"):
            execution, item = conditions[phase][provider]["execution"], conditions[phase][provider]["pr_curve"][headline]
            lines.append(f"| {phase} | {provider} | {execution['attempted_clips']} | {execution['valid_clips']} | {execution['invalid_or_failed_clips']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['always_absent_baseline']} |")
    (evaluation / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value
