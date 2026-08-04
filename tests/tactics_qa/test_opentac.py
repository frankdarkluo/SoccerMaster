from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from pipeline.tactics_qa import opentac
from pipeline.tactics_qa.opentac import (
    build_prompt,
    build_screen_prompt,
    claim_match,
    metrics,
    response_schema,
    scored_cards,
    screen_response_schema,
    validate_evidence_bounds,
    validate_response,
    validate_screen_response,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOSSARY = REPO_ROOT / "data/足球战术数据库_词条表_Grid.csv"
GROUND_TRUTH = REPO_ROOT / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl"
SOURCE_ROWS = REPO_ROOT / "benchmark/tactical_prototypes/source_rows.jsonl"


def card(tactic_id="counter-attack", name="快速反击"):
    return {
        "tactic_id": tactic_id,
        "name_zh": name,
        "name_en": "counter attack",
        "definition": "visible transition",
        "observable_cues": ["cue"],
        "triggers": ["trigger"],
        "confusing": [],
    }


def assessment(**overrides):
    return {
        "tactic_id": "counter-attack",
        "verdict": "present",
        "confidence": 85,
        "reason_zh": "理由",
        "matched_cues": ["cue"],
        "evidence_spans": [{"start_s": 0, "end_s": 1, "visible_movement_zh": "动作", "tactical_link_zh": "联系"}],
        **overrides,
    }


def screen(**overrides):
    return {
        "restart_type": "open_play",
        "possession_transition_s": None,
        "decisive_window": {"start_s": 2.0, "end_s": 7.0},
        "candidate_tactic_ids": ["counter-attack"],
        **overrides,
    }


def test_score_prompt_and_schema_are_phase_independent():
    cards = [card(f"tactic-{index}", f"战术{index}") for index in range(5)]
    tactic_ids = [c["tactic_id"] for c in cards]

    prompt = build_prompt("clip-neutral.mp4", 6.0, cards, screen_facts={"restart_type": "corner"})

    assert "ALL 5 concept cards" in prompt
    assert "prior labels" in prompt
    assert '\"restart_type\": \"corner\"' in prompt
    assert "first causal gate" in prompt
    assert "supporting_sequence_ids" not in response_schema(tactic_ids)["properties"]["assessments"]["items"]["properties"]
    assert "observations" not in response_schema(tactic_ids)["properties"]


def test_score_screen_facts_rebases_transition_and_keeps_questionnaire():
    payload = screen(possession_transition_s=4.5, receiver_relative_to_last_defender="level", defenders_goal_side_count=1, ball_path="ground", terminal_location_zh="禁区")

    assert opentac.score_screen_facts(payload) == {"restart_type": "open_play", "possession_transition_s": 2.5, "receiver_relative_to_last_defender": "level", "defenders_goal_side_count": 1, "ball_path": "ground", "terminal_location_zh": "禁区"}


def test_absent_verdict_does_not_require_evidence():
    payload = {"assessments": [assessment(verdict="absent", matched_cues=[], evidence_spans=[])]}

    assert validate_response(payload, ["counter-attack"]) is payload


def test_present_verdict_requires_matched_cues_and_evidence():
    payload = {"assessments": [assessment(matched_cues=[])]}

    with pytest.raises(ValueError, match="requires matched cues and evidence"):
        validate_response(payload, ["counter-attack"])


def test_validate_response_rejects_non_strict_assessment_and_evidence_fields():
    with pytest.raises(ValueError, match="assessment fields"):
        validate_response({"assessments": [assessment(unexpected=True)]}, ["counter-attack"])
    payload = {"assessments": [assessment()]}
    payload["assessments"][0]["evidence_spans"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="evidence fields"):
        validate_response(payload, ["counter-attack"])


def test_assessments_must_cover_exactly_the_queried_tactic_ids():
    payload = {"assessments": [assessment()]}

    with pytest.raises(ValueError, match="cover every card"):
        validate_response(payload, ["counter-attack", "cutback"])


def test_empty_assessments_are_valid_when_nothing_was_queried():
    assert validate_response({"assessments": []}, []) == {"assessments": []}


def test_aggregate_responses_votes_independently_per_card():
    present = assessment(confidence=90)
    absent = assessment(verdict="absent", confidence=75, matched_cues=[], evidence_spans=[])
    payloads = [
        {"assessments": [present, assessment(tactic_id="cutback", verdict="absent", confidence=80, matched_cues=[], evidence_spans=[])]},
        {"assessments": [absent, assessment(tactic_id="cutback", confidence=70)]},
        {"assessments": [assessment(confidence=70), assessment(tactic_id="cutback", verdict="absent", confidence=60, matched_cues=[], evidence_spans=[])]},
    ]

    result = opentac.aggregate_responses(payloads, ["counter-attack", "cutback"])

    assert [(item["tactic_id"], item["verdict"], item["confidence"]) for item in result["assessments"]] == [
        ("counter-attack", "present", 80),
        ("cutback", "absent", 70),
    ]


def test_aggregate_responses_requires_positive_odd_sample_count():
    with pytest.raises(ValueError, match="positive odd"):
        opentac.aggregate_responses([], ["counter-attack"])
    with pytest.raises(ValueError, match="positive odd"):
        opentac.aggregate_responses([{"assessments": [assessment()]}] * 2, ["counter-attack"])


def test_offset_evidence_translates_window_times_to_source_clip():
    shifted = opentac.offset_evidence({"assessments": [assessment()]}, 4.5)

    assert shifted["assessments"][0]["evidence_spans"][0]["start_s"] == 4.5
    assert shifted["assessments"][0]["evidence_spans"][0]["end_s"] == 5.5


def test_evidence_must_fit_clip_bounds():
    span = {"start_s": 10.0, "end_s": 13.0, "visible_movement_zh": "动作", "tactical_link_zh": "联系"}
    payload = {"assessments": [assessment(evidence_spans=[span])]}

    with pytest.raises(ValueError, match="query bounds"):
        validate_evidence_bounds(payload, 12.0)


def test_claim_match_requires_present_verdict_confidence_and_window_overlap():
    claim = {"clip_uid": "clip", "tactic_id": "counter-attack", "window": {"start_s": 0, "end_s": 1}}
    run = {"assessments": [assessment(confidence=60)]}

    assert claim_match(run, claim, 50) is True
    assert claim_match(run, claim, 70) is False


def test_metrics_reports_precision_recall_f1_and_baseline():
    claims = [
        {"clip_uid": "clip", "tactic_id": "counter-attack", "gt_verdict": "present", "window": {"start_s": 0, "end_s": 1}},
        {"clip_uid": "clip", "tactic_id": "cutback", "gt_verdict": "absent", "window": {"start_s": 0, "end_s": 1}},
    ]
    runs = {"clip": {"assessments": [
        assessment(tactic_id="counter-attack"),
        assessment(tactic_id="cutback", verdict="absent", matched_cues=[], evidence_spans=[]),
    ]}}

    result = metrics(claims, runs, 0)

    assert (result["precision"], result["recall"], result["f1"]) == (1.0, 1.0, 1.0)
    assert result["accuracy"] == 1.0
    assert result["always_absent_baseline"] == 0.5

    no_predictions = metrics(claims, runs, 90)

    assert no_predictions["f1"] == 0.0


def test_scored_cards_is_p0_union_human_reviewed_tactics():
    glossary = REPO_ROOT / "data/足球战术数据库_词条表_Grid.csv"
    ground_truth = REPO_ROOT / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl"

    cards = scored_cards(glossary, ground_truth)

    assert len(cards) == 23
    # P0 tactics are always included regardless of ground-truth coverage.
    assert cards["counter-attack"]["priority"] == "P0"
    # Non-P0 tactics referenced by a human-reviewed primary claim are pulled in too.
    assert cards["corner-near-far-post"]["priority"] == "P1"
    assert any(item["object"] == "已落位后的常规组织推进" for item in cards["counter-attack"]["confusing"])
    assert any("打身后" in item["object"] for item in cards["long-ball"]["confusing"])


def test_run_rejects_even_samples_and_doubao_voting(tmp_path):
    kwargs = (tmp_path, tmp_path / "glossary", tmp_path / "sources", tmp_path / "truth", tmp_path)
    with pytest.raises(ValueError, match="positive odd"):
        opentac.run(*kwargs, phase="phase1_direct", provider="gemini", score_samples=2)
    with pytest.raises(ValueError, match="Gemini-only"):
        opentac.run(*kwargs, phase="phase1_direct", provider="doubao", score_samples=3)
    with pytest.raises(ValueError, match="does not support temperature"):
        opentac.run(*kwargs, phase="phase1_direct", provider="gemini", temperature=0.5)


def test_screen_prompt_asks_questionnaire_only_in_observation_first_phase():
    cards = [card()]

    direct = build_screen_prompt("clip-neutral.mp4", 12.0, cards, "phase1_direct")
    observation_first = build_screen_prompt("clip-neutral.mp4", 12.0, cards, "phase2_observation_first")

    assert "receiver of the key pass" not in direct
    assert "do not name or imply any tactic" in observation_first
    assert "Restart type is the first causal gate" in direct
    assert "receiver_relative_to_last_defender" in screen_response_schema(["counter-attack"], "phase2_observation_first")["required"]
    assert "receiver_relative_to_last_defender" not in screen_response_schema(["counter-attack"], "phase1_direct")["required"]


def test_validate_screen_response_rejects_window_outside_clip_bounds():
    payload = screen(decisive_window={"start_s": 5.0, "end_s": 20.0})

    with pytest.raises(ValueError, match="outside clip bounds"):
        validate_screen_response(payload, ["counter-attack"], "phase1_direct", 12.0)


def test_validate_screen_response_rejects_window_over_five_seconds():
    payload = screen(decisive_window={"start_s": 2.0, "end_s": 8.0})

    with pytest.raises(ValueError, match="outside clip bounds"):
        validate_screen_response(payload, ["counter-attack"], "phase1_direct", 12.0)


def test_validate_screen_response_rejects_unknown_candidate_tactic():
    payload = screen(candidate_tactic_ids=["cutback"])

    with pytest.raises(ValueError, match="invalid candidate_tactic_ids"):
        validate_screen_response(payload, ["counter-attack"], "phase1_direct", 12.0)


def test_validate_screen_response_accepts_empty_candidates():
    payload = screen(candidate_tactic_ids=[])

    assert validate_screen_response(payload, ["counter-attack"], "phase1_direct", 12.0) is payload


def test_validate_screen_response_requires_questionnaire_fields_in_observation_first():
    payload = screen()

    with pytest.raises(ValueError, match="fields are not strict"):
        validate_screen_response(payload, ["counter-attack"], "phase2_observation_first", 12.0)

    full = screen(
        receiver_relative_to_last_defender="ahead", defenders_goal_side_count=2,
        ball_path="ground", terminal_location_zh="禁区中路",
    )
    assert validate_screen_response(full, ["counter-attack"], "phase2_observation_first", 12.0) is full


def test_run_screens_then_scores_only_the_narrowed_window(tmp_path, monkeypatch):
    calls = []
    slowdowns = []

    def fake_generate_json(_provider, _prompt, _schema, *, video_path, fps, **_kwargs):
        calls.append((fps, video_path))
        if len(calls) == 1:
            payload = {
                "restart_type": "open_play", "possession_transition_s": None,
                "decisive_window": {"start_s": 1.0, "end_s": 5.0},
                "candidate_tactic_ids": ["counter-attack"],
            }
        else:
            payload = {"assessments": [{
                "tactic_id": "counter-attack", "verdict": "present", "confidence": 90,
                "reason_zh": "r", "matched_cues": ["c"],
                "evidence_spans": [{"start_s": 3.0, "end_s": 9.0, "visible_movement_zh": "m", "tactical_link_zh": "l"}],
            }]}
        return payload, [{"tokens": 1}], 1

    @contextmanager
    def fake_window_clip(video_path, _start_s, _end_s, *, playback_slowdown):
        slowdowns.append(playback_slowdown)
        yield video_path

    monkeypatch.setattr(opentac, "generate_json", fake_generate_json)
    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "temporary_window_clip", fake_window_clip)

    output_root = tmp_path / "opentac"
    counts = opentac.run(
        output_root, GLOSSARY, SOURCE_ROWS, GROUND_TRUTH, REPO_ROOT,
        phase="phase1_direct", provider="gemini", clip_uids=["event_clips:0006"],
    )

    assert counts["success"] == 1
    assert counts["calls"] == 4
    result = json.loads((output_root / "phase1_direct/gemini/0006.json").read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["screen"]["candidate_tactic_ids"] == ["counter-attack"]
    assert [a["tactic_id"] for a in result["assessments"]] == ["counter-attack"]
    assert result["clip_duration_s"] == 12.0
    assert result["score_samples"] == 3
    assert result["temperature"] is None
    assert len(result["assessment_samples"]) == 3
    assert result["score_playback_slowdown"] == 6.0
    assert result["assessments"][0]["evidence_spans"][0]["start_s"] == 1.5
    assert [fps for fps, _ in calls] == [None] * 4
    assert slowdowns == [6.0]


def test_run_skips_score_pass_when_screen_finds_nothing_plausible(tmp_path, monkeypatch):
    def fake_generate_json(_provider, _prompt, _schema, **_kwargs):
        payload = {
            "restart_type": "open_play", "possession_transition_s": None,
            "decisive_window": {"start_s": 0.0, "end_s": 5.0},
            "candidate_tactic_ids": [],
        }
        return payload, [], 1

    monkeypatch.setattr(opentac, "generate_json", fake_generate_json)
    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)

    output_root = tmp_path / "opentac"
    opentac.run(
        output_root, GLOSSARY, SOURCE_ROWS, GROUND_TRUTH, REPO_ROOT,
        phase="phase1_direct", provider="gemini", clip_uids=["event_clips:0006"], use_topology=False,
    )

    result = json.loads((output_root / "phase1_direct/gemini/0006.json").read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["assessments"] == []
    assert result["score_prompt_sha256"] is None
    assert result["score_fps"] is None


    assert result["topology_enabled"] is False
def test_run_skips_already_written_clips_unless_forced(tmp_path, monkeypatch):
    calls = []

    def fake_generate_json(_provider, _prompt, _schema, **_kwargs):
        calls.append(None)
        payload = {
            "restart_type": "open_play", "possession_transition_s": None,
            "decisive_window": {"start_s": 0.0, "end_s": 5.0},
            "candidate_tactic_ids": [],
        }
        return payload, [], 1

    monkeypatch.setattr(opentac, "generate_json", fake_generate_json)
    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)

    output_root = tmp_path / "opentac"
    run_kwargs = dict(
        output_root=output_root, glossary_path=GLOSSARY, source_rows=SOURCE_ROWS, ground_truth=GROUND_TRUTH,
        repo_root=REPO_ROOT, phase="phase1_direct", provider="gemini", clip_uids=["event_clips:0006"],
    )

    first = opentac.run(**run_kwargs)
    second = opentac.run(**run_kwargs)
    forced = opentac.run(**run_kwargs, force=True)

    assert (first["success"], first["skipped"]) == (1, 0)
    assert (second["success"], second["skipped"]) == (0, 1)
    assert (forced["success"], forced["skipped"]) == (1, 0)
    assert len(calls) == 2  # one screen call per non-skipped run (candidates empty, no score call)
