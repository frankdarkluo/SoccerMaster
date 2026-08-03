import csv
import json
from pathlib import Path

import pytest

from pipeline.tactics_qa import opentac
from pipeline.tactics_qa.opentac import (
    ONE_SHOT_V2_TEACHING,
    build_one_shot_manifest,
    build_one_shot_prompt,
    build_one_shot_v2_manifest,
    build_one_shot_v2_observation_prompt,
    build_one_shot_v2_nomination_prompt,
    build_one_shot_v2_video_instructions,
    build_prompt,
    load_jsonl,
    one_shot_v2_observation_schema,
    one_shot_v2_nomination_schema,
    response_schema,
    validate_evidence_bounds,
    validate_one_shot_v2_nomination,
    validate_one_shot_v2_observations,
    validate_response,
)


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


def observation():
    return {
        "sequence_id": "s1", "start_s": 1.0, "end_s": 4.0,
        "state_before_zh": "双方保持阵型", "position_or_action_zh": "白队向前传球",
        "observed_change_zh": "黑队转身", "space_origin": "existing",
        "space_evidence_zh": "两线之间已有空档", "opponent_constraint_zh": "黑队回追",
        "space_change_zh": "纵向空档被利用", "functional_effect_zh": "白队向前推进",
        "terminal_result_zh": "继续控球", "mechanism_chain_zh": "传球后黑队回追，白队利用空档推进",
        "confidence": 80, "confidence_reasons": ["过程连续可见"], "evidence_gaps": [],
    }


def test_validated_json_retries_only_local_schema_failure(monkeypatch):
    payloads = iter([{"ok": False}, {"ok": True}])

    monkeypatch.setattr(opentac, "generate_json", lambda *_args, **_kwargs: (next(payloads), [{"tokens": 1}], 1))

    def validate(payload):
        if not payload["ok"]:
            raise ValueError("invalid local schema")

    payload, usage, attempts, validation_attempts = opentac.generate_validated_json(
        "gemini", "prompt", {}, validate,
    )

    assert payload == {"ok": True}
    assert usage == [{"tokens": 1}, {"tokens": 1}]
    assert (attempts, validation_attempts) == (2, 2)


    cards = [card(f"tactic-{index}", f"战术{index}") for index in range(19)]
    first = build_prompt("clip-neutral.mp4", 12.0, cards, "phase1_direct")
    second = build_prompt("clip-neutral.mp4", 12.0, cards, "phase2_observation_first")
    assert "ALL 19 P0 concept cards" in first
    assert "prior labels" in first
    assert "supporting_sequence_ids" not in response_schema([c["tactic_id"] for c in cards], "phase1_direct")["properties"]["candidates"]["items"]["properties"]
    assert "Before considering tactic names" in second
    assert "observations" in response_schema([c["tactic_id"] for c in cards], "phase2_observation_first")["properties"]


def test_phase2_observations_cannot_leak_tactic_name():
    cards = {"counter-attack": card()}
    payload = {
        "observations": [{
            "sequence_id": "s1", "start_s": 0.0, "end_s": 2.0,
            "action_zh": "这是快速反击", "defensive_change_zh": "回撤",
            "space_change_or_use_zh": "利用纵深", "attacking_effect_zh": "推进",
            "terminal_result_zh": "继续控球",
        }],
        "candidates": [],
        "no_nomination_reason_zh": "无候选",
    }
    with pytest.raises(ValueError, match="leaks tactic name"):
        validate_response(payload, list(cards), cards, "phase2_observation_first")


def test_one_shot_manifest_has_frozen_target_counts():
    root = Path(__file__).resolve().parents[2]
    claims = load_jsonl(
        root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl"
    )

    manifest = build_one_shot_manifest(claims)

    assert len(manifest) == 49
    assert sum(len(item["claims"]) for item in manifest) == 50
    assert {
        tactic_id: (
            sum(len(item["claims"]) for item in manifest if item["tactic_id"] == tactic_id),
            sum(item["tactic_id"] == tactic_id for item in manifest),
        )
        for tactic_id in {item["tactic_id"] for item in manifest}
    } == {
        "counter-attack": (24, 23),
        "cutback": (3, 3),
        "line-breaking-pass": (9, 9),
        "run-in-behind": (14, 14),
    }
    sngs118 = next(
        item for item in manifest
        if item["tactic_id"] == "counter-attack" and item["clip_uid"].endswith("SNGS-118")
    )
    assert {claim["gt_verdict"] for claim in sngs118["claims"]} == {"present", "absent"}


def test_one_shot_v2_manifest_has_frozen_positive_and_negative_counts():
    root = Path(__file__).resolve().parents[2]
    claims = load_jsonl(
        root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl"
    )

    manifest = build_one_shot_v2_manifest(claims)

    assert len(manifest) == 23
    assert {
        tactic_id: {
            verdict: sum(
                claim["gt_verdict"] == verdict
                for item in manifest if item["tactic_id"] == tactic_id
                for claim in item["claims"]
            )
            for verdict in ("present", "absent")
        }
        for tactic_id in {item["tactic_id"] for item in manifest}
    } == {
        "line-breaking-pass": {"present": 3, "absent": 6},
        "run-in-behind": {"present": 10, "absent": 4},
    }


def test_one_shot_v2_observation_is_query_only_mechanism_analysis():
    prompt = build_one_shot_v2_observation_prompt("clip-deadbeef.mp4", 30.0)
    fields = one_shot_v2_observation_schema()["properties"]["observations"]["items"]["properties"]

    assert "position_or_action_zh" in fields
    assert "observed_change_zh" in fields
    assert "space_origin" in fields
    assert "opponent_constraint_zh" in fields
    assert "functional_effect_zh" in fields
    assert "terminal_result_zh" in fields
    assert "mechanism_chain_zh" in fields
    assert "evidence_gaps" in fields
    assert "P0 CONCEPT CARDS" not in prompt
    assert "run-in-behind" not in prompt
    assert "line-breaking-pass" not in prompt
    assert "gt_verdict" not in prompt
    assert "clip-deadbeef.mp4" in prompt


def test_one_shot_v2_observation_validation_blocks_leaks_and_bad_time():
    cards = {
        "run-in-behind": card("run-in-behind", "打身后 / 反越位跑位"),
    }
    payload = {"observations": [observation()]}

    assert validate_one_shot_v2_observations(payload, cards, 12.0) is payload

    payload["observations"][0]["position_or_action_zh"] = "这是run-in-behind"
    with pytest.raises(ValueError, match="leaks tactic name"):
        validate_one_shot_v2_observations(payload, cards, 12.0)

    payload["observations"][0] = {**observation(), "end_s": 13.0}
    with pytest.raises(ValueError, match="query bounds"):
        validate_one_shot_v2_observations(payload, cards, 12.0)


def test_one_shot_v2_nomination_uses_teaching_observations_and_all_cards():
    cards = [card(f"tactic-{index}", f"战术{index}") for index in range(19)]
    cards[0] = card("run-in-behind", "打身后 / 反越位跑位")
    observations = [observation()]

    instructions = build_one_shot_v2_video_instructions(
        cards[0], ONE_SHOT_V2_TEACHING["run-in-behind"], "clip-deadbeef.mp4",
    )
    prompt = build_one_shot_v2_nomination_prompt(
        "clip-deadbeef.mp4", 12.0, cards, observations,
    )
    candidate_fields = one_shot_v2_nomination_schema(
        [item["tactic_id"] for item in cards]
    )["properties"]["candidates"]["items"]["properties"]

    assert "supporting_sequence_ids" in candidate_fields
    assert "reason_zh" not in candidate_fields
    assert set(one_shot_v2_nomination_schema([item["tactic_id"] for item in cards])["properties"]) == {"candidates"}
    assert set(candidate_fields["evidence_spans"]["items"]["properties"]) == {"start_s", "end_s"}
    assert "3–8s" in instructions[0]
    assert "ordinary forward support" in instructions[0]
    assert instructions[1].startswith("QUERY VIDEO")
    assert '"sequence_id": "s1"' in prompt
    assert "ALL 19 P0 concept cards" in prompt
    assert "SNGS" not in prompt
    assert "gt_verdict" not in prompt


def test_one_shot_v2_nomination_evidence_is_locked_to_cited_observation():
    cards = {"run-in-behind": card("run-in-behind", "打身后 / 反越位跑位")}
    observations = [observation()]
    payload = {
        "candidates": [{
            "rank": 3, "tactic_id": "run-in-behind", "confidence": 80,
            "matched_cues": ["cue"], "supporting_sequence_ids": ["s1"],
            "evidence_spans": [{"start_s": 1.5, "end_s": 3.0}],
        }],
    }

    assert validate_one_shot_v2_nomination(payload, cards, observations, 12.0) is payload

    payload["candidates"][0]["evidence_spans"][0]["start_s"] = 0.5
    with pytest.raises(ValueError, match="locked observation"):
        validate_one_shot_v2_nomination(payload, cards, observations, 12.0)

    payload["candidates"][0]["evidence_spans"][0]["start_s"] = 1.5
    payload["candidates"][0]["supporting_sequence_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown observation"):
        validate_one_shot_v2_nomination(payload, cards, observations, 12.0)


def test_run_one_shot_v2_locks_observation_then_matches_and_resumes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    calls = []

    def fake_generate(provider, prompt, _schema, **kwargs):
        calls.append((provider, prompt, kwargs))
        if "video_path" in kwargs:
            return {"observations": [observation()]}, [], 1
        return {
            "candidates": [{
                "rank": 1, "tactic_id": "run-in-behind", "confidence": 80,
                "matched_cues": ["1. 跑动者起步时位于最后一线身前或与其平齐"],
                "supporting_sequence_ids": ["s1"],
                "evidence_spans": [{"start_s": 1.5, "end_s": 3.0}],
            }],
        }, [], 1

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(
        opentac, "sha256",
        lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64,
    )
    monkeypatch.setattr(opentac, "generate_json", fake_generate)
    common = {
        "output_root": tmp_path,
        "glossary_path": root / "data/足球战术数据库_词条表_Grid.csv",
        "source_rows": root / "benchmark/tactical_prototypes/source_rows.jsonl",
        "ground_truth": root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl",
        "repo_root": root,
        "provider": "gemini",
        "tactic_ids": ["run-in-behind"],
        "clip_uids": ["event_clips:0013"],
    }

    first = opentac.run_one_shot_v2(**common)
    second = opentac.run_one_shot_v2(**common)

    assert first["observation_calls"] == first["nomination_calls"] == first["success"] == 1
    assert second["skipped"] == 1
    assert len(calls) == 2
    assert calls[0][2]["video_path"].name.startswith("0013_")
    assert "P0 CONCEPT CARDS" not in calls[0][1]
    assert "run-in-behind" not in calls[0][1]
    assert calls[1][2]["video_paths"][0].name.startswith("Gakpo")
    assert calls[1][2]["video_paths"][1].name.startswith("0013_")
    assert calls[1][2]["video_instructions"][0].startswith("REFERENCE VIDEO")
    assert calls[1][2]["video_instructions"][1].startswith("QUERY VIDEO")
    target = tmp_path / "video_one_shot_v2/gemini/run-in-behind/0013.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert record["observations"] == [observation()]
    assert record["calls"]["observation"]["status"] == "success"
    assert record["calls"]["nomination"]["status"] == "success"


def test_one_shot_prompt_labels_only_the_first_video():
    cards = [card(f"tactic-{index}", f"战术{index}") for index in range(19)]
    cards[0] = card()

    prompt = build_one_shot_prompt(
        "clip-deadbeef.mp4", 12.0, cards, cards[0],
    )

    assert "FIRST video" in prompt
    assert "confirmed positive example of counter-attack" in prompt
    assert "SECOND video" in prompt
    assert "evidence spans must refer only to the SECOND video" in prompt
    assert "SNGS" not in prompt
    assert "gt_verdict" not in prompt
    assert "ALL 19 P0 concept cards" in prompt


def test_one_shot_evidence_must_fit_query_video():
    payload = {
        "candidates": [{
            "rank": 1,
            "tactic_id": "counter-attack",
            "confidence": 80,
            "reason_zh": "理由",
            "matched_cues": ["cue"],
            "evidence_spans": [{
                "start_s": 10.0,
                "end_s": 13.0,
                "visible_movement_zh": "动作",
                "tactical_link_zh": "联系",
            }],
        }],
        "no_nomination_reason_zh": "",
    }

    with pytest.raises(ValueError, match="query bounds"):
        validate_evidence_bounds(payload, 12.0)


def test_run_one_shot_uploads_example_then_query_and_resumes(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    calls = []

    def fake_generate(_provider, prompt, _schema, **kwargs):
        calls.append((prompt, kwargs["video_paths"]))
        payload = {
            "candidates": [{
                "rank": 1,
                "tactic_id": "cutback",
                "confidence": 80,
                "reason_zh": "理由",
                "matched_cues": ["cue"],
                "evidence_spans": [{
                    "start_s": 0.0, "end_s": 1.0,
                    "visible_movement_zh": "动作", "tactical_link_zh": "联系",
                }],
            }],
            "no_nomination_reason_zh": "",
        }
        return payload, [], 1

    monkeypatch.setattr(opentac, "duration", lambda _path: 12.0)
    monkeypatch.setattr(opentac, "sha256", lambda path: ("e" if "FIFA Game Library" in str(path) else "q") * 64)
    monkeypatch.setattr(opentac, "generate_json", fake_generate)

    common = {
        "output_root": tmp_path,
        "glossary_path": root / "data/足球战术数据库_词条表_Grid.csv",
        "source_rows": root / "benchmark/tactical_prototypes/source_rows.jsonl",
        "ground_truth": root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl",
        "repo_root": root,
        "provider": "gemini",
        "tactic_ids": ["cutback"],
        "clip_uids": ["soccernetgs:SNGS-031"],
    }
    first = opentac.run_one_shot(**common)
    second = opentac.run_one_shot(**common)

    assert first["calls"] == first["success"] == 1
    assert second["skipped"] == 1
    prompt, paths = calls[0]
    assert "Bellingham" not in prompt
    assert "Cut-back cross" not in prompt
    assert paths[0].name.startswith("Bellingham")
    assert paths[1].name == "SNGS-031.mp4"
    assert "SNGS-031" not in prompt
    target = tmp_path / "phase1_video_one_shot/gemini/cutback/SNGS-031.json"
    assert json.loads(target.read_text(encoding="utf-8"))["query_sha256"] == "q" * 64


def test_metrics_can_score_runs_keyed_by_tactic_and_clip():
    claims = [
        {"clip_uid": "clip", "tactic_id": "counter-attack", "gt_verdict": "present", "window": {"start_s": 0, "end_s": 1}},
        {"clip_uid": "clip", "tactic_id": "cutback", "gt_verdict": "absent", "window": {"start_s": 0, "end_s": 1}},
    ]
    candidate = {"tactic_id": "counter-attack", "evidence_spans": [{"start_s": 0, "end_s": 1}]}
    runs = {
        ("counter-attack", "clip"): {"candidates": [candidate]},
        ("cutback", "clip"): {"candidates": [candidate]},
    }

    result = opentac.metrics(claims, runs, 1, run_key=lambda claim: (claim["tactic_id"], claim["clip_uid"]))

    assert result["accuracy"] == 1.0


def test_confidence_accepts_any_integer_allowed_by_schema():
    cards = {"counter-attack": card()}
    payload = {
        "candidates": [{
            "rank": 1,
            "tactic_id": "counter-attack",
            "confidence": 85,
            "reason_zh": "理由",
            "matched_cues": ["cue"],
            "evidence_spans": [{"start_s": 0, "end_s": 1, "visible_movement_zh": "动作", "tactical_link_zh": "联系"}],
        }],
        "no_nomination_reason_zh": "",
    }

    assert validate_response(payload, list(cards), cards, "phase1_direct") is payload

    payload["no_nomination_reason_zh"] = "补充说明"
    assert validate_response(payload, list(cards), cards, "phase1_direct") is payload


def test_report_shot_comparison_v2_is_complete_auditable_and_strict(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    glossary = root / "data/足球战术数据库_词条表_Grid.csv"
    ground_truth = root / "outputs/tactical_claim_benchmark/opentac/evaluation/ground_truth.jsonl"
    cards = {
        key: value
        for key, value in opentac.load_glossary(glossary).items()
        if value["priority"] == "P0"
    }
    claims = sorted(
        (
            claim for claim in load_jsonl(ground_truth)
            if claim.get("score_set") == "primary"
            and claim.get("tactic_id") in opentac.ONE_SHOT_V2_TACTICS
        ),
        key=lambda claim: (claim["tactic_id"], claim["clip_uid"]),
    )
    assert len(claims) == 23

    hits = set()
    for tactic_id, positive_hits, negative_hits in (
        ("line-breaking-pass", 2, 0),
        ("run-in-behind", 3, 1),
    ):
        selected = [claim for claim in claims if claim["tactic_id"] == tactic_id]
        hits.update(
            (claim["tactic_id"], claim["clip_uid"])
            for claim in [
                *[claim for claim in selected if claim["gt_verdict"] == "present"][:positive_hits],
                *[claim for claim in selected if claim["gt_verdict"] == "absent"][:negative_hits],
            ]
        )

    for provider in ("gemini", "doubao"):
        for claim in claims:
            tactic_id, clip_uid = claim["tactic_id"], claim["clip_uid"]
            suffix = clip_uid.split(":", 1)[-1]
            start_s, end_s = claim["window"]["start_s"], claim["window"]["end_s"]
            duration_s = float(end_s) + 1.0
            empty = {
                "status": "success",
                "clip_uid": clip_uid,
                "candidates": [],
                "no_nomination_reason_zh": "无可支持候选",
            }
            opentac.write_json(tmp_path / "phase1_direct" / provider / f"{suffix}.json", empty)
            if provider == "gemini":
                opentac.write_json(
                    tmp_path / "phase1_video_one_shot" / provider / tactic_id / f"{suffix}.json",
                    {
                        **empty,
                        "target_tactic_id": tactic_id,
                        "query_duration_s": duration_s,
                    },
                )

            observed = {**observation(), "start_s": start_s, "end_s": end_s}
            candidates = []
            if (tactic_id, clip_uid) in hits:
                cue = (cards[tactic_id]["observable_cues"] + cards[tactic_id]["triggers"])[0]
                candidates = [{
                    "rank": 1,
                    "tactic_id": tactic_id,
                    "confidence": 80,
                    "matched_cues": [cue],
                    "supporting_sequence_ids": ["s1"],
                    "evidence_spans": [{"start_s": start_s, "end_s": end_s}],
                }]
            opentac.write_json(
                tmp_path / "video_one_shot_v2" / provider / tactic_id / f"{suffix}.json",
                {
                    "status": "success",
                    "target_tactic_id": tactic_id,
                    "clip_uid": clip_uid,
                    "query_duration_s": duration_s,
                    "observations": [observed],
                    "candidates": candidates,
                },
            )

    ignored_v1 = tmp_path / "phase1_video_one_shot/doubao/run-in-behind/ignored.json"
    ignored_v1.parent.mkdir(parents=True)
    ignored_v1.write_text("{", encoding="utf-8")
    monkeypatch.setattr(opentac, "_validate_one_shot_v2_metadata", lambda *_: None)

    summary = opentac.report_shot_comparison_v2(tmp_path, glossary, ground_truth)

    assert summary["expected_claims"] == summary["expected_pairs"] == 23
    for provider in ("gemini", "doubao"):
        provider_value = summary["providers"][provider]
        overall = provider_value["overall"]
        assert overall["zero_shot"] == {
            "valid_claims": 23,
            "expected_claims": 23,
            "top1_accuracy": pytest.approx(10 / 23),
            "top3_accuracy": pytest.approx(10 / 23),
        }
        assert overall["video_one_shot_v2"]["valid_claims"] == 23
        assert overall["video_one_shot_v2"]["top1_accuracy"] == pytest.approx(14 / 23)
        assert overall["video_one_shot_v2"]["top3_accuracy"] == pytest.approx(14 / 23)
        for tactic_id, expected in (("line-breaking-pass", 9), ("run-in-behind", 14)):
            scope = provider_value["tactics"][tactic_id]
            assert scope["zero_shot"]["valid_claims"] == expected
            assert scope["video_one_shot_v2"]["valid_claims"] == expected
        assert provider_value["top1_gates"]["line-breaking-pass"] == {
            "valid_claims": 9,
            "expected_claims": 9,
            "true_positives": 2,
            "positive_claims": 3,
            "required_true_positives": 2,
            "false_positives": 0,
            "negative_claims": 6,
            "allowed_false_positives": 0,
            "passed": True,
        }
        assert provider_value["top1_gates"]["run-in-behind"] == {
            "valid_claims": 14,
            "expected_claims": 14,
            "true_positives": 3,
            "positive_claims": 10,
            "required_true_positives": 3,
            "false_positives": 1,
            "negative_claims": 4,
            "allowed_false_positives": 1,
            "passed": True,
        }

    gemini = summary["providers"]["gemini"]["overall"]
    assert gemini["video_one_shot_v1"] == gemini["zero_shot"]
    assert gemini["delta_v2_minus_v1"]["top1"] == pytest.approx(4 / 23)
    doubao = summary["providers"]["doubao"]["overall"]
    assert doubao["video_one_shot_v1"] is None
    assert doubao["delta_v2_minus_zero"]["top1"] == pytest.approx(4 / 23)

    evaluation = tmp_path / "evaluation/shot_comparison_v2"
    assert {path.name for path in evaluation.iterdir()} == {
        "manifest.jsonl", "per_claim.csv", "summary.json", "summary.md",
    }
    assert len(load_jsonl(evaluation / "manifest.jsonl")) == 23
    with (evaluation / "per_claim.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 46
    assert reader.fieldnames == [
        "provider", "tactic_id", "claim_id", "clip_uid", "gt_verdict",
        "window_start_s", "window_end_s",
        "zero_shot_status", "zero_shot_top1_hit", "zero_shot_top3_hit",
        "video_one_shot_v1_status", "video_one_shot_v1_top1_hit", "video_one_shot_v1_top3_hit",
        "video_one_shot_v2_status", "video_one_shot_v2_top1_hit", "video_one_shot_v2_top3_hit",
    ]
    assert all(row["video_one_shot_v1_status"] == "" for row in rows if row["provider"] == "doubao")

    broken_claim = next(
        claim for claim in claims
        if claim["tactic_id"] == "line-breaking-pass"
        and claim["gt_verdict"] == "present"
        and (claim["tactic_id"], claim["clip_uid"]) in hits
    )
    broken_path = (
        tmp_path / "video_one_shot_v2/gemini" / broken_claim["tactic_id"]
        / f"{broken_claim['clip_uid'].split(':', 1)[-1]}.json"
    )
    original = json.loads(broken_path.read_text(encoding="utf-8"))
    for section in ("observations", "candidates"):
        broken = json.loads(json.dumps(original))
        broken[section][0]["unexpected"] = True
        opentac.write_json(broken_path, broken)

        partial = opentac.report_shot_comparison_v2(tmp_path, glossary, ground_truth)
        condition = partial["providers"]["gemini"]["overall"]["video_one_shot_v2"]
        assert condition["valid_claims"] == 22
        assert condition["expected_claims"] == 23
        assert condition["top1_accuracy"] is None
        assert condition["top3_accuracy"] is None
        assert partial["providers"]["gemini"]["overall"]["delta_v2_minus_v1"] == {
            "top1": None,
            "top3": None,
        }
        assert partial["providers"]["gemini"]["top1_gates"]["line-breaking-pass"]["passed"] is None
        with (evaluation / "per_claim.csv").open(encoding="utf-8-sig", newline="") as handle:
            broken_row = next(
                row for row in csv.DictReader(handle)
                if row["provider"] == "gemini" and row["claim_id"] == broken_claim["claim_id"]
            )
        assert broken_row["video_one_shot_v2_status"] == "invalid"
        assert broken_row["video_one_shot_v2_top1_hit"] == ""
