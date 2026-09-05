#!/usr/bin/env python3
"""Offline tests for the GSR-2025 multi-agent pilot."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pilot


def spatial(*facts: str, progression: str = "no_clear_progression", change: float | None = None) -> dict:
    present = set(facts)
    return {
        "possession_change": change,
        "ball_progression": progression,
        "key_spatial_facts": {fact: ("yes" if fact in present else "no") for fact in pilot.SPATIAL_FACTS},
    }


def temporal(*events, defence: str = "no_visible_change") -> dict:
    """Each event is (time, action) or (time, action, actor); actor defaults to A1."""
    return {
        "events": [
            {"time": e[0], "action": e[1], "actor": e[2] if len(e) > 2 else "A1"}
            for e in events
        ],
        "defensive_change": defence,
    }


FIXTURES = {
    "counter-attack": (
        spatial("multiple_attackers_run_forward", "defenders_not_set", progression="defensive_third -> final_third", change=5.0),
        temporal((5.0, "interception"), (6.0, "forward_dribble"), (15.0, "shot"), defence="retreat_toward_own_goal"),
    ),
    "long-ball": (
        spatial("forward_target_or_landing_space_beyond_midfield", "teammates_positioned_for_second_ball", progression="defensive_third -> final_third"),
        temporal((5.0, "forward_aerial_pass"), (10.0, "header"), (11.0, "duel"), defence="hold_shape"),
    ),
    "one-two": (
        spatial("first_passer_runs_beyond_nearby_defender", progression="middle_third -> final_third"),
        temporal((5.0, "forward_ground_pass"), (6.5, "one_touch_return_pass"), (8.0, "reception")),
    ),
    "tactical-foul": (
        spatial("contact_outside_defending_penalty_area", "defenders_not_set", progression="within_middle_third", change=10.0),
        temporal((10.0, "possession_gain"), (11.0, "forward_dribble"), (14.0, "pull_or_block"), (14.5, "whistle"), defence="retreat_toward_own_goal"),
    ),
    "cutback": (
        spatial("ball_carrier_reaches_right_byline", "backward_pass_target_in_central_penalty_area", progression="within_final_third"),
        temporal((5.0, "forward_dribble"), (7.0, "backward_ground_pass"), (10.0, "reception"), defence="collapse_toward_ball"),
    ),
    "run-in-behind": (
        spatial("runner_moves_from_last_line_front_to_behind", "pass_path_crosses_defensive_line", progression="middle_third -> final_third"),
        temporal((10.0, "forward_ground_pass"), (13.0, "reception"), defence="retreat_toward_own_goal"),
    ),
    "line-breaking-pass": (
        spatial("pass_path_crosses_defensive_line", "receiver_controls_between_or_behind_lines", progression="middle_third -> final_third"),
        temporal((10.0, "forward_ground_pass"), (13.0, "reception")),
    ),
    "back-line-possession-management": (
        spatial("ball_remains_in_own_half", "ball_circulates_across_back_line", progression="within_defensive_third"),
        temporal((0.0, "lateral_ground_pass"), (5.0, "backward_ground_pass"), (12.0, "lateral_ground_pass"), defence="hold_shape"),
    ),
}


def fake_generator_with(a: dict, b: dict, c: dict | None = None, calls: list | None = None):
    lock = threading.Lock()
    c = c or {"label": None, "rationale": "no candidate matches", "objections": []}

    def generate(_provider, prompt, schema, **kwargs):
        role = "A" if schema is pilot.A_SCHEMA else "B" if schema is pilot.B_SCHEMA else "C"
        with lock:
            if role == "C" and calls is not None:
                assert {item[0] for item in calls} >= {"A", "B"}
            if calls is not None:
                calls.append((role, prompt, kwargs))
        if role in {"A", "B"}:
            time.sleep(0.01)
        return copy.deepcopy({"A": a, "B": b, "C": c}[role]), [{"tokens": 1}], 1

    return generate


class SchemaTests(unittest.TestCase):
    def test_compact_valid_outputs_allow_empty_evidence(self):
        self.assertEqual(pilot.validate_a(spatial()), spatial())
        self.assertEqual(pilot.validate_b(temporal())["events"], [])
        self.assertIsNone(pilot.validate_c({"label": None, "rationale": "none fit", "objections": []})["label"])

    def test_schema_rejects_extra_fields_bad_precision_and_bad_order(self):
        with self.assertRaisesRegex(ValueError, "strict"):
            pilot.validate_a({**spatial(), "extra": 1})
        with self.assertRaisesRegex(ValueError, "0.5-second"):
            pilot.validate_a(spatial(change=1.25))
        with self.assertRaisesRegex(ValueError, "time ordered"):
            pilot.validate_b(temporal((2.0, "shot"), (1.0, "reception")))

    def test_c_requires_concrete_paths_and_a_rationale(self):
        with self.assertRaisesRegex(ValueError, "concrete"):
            pilot.validate_c({"label": None, "rationale": "x", "objections": ["evidence is weak"]})
        with self.assertRaisesRegex(ValueError, "rationale"):
            pilot.validate_c({"label": None, "rationale": "   ", "objections": []})
        value = {"label": "one-two", "rationale": "wall pass is visible", "objections": ["B.events[1].action: movement is lateral"]}
        self.assertEqual(pilot.validate_c(value), value)

    def test_c_accepts_full_width_colon(self):
        value = {"label": None, "rationale": "都不成立", "objections": ["A.possession_change：该时刻球权未转换"]}
        self.assertEqual(pilot.validate_c(value), value)

    def test_c_label_must_come_from_the_offered_candidates(self):
        payload = {"label": "cutback", "rationale": "byline cross", "objections": []}
        self.assertEqual(pilot.validate_c(payload, ["cutback", "long-ball"]), payload)
        with self.assertRaisesRegex(ValueError, "candidates"):
            pilot.validate_c(payload, ["long-ball", "counter-attack"])

    def test_a_spatial_facts_are_a_complete_ternary_map(self):
        value = spatial("multiple_attackers_run_forward")
        self.assertEqual(pilot.validate_a(value), value)
        missing = copy.deepcopy(value)
        missing["key_spatial_facts"].pop("ball_remains_in_own_half")
        with self.assertRaisesRegex(ValueError, "key_spatial_facts"):
            pilot.validate_a(missing)
        bad = copy.deepcopy(value)
        bad["key_spatial_facts"]["ball_remains_in_own_half"] = "maybe"
        with self.assertRaisesRegex(ValueError, "key_spatial_facts"):
            pilot.validate_a(bad)


class ActorTests(unittest.TestCase):
    def test_one_two_is_derived_from_actor_identity_without_the_token(self):
        """The r2_pro failure mode: B describes a wall pass but never emits one_touch_return_pass."""
        a, _ = FIXTURES["one-two"]
        b = temporal(
            (5.0, "forward_ground_pass", "A1"),
            (6.5, "forward_ground_pass", "A2"),
            (8.0, "reception", "A1"),
        )
        self.assertNotIn("one_touch_return_pass", [e["action"] for e in b["events"]])
        result = dict(zip(pilot.TACTICS, pilot.RULES))["one-two"](a, b)
        self.assertTrue(result["detail"]["hard"]["pass_then_return"])

    def test_weak_return_is_accepted_but_scores_below_a_confirmed_wall_pass(self):
        """Codex describes X->Y then attributes the continuation to Y; accept it, but rank it lower."""
        a, _ = FIXTURES["one-two"]
        rule = dict(zip(pilot.TACTICS, pilot.RULES))["one-two"]
        confirmed = rule(a, temporal((5.0, "forward_ground_pass", "A1"),
                                     (6.0, "forward_ground_pass", "A2"),
                                     (7.0, "reception", "A1")))
        weak = rule(a, temporal((5.0, "forward_ground_pass", "A1"),
                                (6.0, "forward_ground_pass", "A2"),
                                (7.0, "forward_dribble", "A2")))
        self.assertTrue(confirmed["detail"]["hard"]["pass_then_return"])
        self.assertTrue(weak["detail"]["hard"]["pass_then_return"])
        self.assertTrue(confirmed["detail"]["soft"]["confirmed_return_to_passer"]["matched"])
        self.assertFalse(weak["detail"]["soft"]["confirmed_return_to_passer"]["matched"])
        self.assertGreater(confirmed["score"], weak["score"])

    def test_a_slow_second_pass_is_not_a_wall_pass(self):
        a, _ = FIXTURES["one-two"]
        slow = temporal((5.0, "forward_ground_pass", "A1"), (20.0, "forward_ground_pass", "A2"))
        result = dict(zip(pilot.TACTICS, pilot.RULES))["one-two"](a, slow)
        self.assertFalse(result["detail"]["hard"]["pass_then_return"])

    def test_same_actor_passing_twice_is_not_a_one_two(self):
        a, _ = FIXTURES["one-two"]
        b = temporal((5.0, "forward_ground_pass", "A1"), (6.5, "forward_ground_pass", "A1"), (8.0, "reception", "A1"))
        result = dict(zip(pilot.TACTICS, pilot.RULES))["one-two"](a, b)
        self.assertFalse(result["detail"]["hard"]["pass_then_return"])

    def test_actor_is_required_and_validated(self):
        bad = temporal((1.0, "shot"))
        bad["events"][0]["actor"] = "Z9"
        with self.assertRaisesRegex(ValueError, "actor"):
            pilot.validate_b(bad)


class ScoringTests(unittest.TestCase):
    def test_missing_gate_lowers_the_score_instead_of_zeroing_it(self):
        """P0: gates are weighted terms now, so a near miss still ranks above an unrelated tactic."""
        a, b = FIXTURES["long-ball"]
        rule = dict(zip(pilot.TACTICS, pilot.RULES))["long-ball"]
        full = rule(a, b)["score"]
        broken = copy.deepcopy(a)
        broken["key_spatial_facts"]["forward_target_or_landing_space_beyond_midfield"] = "no"
        partial = rule(broken, b)
        self.assertFalse(partial["hard_pass"])
        self.assertGreater(partial["score"], 0.0)
        self.assertLess(partial["score"], full)

    def test_scores_are_normalised_so_tactics_stay_comparable(self):
        """Every rule maps onto the same 0..1 range, otherwise cross-tactic ranking is meaningless."""
        for tactic, (a, b) in FIXTURES.items():
            with self.subTest(tactic=tactic):
                score = dict(zip(pilot.TACTICS, pilot.RULES))[tactic](a, b)["score"]
                self.assertGreaterEqual(score, pilot.MIN_SCORE)
                self.assertLessEqual(score, 1.0)
        empty = dict(zip(pilot.TACTICS, pilot.RULES))["cutback"](spatial(), temporal())
        self.assertEqual(empty["score"], 0.0)


class RuleTests(unittest.TestCase):
    def test_all_eight_rules_accept_their_canonical_fixture(self):
        rules = dict(zip(pilot.TACTICS, pilot.RULES))
        for tactic, (a, b) in FIXTURES.items():
            with self.subTest(tactic=tactic):
                result = rules[tactic](a, b)
                self.assertTrue(result["hard_pass"])
                self.assertGreaterEqual(result["score"], pilot.MIN_SCORE)
                if tactic != "back-line-possession-management":
                    self.assertIsNotNone(result["interval"])

    def test_each_rule_rejects_a_missing_hard_gate(self):
        mutations = {
            "counter-attack": lambda a, b: a.update(possession_change=None),
            "long-ball": lambda a, b: a["key_spatial_facts"].update(forward_target_or_landing_space_beyond_midfield="no"),
            "one-two": lambda a, b: a["key_spatial_facts"].update(first_passer_runs_beyond_nearby_defender="no"),
            "tactical-foul": lambda a, b: a["key_spatial_facts"].update(contact_outside_defending_penalty_area="no"),
            "cutback": lambda a, b: a["key_spatial_facts"].update(ball_carrier_reaches_right_byline="no"),
            "run-in-behind": lambda a, b: a["key_spatial_facts"].update(runner_moves_from_last_line_front_to_behind="no"),
            "line-breaking-pass": lambda a, b: a["key_spatial_facts"].update(pass_path_crosses_defensive_line="no"),
            "back-line-possession-management": lambda a, b: b["events"].pop(),
        }
        rules = dict(zip(pilot.TACTICS, pilot.RULES))
        for tactic, (source_a, source_b) in FIXTURES.items():
            with self.subTest(tactic=tactic):
                a, b = copy.deepcopy(source_a), copy.deepcopy(source_b)
                mutations[tactic](a, b)
                self.assertFalse(rules[tactic](a, b)["hard_pass"])

    def test_tie_is_reported_and_no_match_is_not_guessed(self):
        a = spatial(
            "runner_moves_from_last_line_front_to_behind",
            "pass_path_crosses_defensive_line",
            "receiver_controls_between_or_behind_lines",
            progression="middle_third -> final_third",
        )
        b = temporal((10.0, "forward_ground_pass"), (13.0, "reception"), defence="retreat_toward_own_goal")
        tied = pilot.classify(a, b)
        self.assertEqual(tied["reason"], "tie")
        self.assertEqual(set(tied["tied"]), {"run-in-behind", "line-breaking-pass"})
        empty = pilot.classify(spatial(), temporal())
        self.assertEqual(empty["reason"], "no_match")
        self.assertIsNone(empty["label"])

    def test_back_line_has_no_interval(self):
        result = pilot.classify(*FIXTURES["back-line-possession-management"])
        self.assertEqual(result["label"], "back-line-possession-management")
        self.assertIsNone(result["interval"])

    def test_not_observable_does_not_satisfy_a_hard_gate(self):
        a, b = copy.deepcopy(FIXTURES["one-two"])
        a["key_spatial_facts"]["first_passer_runs_beyond_nearby_defender"] = "not_observable"
        result = dict(zip(pilot.TACTICS, pilot.RULES))["one-two"](a, b)
        self.assertFalse(result["hard_pass"])


class InputAndRunTests(unittest.TestCase):
    def test_authoritative_gt_resolves_all_37_videos(self):
        rows = pilot.load_gt()
        self.assertEqual(len(rows), 37)
        self.assertEqual(len({row["clip"] for row in rows}), 37)
        self.assertTrue(all(row["video_path"].parent == pilot.VIDEO_ROOT for row in rows))

    def test_missing_video_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing video"):
                pilot.load_gt(video_root=Path(directory))

    def test_a_and_b_are_independent_and_the_arbiter_runs_after_the_rule(self):
        calls = []
        a, b = FIXTURES["counter-attack"]
        verdict = {"label": "counter-attack", "rationale": "turnover then direct attack",
                   "objections": ["B.events[1].action: lateral"]}
        generator = fake_generator_with(a, b, verdict, calls)
        result = pilot.run_clip(pilot.VIDEO_ROOT / "SNGS-010.mp4", pilot.load_criteria(), generator)
        self.assertEqual(result["rule"]["label"], "counter-attack")
        self.assertEqual(result["final"], {**result["final"], "label": "counter-attack", "source": "arbiter"})
        self.assertEqual({role for role, _, _ in calls[:2]}, {"A", "B"})
        self.assertEqual(calls[-1][0], "C")
        for role, prompt, _ in calls[:2]:
            self.assertNotIn("counter-attack", prompt, role)
            self.assertNotIn("00:08-00:18", prompt, role)

    def test_c_failure_does_not_discard_the_prediction(self):
        a, b = FIXTURES["counter-attack"]
        base = fake_generator_with(a, b)

        def generator(provider, prompt, schema, **kwargs):
            if schema is pilot.C_SCHEMA:
                raise ValueError("C objections must target a concrete A.* or B.* field")
            return base(provider, prompt, schema, **kwargs)

        result = pilot.run_clip(pilot.VIDEO_ROOT / "SNGS-010.mp4", pilot.load_criteria(), generator)
        self.assertEqual(result["rule"]["label"], "counter-attack")
        self.assertEqual(result["agents"]["C"]["error_type"], "ValueError")
        self.assertNotIn("output", result["agents"]["C"])

    def test_arbiter_failure_falls_back_to_the_rule_label(self):
        a, b = FIXTURES["counter-attack"]
        base = fake_generator_with(a, b)

        def generator(provider, prompt, schema, **kwargs):
            if schema is pilot.C_SCHEMA:
                raise ValueError("C label must be one of the offered candidates or null")
            return base(provider, prompt, schema, **kwargs)

        result = pilot.run_clip(pilot.VIDEO_ROOT / "SNGS-010.mp4", pilot.load_criteria(), generator)
        self.assertEqual(result["final"]["label"], "counter-attack")
        self.assertEqual(result["final"]["source"], "rule")
        self.assertEqual(result["agents"]["C"]["error_type"], "ValueError")

    def test_arbiter_rejection_is_recorded_without_losing_the_rule_label(self):
        a, b = FIXTURES["counter-attack"]
        verdict = {"label": None, "rationale": "no candidate is supported by the footage", "objections": []}
        result = pilot.run_clip(pilot.VIDEO_ROOT / "SNGS-010.mp4", pilot.load_criteria(),
                                fake_generator_with(a, b, verdict))
        self.assertEqual(result["rule"]["label"], "counter-attack")
        self.assertIsNone(result["final"]["label"])
        self.assertEqual(result["final"]["source"], "arbiter")

    def test_manifest_freezes_criteria(self):
        criteria = pilot.load_criteria()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "r0"
            pilot.ensure_manifest(run_dir, "r0", criteria)
            changed = copy.deepcopy(criteria)
            changed["counter-attack"]["反例① 描述"] += " changed"
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                pilot.ensure_manifest(run_dir, "r0", changed)

    def test_run_resumes_and_retries_only_when_requested(self):
        good = fake_generator_with(*FIXTURES["counter-attack"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = pilot.run_pilot("r0", clip_ids=["SNGS-010"], generator=good, output_root=output)
            second = pilot.run_pilot("r0", clip_ids=["SNGS-010"], generator=good, output_root=output)
            self.assertEqual((first["success"], second["skipped"]), (1, 1))

            def broken(*_args, **_kwargs):
                raise RuntimeError("provider down")

            failed = pilot.run_pilot("r1", clip_ids=["SNGS-010"], generator=broken, output_root=output)
            skipped = pilot.run_pilot("r1", clip_ids=["SNGS-010"], generator=good, output_root=output)
            retried = pilot.run_pilot("r1", clip_ids=["SNGS-010"], retry_failed=True, generator=good, output_root=output)
            self.assertEqual((failed["failed"], skipped["skipped"], retried["success"]), (1, 1, 1))

    def test_r4_uses_fifteen_event_b_budget(self):
        seen = []
        a, b = FIXTURES["counter-attack"]

        def generator(_provider, prompt, schema, **_kwargs):
            events = schema.get("properties", {}).get("events")
            if schema is pilot.A_SCHEMA:
                return copy.deepcopy(a), [{"tokens": 1}], 1
            if isinstance(events, dict):
                seen.append(events.get("maxItems"))
                self.assertIn("最多15条", prompt)
                self.assertIn("不要补满15条", prompt)
                return copy.deepcopy(b), [{"tokens": 1}], 1
            return copy.deepcopy({"label": None, "rationale": "none fit", "objections": []}), [{"tokens": 1}], 1

        with tempfile.TemporaryDirectory() as directory:
            counts = pilot.run_pilot("r4", clip_ids=["SNGS-010"], generator=generator, output_root=Path(directory))
        self.assertEqual(counts["success"], 1)
        self.assertEqual(seen, [15])


class ReportingTests(unittest.TestCase):
    @staticmethod
    def perfect_results() -> list[dict]:
        return [
            {
                "status": "success",
                "clip": row["clip"],
                "gt": {"tactic": row["tactic"], "interval": row["interval"]},
                "rule": {"label": row["tactic"], "reason": "matched", "interval": row["interval"]},
            }
            for row in pilot.load_gt()
        ]

    def test_metrics_cover_classification_baseline_and_temporal_grounding(self):
        results = self.perfect_results()
        metrics = pilot.classification_metrics(results)
        self.assertEqual(metrics["correct_n"], 37)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["majority_baseline"], 15 / 37)
        self.assertEqual(metrics["temporal"]["eligible_n"], 36)
        self.assertEqual(metrics["temporal"]["mean_iou"], 1.0)
        self.assertTrue(metrics["pilot_success"])

        results[0]["rule"]["label"] = "long-ball"
        metrics = pilot.classification_metrics(results)
        self.assertAlmostEqual(metrics["temporal"]["mean_iou"], 35 / 36)

    def test_missing_interval_counts_as_zero(self):
        results = self.perfect_results()
        results[0]["rule"]["interval"] = None
        metrics = pilot.classification_metrics(results)
        self.assertAlmostEqual(metrics["temporal"]["mean_iou"], 35 / 36)

    def test_manual_review_descriptive_metrics(self):
        rows = [
            {"review_seconds": 10, "replayed_video": False, "a_material_error": True, "b_material_error": False, "c_objections_valid": [True]},
            {"review_seconds": 20, "replayed_video": True, "a_material_error": False, "b_material_error": True, "c_objections_valid": [False]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metrics = pilot.manual_review_metrics(path)
        self.assertEqual(metrics["review_seconds_median"], 15)
        self.assertEqual(metrics["video_replay_rate"], 0.5)
        self.assertEqual(metrics["c_objection_precision"], 0.5)
        self.assertEqual(metrics["c_clip_error_recall"], 0.5)

    def test_report_writes_requested_artifacts(self):
        a, b = FIXTURES["counter-attack"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pilot.run_pilot("r0", clip_ids=["SNGS-010"], generator=fake_generator_with(a, b), output_root=output)
            metrics = pilot.report("r0", output)
            run_dir = output / "r0"
            self.assertTrue(all((run_dir / name).is_file() for name in ("results.jsonl", "metrics.json", "review.md", "manual_review.jsonl")))
            self.assertEqual(metrics["n_success"], 1)
            self.assertIn("GT: counter-attack", (run_dir / "review.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
