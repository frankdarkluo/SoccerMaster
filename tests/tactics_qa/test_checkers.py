from pipeline.tactics_qa.checkers import (
    clean_regain, corner_landing, delivery_gate, restart_gate, run_checkers,
)
from pipeline.tactics_qa.evidence import ClipEvidence, PossessionEvent as E

W = {"start_s": 10.0, "end_s": 20.0}


def _ev(tactic_id, events=(), delivery=None, zone=None):
    return ClipEvidence(
        clip_uid="test:clip", tactic_id=tactic_id, events=list(events),
        delivery_kind=delivery, corner_landing_zone=zone,
    )


def test_restart_gate_vetoes_kickoff_origin():
    result = restart_gate(_ev("fast_break_pattern", [E(9.0, "attacking", "kickoff")]), W)
    assert not result.passed and "kickoff" in result.reason


def test_restart_gate_passes_open_play():
    assert restart_gate(_ev("fast_break_pattern", [E(10.5, "attacking", "interception")]), W).passed


def test_restart_gate_ignores_old_restart():
    evidence = _ev("fast_break_pattern", [
        E(2.0, "defending", "corner"), E(10.5, "attacking", "interception"),
    ])
    assert restart_gate(evidence, W).passed


def test_restart_gate_allows_opponent_restart_then_controlled_regain():
    evidence = _ev("fast_break_pattern", [
        E(8.0, "defending", "free_kick"),
        E(9.5, "attacking", "controlled_recovery"),
    ])
    assert restart_gate(evidence, W).passed


def test_delivery_gate_vetoes_cross_for_run_in_behind():
    assert not delivery_gate(_ev("run_in_behind", delivery="cross")).passed


def test_delivery_gate_vetoes_throw_in_for_line_break():
    assert not delivery_gate(_ev("line_break", delivery="throw_in")).passed


def test_delivery_gate_vetoes_corner_cross_for_cutback():
    assert not delivery_gate(_ev("cutback", delivery="corner_cross")).passed


def test_delivery_gate_passes_open_play_pass():
    assert delivery_gate(_ev("run_in_behind", delivery="open_play_pass")).passed


def test_delivery_gate_insufficient_when_unknown():
    result = delivery_gate(_ev("run_in_behind"))
    assert not result.passed and result.verdict == "insufficient"


def test_clean_regain_passes_single_controlled_regain():
    assert clean_regain(_ev("fast_break_pattern", [E(10.0, "attacking", "tackle")]), W).passed


def test_clean_regain_passes_cleared_set_piece_then_control():
    evidence = _ev("fast_break_pattern", [
        E(8.5, "attacking", "clearance"), E(9.5, "attacking", "controlled_recovery"),
    ])
    assert clean_regain(evidence, W).passed


def test_clean_regain_vetoes_multiple_controlled_regains():
    evidence = _ev("fast_break_pattern", [
        E(9.0, "attacking", "tackle"),
        E(11.0, "attacking", "controlled_recovery"),
    ])
    assert not clean_regain(evidence, W).passed


def test_clean_regain_vetoes_chaotic_chain():
    evidence = _ev("fast_break_pattern", [
        E(7.0, "defending", "header_flick"), E(8.0, "attacking", "clearance"),
        E(9.0, "defending", "aerial_duel"), E(10.0, "attacking", "header_flick"),
    ])
    result = clean_regain(evidence, W)
    assert not result.passed and "chaotic" in result.reason


def test_clean_regain_vetoes_no_regain():
    evidence = _ev("fast_break_pattern", [
        E(10.0, "attacking", "pass"), E(12.0, "attacking", "dribble"),
    ])
    result = clean_regain(evidence, W)
    assert not result.passed and "no controlled regain" in result.reason


def test_corner_landing_vetoes_center():
    assert not corner_landing(_ev("corner-near-far-post", zone="center")).passed


def test_corner_landing_passes_near_post():
    assert corner_landing(_ev("corner-near-far-post", zone="near_post")).passed


def test_corner_landing_insufficient_when_unknown():
    result = corner_landing(_ev("corner-near-far-post"))
    assert not result.passed and result.verdict == "insufficient"


def test_run_checkers_applies_only_relevant_checks():
    evidence = _ev("corner-near-far-post", zone="far_post")
    assert [result.checker for result in run_checkers(evidence, W)] == ["corner_landing"]
    fast_break = _ev("fast_break_pattern", [E(10.0, "attacking", "interception")])
    assert {result.checker for result in run_checkers(fast_break, W)} == {"restart_gate", "clean_regain"}
