"""Compare checker-relevant predicates from hand and trajectory evidence."""
from .checkers import REGAIN_WINDOW_S, RESTART_LOOKBACK_S, run_checkers
from .evidence import CHAOS_KINDS, CONTROLLED_REGAIN_KINDS, RESTART_KINDS


def _restart_at_origin(evidence, window):
    start = window["start_s"]
    kinds = [
        event.kind for event in evidence.events
        if event.kind in RESTART_KINDS
        and start - RESTART_LOOKBACK_S <= event.t <= start + 1.0
    ]
    return kinds[0] if kinds else None


def _regain_predicates(evidence, window):
    start = window["start_s"]
    near = [
        event for event in evidence.events
        if start - REGAIN_WINDOW_S <= event.t <= start + REGAIN_WINDOW_S
    ]
    regained = any(
        event.team == "attacking" and event.kind in CONTROLLED_REGAIN_KINDS
        for event in near)
    chaotic = sum(event.kind in CHAOS_KINDS for event in near) > 2
    return regained, chaotic


def _compare(hand, auto):
    if hand is None and auto is None:
        return "both_missing"
    if auto is None:
        return "auto_missing"
    if hand is None:
        return "hand_missing"
    return "match" if hand == auto else "mismatch"


def claim_agreement(hand, auto, window) -> dict:
    hand_regain, hand_chaos = _regain_predicates(hand, window)
    auto_regain, auto_chaos = _regain_predicates(auto, window)
    hand_verdicts = sorted(
        (result.checker, result.verdict) for result in run_checkers(hand, window))
    auto_verdicts = sorted(
        (result.checker, result.verdict) for result in run_checkers(auto, window))
    possession_missing = (
        "attacking side ambiguous" in auto.notes)
    hand_restart = _restart_at_origin(hand, window)
    auto_restart = _restart_at_origin(auto, window)
    restart_agreement = _compare(hand_restart, auto_restart)
    if (hand_restart is not None and auto_restart is None
            and any(event.kind in RESTART_KINDS for event in auto.events)):
        restart_agreement = "mismatch"
    return {
        "clip_uid": hand.clip_uid,
        "tactic_id": hand.tactic_id,
        "restart_at_origin": restart_agreement,
        "has_controlled_regain": (
            "auto_missing" if possession_missing
            else "match" if hand_regain == auto_regain else "mismatch"),
        "chaotic": (
            "auto_missing" if possession_missing
            else "match" if hand_chaos == auto_chaos else "mismatch"),
        "delivery_kind": _compare(hand.delivery_kind, auto.delivery_kind),
        "corner_landing_zone": _compare(
            hand.corner_landing_zone, auto.corner_landing_zone),
        "checker_verdicts_equal": hand_verdicts == auto_verdicts,
        "hand_verdicts": hand_verdicts,
        "auto_verdicts": auto_verdicts,
    }
