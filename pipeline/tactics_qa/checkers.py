"""Deterministic acceptance checkers implementing rubric rules R4-R7."""

from dataclasses import dataclass

from .evidence import CHAOS_KINDS, CONTROLLED_REGAIN_KINDS, RESTART_KINDS, ClipEvidence

RESTART_LOOKBACK_S = 3.0
REGAIN_WINDOW_S = 3.0
MAX_CHAOS_EVENTS = 2
OPEN_PLAY_TACTICS = {
    "fast_break_pattern", "line_break", "run_in_behind", "halfspace-penetration",
}
THROUGH_PASS_TACTICS = {"line_break", "run_in_behind"}


@dataclass(frozen=True)
class CheckResult:
    checker: str
    verdict: str
    reason: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def restart_gate(ev: ClipEvidence, window: dict) -> CheckResult:
    ws = window["start_s"]
    for event in ev.events:
        if event.kind in RESTART_KINDS and ws - RESTART_LOOKBACK_S <= event.t <= ws + 1.0:
            recovered = event.team == "defending" and any(
                other.team == "attacking" and other.kind in CONTROLLED_REGAIN_KINDS
                and event.t <= other.t <= ws + 1.0 for other in ev.events
            )
            if recovered:
                continue
            return CheckResult("restart_gate", "veto", f"{event.kind} at t={event.t:.1f}s starts the window")
    return CheckResult("restart_gate", "pass", "no restart at window origin")


def delivery_gate(ev: ClipEvidence) -> CheckResult:
    if ev.delivery_kind is None:
        return CheckResult("delivery_gate", "insufficient", "delivery kind not established")
    if ev.tactic_id in THROUGH_PASS_TACTICS and ev.delivery_kind != "open_play_pass":
        return CheckResult(
            "delivery_gate", "veto",
            f"delivery is {ev.delivery_kind}, not an open-play penetrating pass",
        )
    if ev.tactic_id == "cutback" and ev.delivery_kind == "corner_cross":
        return CheckResult("delivery_gate", "veto", "cutback claimed on a corner")
    return CheckResult("delivery_gate", "pass", f"delivery {ev.delivery_kind} ok")


def clean_regain(ev: ClipEvidence, window: dict) -> CheckResult:
    ws = window["start_s"]
    near = [event for event in ev.events if ws - REGAIN_WINDOW_S <= event.t <= ws + REGAIN_WINDOW_S]
    chaos = [event for event in near if event.kind in CHAOS_KINDS]
    if len(chaos) > MAX_CHAOS_EVENTS:
        return CheckResult(
            "clean_regain", "veto",
            f"chaotic second-ball chain ({len(chaos)} contested touches) around window start",
        )
    regains = [
        event for event in near
        if event.team == "attacking" and event.kind in CONTROLLED_REGAIN_KINDS
    ]
    if len(regains) > 1:
        return CheckResult("clean_regain", "veto", f"multiple controlled regains ({len(regains)})")
    if not regains:
        return CheckResult("clean_regain", "veto", "no controlled regain near window start")
    return CheckResult("clean_regain", "pass", "single controlled regain")


def corner_landing(ev: ClipEvidence) -> CheckResult:
    if ev.corner_landing_zone is None:
        return CheckResult("corner_landing", "insufficient", "landing zone not established")
    if ev.corner_landing_zone == "center":
        return CheckResult("corner_landing", "veto", "first contact in central zone, not a post zone")
    return CheckResult("corner_landing", "pass", f"first contact at {ev.corner_landing_zone}")


def run_checkers(ev: ClipEvidence, window: dict) -> list:
    results = []
    if ev.tactic_id in OPEN_PLAY_TACTICS:
        results.append(restart_gate(ev, window))
    if ev.tactic_id in THROUGH_PASS_TACTICS or ev.tactic_id == "cutback":
        results.append(delivery_gate(ev))
    if ev.tactic_id == "fast_break_pattern":
        results.append(clean_regain(ev, window))
    if ev.tactic_id == "corner-near-far-post":
        results.append(corner_landing(ev))
    return results
