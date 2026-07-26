"""Asymmetric evidence verdicts for tactical claims."""

from dataclasses import dataclass

from .checkers import CheckResult


@dataclass(frozen=True)
class Verdict:
    tier: str
    reason: str


def decide(
    claim_tactic_id: str,
    checker_results_trajectory: list[CheckResult],
    checker_results_vlm: list[CheckResult],
    observability_ok: bool = True,
) -> Verdict:
    if not observability_ok:
        return Verdict("unobservable", f"{claim_tactic_id}: required evidence is not observable")

    for result in checker_results_trajectory + checker_results_vlm:
        if result.verdict == "veto":
            return Verdict("rejected", result.reason)

    if checker_results_trajectory and all(result.passed for result in checker_results_trajectory):
        return Verdict("confirmed", f"{claim_tactic_id}: trajectory checks passed")

    if checker_results_vlm and all(result.passed for result in checker_results_vlm):
        return Verdict("possible", f"{claim_tactic_id}: VLM-claimed evidence checks passed")

    return Verdict("insufficient", f"{claim_tactic_id}: no evidence source passed all checks")
