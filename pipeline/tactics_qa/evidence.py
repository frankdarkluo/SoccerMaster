"""Typed model-free evidence records consumed by deterministic checkers."""

from dataclasses import asdict, dataclass, field
import json

RESTART_KINDS = {"kickoff", "goal_kick", "throw_in", "free_kick", "corner"}
CONTROLLED_REGAIN_KINDS = {"tackle", "interception", "controlled_recovery"}
# Ball height is unavailable, so rapid alternating trajectory possession uses
# a generic chaos event rather than guessing aerial versus ground contact.
CHAOS_KINDS = {"clearance", "header_flick", "aerial_duel", "contested_touch"}
OTHER_KINDS = {"pass", "cross", "dribble", "shot"}
EVENT_KINDS = RESTART_KINDS | CONTROLLED_REGAIN_KINDS | CHAOS_KINDS | OTHER_KINDS
DELIVERY_KINDS = {
    "open_play_pass", "cross", "goal_kick", "throw_in", "free_kick",
    "corner_cross", "long_clearance",
}
LANDING_ZONES = {"near_post", "center", "far_post"}


@dataclass(frozen=True)
class PossessionEvent:
    t: float
    team: str
    kind: str

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind}")
        if self.team not in {"attacking", "defending"}:
            raise ValueError(f"unknown team: {self.team}")


@dataclass
class ClipEvidence:
    clip_uid: str
    tactic_id: str
    events: list = field(default_factory=list)
    delivery_kind: str | None = None
    corner_landing_zone: str | None = None
    source: str = "human_review_text"
    notes: str = ""

    def __post_init__(self):
        if self.delivery_kind is not None and self.delivery_kind not in DELIVERY_KINDS:
            raise ValueError(f"unknown delivery kind: {self.delivery_kind}")
        if self.corner_landing_zone is not None and self.corner_landing_zone not in LANDING_ZONES:
            raise ValueError(f"unknown landing zone: {self.corner_landing_zone}")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["events"] = [asdict(event) for event in self.events]
        return result


def from_dict(data: dict) -> ClipEvidence:
    data = dict(data)
    data["events"] = [PossessionEvent(**event) for event in data.get("events", [])]
    return ClipEvidence(**data)


def load_evidence_jsonl(path: str) -> dict:
    result = {}
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            evidence = from_dict(json.loads(line))
            key = (evidence.clip_uid, evidence.tactic_id)
            if key in result:
                raise ValueError(f"duplicate evidence key: {key}")
            result[key] = evidence
    return result
