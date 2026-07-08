"""Load reference_football.csv into a structured event ontology."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import REFERENCE_CSV


@dataclass
class TagValue:
    code: str
    display_name_cn: str
    display_name_en: str
    importance_modifier: float = 0.0


@dataclass
class TagGroup:
    code: str
    display_name_cn: str
    display_name_en: str
    applies_to: List[str] = field(default_factory=list)
    exclusive: bool = True
    values: List[TagValue] = field(default_factory=list)


@dataclass
class EventDef:
    event_id: str
    code: str
    family: str
    display_name_cn: str
    display_name_en: str
    description: str
    level_hint: str
    source_type: str
    importance_base: float
    tags: List[str]
    trigger_notes: str
    negative_flag: bool


COMPUTABLE_TAGS = {"pitch_zone", "shot_distance", "pass_distance", "pass_direction"}


class EventSchema:
    def __init__(self, csv_path: Path = REFERENCE_CSV):
        self._events: Dict[str, EventDef] = {}
        self._tag_groups: Dict[str, TagGroup] = {}
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        current_tag_group: Optional[str] = None
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = (row.get("event_id") or "").strip()
                if not eid or eid.startswith("===") or eid.startswith("---"):
                    continue
                if eid in ("Events", "Sports", "Broadcast"):
                    continue

                if eid == "_TAG_GROUP":
                    code = (row.get("code") or "").strip()
                    applies_raw = row.get("description") or ""
                    applies_to = [
                        a.strip()
                        for a in applies_raw.replace("applies_to:", "").split(",")
                        if a.strip()
                    ]
                    exclusive = (row.get("level_hint") or "").strip() == "exclusive"
                    self._tag_groups[code] = TagGroup(
                        code=code,
                        display_name_cn=(row.get("display_name_cn") or "").strip(),
                        display_name_en=(row.get("display_name_en") or "").strip(),
                        applies_to=applies_to,
                        exclusive=exclusive,
                    )
                    current_tag_group = code
                    continue

                if eid == "_TAG_VALUE" and current_tag_group:
                    code = (row.get("code") or "").strip()
                    imp = row.get("importance_base") or "0"
                    try:
                        imp_f = float(imp)
                    except ValueError:
                        imp_f = 0.0
                    self._tag_groups[current_tag_group].values.append(
                        TagValue(
                            code=code,
                            display_name_cn=(row.get("display_name_cn") or "").strip(),
                            display_name_en=(row.get("display_name_en") or "").strip(),
                            importance_modifier=imp_f,
                        )
                    )
                    continue

                if eid.startswith("football."):
                    current_tag_group = None
                    tags_raw = row.get("tags") or ""
                    tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
                    imp = row.get("importance_base") or "0"
                    try:
                        imp_f = float(imp)
                    except ValueError:
                        imp_f = 0.0
                    neg = bool((row.get("negative_flag") or "").strip())
                    self._events[eid] = EventDef(
                        event_id=eid,
                        code=(row.get("code") or "").strip(),
                        family=(row.get("family") or "").strip(),
                        display_name_cn=(row.get("display_name_cn") or "").strip(),
                        display_name_en=(row.get("display_name_en") or "").strip(),
                        description=(row.get("description") or "").strip(),
                        level_hint=(row.get("level_hint") or "").strip(),
                        source_type=(row.get("source_type") or "").strip(),
                        importance_base=imp_f,
                        tags=tags_list,
                        trigger_notes=(row.get("trigger_notes") or "").strip(),
                        negative_flag=neg,
                    )

    def get_event(self, event_id: str) -> Optional[EventDef]:
        return self._events.get(event_id)

    def core_events(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "core"]

    def narrative_events(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "narrative"]

    def event_qualifiers(self) -> List[EventDef]:
        return [e for e in self._events.values() if e.level_hint == "" and e.importance_base > 0]

    def events_by_source_type(self, source_type: str) -> List[EventDef]:
        return [e for e in self._events.values() if e.source_type == source_type]

    def computable_tag_groups(self) -> List[str]:
        return sorted(COMPUTABLE_TAGS & set(self._tag_groups.keys()))

    def visual_tag_groups(self) -> List[str]:
        return sorted(set(self._tag_groups.keys()) - COMPUTABLE_TAGS)

    def get_tag_group(self, group_code: str) -> Optional[TagGroup]:
        return self._tag_groups.get(group_code)

    def tag_vocabulary_for_prompt(self, event_id: str) -> str:
        ev = self.get_event(event_id)
        if ev is None:
            return ""
        base_code = ev.code
        lines = []
        for tg_code, tg in self._tag_groups.items():
            if base_code in tg.applies_to or any(base_code.startswith(a) for a in tg.applies_to):
                pick = "pick one" if tg.exclusive else "pick any"
                vals = ", ".join(f"{v.code} ({v.display_name_cn})" for v in tg.values)
                lines.append(f"{tg_code} ({pick}): {vals}")
        return "\n".join(lines)

    def event_definitions_for_prompt(self) -> str:
        lines = []
        for eid, ev in sorted(self._events.items()):
            lines.append(f"{eid}: {ev.description} ({ev.display_name_cn} / {ev.display_name_en})")
        return "\n".join(lines)
