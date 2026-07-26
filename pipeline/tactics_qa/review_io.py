"""Parse data/足球战术识别.csv (GB18030) into normalized review records."""

import csv
import json

from .taxonomy import ROOT_CAUSE_BY_CLAIM, TACTIC_ID_BY_ZH


def _parse_window(raw: str) -> dict:
    sep = "–" if "–" in raw else "-"
    start, end = raw.split(sep)

    def to_s(mmss: str) -> float:
        minutes, seconds = mmss.strip().split(":")
        return int(minutes) * 60 + float(seconds)

    return {"start_s": to_s(start), "end_s": to_s(end), "raw": raw.strip()}


def parse_review_rows(fp) -> list:
    rows = []
    for row in csv.DictReader(fp):
        clip_id = row["id"].strip()
        tactic_zh = row["tactics"].strip()
        verdict = "correct" if row["识别准确度"].strip() == "正确" else "wrong"
        rows.append({
            "clip_id": clip_id,
            "clip_uid": f"soccernetgs:{clip_id}" if clip_id.startswith("SNGS-") else f"event_clips:{int(clip_id):04d}",
            "tactic_zh": tactic_zh,
            "tactic_id": TACTIC_ID_BY_ZH[tactic_zh],
            "window": _parse_window(row["时间范围"]),
            "evidence_text": row["判断依据"].strip(),
            "confidence": row["置信度"].strip(),
            "verdict": verdict,
            "correction": row["更正"].strip() or None,
            "root_cause": ROOT_CAUSE_BY_CLAIM.get((clip_id, tactic_zh)) if verdict == "wrong" else None,
        })
    return rows


def load_review(path: str = "data/足球战术识别.csv") -> list:
    with open(path, encoding="gb18030", newline="") as fp:
        return parse_review_rows(fp)


def write_jsonl(records: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
