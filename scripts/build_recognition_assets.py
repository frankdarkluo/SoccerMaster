"""Regenerate recognition-review assets and optionally append P0 hard negatives."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tactics_qa.evidence import load_evidence_jsonl  # noqa: E402
from pipeline.tactics_qa.replay import P0_TACTICS, replay, to_markdown  # noqa: E402
from pipeline.tactics_qa.review_io import load_review, write_jsonl  # noqa: E402

ROOT = Path("benchmark/tactical_prototypes")
REVIEW_OUT = ROOT / "recognition_review.jsonl"
EVIDENCE_PATH = ROOT / "recognition_evidence.jsonl"
EPISODES_PATH = ROOT / "episodes.jsonl"


def migrate_hard_negatives(rows, episodes_path, apply=False):
    """Append reviewed P0 errors as deduplicated gold targeted negatives."""
    with open(episodes_path, encoding="utf-8") as fp:
        existing = {
            (episode["clip_uid"], episode["tactic_id"], episode["prototype_type"])
            for episode in map(json.loads, fp)
        }
    with open(ROOT / "source_rows.jsonl", encoding="utf-8") as fp:
        sources = {row["clip_uid"]: row for row in map(json.loads, fp)}
    new = []
    for row in rows:
        if (row["verdict"] != "wrong" or row["tactic_id"] not in P0_TACTICS
                or row["root_cause"] in {"rubric", "window_partial"}):
            continue
        key = (row["clip_uid"], row["tactic_id"], "targeted_negative")
        if key in existing:
            continue
        source_row = sources.get(row["clip_uid"])
        if source_row is None:
            raise ValueError(f"missing source row for {row['clip_uid']}")
        new.append({
            "actor_bindings": {}, "clip_uid": row["clip_uid"],
            "confidence": row["confidence"],
            "evidence_summary": f"误判为{row['tactic_zh']}：{row['correction']}",
            "failed_conditions": [row["root_cause"]], "missing_evidence": [],
            "observability": {"ball": "unknown", "calibration": "unknown", "tracking": "unknown", "video": "direct"},
            "positive_evidence": [row["evidence_text"]],
            "prototype_id": f"recognition_review/{row['clip_id']}/{row['tactic_id']}/hard_negative",
            "prototype_type": "targeted_negative",
            "provenance": {"migration_version": "recognition-review-v1", "rubric": "scoring_rubric.md v1"},
            "review_status": "gold", "schema_version": "tactical-prototype-v1",
            "source": {
                "clip_id": source_row["clip_id"],
                "dataset_id": source_row["dataset_id"],
                "event_id": source_row.get("event_id"),
                "source_csv": source_row["source_csv"],
                "source_group_id": source_row["source_group_id"],
                "source_row_number": source_row["source_row_number"],
                "video_path": source_row["video_path"],
            },
            "source_tactic_name": row["tactic_zh"], "tactic_id": row["tactic_id"],
            "team_id": None, "window": row["window"],
        })
        existing.add(key)
    print(f"P0 hard negatives: {len(new)} new")
    if apply and new:
        with open(episodes_path, "a", encoding="utf-8") as fp:
            for episode in new:
                fp.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    return new


def main() -> None:
    rows = load_review()
    wrong = [row for row in rows if row["verdict"] == "wrong"]
    unmapped = [row for row in wrong if row["root_cause"] is None]
    if unmapped:
        raise SystemExit("wrong rows missing root cause: " + ", ".join(
            f"({row['clip_id']},{row['tactic_zh']})" for row in unmapped
        ))
    write_jsonl(rows, REVIEW_OUT)
    print(f"{len(rows)} rows ({len(wrong)} wrong) -> {REVIEW_OUT}")

    if EVIDENCE_PATH.exists():
        report = replay(rows, load_evidence_jsonl(EVIDENCE_PATH))
        with open(ROOT / "replay_report.json", "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2, sort_keys=True)
        with open(ROOT / "replay_report.md", "w", encoding="utf-8") as fp:
            fp.write(to_markdown(report))
        print("replay report written")

    migrate_hard_negatives(rows, EPISODES_PATH, apply="--apply-episodes" in sys.argv)


if __name__ == "__main__":
    main()
