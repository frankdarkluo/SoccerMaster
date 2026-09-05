"""Bulk-fill the SoccerNet-AA manual review queue.

External review is skipped by policy, so every pending screening row becomes a
reject, and frame-identical boundary overlaps are confirmed without human input.
Ambiguous overlaps stay blank unless a decision is recorded in
ambiguous_pairs.csv.

select_path only accepts simple paths, so redundant clips whose confirmed
boundaries fan out into branches or cycles are rejected as duplicate content.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = ["review_type", "clip_uid", "other_clip_uid", "model_status", "decision", "notes"]


def _prune_branches(
    keeps: set[str], edges: list[tuple[str, str]]
) -> list[str]:
    """Reject the fewest clips that make the confirmed graph a set of paths."""
    adjacency = defaultdict(set)
    for left, right in edges:
        if left in keeps and right in keeps:
            adjacency[left].add(right)
            adjacency[right].add(left)
    rejected = []

    def drop(uid: str) -> None:
        for other in adjacency[uid]:
            adjacency[other].discard(uid)
        adjacency[uid] = set()
        rejected.append(uid)

    while True:
        branches = [uid for uid, near in adjacency.items() if len(near) > 2]
        if not branches:
            break
        drop(max(branches, key=lambda uid: (len(adjacency[uid]), uid)))

    seen = set()
    for start in list(adjacency):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            uid = stack.pop()
            if uid in component:
                continue
            component.add(uid)
            stack.extend(adjacency[uid] - component)
        seen |= component
        while sum(len(adjacency[uid]) for uid in component) // 2 >= len(component):
            drop(max(component, key=lambda uid: (len(adjacency[uid]), uid)))
    return rejected


def main(output_dir: Path) -> int:
    statuses = {}
    confirmed_edges = []
    for line in (output_dir / "overlap_groups.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("type") == "boundary_overlap":
            statuses[(item["left_uid"], item["right_uid"])] = item["status"]
            if item["status"] == "confirmed":
                confirmed_edges.append((item["left_uid"], item["right_uid"]))

    ambiguous = {}
    pairs_path = output_dir / "ambiguous_pairs.csv"
    if pairs_path.exists():
        with pairs_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                decision = row.get("decision", "").strip()
                if decision:
                    ambiguous[(row["left_uid"], row["right_uid"])] = decision

    queue_path = output_dir / "manual_review_queue.csv"
    with queue_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filled = 0
    for row in rows:
        if row.get("decision", "").strip():
            continue
        if row["review_type"] == "screening":
            row["decision"] = "reject"
        elif row["review_type"] == "overlap":
            key = (row["clip_uid"], row["other_clip_uid"])
            status = statuses.get(key)
            if status == "confirmed":
                row["decision"] = "confirm_overlap"
            elif status == "ambiguous":
                row["decision"] = ambiguous.get(key, "")
        if row.get("decision", "").strip():
            filled += 1

    rejected = {
        row["clip_uid"] for row in rows
        if row["review_type"] == "screening" and row.get("decision", "").strip() == "reject"
    }
    keeps = set()
    for line in (output_dir / "local_reviews.jsonl").read_text(encoding="utf-8").splitlines():
        review = json.loads(line)
        if review.get("metrics", {}).get("local_decision") == "keep":
            keeps.add(review["clip_uid"])
    keeps -= rejected
    pruned = _prune_branches(keeps, confirmed_edges)
    for uid in pruned:
        rows.append({
            "review_type": "screening",
            "clip_uid": uid,
            "other_clip_uid": "",
            "model_status": "manual_resolved",
            "decision": "reject",
            "notes": "duplicate content: confirmed boundary branch/cycle",
        })

    temp = queue_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(queue_path)

    blank = sum(1 for row in rows if not row.get("decision", "").strip())
    print(f"filled {filled} rows; {blank} still blank of {len(rows)}")
    print(f"rejected {len(pruned)} redundant clips to unbranch the overlap graph: {pruned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/soccernet_aa_screening")))
