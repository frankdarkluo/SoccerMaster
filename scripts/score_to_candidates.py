#!/usr/bin/env python3
"""把 AI 的 12 类证据打分转成人工核验用的候选清单。

模型只输出证据分（schema 见 pipeline/tactics_qa/criteria.py），判定全在这里做：
排序、卡阈值、定 primary/secondary、算时间关系、给 confidence 路由建议。

用法：
    python3 scripts/score_to_candidates.py 打分.jsonl 候选.csv --threshold 0.55 --top-k 4

阈值可以随便扫，不用重跑模型 —— 这正是把判决从 prompt 里搬出来的目的。
两个旋钮：--threshold 卡平均证据分（召回侧），--min-floor 卡最弱一条证据（精度侧）。

不合规的行不会被"容错"混进 CSV：全部记进 <输出>.rejects.txt，并以非零码退出。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.tactics_qa import criteria  # noqa: E402

COLUMNS = [
    "clip_id", "rank", "role", "tactic_id", "战术", "team",
    "ai_score", "ai_floor", "n_uncertain", "ai_start", "ai_end",
    "relation_to_primary", "evidence_detail", "ai_note",
    # 以下留空，人工填
    "verification", "tactic_final", "start_sec", "end_sec",
    "spatial", "confidence", "人工判断依据",
]


def relation(primary_win, win):
    """primary 与 secondary 的时间关系，判据同人工核验指南 5.3。"""
    ps, pe = primary_win
    ss, se = win
    if ps <= ss and se <= pe:
        return "contains"
    if max(ps, ss) < min(pe, se):
        return "partial_overlap"
    gap = ss - pe if ss >= pe else ps - se
    return "sequential" if gap <= 0.5 else "separate"


def route(margin):
    """top-1 与 top-2 的分差 -> 复核路由建议。人工仍要自己给最终 confidence。"""
    if margin >= 0.30:
        return "单人核验"
    if margin >= 0.15:
        return "建议二次复核"
    return "建议二次复核+可能裁决"


def rank_clip(clip, names, threshold, top_k, min_keep, min_floor):
    rows = []
    for tactic_id, entry in clip["scores"].items():
        vals = list(entry["evidence"].values())
        rows.append({
            "tactic_id": tactic_id,
            "team": entry.get("team", ""),
            "score": sum(vals) / len(vals),
            "floor": min(vals),
            "n_unc": sum(1 for v in vals if abs(v - 0.5) < 1e-9),
            "window": (entry["window_start_s"], entry["window_end_s"]),
            "evidence": entry["evidence"],
            "note": entry.get("note", ""),
        })

    rows.sort(key=lambda r: (-r["score"], r["tactic_id"]))
    kept = [r for r in rows[:top_k]
            if r["score"] >= threshold and r["floor"] >= min_floor]
    if len(kept) < min_keep:
        kept = rows[:min_keep]          # 永远给人至少一条去看，不弃权

    margin = rows[0]["score"] - (rows[1]["score"] if len(rows) > 1 else 0.0)
    primary_win = kept[0]["window"]

    out = []
    for i, r in enumerate(kept):
        out.append({
            "clip_id": clip["clip_id"],
            "rank": i + 1,
            "role": "primary" if i == 0 else "secondary",
            "tactic_id": r["tactic_id"],
            "战术": names[r["tactic_id"]],
            "team": r["team"],
            "ai_score": f"{r['score']:.3f}",
            "ai_floor": f"{r['floor']:.2f}",
            "n_uncertain": r["n_unc"],
            "ai_start": f"{r['window'][0]:.1f}",
            "ai_end": f"{r['window'][1]:.1f}",
            "relation_to_primary": "-" if i == 0 else relation(primary_win, r["window"]),
            "evidence_detail": ";".join(f"{k}={v}" for k, v in r["evidence"].items()),
            "ai_note": r["note"],
            "confidence": route(margin) if i == 0 else "",
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="模型打分 JSONL")
    ap.add_argument("dst", type=Path, help="输出候选 CSV")
    ap.add_argument("--threshold", type=float, default=0.55, help="进入候选的最低平均证据分")
    ap.add_argument("--top-k", type=int, default=4, help="每个 clip 最多给人看几条")
    ap.add_argument("--min-keep", type=int, default=1, help="即使全部低于阈值也至少保留几条")
    ap.add_argument("--min-floor", type=float, default=0.0,
                    help="最弱一条证据的下限；卡精度用，某条判据明确不成立时直接排除")
    args = ap.parse_args()

    doc = criteria.load()
    names = criteria.names_cn(doc)

    rows, rejects, n_ok = [], [], 0
    with args.src.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                clip = json.loads(line)
            except json.JSONDecodeError as exc:
                rejects.append((f"第 {lineno} 行", f"JSON 解析失败：{exc}"))
                continue
            clip_id = clip.get("clip_id") or f"第 {lineno} 行"
            try:
                criteria.validate(clip, doc)
            except ValueError as exc:
                rejects.append((clip_id, str(exc)))
                continue
            rows.extend(rank_clip(clip, names, args.threshold, args.top_k,
                                  args.min_keep, args.min_floor))
            n_ok += 1

    with args.dst.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"{n_ok} 个合规 clip -> {len(rows)} 条候选"
          f"（均 {len(rows)/max(n_ok, 1):.2f} 条/clip），已写入 {args.dst}")

    if rejects:
        report = args.dst.with_suffix(args.dst.suffix + ".rejects.txt")
        report.write_text("\n".join(f"{cid}\t{why}" for cid, why in rejects) + "\n",
                          encoding="utf-8")
        print(f"{len(rejects)} 条不合规，未计入 CSV，明细见 {report}", file=sys.stderr)
        for cid, why in rejects[:5]:
            print(f"  {cid}\t{why}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
