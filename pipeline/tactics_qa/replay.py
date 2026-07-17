"""Offline replay of deterministic checkers against reviewed claims."""

from .checkers import run_checkers

P0_TACTICS = [
    "fast_break_pattern", "run_in_behind", "corner-near-far-post", "cutback",
]
P1_TACTICS = ["line_break"]


def replay(review: list, evidence: dict) -> dict:
    per_tactic = {}
    rows_out = []
    for tactic in sorted({row["tactic_id"] for row in review}):
        claims = [row for row in review if row["tactic_id"] == tactic]
        scope = "P0" if tactic in P0_TACTICS else "P1" if tactic in P1_TACTICS else "deferred"
        stats = {
            "n_claims": len(claims), "scope": scope, "flipped_errors": [],
            "false_vetoes": [], "insufficient": [], "surviving_wrong": [],
        }
        n_correct = sum(row["verdict"] == "correct" for row in claims)
        survivors = 0
        surviving_correct = 0
        for row in claims:
            ev = evidence.get((row["clip_uid"], row["tactic_id"]))
            results = run_checkers(ev, row["window"]) if ev else []
            verdicts = {result.verdict for result in results}
            rows_out.append({
                "clip_uid": row["clip_uid"], "tactic_id": tactic,
                "verdict": row["verdict"], "root_cause": row.get("root_cause"),
                "checks": [result.__dict__ for result in results],
            })
            if "veto" in verdicts:
                key = "false_vetoes" if row["verdict"] == "correct" else "flipped_errors"
                stats[key].append(row["clip_uid"])
            elif "insufficient" in verdicts:
                stats["insufficient"].append(row["clip_uid"])
            else:
                survivors += 1
                if row["verdict"] == "correct":
                    surviving_correct += 1
                else:
                    stats["surviving_wrong"].append(row["clip_uid"])
        stats["precision_before"] = round(n_correct / len(claims), 3) if claims else None
        stats["precision_after"] = round(surviving_correct / survivors, 3) if survivors else None
        per_tactic[tactic] = stats
    return {"schema_version": "checker-replay-v1", "per_tactic": per_tactic, "rows": rows_out}


def to_markdown(report: dict) -> str:
    p0 = {tactic: stats for tactic, stats in report["per_tactic"].items() if stats["scope"] == "P0"}
    flipped = sum(len(stats["flipped_errors"]) for stats in p0.values())
    false_vetoes = sum(len(stats["false_vetoes"]) for stats in p0.values())
    residual_keys = {
        (clip_uid, tactic)
        for tactic, stats in p0.items()
        for clip_uid in stats["surviving_wrong"]
    }
    residuals = [
        f"{row['clip_uid']} ({row['root_cause']})"
        for row in report["rows"]
        if (row["clip_uid"], row["tactic_id"]) in residual_keys
    ]
    precision = "；".join(
        f"{tactic} {stats['precision_before']}→{stats['precision_after']}"
        for tactic, stats in p0.items()
    )
    lines = [
        "# Checker 离线回放报告", "", "## 结果摘要", "",
        f"- P0 precision：{precision}。",
        f"- P0 共拦截 {flipped} 条既有误报，false vetoes={false_vetoes}。",
        f"- P0 residual：{('；'.join(residuals) if residuals else '无')}。",
        "- 其余 surviving wrong 属于 P1 或 deferred，只用于定位下一轮 checker 范围。",
        "",
        "| tactic | scope | claims | precision before | precision after | flipped | false vetoes | insufficient | surviving wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tactic, stats in report["per_tactic"].items():
        lines.append(
            f"| {tactic} | {stats['scope']} | {stats['n_claims']} | "
            f"{stats['precision_before']} | {stats['precision_after']} | "
            f"{len(stats['flipped_errors'])} | {len(stats['false_vetoes'])} | "
            f"{len(stats['insufficient'])} | {len(stats['surviving_wrong'])} |"
        )
    for tactic, stats in report["per_tactic"].items():
        if stats["false_vetoes"]:
            lines += ["", f"**False vetoes ({tactic})**: {', '.join(stats['false_vetoes'])}"]
        if stats["surviving_wrong"]:
            lines += [
                "", f"**Surviving wrong ({tactic})**: "
                f"{', '.join(stats['surviving_wrong'])} — checker 覆盖不到的根因",
            ]
    return "\n".join(lines) + "\n"
