"""Generate trajectory evidence, agreement metrics, and same-subset replay."""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tactics_qa.agreement import claim_agreement
from pipeline.tactics_qa.auto_evidence import build_claim_evidence
from pipeline.tactics_qa.evidence import load_evidence_jsonl
from pipeline.tactics_qa.gsr_io import load_gsr
from pipeline.tactics_qa.replay import P0_TACTICS, replay, to_markdown
from pipeline.tactics_qa import auto_evidence, delivery_infer, kinematics
from pipeline.tactics_qa import possession_timeline, restart_infer

FIELDS = [
    "restart_at_origin",
    "has_controlled_regain",
    "chaotic",
    "delivery_kind",
    "corner_landing_zone",
]


def _thresholds():
    modules = [
        auto_evidence, delivery_infer, kinematics,
        possession_timeline, restart_infer,
    ]
    return {
        f"{module.__name__.rsplit('.', 1)[-1]}.{name}": value
        for module in modules
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, (int, float))
    }


def _missing_causes(coverage, agreements, auto):
    causes = Counter()
    if coverage["no_data"]:
        causes["no GSR predictions.json"] = coverage["no_data"]
    for claim in agreements:
        evidence = auto[(claim["clip_uid"], claim["tactic_id"])]
        if claim["delivery_kind"] == "auto_missing":
            if "no key kick detected" in evidence.notes:
                causes["key-pass kick not detected"] += 1
            elif "attack direction ambiguous" in evidence.notes:
                causes["attack-direction ambiguity"] += 1
            else:
                causes["delivery geometry unobservable"] += 1
        if (claim["has_controlled_regain"] == "auto_missing"
                or claim["chaotic"] == "auto_missing"):
            causes["attacking-side ambiguity"] += 1
        if claim["corner_landing_zone"] == "auto_missing":
            causes["corner landing unobservable"] += 1
        if claim["restart_at_origin"] == "auto_missing":
            causes["restart not inferred"] += 1
    return causes


def _write_markdown(path, report):
    coverage = report["coverage"]
    field_counts = report["field_agreement"]
    equal = field_counts["checker_verdicts_equal"]
    total = sum(equal.values())
    equal_count = equal.get("true", 0)
    false_vetoes = [
        (tactic, clip)
        for tactic, stats in report["replay_auto"].items()
        for clip in stats["false_vetoes"]
    ]
    lines = [
        "# 自动证据一致率报告",
        "",
        "## 结论摘要",
        "",
        f"- 本机 72 条 claim 中仅 {coverage.get('covered', 0)} 条有 GSR 输出，"
        f"{coverage.get('no_data', 0)} 条为 no_data；以下结果只是适配器诊断，不是系统泛化能力。",
        f"- checker 判定完全一致：{equal_count}/{total} "
        f"({(100 * equal_count / total if total else 0):.1f}%)。",
        f"- 自动证据 false veto：{len(false_vetoes)}"
        + (f"（{', '.join(f'{t}:{c}' for t, c in false_vetoes)}）" if false_vetoes else "。"),
        "- 规则阈值未为提高覆盖率而调整；缺证据保留为 auto_missing/insufficient。",
        "",
        "### 优先感知 backlog",
        "",
    ]
    causes = report["auto_missing_causes"]
    lines.extend(
        f"{index}. {name}：{count} 条"
        for index, (name, count) in enumerate(causes[:3], 1)
    )
    if not causes:
        lines.append("1. 当前覆盖子集未观察到自动证据缺失。")
    elif len(causes) < 3:
        lines.append(f"{len(causes) + 1}. 覆盖过小，未观察到第三类原因，不补造结论。")

    lines += [
        "",
        "## 逐字段一致率",
        "",
        "| 字段 | match | mismatch | auto_missing | hand_missing | both_missing | 可比较项一致率 | 自动缺失率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field in FIELDS:
        counts = field_counts[field]
        comparable = counts.get("match", 0) + counts.get("mismatch", 0)
        all_claims = sum(counts.values())
        agreement = 100 * counts.get("match", 0) / comparable if comparable else 0
        missing = 100 * counts.get("auto_missing", 0) / all_claims if all_claims else 0
        lines.append(
            f"| {field} | {counts.get('match', 0)} | {counts.get('mismatch', 0)} | "
            f"{counts.get('auto_missing', 0)} | {counts.get('hand_missing', 0)} | "
            f"{counts.get('both_missing', 0)} | {agreement:.1f}% | {missing:.1f}% |"
        )
    mismatches = [
        (claim["clip_uid"], field)
        for claim in report["claims"]
        for field in FIELDS
        if claim[field] == "mismatch"
    ]
    lines += ["", "### 数据审计差异", ""]
    if mismatches:
        lines.extend(
            f"- {clip_uid}: {field}"
            for clip_uid, field in mismatches
        )
        lines.append(
            "- SNGS-116 的手工 corner 时间是窗口起点 3.0s，轨迹检测为实际开球 6.2s；"
            "保留原记录，需另行审计，不计作 auto_missing。")
    else:
        lines.append("- 当前覆盖子集没有 mismatch。")

    lines += [
        "",
        "## P0 precision_after：自动证据 vs 手工证据（同一覆盖子集）",
        "",
        "| tactic | covered claims | auto | hand | auto false vetoes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tactic in P0_TACTICS:
        auto_stats = report["replay_auto"].get(tactic)
        hand_stats = report["replay_hand_same_subset"].get(tactic)
        if auto_stats is None:
            lines.append(f"| {tactic} | 0 | N/A | N/A | 0 |")
        else:
            lines.append(
                f"| {tactic} | {auto_stats['n_claims']} | "
                f"{auto_stats['precision_after']} | {hand_stats['precision_after']} | "
                f"{len(auto_stats['false_vetoes'])} |"
            )

    lines += [
        "",
        "## 自动证据回放明细",
        "",
        to_markdown({"per_tactic": report["replay_auto"], "rows": report["replay_auto_rows"]}),
        "## 手工证据回放明细（同一覆盖子集）",
        "",
        to_markdown({
            "per_tactic": report["replay_hand_same_subset"],
            "rows": report["replay_hand_same_subset_rows"],
        }),
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark/tactical_prototypes"))
    args = parser.parse_args()

    review = [
        json.loads(line)
        for line in (args.root / "recognition_review.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    hand = load_evidence_jsonl(str(args.root / "recognition_evidence.jsonl"))
    auto, coverage, lines = {}, Counter(), []
    cache = {}
    for row in review:
        predictions = args.outputs_dir / row["clip_id"] / "predictions.json"
        if not predictions.is_file():
            coverage["no_data"] += 1
            continue
        if predictions not in cache:
            cache[predictions] = load_gsr(predictions)
        evidence = build_claim_evidence(
            cache[predictions], row["clip_uid"], row["tactic_id"], row["window"])
        auto[(evidence.clip_uid, evidence.tactic_id)] = evidence
        lines.append(json.dumps(
            evidence.to_dict(), ensure_ascii=False, sort_keys=True))
        coverage["covered"] += 1
    (args.root / "recognition_evidence_auto.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    covered_review = [
        row for row in review
        if (row["clip_uid"], row["tactic_id"]) in auto
    ]
    agreements = [
        claim_agreement(
            hand[(row["clip_uid"], row["tactic_id"])],
            auto[(row["clip_uid"], row["tactic_id"])],
            row["window"],
        )
        for row in covered_review
    ]
    auto_replay = replay(covered_review, auto)
    hand_replay = replay(covered_review, hand)
    summary = {
        field: dict(Counter(claim[field] for claim in agreements))
        for field in FIELDS
    }
    summary["checker_verdicts_equal"] = {
        str(key).lower(): value
        for key, value in Counter(
            claim["checker_verdicts_equal"] for claim in agreements).items()
    }
    report = {
        "schema_version": "evidence-agreement-v1",
        "coverage": dict(coverage),
        "thresholds": _thresholds(),
        "field_agreement": summary,
        "claims": agreements,
        "auto_missing_causes": _missing_causes(
            coverage, agreements, auto).most_common(),
        "replay_auto": auto_replay["per_tactic"],
        "replay_auto_rows": auto_replay["rows"],
        "replay_hand_same_subset": hand_replay["per_tactic"],
        "replay_hand_same_subset_rows": hand_replay["rows"],
    }
    (args.root / "evidence_agreement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.root / "evidence_agreement_report.md", report)
    equal = summary["checker_verdicts_equal"].get("true", 0)
    print(
        f"covered={coverage['covered']} no_data={coverage['no_data']} "
        f"verdict_equal={equal}/{len(agreements)}"
    )


if __name__ == "__main__":
    main()
