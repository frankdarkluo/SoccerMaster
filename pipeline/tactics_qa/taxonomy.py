"""Frozen mapping tables for the 2026-07 recognition review.

``one_two`` is provisional because concepts.jsonl has no matching concept card.
ROOT_CAUSE_BY_CLAIM is keyed by (clip_id, tactic_zh) because SNGS-061 and
SNGS-134 each carry two claims. Causes were assigned from the 更正 column.
"""

TACTIC_ID_BY_ZH = {
    "快速反击": "fast_break_pattern",
    "破线传球": "line_break",
    "打身后 / 反越位跑位": "run_in_behind",
    "线间接应": "between-the-lines",
    "门将参与出球": "gk-in-buildup",
    "角球后点/前点战术": "corner-near-far-post",
    "虚跑/假跑扯动": "dummy-run",
    "下底倒三角": "cutback",
    "肋部渗透": "halfspace-penetration",
    "大范围转移": "switch-of-play",
    "二过一 / 撞墙配合": "one_two",
}

ROOT_CAUSES = {
    "restart_delivery", "chaotic_chain", "normal_progression", "geometry",
    "concept_confusion", "rubric", "window_partial", "role_relation",
}

ROOT_CAUSE_BY_CLAIM = {
    ("13", "打身后 / 反越位跑位"): "restart_delivery",
    ("14", "打身后 / 反越位跑位"): "restart_delivery",
    ("SNGS-001", "快速反击"): "restart_delivery",
    ("SNGS-003", "破线传球"): "restart_delivery",
    ("SNGS-007", "快速反击"): "normal_progression",
    ("SNGS-020", "破线传球"): "restart_delivery",
    ("SNGS-022", "门将参与出球"): "restart_delivery",
    ("SNGS-023", "打身后 / 反越位跑位"): "restart_delivery",
    ("SNGS-025", "角球后点/前点战术"): "geometry",
    ("SNGS-028", "破线传球"): "geometry",
    ("SNGS-030", "虚跑/假跑扯动"): "role_relation",
    ("SNGS-039", "线间接应"): "role_relation",
    ("SNGS-040", "快速反击"): "normal_progression",
    ("SNGS-041", "快速反击"): "restart_delivery",
    ("SNGS-043", "快速反击"): "chaotic_chain",
    ("SNGS-060", "快速反击"): "chaotic_chain",
    ("SNGS-061", "快速反击"): "normal_progression",
    ("SNGS-061", "肋部渗透"): "geometry",
    ("SNGS-062", "肋部渗透"): "concept_confusion",
    ("SNGS-067", "角球后点/前点战术"): "geometry",
    ("SNGS-070", "破线传球"): "geometry",
    ("SNGS-075", "角球后点/前点战术"): "geometry",
    ("SNGS-078", "快速反击"): "normal_progression",
    ("SNGS-079", "角球后点/前点战术"): "geometry",
    ("SNGS-084", "快速反击"): "chaotic_chain",
    ("SNGS-087", "快速反击"): "normal_progression",
    ("SNGS-097", "破线传球"): "restart_delivery",
    ("SNGS-101", "打身后 / 反越位跑位"): "rubric",
    ("SNGS-103", "角球后点/前点战术"): "geometry",
    ("SNGS-106", "下底倒三角"): "restart_delivery",
    ("SNGS-110", "角球后点/前点战术"): "geometry",
    ("SNGS-115", "大范围转移"): "concept_confusion",
    ("SNGS-118", "快速反击"): "window_partial",
    ("SNGS-134", "快速反击"): "normal_progression",
    ("SNGS-134", "打身后 / 反越位跑位"): "concept_confusion",
    ("SNGS-140", "角球后点/前点战术"): "geometry",
    ("SNGS-177", "破线传球"): "concept_confusion",
    ("SNGS-190", "肋部渗透"): "concept_confusion",
    ("SNGS-200", "二过一 / 撞墙配合"): "role_relation",
}
