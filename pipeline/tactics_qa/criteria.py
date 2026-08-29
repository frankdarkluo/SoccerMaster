"""12 类战术证据判据：prompt 文本、JSON schema 与校验的单一真源。

真源是 `tactic_criteria.json`。这个模块从它派生三样东西，保证三者不会漂移：

* `response_schema()` 交给 `pipeline.video_models.generate_json`，用 schema 的 `required`
  从结构上强制模型输出全部 12 类、每类全部证据键 —— 模型没有"少答一类"这个选项。
* `prompt_block()` 生成 prompt 里的判据清单文本。
* `validate()` 在落盘前再查一遍，防止 provider 绕过 schema。

判据本身是研究决策，只在 JSON 里改，不要在生成物里改。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

CRITERIA_PATH = Path(__file__).with_name("tactic_criteria.json")

# 稳定命名而非位置编号：插入 / 删除条目时，已落盘数据里的键含义不会静默改变。
KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*$")

SCORE_MIN, SCORE_MAX = 0.0, 1.0


@lru_cache(maxsize=None)
def load(path: Path = CRITERIA_PATH) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [t["id"] for t in doc["tactics"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: 战术 id 重复")
    for tactic in doc["tactics"]:
        keys = [e["key"] for e in tactic["evidence"]]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{path}: {tactic['id']} 的证据 key 重复")
        if not keys:
            raise ValueError(f"{path}: {tactic['id']} 没有证据条目")
        bad = [k for k in keys if not KEY_PATTERN.fullmatch(k)]
        if bad:
            raise ValueError(f"{path}: {tactic['id']} 的证据 key 不是 snake_case：{bad}")
    return doc


def tactic_ids(doc: dict[str, Any] | None = None) -> list[str]:
    return [t["id"] for t in (doc or load())["tactics"]]


def names_cn(doc: dict[str, Any] | None = None) -> dict[str, str]:
    return {t["id"]: t["name_cn"] for t in (doc or load())["tactics"]}


def evidence_keys(tactic_id: str, doc: dict[str, Any] | None = None) -> list[str]:
    for tactic in (doc or load())["tactics"]:
        if tactic["id"] == tactic_id:
            return [e["key"] for e in tactic["evidence"]]
    raise KeyError(tactic_id)


def response_schema(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """每类一个必填字段，模型无法省略任何一类或任何一条证据。"""
    doc = doc or load()
    per_tactic = {}
    for tactic in doc["tactics"]:
        keys = [e["key"] for e in tactic["evidence"]]
        per_tactic[tactic["id"]] = {
            "type": "object",
            "properties": {
                "team": {"type": "string"},
                "evidence": {
                    "type": "object",
                    "properties": {
                        k: {"type": "number", "minimum": SCORE_MIN, "maximum": SCORE_MAX}
                        for k in keys
                    },
                    "required": keys,
                    "additionalProperties": False,
                },
                "window_start_s": {"type": "number"},
                "window_end_s": {"type": "number"},
                "note": {"type": "string"},
            },
            # Ark 的 strict json_schema 要求每个属性都出现在 required 里，
            # 所以 note 也是必填 —— 没什么可说时给空串。
            "required": ["team", "evidence", "window_start_s", "window_end_s", "note"],
            "additionalProperties": False,
        }
    ids = list(per_tactic)
    return {
        "type": "object",
        "properties": {
            "teams": {
                "type": "object",
                "properties": {"A": {"type": "string"}, "B": {"type": "string"}},
                "required": ["A", "B"],
                "additionalProperties": False,
            },
            "scores": {
                "type": "object",
                "properties": per_tactic,
                "required": ids,
                "additionalProperties": False,
            },
        },
        "required": ["teams", "scores"],
        "additionalProperties": False,
    }


def prompt_block(doc: dict[str, Any] | None = None) -> str:
    """prompt 里的「12 类证据清单」正文，也是 Markdown 文档里同名小节的内容。"""
    doc = doc or load()
    lines = []
    for tactic in doc["tactics"]:
        lines.append(f"### {tactic['id']} {tactic['name_cn']}")
        lines.append("")
        for item in tactic["evidence"]:
            lines.append(f"- {item['key']} {item['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate(payload: Any, doc: dict[str, Any] | None = None,
             *, duration_s: float | None = None) -> None:
    """校验一条模型输出。不合格直接抛错，绝不降级放行。

    静默放行的代价是那一类在这个 clip 上凭空消失，而候选清单照样算得出来，
    事后从 CSV 里看不出来 —— 所以这里宁可失败重试。
    """
    doc = doc or load()
    if not isinstance(payload, dict):
        raise ValueError(f"输出不是 JSON 对象，而是 {type(payload).__name__}")

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("缺少 scores 对象")

    expected = set(tactic_ids(doc))
    got = set(scores)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(f"scores 战术不匹配；缺少 {missing}；多出 {extra}")

    for tactic_id, entry in scores.items():
        where = f"scores.{tactic_id}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} 不是对象")

        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"{where}.evidence 缺失或不是对象")
        want = set(evidence_keys(tactic_id, doc))
        if set(evidence) != want:
            raise ValueError(
                f"{where}.evidence 键不匹配；缺少 {sorted(want - set(evidence))}；"
                f"多出 {sorted(set(evidence) - want)}"
            )
        for key, value in evidence.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{where}.evidence.{key} 不是数值：{value!r}")
            if not SCORE_MIN <= value <= SCORE_MAX:
                raise ValueError(f"{where}.evidence.{key} 越界：{value}")

        start, end = entry.get("window_start_s"), entry.get("window_end_s")
        for name, value in (("window_start_s", start), ("window_end_s", end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{where}.{name} 不是数值：{value!r}")
            if value < 0:
                raise ValueError(f"{where}.{name} 为负：{value}")
        if start > end:
            raise ValueError(f"{where} 窗口反向：{start} > {end}")
        if duration_s is not None and end > duration_s + 0.5:
            raise ValueError(f"{where}.window_end_s={end} 超出片段时长 {duration_s}")


if __name__ == "__main__":  # 生成判据清单，便于同步进 Markdown 文档
    print(prompt_block(), end="")
