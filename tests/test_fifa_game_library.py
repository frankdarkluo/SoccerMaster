from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline import fifa_game_library as fifa
from scripts import run_fifa_game_library as cli


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fifa.INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _index_row(
    relative: str,
    digest: str,
    *,
    source_label: str = "Official label",
    tactic_id: str = "",
    mapping_status: str = "candidate_new",
) -> dict[str, str]:
    filename = Path(relative).name
    return {
        "video_path": f"FIFA Game Library/{relative}",
        "source_folder": relative.split("/", 1)[0],
        "filename": filename,
        "duration_s": "1.000",
        "sha256": digest,
        "duplicate_of": "",
        "raw_label": Path(filename).stem,
        "tactic_id": tactic_id,
        "label_status": "source_gt_reviewed" if tactic_id else "source_gt_unmapped",
        "description_zh": "",
        "description_source": "",
        "canonical_video_path": f"FIFA Game Library/{relative}",
        "analysis_result_path": f"analysis/{Path(relative).with_suffix('.json').as_posix()}",
        "source_label": source_label,
        "label_kind": "tactic",
        "label_role": "primary",
        "mapping_status": mapping_status,
        "evidence_spans_json": "",
    }


def _payload(
    source_labels: list[str],
    *,
    normalized_tactic_id: str = "",
    suggestions: bool = False,
) -> dict:
    return {
        "video_summary_zh": "视频展示了清晰可见的足球行动。",
        "source_labels": [
            {
                "source_label": label,
                "label_kind": "tactic",
                "label_role": "primary",
                "normalized_tactic_id": normalized_tactic_id,
                "description_zh": "进攻方通过连续移动和传递牵制防守，并清晰利用了由此形成的空间优势。",
                "confidence": 80,
                "evidence_spans": [],
                "evidence_gaps": [],
            }
            for label in source_labels
        ],
        "action_events": [],
        "model_suggestions": (
            [{
                "suggested_label_zh": "标题之外的模型建议",
                "normalized_tactic_id": normalized_tactic_id,
                "evidence_event_ids": [],
                "rationale_zh": "仅为模型观察，不是来源标签。",
                "confidence": 60,
            }]
            if suggestions else []
        ),
    }


def _write_grid(
    path: Path,
    *,
    include_candidate: bool = False,
    full_examples: int = 0,
) -> None:
    header = ["优先级", "id（英文标识）", "状态"]
    for marker in ("①", "②", "③", "④"):
        header.extend([f"正例{marker} 比赛", f"正例{marker} 描述", f"正例{marker} 来源"])
    active = ["P0", "active-tactic", "active"]
    for index in range(4):
        active.extend(
            [f"旧比赛{index + 1}", f"旧描述{index + 1}", f"旧来源{index + 1}"]
            if index < full_examples else ["", "", ""]
        )
    rows = [header, active]
    if include_candidate:
        rows.append(["", "candidate-tactic", "candidate_unreviewed", *([""] * 12)])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _setup_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relatives: tuple[str, ...] = ("A/one.mp4",),
) -> tuple[Path, Path, Path, Path, list[dict[str, str]]]:
    library = tmp_path / "library"
    rows = []
    for relative in relatives:
        video = library / relative
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(relative.encode())
        rows.append(_index_row(relative, fifa.sha256(video)))
    index = tmp_path / "index.csv"
    _write_index(index, rows)
    grid = tmp_path / "grid.csv"
    _write_grid(grid, include_candidate=True)
    output = tmp_path / "output"
    cards = {
        tactic_id: {
            "tactic_id": tactic_id,
            "definition": f"{tactic_id} definition",
            "observable_cues": ["cue"],
            "triggers": ["trigger"],
            "confusing": [],
        }
        for tactic_id in ("active-tactic", "candidate-tactic")
    }
    monkeypatch.setattr(fifa, "load_glossary", lambda _: cards)
    monkeypatch.setattr(fifa, "model_name", lambda provider: f"{provider}-model")
    return library, index, grid, output, rows


def test_prepare_counts_626_paths_575_hashes_and_51_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library = tmp_path / "library"
    folder = library / "A"
    folder.mkdir(parents=True)
    originals = []
    for index in range(575):
        path = folder / f"video-{index:03}.mp4"
        path.write_bytes(f"unique-video-{index}".encode())
        originals.append(path)
    for index in range(51):
        (folder / f"zz-copy-{index:03}.mp4").write_bytes(originals[index].read_bytes())
    monkeypatch.setattr(fifa, "FOLDERS", ("A",))
    monkeypatch.setattr(fifa, "APPROVED", {})
    monkeypatch.setattr(fifa, "APPROVED_SOURCE_LABELS", {})
    monkeypatch.setattr(fifa, "get_video_info", lambda _: {"duration_s": 2.5})

    index = tmp_path / "index.csv"
    result = fifa.prepare_index(library, index)
    with index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert result["videos"] == 626
    assert result["unique_hashes"] == 575
    assert result["duplicate_videos"] == 51
    assert len({row["video_path"] for row in rows}) == 626
    assert len({row["canonical_video_path"] for row in rows}) == 575
    assert sum(bool(row["duplicate_of"]) for row in rows) == 51


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("Counter-pressing", "tactic"),
        ("Breaking lines", "tactic"),
        ("Shooting", "outcome"),
        ("Goalkeeper save", "role"),
        ("Save", "outcome"),
        ("Goal-kick", "event"),
        ("Taking on", "technique"),
        ("Use of wide areas", "tactic"),
        ("Balls in behind", "tactic"),
        ("Counterattack", "tactic"),
        ("FU20WCTSGVerticalProgressivePlay001", "tactic"),
        ("Third-player combinations 1", "tactic"),
        ("Cut-back cross", "tactic"),
        ("Short corner", "tactic"),
        ("Defence-splitting passes 4", "tactic"),
        ("Free Kicks 1", "event"),
        ("Penalty kick 1", "event"),
        ("Throw-ins 1", "event"),
        ("Playmaker analysis 1", "role"),
        ("Wing play full back 1", "tactic"),
        ("Full-backs", "role"),
        ("First Touch 004", "technique"),
        ("Receiving between the lines", "tactic"),
        ("Centre-back building 1", "tactic"),
        ("Attacking space behind opposition full-backs", "tactic"),
        ("Reaction after loss", "tactic"),
        ("Verticality after regaining", "tactic"),
        ("Closing passing lines", "tactic"),
        ("Forcing a long pass", "tactic"),
        ("Blocking", "technique"),
        ("X-block", "technique"),
        ("Defending the area", "technique"),
        ("In-swing", "technique"),
        ("Outswing", "technique"),
        ("Explosiveness", "technique"),
        ("Milinković-Savić (2022)", "context"),
        ("Bright", "context"),
        ("Rolfo\u0308", "context"),
    ],
)
def test_title_label_kind_keeps_actions_out_of_tactics(label: str, kind: str):
    assert fifa._label_kind(label) == kind



def test_prepare_treats_nonfirst_official_concept_as_gt_and_replaces_old_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library = tmp_path / "library"
    video = library / "A/Team (2025) — Use of wide areas — Shooting.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    monkeypatch.setattr(fifa, "FOLDERS", ("A",))
    monkeypatch.setattr(fifa, "APPROVED", {})
    monkeypatch.setattr(fifa, "APPROVED_SOURCE_LABELS", {})
    monkeypatch.setattr(fifa, "get_video_info", lambda _: {"duration_s": 1.0})
    index = tmp_path / "index.csv"

    fifa.prepare_index(library, index, expected_videos=None)
    with index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["source_label"] == "Use of wide areas":
            row["label_kind"] = "context"
            row["mapping_status"] = "not_applicable"
    _write_index(index, rows)

    fifa.prepare_index(library, index, expected_videos=None)
    with index.open(encoding="utf-8-sig", newline="") as handle:
        by_label = {row["source_label"]: row for row in csv.DictReader(handle)}

    assert by_label["Team (2025)"]["label_kind"] == "context"
    assert by_label["Use of wide areas"]["label_kind"] == "tactic"
    assert by_label["Use of wide areas"]["mapping_status"] == "candidate_new"
    assert by_label["Shooting"]["label_kind"] == "outcome"


def test_validate_locks_action_source_gt_fields():
    inputs = [{
        "source_label": "Shooting",
        "source_tactic_id": "",
        "label_kind": "outcome",
        "label_role": "primary",
    }]
    payload = _payload(["Shooting"])
    payload["source_labels"][0]["normalized_tactic_id"] = "active-tactic"

    fifa.validate_analysis(payload, inputs, ["active-tactic"], 1.0)

    assert payload["source_labels"][0]["label_kind"] == "outcome"
    assert payload["source_labels"][0]["normalized_tactic_id"] == ""

    inputs[0].update({"label_kind": "context", "label_role": "context"})
    payload["source_labels"][0].update({
        "label_kind": "tactic",
        "label_role": "primary",
        "normalized_tactic_id": "active-tactic",
    })
    fifa.validate_analysis(payload, inputs, ["active-tactic"], 1.0)

    assert payload["source_labels"][0]["label_kind"] == "context"
    assert payload["source_labels"][0]["label_role"] == "context"
    assert payload["source_labels"][0]["normalized_tactic_id"] == ""


def test_validate_normalizes_unicode_label_and_restores_source_gt():
    source_label = "Rolfo\u0308"
    inputs = [{
        "source_label": source_label,
        "source_tactic_id": "",
        "label_kind": "tactic",
        "label_role": "primary",
    }]
    payload = _payload(["Rolfö"])

    fifa.validate_analysis(payload, inputs, [], 1.0)

    assert payload["source_labels"][0]["source_label"] == source_label


def test_validate_multilabel_spans_and_rejects_unknown_evidence_event():
    inputs = [
        {
            "source_label": "Counter-attack",
            "source_tactic_id": "counter-attack",
            "label_kind": "tactic",
            "label_role": "primary",
        },
        {
            "source_label": "Run in behind",
            "source_tactic_id": "run-in-behind",
            "label_kind": "tactic",
            "label_role": "secondary",
        },
    ]
    payload = _payload(["Counter-attack", "Run in behind"])
    payload["source_labels"][0]["normalized_tactic_id"] = "counter-attack"
    payload["source_labels"][0]["evidence_spans"] = [{
        "start_s": 0.1,
        "end_s": 0.4,
        "observation_zh": "夺回球权后立即向前推进。",
    }]
    payload["source_labels"][1]["normalized_tactic_id"] = "run-in-behind"
    payload["source_labels"][1]["label_role"] = "secondary"
    payload["source_labels"][1]["evidence_spans"] = [{
        "start_s": 0.3,
        "end_s": 0.8,
        "observation_zh": "无球队员向防线身后前插。",
    }]
    payload["action_events"] = [{
        "event_id": 1,
        "start_s": 0.1,
        "end_s": 0.8,
        "phase": "transition_attacking",
        "action_zh": "夺回球权后向前传递并前插。",
        "space_change_zh": "防线身后空间被利用。",
        "result_zh": "进攻推进至前场。",
    }]

    fifa.validate_analysis(
        payload,
        inputs,
        ["counter-attack", "run-in-behind"],
        1.0,
    )
    payload["model_suggestions"] = [{
        "suggested_label_zh": "标题外建议",
        "normalized_tactic_id": "",
        "evidence_event_ids": [2],
        "rationale_zh": "引用了不存在的事件。",
        "confidence": 50,
    }]
    with pytest.raises(ValueError, match="model suggestion"):
        fifa.validate_analysis(
            payload,
            inputs,
            ["counter-attack", "run-in-behind"],
            1.0,
        )


def test_migrate_reuses_18_results_and_creates_557_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rows = []
    approved = {}
    output = tmp_path / "output"
    for index in range(575):
        relative = f"A/clip-{index:03}.mp4"
        digest = f"sha-{index:03}"
        is_approved = index < 18
        rows.append(
            _index_row(
                relative,
                digest,
                source_label=f"Label {index}",
                tactic_id="active-tactic" if is_approved else "",
                mapping_status="direct_exact" if is_approved else "candidate_new",
            )
        )
        if is_approved:
            approved[relative] = ("active-tactic",)
            legacy = output / "gemini" / f"clip-{index:03}.json"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
                json.dumps({
                    "status": "success",
                    "clip_sha256": digest,
                    "model": "gemini-old",
                    "prompt_version": "old",
                    "prompt_sha256": f"prompt-{index}",
                    "analyses": [{
                        "tactic_id": "active-tactic",
                        "evidence_spans": [],
                        "evidence_gaps": [],
                        "positive_description_zh": "已有成功结果直接迁移，不进行任何新的外部模型调用。",
                    }],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
    index = tmp_path / "index.csv"
    _write_index(index, rows)
    monkeypatch.setattr(fifa, "APPROVED", approved)
    monkeypatch.setattr(
        fifa,
        "generate_json",
        lambda *args, **kwargs: pytest.fail("migration must not call an API"),
    )

    result = fifa.migrate_results(index, output)

    assert result == {
        "clips": 575,
        "migrated_success": 18,
        "created_pending": 557,
        "existing": 0,
        "recovered": 0,
    }
    analyses = list((output / "analysis").rglob("*.json"))
    assert len(analyses) == 575
    assert sum(json.loads(path.read_text())["status"] == "success" for path in analyses) == 18
    assert sum(json.loads(path.read_text())["status"] == "pending" for path in analyses) == 557



def test_migrate_refreshes_success_metadata_without_reprocessing(tmp_path: Path):
    row = _index_row(
        "A/approved.mp4",
        "sha-approved",
        tactic_id="active-tactic",
        mapping_status="direct_exact",
    )
    row["label_kind"] = "context"
    index = tmp_path / "index.csv"
    _write_index(index, [row])
    output = tmp_path / "output"
    target = output / row["analysis_result_path"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({
        "status": "success",
        "schema_version": fifa.SCHEMA_VERSION,
        "video_path": "stale-path",
        "clip_sha256": row["sha256"],
        "source_titles": [],
        "source_ground_truth": [{
            "source_label": row["source_label"],
            "label_kind": "context",
            "label_role": "primary",
            "tactic_id": row["tactic_id"],
        }],
        "source_labels": [{"source_label": row["source_label"]}],
        "provider": "preserved-provider",
    }), encoding="utf-8")

    result = fifa.migrate_results(index, output)
    refreshed = json.loads(target.read_text(encoding="utf-8"))

    assert result["existing"] == 1
    assert refreshed["provider"] == "preserved-provider"
    assert refreshed["source_ground_truth"][0]["label_kind"] == "tactic"
    assert refreshed["video_path"] == row["canonical_video_path"]

def test_migrate_bad_json_requires_force_and_recovers(tmp_path: Path):
    index = tmp_path / "index.csv"
    _write_index(index, [_index_row("A/bad.mp4", "sha-bad")])
    output = tmp_path / "output"
    target = output / "analysis/A/bad.json"
    target.parent.mkdir(parents=True)
    target.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="use force"):
        fifa.migrate_results(index, output)

    assert fifa.migrate_results(index, output, force=True) == {
        "clips": 1,
        "migrated_success": 0,
        "created_pending": 1,
        "existing": 0,
        "recovered": 1,
    }
    assert json.loads(target.read_text())["status"] == "pending"



def test_targeted_force_migration_does_not_touch_unrelated_bad_json(tmp_path: Path):
    selected = _index_row("A/selected.mp4", "sha-selected")
    unrelated = _index_row("A/unrelated.mp4", "sha-unrelated")
    index = tmp_path / "index.csv"
    _write_index(index, [selected, unrelated])
    output = tmp_path / "output"
    selected_target = output / selected["analysis_result_path"]
    unrelated_target = output / unrelated["analysis_result_path"]
    selected_target.parent.mkdir(parents=True)
    selected_target.write_text("{bad selected", encoding="utf-8")
    unrelated_target.write_text("{bad unrelated", encoding="utf-8")

    result = fifa.migrate_results(
        index,
        output,
        force=True,
        clip_shas=[selected["sha256"]],
    )

    assert result["clips"] == 1
    assert json.loads(selected_target.read_text(encoding="utf-8"))["status"] == "pending"
    assert unrelated_target.read_text(encoding="utf-8") == "{bad unrelated"


def test_hard_timeout_is_a_total_deadline():
    import time

    with pytest.raises(TimeoutError, match="exceeded"):
        with fifa._hard_timeout(0.01):
            time.sleep(0.1)
    time.sleep(0.02)


@pytest.mark.parametrize(
    ("gemini_keys", "ark_key", "message"),
    [
        (("", "", ""), "ark", "three distinct Gemini"),
        (("same", "same", "other"), "ark", "three distinct Gemini"),
        (("one", "two", "three"), "", "ARK_API_KEY"),
    ],
)
def test_run_preflights_three_distinct_gemini_keys_and_ark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gemini_keys: tuple[str, str, str],
    ark_key: str,
    message: str,
):
    library, index, grid, output, _ = _setup_run(tmp_path, monkeypatch)
    for name, value in zip(
        ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2"),
        gemini_keys,
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ARK_API_KEY", ark_key)
    monkeypatch.setattr(
        fifa,
        "generate_json",
        lambda *args, **kwargs: pytest.fail("preflight must happen before an API call"),
    )

    with pytest.raises(RuntimeError, match=message):
        fifa.run_library(library, index, grid, output)


def test_run_filters_sha_falls_back_and_keeps_suggestions_out_of_gt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library, index, grid, output, rows = _setup_run(
        tmp_path, monkeypatch, ("A/first.mp4", "A/second.mp4"),
    )
    for offset, name in enumerate(
        ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2"),
    ):
        monkeypatch.setenv(name, f"gemini-{offset}")
    monkeypatch.setenv("ARK_API_KEY", "doubao")
    calls = []

    def generate(provider, prompt, schema, *, video_path, **kwargs):
        calls.append((provider, prompt, video_path.name))
        assert kwargs == ({"retries": 0} if provider == "doubao" else {})
        assert re.fullmatch(r"clip-[0-9a-f]{16}\.mp4", video_path.name)
        if provider == "gemini":
            raise RuntimeError("Gemini exhausted")
        labels = schema["properties"]["source_labels"]["items"]["properties"][
            "source_label"
        ]["enum"]
        return _payload(labels, normalized_tactic_id="active-tactic", suggestions=True), [], 1

    monkeypatch.setattr(fifa, "generate_json", generate)
    proxies = []

    def proxy(source, target):
        proxies.append((source.name, target.name))
        target.write_bytes(source.read_bytes())

    monkeypatch.setattr(fifa, "_doubao_proxy", proxy)
    target_sha = rows[1]["sha256"]

    result = fifa.run_library(
        library,
        index,
        grid,
        output,
        clip_shas=[target_sha],
    )

    assert result == {
        "clips": 1,
        "api_calls": 2,
        "gemini_calls": 1,
        "doubao_calls": 1,
        "success": 1,
        "failed": 0,
        "skipped": 0,
    }
    assert [provider for provider, _, _ in calls] == ["gemini", "doubao"]
    assert proxies == [("second.mp4", f"clip-{target_sha[:16]}.mp4")]
    assert all("active-tactic" in prompt for _, prompt, _ in calls)
    assert all("candidate-tactic" not in prompt for _, prompt, _ in calls)
    first_path = output / "analysis/A/first.json"
    second = json.loads((output / "analysis/A/second.json").read_text())
    assert not first_path.exists()
    assert second["provider"] == "doubao"
    assert second["upload_profile"] == fifa.DOUBAO_UPLOAD_PROFILE
    assert second["source_ground_truth"][0]["tactic_id"] == ""
    assert second["source_labels"][0]["mapping_status"] == "model_mapped_unreviewed"
    assert second["model_suggestions"][0]["suggested_label_zh"] == "标题之外的模型建议"


def test_run_retries_invalid_gemini_once_before_doubao(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library, index, grid, output, rows = _setup_run(tmp_path, monkeypatch)
    for offset, name in enumerate(
        ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2"),
    ):
        monkeypatch.setenv(name, f"gemini-{offset}")
    monkeypatch.setenv("ARK_API_KEY", "doubao")
    calls = []

    def generate(provider, prompt, schema, *, video_path, **kwargs):
        calls.append(provider)
        labels = schema["properties"]["source_labels"]["items"]["properties"][
            "source_label"
        ]["enum"]
        payload = _payload(labels)
        if len(calls) == 1:
            payload["source_labels"] = []
        return payload, [], 1

    monkeypatch.setattr(fifa, "generate_json", generate)

    result = fifa.run_library(library, index, grid, output)

    assert calls == ["gemini", "gemini"]
    assert result["api_calls"] == 2
    assert result["doubao_calls"] == 0
    saved = json.loads((output / rows[0]["analysis_result_path"]).read_text())
    assert saved["provider"] == "gemini"
    assert saved["validation_retries"] == 1


def test_run_retries_invalid_doubao_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library, index, grid, output, rows = _setup_run(tmp_path, monkeypatch)
    for offset, name in enumerate(
        ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2"),
    ):
        monkeypatch.setenv(name, f"gemini-{offset}")
    monkeypatch.setenv("ARK_API_KEY", "doubao")
    calls = []

    def generate(provider, prompt, schema, *, video_path, **kwargs):
        calls.append(provider)
        if provider == "gemini":
            raise RuntimeError("Gemini exhausted")
        labels = schema["properties"]["source_labels"]["items"]["properties"][
            "source_label"
        ]["enum"]
        payload = _payload(labels)
        if calls.count("doubao") == 1:
            payload["source_labels"] = []
        return payload, [], 1

    monkeypatch.setattr(fifa, "generate_json", generate)
    monkeypatch.setattr(
        fifa, "_doubao_proxy",
        lambda source, target: target.write_bytes(source.read_bytes()),
    )

    result = fifa.run_library(library, index, grid, output)

    assert calls == ["gemini", "doubao", "doubao"]
    assert result["doubao_calls"] == 2
    assert result["success"] == 1
    saved = json.loads((output / rows[0]["analysis_result_path"]).read_text())
    assert saved["provider"] == "doubao"
    assert saved["validation_retries"] == 1


def test_duplicate_titles_share_one_call_and_one_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library = tmp_path / "library"
    video = library / "A/canonical.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"same-video")
    digest = fifa.sha256(video)
    first = _index_row(
        "A/canonical.mp4",
        digest,
        source_label="Counter-attack",
    )
    second = _index_row(
        "B/alias-title.mp4",
        digest,
        source_label="Run in behind",
    )
    second.update({
        "duplicate_of": first["video_path"],
        "canonical_video_path": first["canonical_video_path"],
        "analysis_result_path": first["analysis_result_path"],
    })
    index = tmp_path / "index.csv"
    _write_index(index, [first, second])
    grid = tmp_path / "grid.csv"
    _write_grid(grid)
    output = tmp_path / "output"
    monkeypatch.setattr(fifa, "load_glossary", lambda _: {
        "active-tactic": {
            "tactic_id": "active-tactic",
            "definition": "definition",
            "observable_cues": ["cue"],
            "triggers": [],
            "confusing": [],
        },
    })
    monkeypatch.setattr(fifa, "model_name", lambda provider: f"{provider}-model")
    for offset, name in enumerate(
        ("GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2"),
    ):
        monkeypatch.setenv(name, f"gemini-{offset}")
    monkeypatch.setenv("ARK_API_KEY", "doubao")
    calls = []

    def generate(provider, prompt, schema, *, video_path):
        calls.append(provider)
        labels = schema["properties"]["source_labels"]["items"]["properties"][
            "source_label"
        ]["enum"]
        return _payload(labels), [], 1

    monkeypatch.setattr(fifa, "generate_json", generate)

    result = fifa.run_library(library, index, grid, output)

    assert result["clips"] == 1
    assert calls == ["gemini"]
    files = list((output / "analysis").rglob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["source_titles"] == ["canonical", "alias-title"]
    assert {item["source_label"] for item in payload["source_ground_truth"]} == {
        "Counter-attack",
        "Run in behind",
    }


def test_report_summarises_results_and_lists_failures_and_candidates(tmp_path: Path):
    rows = [
        _index_row("A/success.mp4", "sha-success"),
        _index_row("A/failed.mp4", "sha-failed"),
        _index_row("A/missing.mp4", "sha-missing"),
    ]
    index = tmp_path / "index.csv"
    _write_index(index, rows)
    output = tmp_path / "output"
    success = output / rows[0]["analysis_result_path"]
    failed = output / rows[1]["analysis_result_path"]
    success.parent.mkdir(parents=True)
    success.write_text(
        json.dumps({
            "status": "success",
            "provider": "gemini",
            "clip_sha256": "sha-success",
            **_payload(["Official label"], suggestions=True),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps({
            "status": "failed",
            "provider": "doubao",
            "clip_sha256": "sha-failed",
            "error_type": "RuntimeError",
            "error": "both providers failed",
            "source_labels": [{"source_label": "Official label"}],
            "model_suggestions": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    grid = tmp_path / "grid.csv"
    _write_grid(grid, include_candidate=True)
    summary = fifa.report(index, output, grid)

    assert summary["physical_videos"] == 3
    assert summary["unique_hashes"] == 3
    assert summary["expected_analyses"] == 3
    assert summary["attempted"] == 2
    assert summary["missing"] == 1
    assert summary["statuses"] == {"failed": 1, "success": 1}
    assert summary["providers"] == {"doubao": 1, "gemini": 1}
    assert summary["evidence"]["tactic_labels"] == 1
    assert (output / "evidence_spans.csv").is_file()
    assert summary["model_suggestions"] == 1
    assert summary["candidate_grid_entries"][0]["tactic_id"] == "candidate-tactic"
    assert summary["failed_records"][0]["video_path"] == "FIFA Game Library/A/failed.mp4"
    assert any(
        item.get("source_label") == "Official label"
        for item in summary["candidate_labels"]
    )
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()


def test_apply_grid_preserves_old_examples_and_caps_total_at_six(tmp_path: Path):
    rows = [
        _index_row(
            f"A/example-{index}.mp4",
            f"sha-{index}",
            tactic_id="active-tactic",
            mapping_status="direct_exact",
        )
        for index in range(3)
    ]
    rows.append(_index_row("A/model-only.mp4", "sha-model"))
    index = tmp_path / "index.csv"
    _write_index(index, rows)
    output = tmp_path / "output"
    for number, row in enumerate(rows):
        target = output / row["analysis_result_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload(
            [row["source_label"]],
            normalized_tactic_id="active-tactic",
        )
        payload["source_labels"][0]["mapping_status"] = (
            "direct_exact" if number < 3 else "model_mapped_unreviewed"
        )
        target.write_text(
            json.dumps({
                "status": "success",
                "provider": "gemini",
                "clip_sha256": row["sha256"],
                **payload,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    grid = tmp_path / "grid.csv"
    _write_grid(grid, full_examples=4)
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        before = next(csv.DictReader(handle))

    result = fifa.apply_grid(index, grid, output)

    assert result["grid_mappings_inserted"] == 2
    assert result["grid_full_skipped"] == 1
    assert result["duplicate_sha_skipped"] == 0
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        after = next(csv.DictReader(handle))
    for marker in ("①", "②", "③", "④"):
        for suffix in ("比赛", "描述", "来源"):
            assert after[f"正例{marker} {suffix}"] == before[f"正例{marker} {suffix}"]
    assert after["正例⑤ 比赛"] == "example-0"
    assert after["正例⑥ 比赛"] == "example-1"
    assert sum(bool(after[f"正例{marker} 比赛"]) for marker in "①②③④⑤⑥") == 6
    assert "model-only" not in after.values()




def test_apply_grid_accepts_only_explicitly_approved_model_mapping(tmp_path: Path):
    approved = _index_row(
        "A/approved.mp4", "sha-approved", source_label="Clear alias",
    )
    rejected = _index_row(
        "A/rejected.mp4", "sha-rejected", source_label="Ambiguous label",
    )
    index = tmp_path / "index.csv"
    _write_index(index, [approved, rejected])
    output = tmp_path / "output"
    for row in (approved, rejected):
        target = output / row["analysis_result_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload([row["source_label"]], normalized_tactic_id="active-tactic")
        payload["source_labels"][0].update({
            "mapping_status": "model_mapped_unreviewed",
            "evidence_spans": [{
                "start_s": 0.1,
                "end_s": 0.5,
                "observation_zh": "清晰战术证据。",
            }],
        })
        target.write_text(json.dumps({
            "status": "success",
            "provider": "gemini",
            "clip_sha256": row["sha256"],
            **payload,
        }, ensure_ascii=False), encoding="utf-8")
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fifa.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerow({
            "source_label": "Clear alias",
            "proposed_tactic_id": "active-tactic",
            "decision": "approve",
            "note": "official title is an exact synonym",
        })
    grid = tmp_path / "grid.csv"
    _write_grid(grid)

    result = fifa.apply_grid(index, grid, output, review)

    assert result["reviewed_mappings_recorded"] == 1
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["正例① 比赛"] == "approved"
    assert "rejected" not in row.values()
    assert "证据 0.1–0.5s" in row["正例① 来源"]
    with index.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = next(
            row for row in csv.DictReader(handle) if row["source_label"] == "Clear alias"
        )
    assert fifa._source_inputs([reviewed])[0]["source_tactic_id"] == "active-tactic"

def test_apply_grid_ranks_mechanism_evidence_before_confidence(tmp_path: Path):
    low = _index_row(
        "A/Low Team (2024) — Example.mp4",
        "sha-low",
        tactic_id="active-tactic",
        mapping_status="direct_exact",
    )
    high = _index_row(
        "A/High Team (2024) — Example.mp4",
        "sha-high",
        tactic_id="active-tactic",
        mapping_status="direct_exact",
    )
    index = tmp_path / "index.csv"
    _write_index(index, [low, high])
    output = tmp_path / "output"
    for row, confidence, clear in ((low, 20, True), (high, 95, False)):
        target = output / row["analysis_result_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload([row["source_label"]])
        payload["source_labels"][0].update({
            "normalized_tactic_id": "active-tactic",
            "mapping_status": "direct_exact",
            "confidence": confidence,
            "evidence_spans": ([{
                "start_s": 0.1,
                "end_s": 0.5,
                "observation_zh": "清晰战术证据。",
            }] if clear else []),
            "evidence_gaps": [] if clear else ["缺少机制证据"],
        })
        target.write_text(json.dumps({
            "status": "success",
            "provider": "gemini",
            "clip_sha256": row["sha256"],
            **payload,
        }, ensure_ascii=False), encoding="utf-8")
    grid = tmp_path / "grid.csv"
    _write_grid(grid, full_examples=4)
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        grid_rows = list(csv.reader(handle))
    header = grid_rows[0]
    fifa._extend_grid_header(header)
    active = grid_rows[1]
    active.extend([""] * (len(header) - len(active)))
    fifth = header.index("正例⑤ 比赛")
    active[fifth:fifth + 3] = ["旧比赛5", "旧描述5", "旧来源5"]
    with grid.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(grid_rows)

    result = fifa.apply_grid(index, grid, output)

    assert result["grid_mappings_inserted"] == 1
    assert result["grid_full_skipped"] == 1
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["正例⑥ 比赛"] == "Low Team (2024) — Example"



def test_apply_grid_uses_existing_team_and_edition_for_diversity(tmp_path: Path):
    same = _index_row(
        "A/Same Team (2025) — Short.mp4",
        "sha-same",
        tactic_id="active-tactic",
        mapping_status="direct_exact",
    )
    diverse = _index_row(
        "A/Diverse Team (2025) — Long.mp4",
        "sha-diverse",
        tactic_id="active-tactic",
        mapping_status="direct_exact",
    )
    same["duration_s"], diverse["duration_s"] = "0.100", "1.000"
    index = tmp_path / "index.csv"
    _write_index(index, [same, diverse])
    output = tmp_path / "output"
    for item in (same, diverse):
        target = output / item["analysis_result_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload([item["source_label"]], normalized_tactic_id="active-tactic")
        payload["source_labels"][0].update({
            "mapping_status": "direct_exact",
            "evidence_spans": [{
                "start_s": 0.1,
                "end_s": 0.5,
                "observation_zh": "同等清晰的机制证据。",
            }],
        })
        target.write_text(json.dumps({
            "status": "success",
            "provider": "gemini",
            "clip_sha256": item["sha256"],
            **payload,
        }, ensure_ascii=False), encoding="utf-8")

    grid = tmp_path / "grid.csv"
    _write_grid(grid, full_examples=4)
    with grid.open(encoding="utf-8-sig", newline="") as handle:
        grid_rows = list(csv.reader(handle))
    fifa._extend_grid_header(grid_rows[0])
    grid_rows[1].extend(["", "", "", "", "", ""])
    fifth = grid_rows[0].index("正例⑤ 比赛")
    grid_rows[1][fifth:fifth + 3] = [
        "Same Team (2024) — Existing", "旧描述5", "旧来源5",
    ]
    with grid.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(grid_rows)

    fifa.apply_grid(index, grid, output)

    with grid.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["正例⑥ 比赛"] == "Diverse Team (2025) — Long"

def test_cli_routes_prepare_run_report_and_apply_grid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls = []
    monkeypatch.setattr(cli, "prepare_index", lambda *args: calls.append(("prepare", args)) or {"videos": 626})
    monkeypatch.setattr(cli, "migrate_results", lambda *args: calls.append(("migrate", args)) or {"clips": 575})
    monkeypatch.setattr(cli, "run_library", lambda *args, **kwargs: calls.append(("run", args, kwargs)) or {"clips": 2})
    monkeypatch.setattr(cli, "report", lambda *args: calls.append(("report", args)) or {"attempted": 575})
    monkeypatch.setattr(cli, "apply_grid", lambda *args: calls.append(("apply-grid", args)) or {"grid_mappings_inserted": 1})

    monkeypatch.setattr("sys.argv", ["run_fifa_game_library.py", "prepare"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["analysis"]["clips"] == 575

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_fifa_game_library.py", "run", "--allow-external-upload",
            "--clip-sha", "sha-a", "--clip-sha", "sha-b", "--retry-failed",
        ],
    )
    cli.main()
    capsys.readouterr()
    run_call = next(call for call in calls if call[0] == "run")
    assert run_call[2]["clip_shas"] == ["sha-a", "sha-b"]
    assert run_call[2]["retry_failed"] is True

    for command, name in (("report", "report"), ("apply-grid", "apply-grid")):
        monkeypatch.setattr("sys.argv", ["run_fifa_game_library.py", command])
        cli.main()
        capsys.readouterr()
        assert any(call[0] == name for call in calls)


def test_cli_refuses_upload_without_explicit_consent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["run_fifa_game_library.py", "run"])
    with pytest.raises(SystemExit, match="allow-external-upload"):
        cli.main()


def test_visual_scan_streams_video_without_frame_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    library = tmp_path / "library"
    folder = library / "A"
    folder.mkdir(parents=True)
    video = folder / "short.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (96, 64),
    )
    assert writer.isOpened()
    for index in range(12):
        color = (230, 230, 230) if index == 3 else (20, 110 + index, 20)
        frame = np.full((64, 96, 3), color, dtype=np.uint8)
        cv2.circle(frame, (20 + index * 3, 32), 8, (0, 0, 255), 2)
        cv2.line(frame, (5, 55), (90, 10 + index), (0, 0, 255), 2)
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(fifa, "FOLDERS", ("A",))

    output = tmp_path / "output"
    result = fifa.visual_scan(
        library, output, expected_videos=1, sample_hz=1.0, evidence_cap=2,
    )

    evidence = output / "visual_patterns/evidence"
    assert result["videos_scanned"] == 1
    assert result["frames_expected"] == 12
    assert result["frames_decoded"] == 12
    assert result["frames_analysed"] >= 3
    assert not list(library.rglob("*.jpg"))
    assert all(len(list(directory.glob("*.jpg"))) <= 2 for directory in evidence.iterdir())
