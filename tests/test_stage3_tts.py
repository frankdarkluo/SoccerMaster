import json

from pipeline.stage3_tts.run import _load_segments


def test_current_commentary_record_becomes_one_tts_segment(tmp_path):
    record = tmp_path / "commentary.json"
    record.write_text(json.dumps({
        "commentary_zh": "白队通过快速反击推进，并迫使对手持续回撤。",
        "locked_fact": {
            "tactic_zh": "快速反击",
            "window": {"start_s": 2.0, "end_s": 6.0},
        },
    }), encoding="utf-8")

    assert _load_segments(record, 12.0) == [{
        "timestamp_s": 2.0,
        "end_s": 12.0,
        "text_zh": "白队通过快速反击推进，并迫使对手持续回撤。",
        "fallback_text_zh": "快速反击形成威胁。",
        "human_review_reference": {
            "tactic_zh": "快速反击",
            "window": {"start_s": 2.0, "end_s": 6.0},
        },
    }]
