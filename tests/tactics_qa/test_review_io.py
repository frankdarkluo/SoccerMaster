import io

from pipeline.tactics_qa.review_io import parse_review_rows

SAMPLE = (
    "id,tactics,时间范围,判断依据,置信度,识别准确度,更正\n"
    "SNGS-001,快速反击,00:13–00:23,白队在中场对抗后夺回球权。,中高,错误,开球后开大脚打到前场\n"
    "SNGS-045,打身后 / 反越位跑位,00:20–00:25,前锋从后卫线前沿启动。,高,正确,\n"
)


def test_parse_review_rows():
    rows = parse_review_rows(io.StringIO(SAMPLE))
    assert len(rows) == 2
    row = rows[0]
    assert row["clip_id"] == "SNGS-001"
    assert row["tactic_id"] == "fast_break_pattern"
    assert row["window"] == {"start_s": 13.0, "end_s": 23.0, "raw": "00:13–00:23"}
    assert row["verdict"] == "wrong"
    assert row["root_cause"] == "restart_delivery"
    assert rows[1]["verdict"] == "correct"
    assert rows[1]["root_cause"] is None


def test_numeric_ids_map_to_event_clips_source():
    sample = io.StringIO(
        "id,tactics,时间范围,判断依据,置信度,识别准确度,更正\n"
        "6,快速反击,00:01–00:12,解围后反击。,中高,正确,\n"
    )
    assert parse_review_rows(sample)[0]["clip_uid"] == "event_clips:0006"
