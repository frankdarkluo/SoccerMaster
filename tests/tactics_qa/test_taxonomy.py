from pipeline.tactics_qa.taxonomy import TACTIC_ID_BY_ZH, ROOT_CAUSE_BY_CLAIM, ROOT_CAUSES


def test_all_csv_tactic_names_mapped():
    names = [
        "快速反击", "破线传球", "打身后 / 反越位跑位", "线间接应",
        "门将参与出球", "角球后点/前点战术", "虚跑/假跑扯动", "下底倒三角",
        "肋部渗透", "大范围转移", "二过一 / 撞墙配合",
    ]
    for name in names:
        assert name in TACTIC_ID_BY_ZH, name


def test_root_cause_table_covers_39_errors():
    assert len(ROOT_CAUSE_BY_CLAIM) == 39
    assert set(ROOT_CAUSE_BY_CLAIM.values()) <= ROOT_CAUSES


def test_duplicate_clips_disambiguated_by_tactic():
    assert ("SNGS-061", "快速反击") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-061", "肋部渗透") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-134", "快速反击") in ROOT_CAUSE_BY_CLAIM
    assert ("SNGS-134", "打身后 / 反越位跑位") in ROOT_CAUSE_BY_CLAIM
