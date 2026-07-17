from pipeline.tactics_qa.possession_timeline import team_runs, transition_events


def run(team, start, end):
    return {"team": team, "start_s": start, "end_s": end}


def test_team_runs_merge_same_team_players():
    runs = team_runs([(1, "left", 0, 49), (2, "left", 55, 99),
                      (3, "right", 110, 200)], 25)
    assert [r["team"] for r in runs] == ["left", "right"]
    assert runs[0]["start_s"] == 0.0 and runs[0]["end_s"] == 99 / 25


def test_controlled_regain():
    events = transition_events([run("right", 0, 5), run("left", 5.5, 9)], "left")
    assert [(e.kind, e.team) for e in events] == [("controlled_recovery", "attacking")]


def test_short_alternation_is_chaos_then_stable_regain():
    runs = [run("right", 0, 3.5), run("left", 3.7, 4.3),
            run("right", 4.5, 5.1), run("left", 5.3, 5.8),
            run("right", 6.0, 6.5), run("left", 6.7, 12)]
    events = transition_events(runs, "left")
    assert [e.kind for e in events].count("contested_touch") >= 3
    assert events[-1].kind == "controlled_recovery"


def test_no_transition():
    assert transition_events([run("left", 0, 12)], "left") == []
