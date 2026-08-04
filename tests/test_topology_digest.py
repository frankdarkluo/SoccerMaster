from pipeline.topology.digest import build_digest, render_digest, window_digest


def line(role, team, x_values, status="renderable", gaps=None):
    return {
        "line_id": f"w:{role}",
        "team": team,
        "role": role,
        "status": status,
        "members": [{"x_m": x, "y_m": 0.0} for x in x_values],
        "lateral_span_m": round(max(x_values) - min(x_values), 2) if x_values else 0.0,
        "evidence_gaps": gaps or [],
    }


def topology(windows, position_source="gt"):
    return {"position_source": position_source, "windows": windows}


def test_build_digest_aggregates_renderable_lines_and_gap_zone():
    window = {
        "start_s": 1.5,
        "attacking_team": "left",
        "lines": [
            line("attacking_forward", "left", [40.0, 44.0]),
            line("attacking_midfield", "left", [30.0, 32.0]),
            line("defending_defensive", "right", [46.0, 48.0]),
        ],
        "zones": [{"type": "inter_line_space", "polygon_pitch": [[31.0, -5], [42.0, -5], [42.0, 5], [31.0, 5]]}],
    }

    rows = build_digest(topology([window]))

    assert rows == [{
        "t": 1.5,
        "attacking_team": "left",
        "forward": {"team": "left", "x_m": 42.0, "span_m": 4.0},
        "midfield": {"team": "left", "x_m": 31.0, "span_m": 2.0},
        "last_line": {"team": "right", "x_m": 47.0, "span_m": 2.0},
        "inter_line_gap_m": 11.0,
    }]


def test_build_digest_skips_non_renderable_lines_and_surfaces_evidence_gaps():
    window = {
        "start_s": 0.0,
        "attacking_team": None,
        "lines": [line("attacking_forward", "left", [], status="candidate", gaps=["fewer than two stable visible members"])],
        "zones": [],
    }

    rows = build_digest(topology([window]))

    assert rows == [{"t": 0.0, "attacking_team": None, "evidence_gaps": ["fewer than two stable visible members"]}]


def test_digest_surfaces_measured_ball_position_and_movement():
    window = {
        "start_s": 4.0, "attacking_team": "left", "lines": [], "zones": [],
        "ball": {"x_m": -52.1, "y_m": 33.7, "displacement_m": 0.2, "sample_count": 25},
    }

    rows = build_digest(topology([window]))

    assert rows[0]["ball"] == window["ball"]
    assert "ball=x-52.1m/y33.7m/move0.2m" in render_digest(topology([window]))


def test_render_digest_reports_absence_honestly_without_windows():
    text = render_digest(topology([], position_source="soccer_master"))

    assert text == "position_source=soccer_master\nno renderable topology windows (insufficient ball/goalkeeper/possession evidence)"


def test_render_digest_produces_one_compact_line_per_window():
    window = {
        "start_s": 2.0,
        "attacking_team": "right",
        "lines": [line("defending_defensive", "left", [10.0])],
        "zones": [],
    }

    text = render_digest(topology([window]))

    assert text == "position_source=gt\nt=2.0s attack=right last_line[left]=x10.0m/span0.0m"


def test_window_digest_keeps_overlapping_windows_and_rezeros_time():
    windows = [
        {"start_s": 0.0, "end_s": 1.0, "attacking_team": "left", "lines": [], "zones": []},
        {"start_s": 5.0, "end_s": 6.0, "attacking_team": "right", "lines": [], "zones": []},
        {"start_s": 5.5, "end_s": 6.5, "attacking_team": "right", "lines": [], "zones": []},
        {"start_s": 10.0, "end_s": 11.0, "attacking_team": "left", "lines": [], "zones": []},
    ]

    windowed = window_digest(topology(windows, position_source="soccer_master"), 5.0, 6.5, time_scale=6)

    assert windowed["position_source"] == "soccer_master"
    assert [(w["start_s"], w["end_s"]) for w in windowed["windows"]] == [(0.0, 6.0), (3.0, 9.0)]


def test_window_digest_clamps_a_partially_overlapping_window_to_zero():
    windows = [{"start_s": 4.5, "end_s": 5.5, "attacking_team": "left", "lines": [], "zones": []}]

    windowed = window_digest(topology(windows), 5.0, 6.0)

    assert [(w["start_s"], w["end_s"]) for w in windowed["windows"]] == [(0.0, 0.5)]
