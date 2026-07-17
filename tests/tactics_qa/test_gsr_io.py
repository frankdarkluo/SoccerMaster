import json

from pipeline.tactics_qa.gsr_io import GsrClip, load_gsr


def make_predictions(ball, players=()):
    fids = sorted(set(ball) | {row[0] for row in players})
    images = [{"image_id": f"900{f:07d}", "file_name": f"{f:06d}.jpg"} for f in fids]
    annotations = [
        {"image_id": f"900{f:07d}", "category_id": 4, "track_id": 99,
         "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
         "attributes": {"role": "ball", "team": None}}
        for f, (x, y) in ball.items()
    ]
    annotations += [
        {"image_id": f"900{f:07d}", "category_id": 1, "track_id": track,
         "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
         "attributes": {"role": "player", "team": team, "jersey": ""}}
        for f, track, team, x, y in players
    ]
    return {"info": {"name": "test", "n_frames": max(fids), "fps": 25},
            "images": images, "annotations": annotations, "categories": []}


def test_load_gsr_ball_and_players(tmp_path):
    data = make_predictions({1: (0.0, 0.0), 2: (1.0, 0.5)},
                            [(1, 7, "left", 10.0, 5.0)])
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(data))
    clip = load_gsr(path)
    assert isinstance(clip, GsrClip)
    assert clip.fps == 25
    assert clip.ball[1] == (0.0, 0.0)
    assert clip.players[1][0] == (7, "left", 10.0, 5.0)


def test_missing_ball_frames_stay_missing(tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(make_predictions({1: (0.0, 0.0), 5: (2.0, 0.0)})))
    assert 3 not in load_gsr(path).ball
