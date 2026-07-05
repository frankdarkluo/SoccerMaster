from pipeline.stage2_events.evidence import (
    load_homography, project_pitch_to_image, build_bbox_index, build_evidence_frames,
)
from pipeline.stage2_events.types import Event
from pipeline.stage2_events.schema import EventSchema
from pipeline.stage2_events.detector import EventDetector


def _count_color_pixels(img, x0, y0, x1, y1, predicate):
    region = img[y0:y1, x0:x1].reshape(-1, 3)
    return sum(1 for b, g, r in region if predicate(b, g, r))


def test_project_pitch_to_image_uses_h(homography_file):
    homo = load_homography(str(homography_file))
    # fixture H maps pitch (0,0) -> image (960,540)
    xy = project_pitch_to_image(homo, "3148000001", 0.0, 0.0)
    assert xy is not None
    x, y = xy
    assert abs(x - 960) < 1.0 and abs(y - 540) < 1.0


def test_build_bbox_index_maps_frame_track(predictions_file):
    idx = build_bbox_index(str(predictions_file))
    assert (1, 7) in idx and "w" in idx[(1, 7)]


def test_build_evidence_frames_writes_boxed_jpgs(tmp_path, predictions_file,
                                                  homography_file, frames_dir):
    events = EventDetector(EventSchema(), fps=25).detect(str(predictions_file))
    pev = next(e for e in events if e.event_code == "football.pass")
    idx = build_bbox_index(str(predictions_file))
    homo = load_homography(str(homography_file))
    paths = build_evidence_frames(pev, frames_dir, idx, homo,
                                  str(predictions_file), tmp_path / "burst",
                                  fps=25, window_s=0.2)
    assert paths and all(p.exists() for p in paths)
    import cv2
    img = cv2.imread(str(paths[0]))
    assert img is not None and img.shape[2] == 3
    assert _count_color_pixels(
        img, 98, 198, 142, 302,
        lambda b, g, r: r > 180 and r > g + 40 and r > b + 40,
    ) > 0
    assert _count_color_pixels(
        img, 96, 196, 145, 305,
        lambda b, g, r: r > 180 and g > 150 and b < 120,
    ) > 0
    assert _count_color_pixels(
        img, 494, 394, 520, 418,
        lambda b, g, r: g > 180 and r < 130 and b < 130,
    ) > 0


def test_build_evidence_frames_draws_shot_arrow(tmp_path, predictions_file,
                                                homography_file, frames_dir):
    idx = build_bbox_index(str(predictions_file))
    homo = load_homography(str(homography_file))
    shot = Event(
        event_id="shot_001",
        timestamp_s=0.0,
        frame_id=18,
        event_code="football.shoot",
        display_name_en="shot",
        display_name_cn="shot",
        importance=1.0,
        player_jersey="4",
        player_team="right",
        track_id=4,
    )
    paths = build_evidence_frames(
        shot, frames_dir, idx, homo, str(predictions_file), tmp_path / "shot",
        fps=25, window_s=0.0
    )
    assert paths and paths[0].exists()
    import cv2
    img = cv2.imread(str(paths[0]))
    assert img is not None and img.shape[2] == 3
    assert _count_color_pixels(
        img, 0, 0, img.shape[1], img.shape[0],
        lambda b, g, r: r > 170 and g > 170 and b < 140,
    ) > 500
