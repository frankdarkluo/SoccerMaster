#!/usr/bin/env python3
"""Strip mask arrays from a fat state.pkl into state.lite.pkl (standalone subprocess)."""
import gc
import os
import pickle
import sys


def _strip_matched_detection_masks(unprocessed_data):
    cleared = 0
    for tracklet in unprocessed_data.unprocessed_tracklets:
        for segment in tracklet.segments:
            for detection in segment.detections:
                if detection.mask is not None:
                    detection.mask = None
                    cleared += 1
    return cleared


def _strip_unmatched_segment_masks(unmatched_segments):
    cleared = 0
    for seg in unmatched_segments:
        if seg.mask is not None:
            seg.mask = None
            cleared += 1
    return cleared


def strip_state(state):
    cleared_matched = _strip_matched_detection_masks(state['unprocessed_data'])
    cleared_unmatched = _strip_unmatched_segment_masks(state['unmatched_segments'])
    state.pop('video_data', None)
    return cleared_matched, cleared_unmatched


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <state.pkl> <state.lite.pkl>", file=sys.stderr)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    src_size_gb = os.path.getsize(src) / (1024 ** 3)
    print(f"Loading {src} ({src_size_gb:.2f} GB)...")
    with open(src, 'rb') as f:
        state = pickle.load(f)
    gc.collect()

    cleared_matched, cleared_unmatched = strip_state(state)
    print(f"Stripped masks: {cleared_matched} matched, {cleared_unmatched} unmatched")

    with open(dst, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    dst_size_gb = os.path.getsize(dst) / (1024 ** 3)
    print(f"Wrote {dst} ({dst_size_gb:.2f} GB)")
    del state
    gc.collect()


if __name__ == '__main__':
    main()
