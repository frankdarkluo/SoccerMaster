import argparse
import json

from .io_gamestate import load_detections
from .pipeline import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Formation topology V0: per-window, per-team team-shape signal."
    )
    parser.add_argument("--input", required=True, help="Labels-GameState.json path")
    parser.add_argument("--output", required=True, help="output records JSON path")
    parser.add_argument("--fps", type=float, required=True, help="video frame rate")
    parser.add_argument("--win", type=float, default=3.0, help="window length in seconds")
    parser.add_argument("--stride", type=float, default=1.0, help="window stride in seconds")
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=8,
        help="min visible outfielders to attempt line split",
    )
    parser.add_argument(
        "--gap-delta",
        type=float,
        default=7.0,
        help="min x-gap in meters to count as a line boundary",
    )
    args = parser.parse_args(argv)

    detections = load_detections(args.input)
    records = analyze(
        detections,
        fps=args.fps,
        win_s=args.win,
        stride_s=args.stride,
        min_coverage=args.min_coverage,
        gap_delta=args.gap_delta,
    )
    with open(args.output, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
