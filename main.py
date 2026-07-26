"""Entry point for the PS3Eye calibration pipeline.

The steps are still plain standalone scripts in `src/` — this just runs them in
the project environment so you don't have to think about where you are:

    uv run main.py board        generate the ChArUco board PNG to display
    uv run main.py capture      capture board images from the whole array
    uv run main.py check        report detection quality on the captures
    uv run main.py calibrate    write camera0.json, camera1.json, ...
    uv run main.py pose         live pose demo using those calibrations

Extra arguments are passed through, e.g. `uv run main.py calibrate 0 2`.
See src/instructions_for_calibration.md for the full procedure.
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

STEPS = {
    "board": "ChArUco_board.py",
    "capture": "capture_images_for_calibration.py",
    "check": "sanity_check.py",
    "calibrate": "calibration.py",
    "pose": "aruco_marker_pose.py",
}


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0 if argv else 1

    step = argv[0]
    if step not in STEPS:
        print(f"Unknown step {step!r}. Choose one of: {', '.join(STEPS)}",
              file=sys.stderr)
        return 1

    return subprocess.call([sys.executable, str(SRC / STEPS[step])] + argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
