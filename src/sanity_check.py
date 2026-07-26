"""Report what was detected in each captured image, without calibrating.

Useful for pruning bad captures before the expensive calibration step: it
prints the marker and charuco-corner count per image and flags the ones
calibration.py would end up discarding.

    python3 sanity_check.py         # every images/cameraN/
    python3 sanity_check.py 0 2     # just cameras 0 and 2
"""

import sys

import cv2

from ChArUco_board import ARUCO_DICT, SQUARES_HORIZONTALLY, SQUARES_VERTICALLY
from calibration import MIN_CORNERS, MIN_VIEWS
from paths import camera_image_dir, captured_camera_indices

# Board dimensions are irrelevant to corner *counting*, so unit lengths are
# fine here — real-world scale only matters in calibration.py.
_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_board = cv2.aruco.CharucoBoard(
    (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), 2.0, 1.0, _dictionary)
_charuco_detector = cv2.aruco.CharucoDetector(_board)


def check_camera(index):
    """Print a per-image report for one camera; return the usable view count."""
    img_dir = camera_image_dir(index)
    image_files = sorted(img_dir.glob("*.bmp"))
    print(f"\ncamera {index}  ({len(image_files)} images in {img_dir})")
    if not image_files:
        return 0

    usable = 0
    for f in image_files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  {f.name}: unreadable")
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            _charuco_detector.detectBoard(gray)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        n_corners = 0 if charuco_ids is None else len(charuco_ids)

        ok = n_corners >= MIN_CORNERS
        usable += ok
        print(f"  {f.name}: {n_markers:2d} markers, {n_corners:2d} corners"
              f"{'' if ok else '   <- unusable'}")

    print(f"  {usable}/{len(image_files)} usable "
          f"(calibration needs at least {MIN_VIEWS})")
    return usable


def main(argv):
    indices = [int(a) for a in argv] if argv else captured_camera_indices()
    if not indices:
        raise SystemExit(
            "No images/cameraN/ directories found — run "
            "capture_images_for_calibration.py first.")
    for index in indices:
        check_camera(index)


if __name__ == "__main__":
    main(sys.argv[1:])
