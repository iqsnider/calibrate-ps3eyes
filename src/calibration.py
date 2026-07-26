"""Calibrate every captured PS3Eye and write one cameraN.json per camera.

Run with no arguments to calibrate every camera that has a capture directory,
or pass indices to do a subset:

    python3 calibration.py         # all of images/cameraN/
    python3 calibration.py 0 2     # just cameras 0 and 2
"""

import json
import sys

import cv2
import numpy as np

from ChArUco_board import ARUCO_DICT, SQUARES_HORIZONTALLY, SQUARES_VERTICALLY
from paths import calibration_path, camera_image_dir, captured_camera_indices

# Measure a black square on the *displayed* board with calipers and put the
# result here. This is the only place real-world scale enters the pipeline, so
# every distance you later estimate is wrong by whatever this is wrong by.
SQUARE_LENGTH_m = 0.029                   # Square side length (in m)
MARKER_LENGTH_m = SQUARE_LENGTH_m / 2     # ArUco marker side length (in m)

SENSOR = 'OV7720_omnivision'
LENS = 'PS3Eye_stock'

MIN_VIEWS = 4       # cv2.calibrateCamera needs several genuinely distinct views
MIN_CORNERS = 6     # per-view charuco corners needed to use the view at all
                    # (4 is the theoretical minimum, but 4 corners on a grid
                    # board are usually collinear — see view_is_degenerate)


def view_is_degenerate(obj_points, img_points):
    """True if OpenCV cannot fit a plane homography to this view.

    calibrateCamera's initIntrinsicParams2D fits one homography per view to
    seed the focal length, and asserts the result is 3x3. findHomography
    returns an empty matrix for collinear or near-collinear point sets, which
    surfaces as the unhelpful

        (-215:Assertion failed) matH0.size() == Size(3, 3)

    from deep inside calibrateCamera. Running the same test here means every
    view we keep is one OpenCV can actually use.
    """
    src = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)[:, :2]
    dst = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    if len(src) < 4:
        return True
    H, _ = cv2.findHomography(src, dst)
    return H is None or H.shape != (3, 3)


def get_calibration_parameters(img_dir):
    """Detect the board in every image in `img_dir` and calibrate from it.

    Returns (mtx, dist, reproj_error, image_size, n_views).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH_m, MARKER_LENGTH_m, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    image_files = sorted(img_dir.glob("*.bmp"))

    all_object_points = []
    all_image_points = []
    image_size = None

    for image_file in image_files:
        gray = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  Could not read {image_file.name}")
            continue

        size = gray.shape[::-1]  # (width, height) — correct order
        if image_size is None:
            image_size = size
        elif size != image_size:
            # Intrinsics are resolution-specific; mixing sizes in one set would
            # silently produce garbage.
            raise RuntimeError(
                f"{image_file.name} is {size[0]}x{size[1]} but earlier images "
                f"in {img_dir} are {image_size[0]}x{image_size[1]}. Capture "
                f"the whole set at one resolution.")

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray)

        n_corners = 0 if charuco_ids is None else len(charuco_ids)
        if n_corners < MIN_CORNERS:
            print(f"  Board not detected in {image_file.name} "
                  f"({n_corners} corners, need {MIN_CORNERS})")
            continue

        obj_points, img_points = board.matchImagePoints(
            charuco_corners, charuco_ids)
        if obj_points is None or len(obj_points) < MIN_CORNERS:
            print(f"  Board not detected in {image_file.name}")
            continue

        if view_is_degenerate(obj_points, img_points):
            # Corners all on one row/column: no homography, and calibrateCamera
            # would abort on the whole set rather than skip the view.
            print(f"  Skipping {image_file.name}: {len(obj_points)} corners "
                  f"are collinear")
            continue

        all_object_points.append(obj_points)
        all_image_points.append(img_points)

    if len(all_object_points) < MIN_VIEWS:
        raise RuntimeError(
            f"Only {len(all_object_points)} usable views in {img_dir}. Fix "
            f"image quality before calibrating.")

    reproj_error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None)
    return mtx, dist, reproj_error, image_size, len(all_object_points)


def calibrate_camera(index):
    """Calibrate one camera and write its cameraN.json. Returns the JSON dict."""
    img_dir = camera_image_dir(index)
    if not img_dir.is_dir():
        raise RuntimeError(f"No capture directory {img_dir}")

    print(f"Camera {index}: calibrating from {img_dir}")
    mtx, dist, reproj_error, image_size, n_views = \
        get_calibration_parameters(img_dir)
    print(f"  {n_views} usable views, reprojection error "
          f"{reproj_error:.4f} px")

    data = {
        "camera_index": index,
        "sensor": SENSOR,
        "lens": LENS,
        "mtx": mtx.tolist(),
        "dist": dist.tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "reproj_error_px": reproj_error,
        "n_views": n_views,
        "square_length_m": SQUARE_LENGTH_m,
    }

    output = calibration_path(index)
    with open(output, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"  Saved {output}")
    return data


def main(argv):
    if argv:
        indices = [int(a) for a in argv]
    else:
        indices = captured_camera_indices()
        if not indices:
            raise SystemExit(
                "No images/cameraN/ directories found — run "
                "capture_images_for_calibration.py first.")

    failures = {}
    for index in indices:
        try:
            calibrate_camera(index)
        except RuntimeError as exc:
            # One bad camera should not stop the rest of the array.
            print(f"  FAILED: {exc}")
            failures[index] = str(exc)

    ok = [i for i in indices if i not in failures]
    print(f"\nCalibrated {len(ok)}/{len(indices)} cameras: "
          f"{', '.join(f'camera{i}.json' for i in ok) or 'none'}")
    if failures:
        print("Failed: " + ", ".join(f"camera {i}" for i in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
