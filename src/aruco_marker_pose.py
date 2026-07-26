"""Marker pose estimation against the calibrations produced by this pipeline.

`get_marker_poses()` is the reusable part — give it a frame and one camera's
intrinsics and it returns the pose of every marker it can see. The
`__main__` block below is a live demo across the whole array, which doubles as
a check that each cameraN.json is sane: hold a marker at a known distance and
see whether the reported range agrees.
"""

import json
import time

import cv2
import numpy as np

from ChArUco_board import ARUCO_DICT
from paths import calibration_path
from ps3eye_camera import (FRAME_H, FRAME_W, annotate, open_cameras,
                           read_frames, tile, to_bgr)

# Physical size of the marker being *tracked*. This is independent of the
# calibration board's marker size — don't confuse the two.
MARKER_SIZE_m = 0.15  # [m] length of black marker
ARUCO_DICT = cv2.aruco.DICT_4X4_250

# aruco detector
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def marker_object_points(marker_size_m):
    """Corners of a marker in its own frame, in the order detectMarkers uses."""
    s = marker_size_m / 2
    return np.array([[-s, s, 0],
                     [s, s, 0],
                     [s, -s, 0],
                     [-s, -s, 0]], dtype=np.float32)


OBJ_POINTS = marker_object_points(MARKER_SIZE_m)


def load_calibration(index):
    """Load one camera's intrinsics as (mtx, dist, (width, height))."""
    path = calibration_path(index)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — calibrate camera {index} first.")
    with open(path) as f:
        calib = json.load(f)

    mtx = np.array(calib["mtx"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64)

    # Prefer the recorded size; fall back to deriving it from the principal
    # point for calibrations written before that field existed.
    width = calib.get("image_width", int(round(2 * mtx[0, 2])))
    height = calib.get("image_height", int(round(2 * mtx[1, 2])))
    return mtx, dist, (width, height)


def get_marker_poses(frame, mtx, dist, obj_points=OBJ_POINTS):
    """Detect markers in `frame` and solve each one's pose.

    Returns (corners, ids, {marker_id: (rvec, tvec)}). `frame` may be
    grayscale (what the PS3Eye pipeline produces) or BGR.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame,
                                                      cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    poses = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners[0], mtx, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                poses[int(marker_id)] = (rvec, tvec)
    return corners, ids, poses


def main():
    cam = open_cameras()
    ids = list(cam.ids)

    calibrations = {}
    for camera_id in ids:
        try:
            mtx, dist, (w, h) = load_calibration(camera_id)
        except FileNotFoundError as exc:
            print(f"camera {camera_id}: {exc}")
            continue
        if (w, h) != (FRAME_W, FRAME_H):
            # Intrinsics do not transfer across resolutions.
            print(f"camera {camera_id}: calibrated at {w}x{h} but capturing "
                  f"at {FRAME_W}x{FRAME_H} — skipping, poses would be wrong.")
            continue
        calibrations[camera_id] = (mtx, dist)

    if not calibrations:
        cam.end()
        raise SystemExit("No usable calibrations for the connected cameras.")

    print(f"Tracking {MARKER_SIZE_m * 1000:.1f} mm markers on camera(s) "
          f"{sorted(calibrations)}.  q = quit")

    last_print = 0.0
    try:
        while True:
            frames = read_frames(cam)

            tiles = []
            report = []
            for camera_id, frame in zip(ids, frames):
                preview = to_bgr(frame)

                if camera_id not in calibrations:
                    annotate(preview, f"cam {camera_id}  no calibration", False)
                    tiles.append(preview)
                    continue

                mtx, dist = calibrations[camera_id]
                corners, marker_ids, poses = get_marker_poses(frame, mtx, dist)
                if marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(preview, corners, marker_ids)

                for marker_id, (rvec, tvec) in poses.items():
                    cv2.drawFrameAxes(preview, mtx, dist, rvec, tvec,
                                      MARKER_SIZE_m * 0.5)
                    x, y, z = tvec.flatten()
                    report.append(
                        f"cam {camera_id} id {marker_id}: "
                        f"x={x:+.3f} y={y:+.3f} z={z:+.3f} m  "
                        f"range={float(np.linalg.norm(tvec)):.3f} m")

                annotate(preview, f"cam {camera_id}  markers: {len(poses)}",
                         bool(poses))
                tiles.append(preview)

            cv2.imshow("pose", tile(tiles))

            # Throttled so the terminal stays readable with several cameras.
            now = time.time()
            if report and now - last_print > 0.5:
                print("\n".join(report))
                last_print = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
