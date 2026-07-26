"""Capture ChArUco board images from every PS3Eye in the array at once.

Each camera gets its own capture directory (`images/cameraN/`) because each
camera is calibrated independently. Pressing SPACE saves a frame from every
camera that currently has a good view of the board, so one pass in front of
the array builds all the image sets together.

Controls:
    SPACE   save from every camera currently seeing >= MIN_MARKERS markers
    a       save from every camera regardless of marker count
    0-9     save from that camera index only
    q       quit
"""

import cv2

from ChArUco_board import ARUCO_DICT
from paths import camera_image_dir
from ps3eye_camera import annotate, open_cameras, read_frames, tile, to_bgr

# A tile turns green (and SPACE saves from it) at this many markers. The 7x5
# board carries 17 markers in total; asking for all of them is unrealistic at
# an angle, but too few makes for a weak view to calibrate from.
MIN_MARKERS = 8


def next_image_index(directory):
    """First unused image number, so repeated runs add to the set."""
    return len(list(directory.glob("image_*.bmp")))


def save_frame(directory, index, frame, camera_id, n_markers):
    path = directory / f"image_{index:03d}.bmp"
    cv2.imwrite(str(path), frame)
    print(f"  camera {camera_id}: saved {path.name} ({n_markers} markers)")


def main():
    cam = open_cameras()
    ids = list(cam.ids)

    dirs, counts = {}, {}
    for camera_id in ids:
        d = camera_image_dir(camera_id)
        d.mkdir(parents=True, exist_ok=True)
        dirs[camera_id] = d
        counts[camera_id] = next_image_index(d)
        if counts[camera_id]:
            print(f"camera {camera_id}: {counts[camera_id]} existing images in "
                  f"{d}, new captures will be appended")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary,
                                       cv2.aruco.DetectorParameters())

    print("Controls:" + __doc__.split("Controls:")[1].rstrip())
    try:
        while True:
            frames = read_frames(cam)

            tiles = []
            marker_counts = {}
            for camera_id, frame in zip(ids, frames):
                # Frames are already grayscale, which is what detection wants.
                corners, marker_ids, _ = detector.detectMarkers(frame)
                n = 0 if marker_ids is None else len(marker_ids)
                marker_counts[camera_id] = n

                preview = to_bgr(frame)
                if n:
                    cv2.aruco.drawDetectedMarkers(preview, corners, marker_ids)
                annotate(
                    preview,
                    f"cam {camera_id}  markers: {n}  saved: {counts[camera_id]}",
                    n >= MIN_MARKERS)
                tiles.append(preview)

            cv2.imshow("calibration capture", tile(tiles))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key in (ord(" "), ord("a")):
                force = key == ord("a")
                chosen = [c for c in ids
                          if force or marker_counts[c] >= MIN_MARKERS]
                if not chosen:
                    print(f"No camera sees {MIN_MARKERS}+ markers — nothing "
                          f"saved (press 'a' to save anyway)")
                for camera_id, frame in zip(ids, frames):
                    if camera_id in chosen:
                        save_frame(dirs[camera_id], counts[camera_id], frame,
                                   camera_id, marker_counts[camera_id])
                        counts[camera_id] += 1

            elif ord("0") <= key <= ord("9"):
                camera_id = key - ord("0")
                if camera_id in dirs:
                    frame = frames[ids.index(camera_id)]
                    save_frame(dirs[camera_id], counts[camera_id], frame,
                               camera_id, marker_counts[camera_id])
                    counts[camera_id] += 1
                else:
                    print(f"No camera {camera_id} in this array ({ids})")
    finally:
        cam.end()
        cv2.destroyAllWindows()

    print("Done.")
    for camera_id in ids:
        print(f"  camera {camera_id}: {counts[camera_id]} images in "
              f"{dirs[camera_id]}")


if __name__ == "__main__":
    main()
