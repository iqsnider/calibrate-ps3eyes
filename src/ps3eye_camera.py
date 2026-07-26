"""Shared PS3Eye capture setup for the calibration pipeline.

Every script that touches hardware opens the array through `open_cameras()` so
that resolution, frame rate and sensor settings are identical between the
images a camera is calibrated from and whatever consumes its `cameraN.json`
afterwards — intrinsics are resolution-specific and do not transfer.

pseyepy requires that *all* cameras be driven by a single `Camera` object, so
these helpers always return one object covering the whole array.
"""

import cv2
import numpy as np
from pseyepy import Camera, cam_count

# --- Capture configuration (the whole pipeline must agree on this) ---
RESOLUTION = Camera.RES_LARGE   # RES_LARGE = 640x480, RES_SMALL = 320x240
FPS = 30                        # RES_LARGE supports 15/30/40/50/60
COLOUR = False                  # grayscale; ArUco detection only ever uses gray

_RESOLUTION_WH = {Camera.RES_SMALL: (320, 240), Camera.RES_LARGE: (640, 480)}
FRAME_W, FRAME_H = _RESOLUTION_WH[RESOLUTION]

# Sensor settings applied to every camera in the array. See pseyepy.Camera for
# valid ranges. `auto_exposure` is deliberately absent — the driver does not
# implement it. If the monitor-displayed board blows out to solid white, turn
# auto_gain off and pin gain/exposure by hand, e.g.
#     CAMERA_SETTINGS = dict(auto_gain=False, gain=20, exposure=60,
#                            auto_whitebalance=True)
CAMERA_SETTINGS = dict(
    auto_gain=True,
    auto_whitebalance=True,
)

# Largest preview window this module will build, in pixels. Tiled previews of a
# big array are scaled down to fit; the saved images are always full-res.
MAX_PREVIEW_W = 1600
MAX_PREVIEW_H = 900


def open_cameras(ids=None):
    """Open PS3Eye cameras and return the single pseyepy `Camera` driving them.

    `ids` defaults to every connected camera. `cam.ids` gives the camera
    indices in the same order that `read_frames()` returns frames.

    Several PS3Eyes on one USB 2.0 host controller can exceed the available
    bandwidth; if cameras fail to open or frames come back black, move some to
    another controller or drop `FPS` (15 is plenty for calibration captures).
    """
    n_connected = cam_count()
    if n_connected == 0:
        raise SystemExit(
            "No PS3Eye cameras found. Check the USB connections, and that you "
            "have permission to talk to them (a udev rule, or run as root).")

    if ids is None:
        ids = list(range(n_connected))
    ids = [int(i) for i in ids]
    for i in ids:
        if i >= n_connected:
            raise SystemExit(
                f"Camera {i} requested but only {n_connected} connected.")

    print(f"Opening camera(s) {ids} at {FRAME_W}x{FRAME_H} @ {FPS} fps "
          f"({'colour' if COLOUR else 'grayscale'})")
    return Camera(ids=ids, resolution=RESOLUTION, fps=FPS, colour=COLOUR,
                  **CAMERA_SETTINGS)


def read_frames(cam):
    """Read one frame per camera, in `cam.ids` order.

    pseyepy returns a read-only view of a buffer it overwrites on the next
    read, so each frame is copied before it leaves this function — otherwise
    holding a frame across loop iterations silently gives you a later one, and
    the OpenCV drawing calls would refuse to write into it.
    """
    frames, _ = cam.read(squeeze=False, timestamp=True)
    return [np.array(f, copy=True) for f in frames]


def to_bgr(frame):
    """Grayscale (or already-colour) frame as a 3-channel BGR image."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if COLOUR:
        # pseyepy hands back RGB in colour mode, OpenCV wants BGR.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def tile(frames, cols=None):
    """Lay out equally-sized BGR frames in a grid for a single preview window.

    Unused grid cells are filled with black. The result is scaled down to fit
    within MAX_PREVIEW_W/H so a large array still fits on screen.
    """
    n = len(frames)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    blank = np.zeros_like(frames[0])
    padded = list(frames) + [blank] * (rows * cols - n)
    grid = np.vstack([np.hstack(padded[r * cols:(r + 1) * cols])
                      for r in range(rows)])

    scale = min(1.0, MAX_PREVIEW_W / grid.shape[1], MAX_PREVIEW_H / grid.shape[0])
    if scale < 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return grid


def annotate(frame_bgr, text, ok):
    """Draw a status line and a pass/fail border onto a preview tile."""
    colour = (0, 255, 0) if ok else (0, 0, 255)
    cv2.rectangle(frame_bgr, (0, 0),
                  (frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1), colour, 3)
    cv2.putText(frame_bgr, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, colour, 2)
    return frame_bgr
