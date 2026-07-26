"""Filesystem layout for the calibration pipeline.

Everything is anchored to this file's directory rather than the working
directory, so the scripts can be run from anywhere (`uv run main.py ...`, an
IDE, or `python3 src/calibration.py` from the repo root).

Layout::

    src/images/camera0/image_000.bmp   captures for pseyepy camera 0
    src/images/camera1/image_000.bmp   captures for pseyepy camera 1
    src/camera0.json                   intrinsics for pseyepy camera 0
    src/camera1.json                   intrinsics for pseyepy camera 1

The number in each name is the pseyepy camera index, which is how the driver
enumerates the array over USB. That ordering is stable for a given set of
cameras in a given set of ports, but it is *not* tied to a specific physical
camera — replugging into different ports can renumber them. Re-run the
pipeline if you rearrange the array.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SRC_DIR / "images"


def camera_image_dir(index):
    """Directory holding the calibration captures for one camera."""
    return IMAGE_DIR / f"camera{index}"


def calibration_path(index):
    """Path of the intrinsics JSON for one camera."""
    return SRC_DIR / f"camera{index}.json"


def captured_camera_indices():
    """Camera indices that have a capture directory, in numeric order."""
    if not IMAGE_DIR.is_dir():
        return []
    indices = []
    for d in IMAGE_DIR.iterdir():
        if d.is_dir() and d.name.startswith("camera"):
            suffix = d.name[len("camera"):]
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)
