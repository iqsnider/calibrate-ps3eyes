# Calibrating a PS3Eye array

Each camera is calibrated independently and gets its own `cameraN.json`, where `N` is the pseyepy
camera index. The capture step drives the whole array at once, so a single pass in front of the
cameras builds every image set together.

Run the steps from the repo root with `uv run main.py <step>`, or run the scripts in `src/`
directly — they work from any working directory.

**STEP 1**: Generate the ChArUco board by running `uv run main.py board`, which writes
`src/ChArUco_Marker.png`.

**STEP 2**: Display the board on a matte screen (or print it). If you display it, set the monitor
zoom and leave it there — the physical size must not change between measuring and capturing.

**STEP 3**: Use calipers to measure the edge length of a black square (not the ArUco marker) as it
is actually displayed.

**STEP 4**: Put that measurement, in metres, in `SQUARE_LENGTH_m` at the top of
`src/calibration.py`. This is the only place real-world scale enters the pipeline; every distance
you later estimate is wrong by whatever this number is wrong by.

**STEP 5**: Capture with `uv run main.py capture`. Every camera gets a tile in the preview window,
green once it sees enough of the board:

- `SPACE` saves a frame from every camera that currently has a good view
- `a` saves from every camera regardless of marker count
- `0`–`9` saves from one camera only — useful when the array is spread out and no single board
  position satisfies all of them at once
- `q` quits

Aim for 15–20 usable views per camera: board close and far, at the frame edges and corners, and
tilted 30–45° in several directions. Views that are all head-on and centred will not constrain the
distortion terms. Re-running the capture step appends to the existing sets rather than replacing
them.

**STEP 6**: *(optional)* `uv run main.py check` reports the marker and corner count for every
captured image and flags the ones calibration would discard. Delete bad captures before continuing.

**STEP 7**: `uv run main.py calibrate` writes `src/camera0.json`, `src/camera1.json`, … Pass
indices (`uv run main.py calibrate 0 2`) to redo just some of them. A reprojection error under
about 0.5 px is good for a PS3Eye; well above 1 px usually means a blurred or mis-measured set.
One camera failing does not stop the rest of the array.

**STEP 8**: Sanity-check the result with `uv run main.py pose` — hold a marker of known size
(set `MARKER_SIZE_m` in `src/aruco_marker_pose.py`) at a known distance and check that the reported
range agrees.

## A note on camera numbering

`N` in `cameraN.json` is the pseyepy index, which comes from USB enumeration order. It is stable
for a given set of cameras in a given set of ports, but it is not bound to a specific physical
camera — moving cameras between ports can renumber them, which would silently pair a camera with
the wrong intrinsics. If you rearrange the array, re-run the pipeline (or at minimum re-check it
with the pose demo).

## USB bandwidth

Several PS3Eyes on one USB 2.0 host controller can exceed the available bandwidth. If cameras fail
to open or frames come back black, spread them across controllers or lower `FPS` in
`src/ps3eye_camera.py` — 15 fps is plenty for calibration captures.
