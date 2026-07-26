# calibrate-ps3eyes

Calibrates an array of PS3Eye cameras against a ChArUco board. Every camera is calibrated
independently and gets its own `src/cameraN.json` of intrinsics and distortion coefficients.

Camera access goes through [`pseyepy`](https://github.com/iqsnider/pseyepy) — the PS3Eye is not a
UVC device, so `cv2.VideoCapture` will not see it.

## Quickstart

```sh
uv sync
uv run main.py board        # generate src/ChArUco_Marker.png, display it, measure a square
                            # with calipers -> SQUARE_LENGTH_m in src/calibration.py
uv run main.py capture      # SPACE saves a view from every camera that can see the board
uv run main.py check        # optional: report detection quality, prune bad captures
uv run main.py calibrate    # writes src/camera0.json, src/camera1.json, ...
uv run main.py pose         # live check: hold a marker at a known distance
```

Full procedure, including how many views to capture and what to watch out for, is in
[`src/instructions_for_calibration.md`](src/instructions_for_calibration.md).
