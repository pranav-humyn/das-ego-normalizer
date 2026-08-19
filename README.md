# DAS-EGO Normalization

Takes one raw DAS-EGO recording (a single MCAP file) and produces a folder
of plain, usable outputs: per-camera video, a rectified stereo pair, and
IMU data.

## Input

One DAS-EGO `.mcap` file — a head-mounted rig (GenRobot AI) with 6 cameras
(`camera0`–`camera5`), 1 IMU, and a microphone, all recorded into a single
container file. Raw files live at
`s3://stage-humyn-egocentric-stereo-data/raw/DAS-EGO/`.

Read via the official `mcap` + `mcap-protobuf-support` libraries — no custom
parsing needed for the container itself. Two things worth knowing about the
data inside it:
- Each camera's "compressed image" messages are actually raw H.264
  elementary-stream chunks, not one-JPEG-per-message — they're concatenated
  in timestamp order before being repackaged as an mp4.
- Each camera's `camera_info` message carries its own intrinsics (Double
  Sphere lens model — see below) and its 3D position/orientation relative
  to the rig body (`T_b_c`). There is no separate calibration file; it's
  embedded in the MCAP.

## Output

Running the normalizer on one `.mcap` produces a folder containing:

| File | What it is |
|---|---|
| `left_raw.mp4`, `right_raw.mp4` | The two stereo-pair cameras, remuxed to mp4 (no re-encode — source is already H.264) |
| `left_rectified.mp4`, `right_rectified.mp4` | The stereo pair, geometrically undistorted + row-aligned so the same real-world point lands on the same pixel row in both — real H.264 encode |
| `sbs_raw.mp4`, `sbs_rectified.mp4` | Left+right stacked side by side, for quick visual review |
| `camera0_raw.mp4`, `camera1_raw.mp4`, `camera4_raw.mp4`, `camera5_raw.mp4` | The 4 non-stereo camera views, remuxed the same way as left/right |
| `calibration.json` | Per-camera intrinsics (raw + rectified), stereo baseline/rotation/translation |
| `meta.json` | Device identity, recording dimensions/fps/duration, source file info, encoding settings |
| `imu.csv` | `timestamp_s,ax,ay,az,gx,gy,gz` — accel in m/s², gyro in rad/s, timestamps relative to the start of the recording |

## Which 2 of the 6 cameras become the stereo pair

Never assumed from camera names — confirmed from geometry every run. Each
camera's `T_b_c` gives its exact position and orientation; for every
possible pair (15 total), the normalizer computes the distance between them
(baseline) and the angle between their facing directions (axis angle). A
real stereo pair has a small baseline (a few cm) **and** a near-zero axis
angle. On the validated sample, only `camera2`/`camera3` qualify (5.78cm
baseline, 0.2° axis angle) — other candidate pairs have a similar baseline
but a 20°+ axis mismatch, meaning they point in different directions.

## The lens model

DAS-EGO's `camera_info.distortion_model` reads `"ds"` — the **Double
Sphere** model (Usenko, Demmel, Cremers, 3DV 2018), a 6-parameter wide-angle
lens model (`fx, fy, cx, cy, xi, alpha`). This isn't supported by OpenCV's
built-in fisheye functions, so the projection/unprojection math is
implemented directly from the paper's equations in `normalize.py`, and
self-tested (project → unproject → project on a grid of pixels, checking
for the same pixel back) before being trusted on real frames.

## Verifying correctness

- **Self-test**: round-trip projection error on a pixel grid, checked
  automatically on every run before rectifying (must be <0.01px).
- **Disparity check**: ORB feature-matching between the rectified left/right
  frames — matched points should shift sideways in one consistent direction
  (confirms which camera is really left vs. right) with ~0 vertical
  offset. On the validated sample: 100% consistently-signed horizontal
  disparity, ~4px mean vertical residual on 1600px-wide frames.
- **IMU sanity**: gravity magnitude at rest should be ~9.8 m/s² once
  converted (DAS-EGO stores raw accel in g-units — the normalizer converts
  this before writing `imu.csv`).

## Running it

Locally:
```
pip install -r requirements.txt
python normalize.py <input.mcap> <output_dir>
```

With Docker:
```
docker build -t das-ego-normalizer .
docker run --rm \
  -v /path/to/input_dir:/input:ro \
  -v /path/to/output_dir:/output \
  das-ego-normalizer /input/<name>.mcap /output
```

## Current status

Validated end-to-end on one sample file (11.9MB, ~4s). Running this across
the full DAS-EGO corpus (files run up to ~24GB) is the next step — not yet
done.
