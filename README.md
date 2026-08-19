# DAS-EGO Normalization

Takes one raw DAS-EGO recording (a single MCAP file) and produces a folder
of plain, usable outputs: per-camera video, a rectified stereo pair, and
IMU data.

## Input

One DAS-EGO `.mcap` file — a head-mounted rig (GenRobot AI) with 6 cameras,
1 IMU, and a microphone, all recorded into a single container file. Raw
files live at `s3://stage-humyn-egocentric-stereo-data/raw/DAS-EGO/`.

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

## Important notes

- **The stereo pair isn't hardcoded.** Of the 6 cameras, only 2 are
  physically mounted as a true stereo pair (currently `camera2`/`camera3`).
  The script confirms this from geometry (position + facing direction) on
  every run rather than assuming it — re-verify if a rig revision changes
  camera placement.
- **Lens model is Double Sphere, not standard fisheye.** DAS-EGO's cameras
  use the Double Sphere model (6 parameters: `fx, fy, cx, cy, xi, alpha`),
  which OpenCV doesn't support natively — the projection math is
  implemented directly in `normalize.py` and self-tested on every run
  before being trusted (must be <0.01px round-trip error, or the script
  aborts).
- **IMU accel is unit-converted.** DAS-EGO stores raw acceleration in
  g-units; `imu.csv` converts this to m/s² automatically. Gravity at rest
  should read ~9.8 in the output file.
- **Validated so far**: 3 real clips from the same rig/session (3.8s,
  ~8s, and 32.9s), checked via stereo self-test, feature-based disparity
  check, and IMU sanity checks. Not yet run across the full DAS-EGO corpus
  or a different rig/session.
