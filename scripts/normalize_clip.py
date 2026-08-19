#!/usr/bin/env python3
"""End-to-end DAS-EGO MCAP normalizer -> the real stereo-rig normalized-output
layout already used in production for akai/oak-d
(stage-stereo-pipeline-v2-data/normalized/<device>/<date>/<session>/<segment>/):
left_raw.mp4, right_raw.mp4, left_rectified.mp4, right_rectified.mp4,
sbs_raw.mp4, sbs_rectified.mp4, calibration.json, meta.json, imu.csv --
plus camera0/1/4/5_raw.mp4 for DAS-EGO's 4 extra non-stereo views (a real
deviation from akai/oak-d, which only have 2 cameras total).

Reuses the already-validated logic from this repo's standalone scripts
(find_stereo_pairs.py's geometry check, ds_rectify.py's Double Sphere math,
build_raw_mp4.py's H.264 remux, export_imu.py's IMU decode) rather than
re-deriving any of it -- only the output-shaping/unit-conversion/schema layer
around them is new.

Two values (session_id, date_str) are NOT derivable from the MCAP file
itself -- they come from where the file sits in S3 (raw/DAS-EGO/<region>/
<date>/<session>/...) -- so they must be passed in, not guessed.
"""
import sys
import os
import json
import shutil
import subprocess
import time
import resource
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import cv2
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

G = 9.80665  # standard gravity, m/s^2 per g -- DAS-EGO accel is in g-units (found+validated earlier)


# ---------- Double Sphere model (Usenko, Demmel, Cremers, 3DV 2018) ----------
# identical math to ds_rectify.py -- copied verbatim, not reimplemented differently

def ds_project(P, fx, fy, cx, cy, xi, alpha):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    zp = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zp * zp)
    denom = alpha * d2 + (1 - alpha) * zp
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    return np.stack([u, v], axis=-1), denom


def ds_unproject(uv, fx, fy, cx, cy, xi, alpha):
    u, v = uv[..., 0], uv[..., 1]
    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx * mx + my * my
    valid = (2 * alpha - 1) * r2 < 1
    disc = np.clip(1 - (2 * alpha - 1) * r2, 0, None)
    mz = (1 - alpha * alpha * r2) / (alpha * np.sqrt(disc) + (1 - alpha))
    scale = (mz * xi + np.sqrt(np.clip(mz * mz + (1 - xi * xi) * r2, 0, None))) / (mz * mz + r2)
    x = scale * mx
    y = scale * my
    z = scale * mz - xi
    P = np.stack([x, y, z], axis=-1)
    P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    return P, valid


def self_test(fx, fy, cx, cy, xi, alpha, w, h):
    uu, vv = np.meshgrid(np.linspace(0, w - 1, 40), np.linspace(0, h - 1, 40))
    uv = np.stack([uu, vv], axis=-1)
    rays, valid = ds_unproject(uv, fx, fy, cx, cy, xi, alpha)
    uv2, denom = ds_project(rays, fx, fy, cx, cy, xi, alpha)
    err = np.linalg.norm((uv - uv2)[valid & (denom > 0)], axis=-1)
    return float(err.max()), float(err.mean())


def quat_xyzw_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    xx, yy, zz = x * X, y * Y, z * Z
    xy, xz, yz = x * Y, x * Z, y * Z
    wx, wy, wz = w * X, w * Y, w * Z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def build_rectifying_rotations(R_b_l, t_b_l, R_b_r, t_b_r):
    baseline = t_b_r - t_b_l
    e1 = baseline / np.linalg.norm(baseline)
    fwd_l = R_b_l @ np.array([0, 0, 1.0])
    fwd_r = R_b_r @ np.array([0, 0, 1.0])
    avg_fwd = fwd_l + fwd_r
    avg_fwd = avg_fwd - np.dot(avg_fwd, e1) * e1
    e3 = avg_fwd / np.linalg.norm(avg_fwd)
    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=0)


def build_remap(fx, fy, cx, cy, xi, alpha, R_cam_from_rect, w, h, fx_r, fy_r, cx_r, cy_r):
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (us - cx_r) / fx_r
    y = (vs - cy_r) / fy_r
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_cam = dirs @ R_cam_from_rect.T
    uv, denom = ds_project(dirs_cam, fx, fy, cx, cy, xi, alpha)
    mapx = uv[..., 0].astype(np.float32)
    mapy = uv[..., 1].astype(np.float32)
    mapx[denom <= 0] = -1
    mapy[denom <= 0] = -1
    return mapx, mapy


# ---------- MCAP reads ----------

def read_camera_infos(mcap_path):
    cams = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if channel.topic.endswith("/camera_info") and "camera_info_resize" not in channel.topic:
                cam_id = channel.topic.split("/")[3]
                D = list(msg.D)
                t = list(msg.T_b_c)
                cams[cam_id] = {
                    "fx": D[0], "fy": D[1], "cx": D[2], "cy": D[3], "xi": D[4], "alpha": D[5],
                    "width": msg.width, "height": msg.height,
                    "t": np.array(t[:3]), "q_xyzw": np.array(t[3:7]),
                }
    return cams


def find_stereo_pair(cams):
    """Confirms the real stereo pair from geometry (baseline + optical-axis
    angle) -- refuses to guess camera2/camera3 by convention."""
    order = sorted(cams.keys(), key=lambda s: int(s.replace("camera", "")))
    best = None
    for a, b in combinations(order, 2):
        ta, tb = cams[a]["t"], cams[b]["t"]
        baseline = float(np.linalg.norm(ta - tb))
        Ra = quat_xyzw_to_R(cams[a]["q_xyzw"])
        Rb = quat_xyzw_to_R(cams[b]["q_xyzw"])
        za = Ra @ np.array([0, 0, 1.0])
        zb = Rb @ np.array([0, 0, 1.0])
        cos_ang = np.clip(np.dot(za, zb) / (np.linalg.norm(za) * np.linalg.norm(zb)), -1, 1)
        angle_deg = float(np.degrees(np.arccos(cos_ang)))
        print(f"  {a}-{b}: baseline={baseline*100:.2f}cm axis_angle={angle_deg:.1f}deg")
        if baseline < 0.15 and angle_deg < 5.0:
            if best is None or baseline < best[0]:
                best = (baseline, angle_deg, a, b)
    if best is None:
        raise RuntimeError("No real stereo pair found (need baseline<15cm AND axis_angle<5deg) "
                            "-- aborting rather than assuming camera2/camera3.")
    return best


def build_raw_mp4_all(mcap_path, tmp_dir):
    """Mirrors build_raw_mp4.py: remux every camera's native H.264 stream,
    no re-encode (source is already H.264, unlike akai's raw MJPEG)."""
    streams = defaultdict(list)
    times = defaultdict(list)
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        msgs = []
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if not channel.topic.endswith("/compressed"):
                continue
            cam_id = channel.topic.split("/")[3]
            msgs.append((cam_id, message.log_time, msg.data))
        msgs.sort(key=lambda m: (m[0], m[1]))
        for cam_id, t, data in msgs:
            streams[cam_id].append(data)
            times[cam_id].append(t)

    result = {}
    for cam_id, chunks in streams.items():
        ts = times[cam_id]
        n = len(chunks)
        duration_s = (ts[-1] - ts[0]) / 1e9 if n > 1 else 0
        fps = round((n - 1) / duration_s, 3) if duration_s > 0 else 30.0

        h264_path = os.path.join(tmp_dir, f"_{cam_id}.h264")
        with open(h264_path, "wb") as out:
            for c in chunks:
                out.write(c)

        mp4_path = os.path.join(tmp_dir, f"_{cam_id}_raw.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "h264", "-r", str(fps), "-i", h264_path,
             "-c:v", "copy", mp4_path], check=True)
        os.remove(h264_path)
        result[cam_id] = {"path": mp4_path, "n_frames": n, "fps": fps,
                           "start_log_time_ns": ts[0], "end_log_time_ns": ts[-1]}
    return result


def read_imu(mcap_path, topic="/robot0/sensor/imu"):
    rows = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            rows.append({
                "log_time_ns": message.log_time,
                "ax_g": msg.linear_acceleration.x,
                "ay_g": msg.linear_acceleration.y,
                "az_g": msg.linear_acceleration.z,
                "gx": msg.angular_velocity.x,
                "gy": msg.angular_velocity.y,
                "gz": msg.angular_velocity.z,
            })
    rows.sort(key=lambda r: r["log_time_ns"])
    return rows


def ffprobe_video(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height,r_frame_rate,duration,codec_name",
         "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def main(mcap_path, output_root, session_id, date_str):
    t_start = time.time()
    fname = os.path.basename(mcap_path)
    parts = fname.replace(".mcap", "").split("_")
    # DAS-Ego_20260807174743_none_none_80e881_bce97180
    ts_str, device_id, file_hash = parts[1], parts[4], parts[5]
    segment_id = f"{ts_str}_{file_hash}"
    recorded_dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    outdir = os.path.join(output_root, "das-ego", date_str, session_id, segment_id)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    tmp = os.path.join(outdir, "_tmp")
    os.makedirs(tmp)

    print(f"=== normalizing {fname} -> {outdir} ===\n")

    print("=== 1. reading camera_info for all cameras ===")
    cams = read_camera_infos(mcap_path)
    print(f"  found {len(cams)} cameras: {sorted(cams.keys())}")

    print("\n=== 2. confirming real stereo pair via geometry (not assumed) ===")
    baseline, angle_deg, left_id, right_id = find_stereo_pair(cams)
    print(f"  -> stereo pair: {left_id} (left) / {right_id} (right), "
          f"baseline={baseline*100:.2f}cm axis_angle={angle_deg:.2f}deg")
    extra_ids = sorted([c for c in cams if c not in (left_id, right_id)],
                        key=lambda s: int(s.replace("camera", "")))

    print("\n=== 3. Double Sphere self-test (must pass before rectifying) ===")
    selftest_results = {}
    for cid in (left_id, right_id):
        c = cams[cid]
        max_err, mean_err = self_test(c["fx"], c["fy"], c["cx"], c["cy"], c["xi"], c["alpha"],
                                       c["width"], c["height"])
        print(f"  {cid}: max_px_err={max_err:.6f} mean_px_err={mean_err:.6f}")
        assert max_err < 0.01, f"Double Sphere self-test FAILED for {cid}: max_err={max_err}"
        selftest_results[cid] = {"max_px_err": max_err, "mean_px_err": mean_err}
    print("  PASS")

    print("\n=== 4. remuxing all 6 cameras' raw H.264 -> mp4 (no re-encode) ===")
    raw = build_raw_mp4_all(mcap_path, tmp)
    for cid in sorted(raw.keys(), key=lambda s: int(s.replace("camera", ""))):
        info = raw[cid]
        print(f"  {cid}: {info['n_frames']} frames @ {info['fps']:.3f}fps")

    w, h = cams[left_id]["width"], cams[left_id]["height"]
    assert (w, h) == (cams[right_id]["width"], cams[right_id]["height"])

    # copy left/right raw + extra-camera raw into final names
    shutil.move(raw[left_id]["path"], os.path.join(outdir, "left_raw.mp4"))
    shutil.move(raw[right_id]["path"], os.path.join(outdir, "right_raw.mp4"))
    for cid in extra_ids:
        shutil.move(raw[cid]["path"], os.path.join(outdir, f"{cid}_raw.mp4"))

    print("\n=== 5. building rectification maps ===")
    cl, cr = cams[left_id], cams[right_id]
    R_b_l = quat_xyzw_to_R(cl["q_xyzw"])
    R_b_r = quat_xyzw_to_R(cr["q_xyzw"])
    R_rect_body = build_rectifying_rotations(R_b_l, cl["t"], R_b_r, cr["t"])
    R_l_from_rect = R_b_l @ R_rect_body.T
    R_r_from_rect = R_b_r @ R_rect_body.T
    fx_r = (cl["fx"] + cr["fx"]) / 2
    fy_r = (cl["fy"] + cr["fy"]) / 2
    cx_r, cy_r = w / 2, h / 2
    mapx_l, mapy_l = build_remap(cl["fx"], cl["fy"], cl["cx"], cl["cy"], cl["xi"], cl["alpha"],
                                  R_l_from_rect, w, h, fx_r, fy_r, cx_r, cy_r)
    mapx_r, mapy_r = build_remap(cr["fx"], cr["fy"], cr["cx"], cr["cy"], cr["xi"], cr["alpha"],
                                  R_r_from_rect, w, h, fx_r, fy_r, cx_r, cy_r)

    print("\n=== 6. rendering rectified videos (real H.264, streamed straight into ffmpeg) ===")
    fps = raw[left_id]["fps"]
    left_raw_path = os.path.join(outdir, "left_raw.mp4")
    right_raw_path = os.path.join(outdir, "right_raw.mp4")
    cap_l = cv2.VideoCapture(left_raw_path)
    cap_r = cv2.VideoCapture(right_raw_path)

    left_rect_path = os.path.join(outdir, "left_rectified.mp4")
    right_rect_path = os.path.join(outdir, "right_rectified.mp4")
    proc_l = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", left_rect_path],
        stdin=subprocess.PIPE)
    proc_r = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", right_rect_path],
        stdin=subprocess.PIPE)

    n = 0
    while True:
        okl, fl = cap_l.read()
        okr, fr = cap_r.read()
        if not okl or not okr:
            break
        rl = cv2.remap(fl, mapx_l, mapy_l, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        rr = cv2.remap(fr, mapx_r, mapy_r, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        proc_l.stdin.write(rl.tobytes())
        proc_r.stdin.write(rr.tobytes())
        n += 1
    cap_l.release(); cap_r.release()
    proc_l.stdin.close(); proc_r.stdin.close()
    proc_l.wait(); proc_r.wait()
    if proc_l.returncode != 0 or proc_r.returncode != 0:
        raise RuntimeError("ffmpeg rectified-video encode failed")
    print(f"  rectified {n} frame pairs -> {left_rect_path}, {right_rect_path}")

    print("\n=== 7. side-by-side videos ===")
    sbs_raw_path = os.path.join(outdir, "sbs_raw.mp4")
    sbs_rect_path = os.path.join(outdir, "sbs_rectified.mp4")
    for a, b, out in ((left_raw_path, right_raw_path, sbs_raw_path),
                      (left_rect_path, right_rect_path, sbs_rect_path)):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", a, "-i", b, "-filter_complex", "hstack=inputs=2",
             "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", out],
            check=True)
        print(f"  wrote {out}")

    print("\n=== 8. IMU export (unit-converted to match akai/oak-d convention) ===")
    imu_rows = read_imu(mcap_path)
    t0 = imu_rows[0]["log_time_ns"]
    imu_csv_path = os.path.join(outdir, "imu.csv")
    with open(imu_csv_path, "w") as f:
        f.write("timestamp_s,ax,ay,az,gx,gy,gz\n")
        for r in imu_rows:
            ts = (r["log_time_ns"] - t0) / 1e9
            f.write(f"{ts:.6f},{r['ax_g']*G:.6f},{r['ay_g']*G:.6f},{r['az_g']*G:.6f},"
                    f"{r['gx']:.6f},{r['gy']:.6f},{r['gz']:.6f}\n")
    print(f"  wrote {len(imu_rows)} rows to {imu_csv_path}")

    print("\n=== 9. calibration.json ===")
    t_r_from_l = R_b_r.T @ (cl["t"] - cr["t"])
    R_r_from_l = R_b_r.T @ R_b_l
    fov_r_h = float(np.degrees(2 * np.arctan(w / (2 * fx_r))))
    fov_r_v = float(np.degrees(2 * np.arctan(h / (2 * fy_r))))
    fov_r_d = float(np.degrees(2 * np.arctan(np.hypot(w / 2, h / 2) / ((fx_r + fy_r) / 2))))
    calibration = {
        "raw": {
            "left": {"fx": cl["fx"], "fy": cl["fy"], "cx": cl["cx"], "cy": cl["cy"],
                     "width": cl["width"], "height": cl["height"],
                     "distortion_model": "double_sphere",
                     "distortion": {"xi": cl["xi"], "alpha": cl["alpha"]}},
            "right": {"fx": cr["fx"], "fy": cr["fy"], "cx": cr["cx"], "cy": cr["cy"],
                      "width": cr["width"], "height": cr["height"],
                      "distortion_model": "double_sphere",
                      "distortion": {"xi": cr["xi"], "alpha": cr["alpha"]}},
            # rotation/translation kept at this level (siblings of left/right/stereo),
            # matching the real akai calibration.json's raw.rotation/raw.translation shape
            "rotation": R_r_from_l.tolist(),
            "translation": t_r_from_l.tolist(),
            "convention": "rotation/translation take a point expressed in the LEFT camera frame "
                          "into the RIGHT camera frame: p_right = R @ p_left + t (t is the left "
                          "camera's origin, expressed in the right camera's frame)",
            "stereo": {
                "baseline_mm": baseline * 1000.0,
                "baseline_meters": baseline,
                "axis_angle_degrees": angle_deg,
            },
        },
        "rectified": {
            "left": {"fx": fx_r, "fy": fy_r, "cx": cx_r, "cy": cy_r, "width": w, "height": h,
                      "h_fov_degrees": fov_r_h, "v_fov_degrees": fov_r_v, "d_fov_degrees": fov_r_d},
            "right": {"fx": fx_r, "fy": fy_r, "cx": cx_r, "cy": cy_r, "width": w, "height": h,
                       "h_fov_degrees": fov_r_h, "v_fov_degrees": fov_r_v, "d_fov_degrees": fov_r_d},
            "stereo": {"baseline_mm": baseline * 1000.0, "baseline_meters": baseline},
        },
        "source": "das_ego_mcap_embedded_camera_info",
        "self_test": selftest_results,
        "note": "raw distortion is Double Sphere (Usenko et al., 3DV 2018), NOT akai/oak-d's "
                "4-coefficient equidistant model -- this is the one intentional schema deviation "
                "from the akai/oak-d calibration.json shape. raw.left/right.h_fov/v_fov/d_fov_degrees "
                "are intentionally omitted: Double Sphere FOV requires model-specific edge-ray "
                "computation not yet implemented here, and we'd rather omit it than publish a "
                "fabricated number. rectified.*.fov_degrees IS a real, correctly-computed pinhole "
                "FOV (post-rectification the model is a standard pinhole, per akai's own convention).",
    }
    with open(os.path.join(outdir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"  wrote {os.path.join(outdir, 'calibration.json')}")

    print("\n=== 10. meta.json ===")
    proc_duration_s = time.time() - t_start
    meta = {
        "schema_version": 1,
        "device": {
            "device_type": "das-ego",
            "device_id": device_id,
            "model": "GenRobot AI DAS Ego",
            "firmware_version": None,
            "hardware_info": {},
        },
        "recording": {
            "width": w, "height": h, "fps": fps,
            "duration_seconds": n / fps if fps else None,
            "frame_count": n,
            "is_stereo": True,
            "is_rectified": True,
            "bit_depth": 8,
            "is_hdr": False,
            "codec": "h264",
        },
        "source": {
            "original_format": "mcap",
            "original_files": [fname],
            "total_bytes": os.path.getsize(mcap_path),
            "recorded_utc": recorded_dt.isoformat(),
            "recorded_utc_source": "parsed_from_filename_timestamp_assumed_utc",
            "normalized_utc": datetime.now(timezone.utc).isoformat(),
        },
        "capabilities": {
            "has_stereo": True,
            "has_calibration": True,
            "has_imu": True,
            "has_depth": False,
            "has_raw_videos": True,
            "has_extra_camera_views": True,
            "extra_camera_ids": extra_ids,
        },
        "encoding": {"codec": "libx264", "crf": 18, "preset": "medium", "lossless": False},
        "processing_metrics": {
            "duration_seconds": round(proc_duration_s, 2),
            "compute_type": "cpu",
            "cpu_utilization_percent": None,
            "memory_used_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "memory_percent": None,
            "gpu_name": None,
            "gpu_memory_used_mb": None,
            "gpu_utilization_percent": None,
        },
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {os.path.join(outdir, 'meta.json')}")

    shutil.rmtree(tmp)
    print(f"\n=== done in {proc_duration_s:.1f}s -> {outdir} ===")
    return outdir


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: normalize_clip.py <mcap_path> <output_root> <session_id> <date_str>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
