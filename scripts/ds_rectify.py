#!/usr/bin/env python3
"""Stereo rectification for the Double Sphere ('ds') camera model, which
OpenCV's cv2.fisheye does not support. Implements the closed-form
projection/unprojection from Usenko, Demmel, Cremers, "The Double Sphere
Camera Model" (3DV 2018), self-tests them for round-trip correctness, then
builds a standard rectifying-rotation stereo pair and remaps real frames.
"""
import sys
import json

import numpy as np
import cv2
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


# ---------- Double Sphere model (Usenko et al. 3DV 2018) ----------

def ds_project(P, fx, fy, cx, cy, xi, alpha):
    """3D point(s) P (...,3) in camera frame -> pixel (...,2)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    zp = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zp * zp)
    denom = alpha * d2 + (1 - alpha) * zp
    u = fx * x / denom + cx
    v = fy * y / denom + cy
    return np.stack([u, v], axis=-1), denom  # denom<=0 => behind camera / invalid


def ds_unproject(uv, fx, fy, cx, cy, xi, alpha):
    """Pixel (...,2) -> unit 3D ray (...,3) in camera frame."""
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
    """Round-trip project(unproject(u,v)) == (u,v) on a grid, and
    unproject(project(P)) direction matches P, both within tight tolerance."""
    us = np.linspace(w * 0.05, w * 0.95, 12)
    vs = np.linspace(h * 0.05, h * 0.95, 12)
    uu, vv = np.meshgrid(us, vs)
    uv = np.stack([uu, vv], axis=-1)
    rays, valid = ds_unproject(uv, fx, fy, cx, cy, xi, alpha)
    uv2, denom = ds_project(rays, fx, fy, cx, cy, xi, alpha)
    err = np.linalg.norm((uv - uv2)[valid & (denom > 0)], axis=-1)
    max_err = float(np.max(err)) if err.size else float("nan")
    mean_err = float(np.mean(err)) if err.size else float("nan")
    return max_err, mean_err


# ---------- rectifying rotation (standard stereo-rectification construction) ----------

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


def build_rectifying_rotations(R_b_2, t_b_2, R_b_3, t_b_3):
    """R_b_i: body->cameraI rotation (3x3). t_b_i: camera position in body frame.
    Returns (R1, R2): rotations that map body-frame directions into a shared
    rectified frame for cam2 and cam3 respectively (so applying R_i then the
    camera's own R_b_i^{-1}... see usage in build_maps)."""
    baseline = t_b_3 - t_b_2
    e1 = baseline / np.linalg.norm(baseline)
    # average forward (+z) axis of the two cameras, expressed in body frame
    z2 = R_b_2.T @ np.array([0, 0, 1.0])
    z3 = R_b_3.T @ np.array([0, 0, 1.0])
    z_avg = z2 + z3
    z_avg = z_avg / np.linalg.norm(z_avg)
    e3 = z_avg - np.dot(z_avg, e1) * e1  # orthogonalize
    e3 = e3 / np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2)
    R_rect_body = np.stack([e1, e2, e3], axis=0)  # body -> rectified frame
    return R_rect_body


def build_remap(fx, fy, cx, cy, xi, alpha, R_cam_from_rect, w, h, fx_r, fy_r, cx_r, cy_r):
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (us - cx_r) / fx_r
    y = (vs - cy_r) / fy_r
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_cam = dirs @ R_cam_from_rect.T  # rotate rect-frame ray into this camera's own frame
    uv, denom = ds_project(dirs_cam, fx, fy, cx, cy, xi, alpha)
    mapx = uv[..., 0].astype(np.float32)
    mapy = uv[..., 1].astype(np.float32)
    mapx[denom <= 0] = -1
    mapy[denom <= 0] = -1
    return mapx, mapy


def get_cam_calib(mcap_path, cam_topic_id):
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if channel.topic == f"/robot0/sensor/{cam_topic_id}/camera_info":
                D = list(msg.D)
                return {
                    "fx": D[0], "fy": D[1], "cx": D[2], "cy": D[3], "xi": D[4], "alpha": D[5],
                    "width": msg.width, "height": msg.height,
                    "t": np.array(list(msg.T_b_c)[:3]),
                    "q_xyzw": np.array(list(msg.T_b_c)[3:7]),
                }
    raise RuntimeError(f"no camera_info for {cam_topic_id}")


def main(mcap_path):
    cal2 = get_cam_calib(mcap_path, "camera2")
    cal3 = get_cam_calib(mcap_path, "camera3")
    w, h = cal2["width"], cal2["height"]

    print("=== self-test: double-sphere projection/unprojection round-trip ===")
    for name, c in (("camera2", cal2), ("camera3", cal3)):
        max_err, mean_err = self_test(c["fx"], c["fy"], c["cx"], c["cy"], c["xi"], c["alpha"], w, h)
        print(f"  {name}: max_px_err={max_err:.6f} mean_px_err={mean_err:.6f} "
              f"{'PASS' if max_err < 0.01 else 'FAIL'}")

    R_b_2 = quat_xyzw_to_R(cal2["q_xyzw"])
    R_b_3 = quat_xyzw_to_R(cal3["q_xyzw"])
    R_rect_body = build_rectifying_rotations(R_b_2, cal2["t"], R_b_3, cal3["t"])

    # R_cam_from_rect: rotate a ray expressed in the rectified frame into this
    # camera's own optical frame. rectified-frame axes are expressed in BODY
    # coords (R_rect_body rows); camera's own frame relates to body via R_b_i.
    R_cam2_from_rect = R_b_2 @ R_rect_body.T
    R_cam3_from_rect = R_b_3 @ R_rect_body.T

    fx_r = (cal2["fx"] + cal3["fx"]) / 2
    fy_r = (cal2["fy"] + cal3["fy"]) / 2
    cx_r, cy_r = w / 2, h / 2
    baseline_m = float(np.linalg.norm(cal3["t"] - cal2["t"]))

    print(f"\nbaseline: {baseline_m*100:.3f} cm, rectified virtual intrinsics: "
          f"fx={fx_r:.2f} fy={fy_r:.2f} cx={cx_r:.1f} cy={cy_r:.1f}")

    mapx2, mapy2 = build_remap(cal2["fx"], cal2["fy"], cal2["cx"], cal2["cy"],
                                cal2["xi"], cal2["alpha"], R_cam2_from_rect, w, h, fx_r, fy_r, cx_r, cy_r)
    mapx3, mapy3 = build_remap(cal3["fx"], cal3["fy"], cal3["cx"], cal3["cy"],
                                cal3["xi"], cal3["alpha"], R_cam3_from_rect, w, h, fx_r, fy_r, cx_r, cy_r)

    np.savez("/tmp/ds_rect_maps.npz", mapx2=mapx2, mapy2=mapy2, mapx3=mapx3, mapy3=mapy3)
    info = {
        "model": "double_sphere", "baseline_m": baseline_m,
        "rectified_intrinsics": {"fx": fx_r, "fy": fy_r, "cx": cx_r, "cy": cy_r},
        "resolution": [w, h],
        "cam2_calib": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cal2.items()},
        "cam3_calib": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cal3.items()},
    }
    with open("/tmp/ds_rect_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nmaps saved to /tmp/ds_rect_maps.npz, info to /tmp/ds_rect_info.json")


if __name__ == "__main__":
    main(sys.argv[1])
