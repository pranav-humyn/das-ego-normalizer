#!/usr/bin/env python3
"""Determine real stereo pairs among DAS-EGO's 6 cameras from their body->camera
extrinsics (T_b_c: tx,ty,tz + quaternion, 7 floats), rather than assuming from
topic names. For every camera pair, report: baseline distance (translation-only,
convention-independent) and the angle between optical (+z) axes under both
plausible quaternion orderings (since the proto doesn't label them and we
should not silently assume one)."""
import sys
from itertools import combinations

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def quat_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
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


def main(path):
    cams = {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if channel.topic.endswith("/camera_info") and "camera_info_resize" not in channel.topic:
                cam_id = channel.topic.split("/")[3]  # .../cameraN/camera_info
                t = list(msg.T_b_c)
                cams[cam_id] = {"t": np.array(t[:3]), "q_raw": t[3:7], "frame_id": msg.frame_id}

    order = sorted(cams.keys(), key=lambda s: int(s.replace("camera", "")))
    print("=== per-camera translation (body frame, meters) + frame_id ===")
    for c in order:
        t = cams[c]["t"]
        print(f"  {c}: t=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  frame_id={cams[c]['frame_id']}")

    print("\n=== pairwise baseline distance (convention-independent) + optical-axis angle ===")
    print("(axis angle computed under both xyzw and wxyz quaternion-order assumptions)")
    results = []
    for a, b in combinations(order, 2):
        ta, tb = cams[a]["t"], cams[b]["t"]
        baseline = float(np.linalg.norm(ta - tb))

        qa_raw, qb_raw = cams[a]["q_raw"], cams[b]["q_raw"]
        angles = {}
        for label, idx in (("xyzw", (0, 1, 2, 3)), ("wxyz", (1, 2, 3, 0))):
            qa = [qa_raw[i] for i in idx]
            qb = [qb_raw[i] for i in idx]
            za = quat_to_R(qa) @ np.array([0, 0, 1.0])
            zb = quat_to_R(qb) @ np.array([0, 0, 1.0])
            cos_ang = np.clip(np.dot(za, zb) / (np.linalg.norm(za) * np.linalg.norm(zb)), -1, 1)
            angles[label] = np.degrees(np.arccos(cos_ang))
        results.append((baseline, a, b, angles))

    results.sort(key=lambda r: r[0])
    for baseline, a, b, angles in results:
        print(f"  {a}-{b}: baseline={baseline*100:.2f}cm  "
              f"axis_angle[xyzw]={angles['xyzw']:.1f}deg  axis_angle[wxyz]={angles['wxyz']:.1f}deg")


if __name__ == "__main__":
    main(sys.argv[1])
