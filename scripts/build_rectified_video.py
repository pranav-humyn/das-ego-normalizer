#!/usr/bin/env python3
"""Apply the saved double-sphere rectification maps to every frame of the
camera2/camera3 raw videos, producing real playable rectified mp4s (not just
one still frame)."""
import sys
import numpy as np
import cv2

d = np.load("/tmp/ds_rect_maps.npz")
mapx2, mapy2, mapx3, mapy3 = d["mapx2"], d["mapy2"], d["mapx3"], d["mapy3"]

cap2 = cv2.VideoCapture(sys.argv[1])
cap3 = cv2.VideoCapture(sys.argv[2])
out2_path, out3_path = sys.argv[3], sys.argv[4]

w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w2 = cv2.VideoWriter(out2_path, fourcc, fps, (w, h))
w3 = cv2.VideoWriter(out3_path, fourcc, fps, (w, h))

n = 0
while True:
    ok2, f2 = cap2.read()
    ok3, f3 = cap3.read()
    if not ok2 or not ok3:
        break
    r2 = cv2.remap(f2, mapx2, mapy2, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    r3 = cv2.remap(f3, mapx3, mapy3, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    w2.write(r2)
    w3.write(r3)
    n += 1

cap2.release(); cap3.release(); w2.release(); w3.release()
print(f"wrote {n} rectified frames to {out2_path} and {out3_path}")
