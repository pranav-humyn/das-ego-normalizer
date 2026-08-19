#!/usr/bin/env python3
"""Apply the saved double-sphere rectification maps to a real camera2/camera3
frame pair, then run the empirical validation: pick matched feature points
and check (a) vertical disparity (dy) is near zero, (b) horizontal disparity
is consistently signed."""
import sys
import numpy as np
import cv2

d = np.load("/tmp/ds_rect_maps.npz")
mapx2, mapy2, mapx3, mapy3 = d["mapx2"], d["mapy2"], d["mapx3"], d["mapy3"]

img2 = cv2.imread(sys.argv[1])  # camera2 raw frame
img3 = cv2.imread(sys.argv[2])  # camera3 raw frame
out2_path, out3_path = sys.argv[3], sys.argv[4]

rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
rect3 = cv2.remap(img3, mapx3, mapy3, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
cv2.imwrite(out2_path, rect2)
cv2.imwrite(out3_path, rect3)
print(f"wrote {out2_path}, {out3_path}")

# feature-match validation: ORB detect+match between the two rectified images
orb = cv2.ORB_create(2000)
k1, d1 = orb.detectAndCompute(cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY), None)
k2, d2 = orb.detectAndCompute(cv2.cvtColor(rect3, cv2.COLOR_BGR2GRAY), None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(d1, d2)
matches = sorted(matches, key=lambda m: m.distance)[:60]

dys, dxs = [], []
for m in matches:
    p1 = np.array(k1[m.queryIdx].pt)
    p2 = np.array(k2[m.trainIdx].pt)
    dy = p2[1] - p1[1]
    dx = p1[0] - p2[0]  # left(cam2) minus right(cam3): positive disparity expected if cam2=left
    if abs(dy) < 50 and abs(dx) < 300:  # drop obvious mismatches
        dys.append(dy)
        dxs.append(dx)

dys = np.array(dys)
dxs = np.array(dxs)
print(f"\nmatched keypoints used: {len(dys)} (after outlier filter)")
print(f"vertical disparity (dy): mean={dys.mean():.2f}px median={np.median(dys):.2f}px std={dys.std():.2f}px")
print(f"horizontal disparity (dx, cam2-cam3): mean={dxs.mean():.2f}px median={np.median(dxs):.2f}px "
      f"positive_fraction={np.mean(dxs > 0):.2f}")
