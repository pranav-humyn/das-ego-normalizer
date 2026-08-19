#!/usr/bin/env python3
"""Build one clear before/after comparison image: raw (unrectified) camera2 vs
camera3 stacked with horizontal reference lines (features land on DIFFERENT
rows), next to rectified camera2 vs camera3 with the same lines (features
land on the SAME row). This is the plain visual proof, no numbers needed.
"""
import sys
import cv2
import numpy as np

raw2 = cv2.imread(sys.argv[1])   # cam2_frame.png (raw)
raw3 = cv2.imread(sys.argv[2])   # cam3_frame.png (raw)
rect2 = cv2.imread(sys.argv[3])  # cam2_rectified.jpg
rect3 = cv2.imread(sys.argv[4])  # cam3_rectified.jpg
out_path = sys.argv[5]

LINE_COLOR = (0, 0, 255)
LINE_Y_FRACS = [0.25, 0.45, 0.65]


def with_lines(img, label):
    img = img.copy()
    h, w = img.shape[:2]
    for f in LINE_Y_FRACS:
        y = int(h * f)
        cv2.line(img, (0, y), (w, y), LINE_COLOR, 3)
    cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(img, label, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return img


def side_by_side(a, b):
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    gap = np.full((h, 8, 3), 255, np.uint8)
    return np.hstack([a, gap, b])


top = side_by_side(with_lines(raw2, "camera2 - RAW (unrectified)"),
                    with_lines(raw3, "camera3 - RAW (unrectified)"))
bottom = side_by_side(with_lines(rect2, "camera2 - RECTIFIED"),
                       with_lines(rect3, "camera3 - RECTIFIED"))

# match widths
w = min(top.shape[1], bottom.shape[1])
top = cv2.resize(top, (w, int(top.shape[0] * w / top.shape[1])))
bottom = cv2.resize(bottom, (w, int(bottom.shape[0] * w / bottom.shape[1])))
gap = np.full((12, w, 3), (0, 200, 0), np.uint8)
final = np.vstack([top, gap, bottom])
cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"wrote {out_path}  ({final.shape[1]}x{final.shape[0]})")
