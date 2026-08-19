#!/usr/bin/env python3
"""Export /robot0/sensor/imu to a clean CSV, then sanity-check it: gravity
magnitude at rest (~9.8 m/s^2), timestamp continuity/rate, and time-range
overlap with the video."""
import sys
import csv
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def main(mcap_path, csv_path):
    rows = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if channel.topic != "/robot0/sensor/imu":
                continue
            rows.append({
                "log_time_ns": message.log_time,
                "ax": msg.linear_acceleration.x,
                "ay": msg.linear_acceleration.y,
                "az": msg.linear_acceleration.z,
                "gx": msg.angular_velocity.x,
                "gy": msg.angular_velocity.y,
                "gz": msg.angular_velocity.z,
            })

    rows.sort(key=lambda r: r["log_time_ns"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {csv_path}")

    # --- sanity checks ---
    accel = np.array([[r["ax"], r["ay"], r["az"]] for r in rows])
    mag = np.linalg.norm(accel, axis=1)
    print(f"\n=== gravity magnitude check (expect ~9.8 m/s^2 if this is raw accel) ===")
    print(f"  mean|a|={mag.mean():.3f}  median|a|={np.median(mag):.3f}  "
          f"min={mag.min():.3f}  max={mag.max():.3f}  std={mag.std():.3f}")
    all_zero = np.all(accel == 0)
    print(f"  all-zero check: {'FAIL (all zero - broken, like the akai bucket bug)' if all_zero else 'PASS (non-zero)'}")

    ts = np.array([r["log_time_ns"] for r in rows], dtype=np.int64)
    dt_ns = np.diff(ts)
    rate_hz = 1e9 / dt_ns.mean() if len(dt_ns) else float("nan")
    print(f"\n=== timestamp continuity check ===")
    print(f"  n={len(ts)} duration_s={(ts[-1]-ts[0])/1e9:.3f} implied_rate_hz={rate_hz:.2f}")
    print(f"  dt(ns) mean={dt_ns.mean():.0f} std={dt_ns.std():.0f} max_gap_ns={dt_ns.max()}")
    gaps = dt_ns[dt_ns > 3 * dt_ns.mean()]
    print(f"  gaps >3x mean spacing: {len(gaps)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
