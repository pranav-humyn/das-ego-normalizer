#!/usr/bin/env python3
"""Build camera{N}_raw.mp4 for every camera in a DAS-EGO MCAP by concatenating
its native H.264 chunks (log_time order) and remuxing (no re-encode, since the
source is already H.264 - unlike akai's raw-MJPEG which genuinely needs a
fresh encode)."""
import sys
import os
import subprocess
from collections import defaultdict

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    streams = defaultdict(list)
    times = defaultdict(list)
    with open(path, "rb") as f:
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

    for cam_id, chunks in streams.items():
        ts = times[cam_id]
        n = len(chunks)
        duration_s = (ts[-1] - ts[0]) / 1e9 if n > 1 else 0
        fps = round((n - 1) / duration_s, 3) if duration_s > 0 else 30.0

        h264_path = os.path.join(outdir, f"{cam_id}.h264")
        with open(h264_path, "wb") as out:
            for c in chunks:
                out.write(c)

        mp4_path = os.path.join(outdir, f"{cam_id}_raw.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "h264", "-r", str(fps), "-i", h264_path,
             "-c:v", "copy", mp4_path],
            capture_output=True, text=True)
        ok = os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0
        os.remove(h264_path)

        # verify with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,width,height,r_frame_rate",
             "-of", "default=noprint_wrappers=1", mp4_path],
            capture_output=True, text=True)
        print(f"{cam_id}: frames={n} computed_fps={fps} -> {mp4_path} "
              f"{'OK' if ok else 'FAILED: ' + r.stderr[-300:]}")
        print(f"  ffprobe: {probe.stdout.strip().replace(chr(10), ' ')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
