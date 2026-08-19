#!/usr/bin/env python3
"""Concatenate each camera's H.264 elementary-stream chunks (in log_time order)
into a single .h264 file, then use ffmpeg to decode one representative frame
per camera as a JPEG for visual FOV-overlap confirmation.
"""
import sys
import os
import subprocess
from collections import defaultdict

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    streams = defaultdict(list)
    formats = {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        msgs = []
        for schema, channel, message, msg in reader.iter_decoded_messages():
            if not channel.topic.endswith("/compressed"):
                continue
            cam_id = channel.topic.split("/")[3]
            msgs.append((message.log_time, cam_id, msg.data, msg.format))
        msgs.sort(key=lambda m: (m[1], m[0]))
        for _, cam_id, data, fmt in msgs:
            streams[cam_id].append(data)
            formats[cam_id] = fmt

    for cam_id, chunks in streams.items():
        fmt = formats[cam_id]
        h264_path = os.path.join(outdir, f"{cam_id}.h264")
        with open(h264_path, "wb") as out:
            for c in chunks:
                out.write(c)
        jpg_path = os.path.join(outdir, f"{cam_id}.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "h264", "-i", h264_path, "-frames:v", "1", jpg_path],
            capture_output=True, text=True)
        ok = os.path.exists(jpg_path) and os.path.getsize(jpg_path) > 0
        print(f"{cam_id}: format={fmt} chunks={len(chunks)} bytes={sum(len(c) for c in chunks)} "
              f"-> {jpg_path} {'OK' if ok else 'FAILED: ' + r.stderr[-300:]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
