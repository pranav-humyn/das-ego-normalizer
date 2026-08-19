#!/usr/bin/env python3
"""Inspect a DAS-EGO MCAP: channels, schemas, message counts, and decode
one representative message per topic (camera_info variants + imu) so we can
see real field values instead of guessing from strings/xxd.
"""
import sys
from collections import Counter

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def main(path):
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()

        print("=== SCHEMAS ===")
        for sid, schema in summary.schemas.items():
            print(f"  id={sid} name={schema.name} encoding={schema.encoding}")

        print("\n=== CHANNELS ===")
        for cid, ch in summary.channels.items():
            schema = summary.schemas.get(ch.schema_id)
            sname = schema.name if schema else "?"
            print(f"  id={cid} topic={ch.topic} schema={sname} encoding={ch.message_encoding}")

        print("\n=== MESSAGE COUNTS PER TOPIC ===")
        counts = Counter()
        first_msg_per_topic = {}
        min_t, max_t = None, None
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            counts[channel.topic] += 1
            if channel.topic not in first_msg_per_topic:
                first_msg_per_topic[channel.topic] = (schema.name, proto_msg)
            t = message.log_time
            min_t = t if min_t is None else min(min_t, t)
            max_t = t if max_t is None else max(max_t, t)
        for topic, c in sorted(counts.items()):
            print(f"  {topic}: {c}")
        print(f"\n  time range (ns): {min_t} -> {max_t}  (duration s: {(max_t-min_t)/1e9:.2f})")

        print("\n=== SAMPLE DECODED MESSAGE PER TOPIC (camera_info* and imu only) ===")
        for topic, (sname, msg) in sorted(first_msg_per_topic.items()):
            if "camera_info" in topic or "imu" in topic:
                print(f"\n--- {topic} ({sname}) ---")
                print(msg)


if __name__ == "__main__":
    main(sys.argv[1])
