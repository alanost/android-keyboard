#!/usr/bin/env python3
"""replay_extract.py <dir> <out-dir>: turn session replay files into training data.

Writes taps.jsonl (real touch points relative to the pressed key), words.jsonl (one row per
committed word with its key sequence and whether it survived to the end of the session) and
prints a summary. Truncated gzip files are read as far as possible.
"""
import glob, json, os, sys, zlib
from collections import Counter

DELETE = -4  # Constants.CODE_DELETE


def read(path):
    raw = open(path, "rb").read()
    text = zlib.decompressobj(31).decompress(raw).decode("utf-8", "replace")
    for line in text.splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass  # last line of a truncated file


def key_rects(event):
    return {k[0]: k for k in event["keys"] if k[0] > 0}


def session(path, taps, words, stats):
    rects, keys, rows, final, meta = {}, [], [], "", {}
    for ev in read(path):
        kind = ev.get("e")
        stats[kind] += 1
        if kind == "start":
            meta = {"pkg": ev.get("pkg"), "locale": ev.get("locale")}
        elif kind == "layout":
            rects = key_rects(ev)
        elif kind == "key":
            final = ev.get("txt") or final
            if ev["kc"] == DELETE:
                if keys:
                    keys.pop()
                stats["backspace"] += 1
            elif ev["cp"] > 0:
                keys.append([ev["cp"], ev["x"], ev["y"]])
                r = rects.get(ev["cp"])
                if r and ev["x"] >= 0:
                    taps.write(json.dumps({"ch": chr(ev["cp"]),
                                           "dx": (ev["x"] - r[2] - r[4] / 2) / r[4],
                                           "dy": (ev["y"] - r[3] - r[5] / 2) / r[5]}) + "\n")
        elif kind == "commit":
            rows.append({**meta, "typed": ev.get("typed"), "to": ev.get("to"), "ct": ev.get("ct"),
                         "ctx": ev.get("txt"), "keys": keys, "reverted": False})
            keys = []
        elif kind == "revert":
            stats["rejected_corrections"] += 1
            if rows and rows[-1]["to"] == ev.get("to"):
                rows[-1]["reverted"] = True
    for row in rows:
        row["survived"] = bool(row["to"]) and row["to"] in final
        stats["corrected"] += row["typed"] != row["to"]
        words.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    stats = Counter()
    paths = sorted(glob.glob(os.path.join(src, "*.jsonl.gz")))
    with open(f"{out}/taps.jsonl", "w") as taps, open(f"{out}/words.jsonl", "w") as words:
        for path in paths:
            session(path, taps, words, stats)
    print(f"Sitzungen: {len(paths)}")
    for name, count in stats.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
