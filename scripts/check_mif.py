"""Check that the memory images Quartus extracted from $readmemh (db/*.mif) hold exactly the words of our hex files.

    python scripts/check_mif.py quartus/snn_demo rtl/weights/kth6

A ROM that came out of synthesis all-zero or shifted would classify wrongly on the board and nothing in simulation would show it
(simulation reads the hex files directly).
"""
import re
import sys
from pathlib import Path


def read_hex(path):
    return [int(l.strip(), 16) for l in Path(path).read_text().splitlines() if l.strip()]


def read_mif(path):
    text = Path(path).read_text()
    depth = int(re.search(r"DEPTH\s*=\s*(\d+)", text).group(1))
    width = int(re.search(r"WIDTH\s*=\s*(\d+)", text).group(1))
    aradix = {"UNS": 10, "DEC": 10, "HEX": 16, "BIN": 2}[re.search(r"ADDRESS_RADIX\s*=\s*(\w+)", text).group(1)]
    dradix = {"UNS": 10, "DEC": 10, "HEX": 16, "BIN": 2}[re.search(r"DATA_RADIX\s*=\s*(\w+)", text).group(1)]
    words = {}
    body = text.split("CONTENT BEGIN", 1)[1]
    for m in re.finditer(r"^\s*(?:\[(\w+)\.\.(\w+)\]|(\w+))\s*:\s*(\w+)\s*;", body, flags=re.M):
        if set(m.group(4).upper()) <= {"X"}:      # word the source left uninitialised
            continue
        v = int(m.group(4), dradix)
        if m.group(3) is not None:
            words[int(m.group(3), aradix)] = v
        else:
            for a in range(int(m.group(1), aradix), int(m.group(2), aradix) + 1):
                words[a] = v
    return depth, width, words


def main():
    proj, wdir = Path(sys.argv[1]), Path(sys.argv[2])
    images = {name: read_hex(wdir / name) for name in ("w1.hex", "w2.hex", "wfc.hex", "clips.hex")}
    mifs = {}
    for f in sorted((proj / "db").glob("*.mif")):
        depth, width, words = read_mif(f)
        mifs[f.name] = (depth, width, words)
    bad = 0
    for name, image in images.items():
        found = None
        for fn, (depth, width, words) in mifs.items():
            if all(words.get(a, 0) == v for a, v in enumerate(image)) and len(words) >= len(image) - image.count(0):
                found = (fn, depth, width)
                break
        print("%-10s %6d words: %s" % (name, len(image), ("matches %s (depth %d, width %d)" % found) if found else "NO MATCHING MIF"))
        bad += found is None
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
