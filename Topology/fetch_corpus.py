"""Fetch + verify the ETH-80 corpus for this session (harness utility).

The canonical source is the MPI-INF D2 datasets page; this sandbox can only
reach github.com / codeload.github.com / api.github.com, so the public mirror
``chenchkx/ETH-80`` (a byte-for-byte copy of the 256x256 object images plus
the ``maps/`` object masks) is used.  It is verified against the layout the
harness expects:

    data/ETH-80-master/<class>/<instance>/<name>-<az>-<el>.png
    data/ETH-80-master/<class>/<instance>/maps/<name>-<az>-<el>-map.png

The mirror ALREADY has this layout, so no normalisation is performed; the
script refuses to proceed if the counts do not match 8 x 10 x 41.

The large image tree is kept in ``.cache/`` (outside git) and exposed through
``data/ETH-80-master`` symlink.  Re-run with --fetch to re-download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
TARBALL = TREE / ".cache" / "eth80.tar.gz"
RAW = TREE / ".cache" / "ETH-80-raw"
LINK = TREE / "data" / "ETH-80-master"
URL = "https://codeload.github.com/chenchkx/ETH-80/tar.gz/refs/heads/master"
CLASSES = ("apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato")
CLASS_DIR = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
             "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}
NAME = re.compile(r"^([a-z]+)(\d+)-(\d{3})-(\d{3})\.png$")
MAP_NAME = re.compile(r"^([a-z]+)(\d+)-(\d{3})-(\d{3})-map\.png$")


def fetch() -> None:
    TARBALL.parent.mkdir(parents=True, exist_ok=True)
    if TARBALL.is_file() and TARBALL.stat().st_size > 100_000_000:
        print("using cached tarball", TARBALL)
    else:
        print("downloading", URL)
        with urllib.request.urlopen(URL, timeout=900) as src, open(TARBALL, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    if not RAW.is_dir() or not (RAW / "1" / "2" / "apple2-022-000.png").is_file():
        RAW.mkdir(parents=True, exist_ok=True)
        with tarfile.open(TARBALL, "r:gz") as tf:
            members = [m for m in tf.getmembers()
                       if m.name.split("/")[-1] != ""]
            for m in members:
                parts = m.name.split("/")
                m.name = "/".join(parts[1:])       # strip the repo root dir
                tf.extractall(RAW, members=members)
    LINK.parent.mkdir(parents=True, exist_ok=True)
    if LINK.is_symlink() or LINK.is_file():
        LINK.unlink()
    elif LINK.is_dir():
        raise RuntimeError("data/ETH-80-master is a real directory; refusing")
    LINK.symlink_to(os.path.relpath(RAW, LINK.parent), target_is_directory=True)


def verify() -> dict:
    out: dict = {"source": URL, "layout": str(LINK), "normalised": False,
                 "note": "mirror already has <class>/<instance>/<name>.png + maps/; "
                         "no layout change was needed"}
    from PIL import Image
    per_class = {}
    total_obj = total_map = 0
    view_counts = Counter()
    sizes = Counter()
    for cls in CLASSES:
        root = LINK / CLASS_DIR[cls]
        inst = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "maps"):
            objs = sorted(f.name for f in d.iterdir() if NAME.match(f.name))
            others = sorted(f.name for f in d.iterdir()
                            if f.is_file() and f.suffix == ".png"
                            and not NAME.match(f.name))
            if others:
                raise RuntimeError("%s: stray pngs %r" % (d, others[:3]))
            maps_dir = d / "maps"
            maps = (sorted(f.name for f in maps_dir.iterdir()
                           if MAP_NAME.match(f.name))
                    if maps_dir.is_dir() else [])
            inst[d.name] = {"images": len(objs), "masks": len(maps),
                            "views": len({NAME.match(f).group(0)[-11:]
                                          for f in objs})}
            view_counts.update("-" + NAME.match(f).group(3) + "-" +
                               NAME.match(f).group(4) for f in objs)
            total_obj += len(objs)
            total_map += len(maps)
            if objs:
                with Image.open(d / objs[0]) as im:
                    sizes["%dx%d" % im.size] += 1
        per_class[cls] = {"instances": len(inst), "detail": inst}
    out["counts"] = {
        "classes": len(per_class),
        "instances_per_class": {c: per_class[c]["instances"] for c in CLASSES},
        "views_per_instance": sorted({v["images"] for c in CLASSES
                                      for v in per_class[c]["detail"].values()}),
        "object_images": total_obj, "mask_images": total_map,
        "views_used": {"-090-000": view_counts["-090-000"],
                       "-035-135": view_counts["-035-135"]},
        "image_sizes": dict(sizes),
        "distinct_views": len(view_counts),
    }
    expected = (len(per_class) == 8
                and all(per_class[c]["instances"] == 10 for c in CLASSES)
                and out["counts"]["views_per_instance"] == [41]
                and total_obj == 3280 and total_map == 3280
                and out["counts"]["views_used"]["-090-000"] == 80
                and out["counts"]["views_used"]["-035-135"] == 80
                and dict(sizes) == {"256x256": 80}
                and (LINK / "1" / "2" / "apple2-022-000.png").is_file())
    out["verified"] = bool(expected)
    if not expected:
        raise RuntimeError("ETH-80 layout verification FAILED: " +
                           json.dumps(out["counts"], indent=2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" /
                                         "corpus_verify.json"))
    a = ap.parse_args(argv)
    if a.fetch or not LINK.is_dir():
        fetch()
    result = verify()
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verified", "counts", "note")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
