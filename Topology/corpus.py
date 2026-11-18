"""The pinned ETH-80 corpus and the instance-disjoint split.

The split is by OBJECT INSTANCE: the stored graphs come from the first four
instances of each class and the test images from the next four instances of the
same classes.  No test image is a new view of a stored object.

Instances are ordered by the ETH-80 instance number in the FILENAME
(``apple7-090-000.png`` is instance 7 of apple), not by directory name: this
corpus keeps one image set per directory, and in the car class the directory
names do not follow the instance numbers (directory ``4`` holds ``car11-*``).
The split is therefore defined by RANK in the class's sorted instance list and
the resolved instance numbers are reported with the split, so it is auditable.

Views are named by their ``-<elevation>-<azimuth>`` suffix.  Every instance is
checked to carry the same 41 views (``maps/`` ground-truth directories beside
the images are excluded).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from guard import TREE

DATA = TREE / "data" / "ETH-80-master"
_NAME = re.compile(r"^([a-z]+)(\d+)-(\d{3})-(\d{3})\.png$")

CLASSES = ("apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato")
CLASS_DIR = {"apple": "1", "car": "2", "cow": "3", "cup": "4",
             "dog": "5", "horse": "6", "pear": "7", "tomato": "8"}

#: ranks are 0-based positions in the class's sorted instance list
TRAIN_RANKS = (0, 1, 2, 3)
TRAIN_VIEWS = ("-090-000",)
TEST_RANKS = (4, 5, 6, 7)
TEST_VIEWS = ("-090-000", "-035-135")

#: [TUNE] instance-disjoint reference / dev / held-out split.  Three reference
#: instances per class (the user's minimum), three dev instances and three
#: held-out instances; rank 9 stays unused.  Dev and held-out queries each
#: carry two views, so no query is a reference view.
REF_RANKS = (0, 1, 2)
DEV_RANKS = (3, 4, 5)
HELDOUT_RANKS = (6, 7, 8)
REF_VIEWS = ("-090-000",)
QUERY_VIEWS = ("-090-000", "-035-135")

VIEWS_PER_INSTANCE = 41


def _instances(cls: str) -> dict:
    """instance number -> {'-elev-azim': path} for one class."""
    root = DATA / CLASS_DIR[cls]
    if not root.is_dir():
        raise FileNotFoundError(root)
    out: dict = {}
    for base, dirs, files in root.walk() if hasattr(root, "walk") else _walk(root):
        dirs[:] = [d for d in dirs if d != "maps"]
        for name in files:
            match = _NAME.match(name)
            if match is None:
                if name.endswith(".png"):
                    raise RuntimeError("%s: unparsed image name %r" % (cls, name))
                continue
            instance = int(match.group(2))
            view = "-%s-%s" % (match.group(3), match.group(4))
            slot = out.setdefault(instance, {})
            if view in slot:
                raise RuntimeError("%s instance %d: duplicate view %s"
                                   % (cls, instance, view))
            slot[view] = str(Path(base) / name)
    return out


def _walk(root: Path):
    import os
    for base, dirs, files in os.walk(root):
        yield Path(base), dirs, files


def _entry(cls: str, instance: int, view: str, path: str) -> dict:
    return {"class": cls, "object": instance, "view": view, "path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "bytes": Path(path).stat().st_size}


def split() -> dict:
    """The pre-registered train/test entries, with the corpus checked."""
    checks, split_ids = {}, {}
    train, test = [], []
    for cls in CLASSES:
        instances = _instances(cls)
        ids = sorted(instances)
        if len(ids) != 10:
            raise RuntimeError("%s: %d instances, expected 10" % (cls, len(ids)))
        reference = tuple(sorted(instances[ids[0]]))
        if len(reference) != VIEWS_PER_INSTANCE:
            raise RuntimeError("%s: %d views, expected %d"
                               % (cls, len(reference), VIEWS_PER_INSTANCE))
        for instance in ids:
            if tuple(sorted(instances[instance])) != reference:
                raise RuntimeError("%s instance %d: view set differs" % (cls, instance))
        for view in TRAIN_VIEWS + TEST_VIEWS:
            if view not in reference:
                raise RuntimeError("%s: view %s absent" % (cls, view))
        train_ids = [ids[r] for r in TRAIN_RANKS]
        test_ids = [ids[r] for r in TEST_RANKS]
        if set(train_ids) & set(test_ids):
            raise RuntimeError("%s: train and test instances overlap" % cls)
        split_ids[cls] = {"train_instances": train_ids,
                          "test_instances": test_ids}
        checks[cls] = {"instances": len(ids), "views_per_instance": len(reference),
                       "images": sum(len(v) for v in instances.values())}
        for instance in train_ids:
            for view in TRAIN_VIEWS:
                train.append(_entry(cls, instance, view, instances[instance][view]))
        for instance in test_ids:
            for view in TEST_VIEWS:
                test.append(_entry(cls, instance, view, instances[instance][view]))
    if {(e["class"], e["object"]) for e in train} & {
            (e["class"], e["object"]) for e in test}:
        raise RuntimeError("split is not instance-disjoint")
    return {"corpus": "ETH-80 (data/ETH-80-master: 8 classes x 10 instances x 41 views)",
            "split": {"by": "object instance (rank in the sorted instance list)",
                      "train_ranks": list(TRAIN_RANKS),
                      "train_views": list(TRAIN_VIEWS),
                      "test_ranks": list(TEST_RANKS),
                      "test_views": list(TEST_VIEWS),
                      "instances": split_ids},
            "checks": checks, "train": train, "test": test,
            "counts": {"train": len(train), "test": len(test)}}


def name_of(entry: dict) -> str:
    return "%s%d%s" % (entry["class"], int(entry["object"]), entry["view"])


def split_tuning() -> dict:
    """[TUNE] reference / dev / held-out entries, instance-disjoint.

    References are ranks 0-2 (view -090-000).  Dev queries are ranks 3-5 and
    held-out queries ranks 6-8, each at views -090-000 and -035-135.  No query
    instance appears as a reference; rank 9 is unused.
    """
    refs, dev, held = [], [], []
    resolved = {}
    for cls in CLASSES:
        instances = _instances(cls)
        ids = sorted(instances)
        if len(ids) != 10:
            raise RuntimeError("%s: %d instances, expected 10" % (cls, len(ids)))
        for view in REF_VIEWS + QUERY_VIEWS:
            if view not in instances[ids[0]]:
                raise RuntimeError("%s: view %s absent" % (cls, view))
        ref_ids = [ids[r] for r in REF_RANKS]
        dev_ids = [ids[r] for r in DEV_RANKS]
        held_ids = [ids[r] for r in HELDOUT_RANKS]
        sets = [set(ref_ids), set(dev_ids), set(held_ids)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise RuntimeError("%s: tuning split overlaps" % cls)
        resolved[cls] = {"reference_instances": ref_ids,
                         "dev_instances": dev_ids,
                         "heldout_instances": held_ids}
        for instance in ref_ids:
            for view in REF_VIEWS:
                refs.append(_entry(cls, instance, view, instances[instance][view]))
        for instance in dev_ids:
            for view in QUERY_VIEWS:
                dev.append(_entry(cls, instance, view, instances[instance][view]))
        for instance in held_ids:
            for view in QUERY_VIEWS:
                held.append(_entry(cls, instance, view, instances[instance][view]))
    keys = lambda rows: {(e["class"], e["object"], e["view"]) for e in rows}
    if keys(refs) & keys(dev) or keys(refs) & keys(held) or keys(dev) & keys(held):
        raise RuntimeError("tuning split is not disjoint by (class, instance, view)")
    return {"corpus": "ETH-80 (data/ETH-80-master: 8 classes x 10 instances x 41 views)",
            "split": {"by": "object instance (rank in the sorted instance list)",
                      "reference_ranks": list(REF_RANKS),
                      "reference_views": list(REF_VIEWS),
                      "dev_ranks": list(DEV_RANKS),
                      "heldout_ranks": list(HELDOUT_RANKS),
                      "query_views": list(QUERY_VIEWS),
                      "instances": resolved},
            "references": refs, "dev": dev, "heldout": held,
            "counts": {"references": len(refs), "dev": len(dev),
                       "heldout": len(held)}}


if __name__ == "__main__":
    import json
    import guard
    guard.activate()
    value = split()
    print(json.dumps({k: v for k, v in value.items()
                      if k not in ("train", "test")}, indent=2, sort_keys=True))
