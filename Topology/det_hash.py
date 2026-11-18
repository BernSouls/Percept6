"""Hash the raw joint_mcs output per known-transform shift."""
import hashlib, json, sys
import guard
guard.activate()
import numpy as np
from PIL import Image
import eth80_frontend as FE
from matching.joint_mcs import joint_mcs
from pathlib import Path
IMG = str(Path(guard.TREE) / "data" / "ETH-80-master" / "1" / "2" / "apple2-022-000.png")

def build(shift):
    with Image.open(IMG) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if shift:
        image = np.roll(image, (0, shift), axis=(0, 1))
    trace = {}
    g = FE.build_graph(image, name="det_%d" % shift, record_co_located=False, trace=trace)
    return g

base = build(0)
for shift in (0, 2, -2, 3, -3):
    other = base if shift == 0 else build(shift)
    res = joint_mcs(base.subgraph, other.subgraph, base.sigs, other.sigs)
    payload = {
        "shift": shift,
        "mapping": sorted(res.mapping.items()),
        "score": res.score,
        "matched_edges": res.matched_edges,
        "node_credit": res.node_credit,
        "edge_credit": res.edge_credit,
        "denominator": res.denominator,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print("shift %+d  score %.6f  pairs %4d  sha256 %s" % (shift, res.score, len(res.mapping), h))
