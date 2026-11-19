"""Pre-registered STEP 0 calibration measurement: the tracer's own
consecutive-peak gap and angle distribution on the dev split.

Measures, over the dev scenes, the inter-peak distance and orientation
difference between adjacent PEAK_CHAIN members -- the two quantities the L3
spine reach and the L4 tangent cone are calibrated against. Read-only against
the product; no product edits. Writes JSON only.

Run: .venv/bin/python Topology/_calib_tracer.py
"""
import sys, math, json
from collections import Counter
sys.path[:0] = ['Topology', '.']

import numpy as np
from PIL import Image

import parameters as P
P.FE_L3_SPINE = True
P.FE_L4_JUNCTIONS = True

import eth80_frontend as FE
import perception.front_end_mechanisms as FEM
from corpus import split_tuning
from perception import attachment as _att

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def measure():
    split = split_tuning()
    dev = split['dev'][:CAP]
    records = []

    orig = FEM.apply_once

    def hooked(binder):
        records.append((binder._stage6_detections(),
                        binder._stage6_positions()))
        return orig(binder)

    FEM.apply_once = hooked
    try:
        for e in dev:
            img = np.asarray(Image.open(e['path']).convert('RGB'), dtype=np.uint8)
            tr = {}
            FE.build_graph(img, name='cal_%s%d' % (e['class'], e['object']),
                           record_co_located=False, trace=tr)
    finally:
        FEM.apply_once = orig

    gaps, angs = [], []
    chain_types = Counter()
    per_scene = []
    for (dets, positions), e in zip(records, dev):
        amap = {}
        peak_of = {}
        for d, nid in dets:
            if d.k == 'CONTOUR_PEAK':
                p = _att._xy(positions, d.alpha)
                if p is not None:
                    peak_of[d.alpha] = (d, p, nid)
        g_local, a_local, npeak = [], [], len(peak_of)
        for d, nid in dets:
            if d.k != 'PEAK_CHAIN':
                continue
            members = _att._members(d)
            q = d.sigma.quantized
            closed = bool(len(q) > 3 and float(q[3]) >= 0.5 and len(members) >= 3)
            pairs = list(zip(members[:-1], members[1:]))
            if closed:
                pairs.append((members[-1], members[0]))
            for a, b in pairs:
                if a not in peak_of or b not in peak_of:
                    continue
                da, pa, _na = peak_of[a]
                db, pb, _nb = peak_of[b]
                g_local.append(math.hypot(pb[0] - pa[0], pb[1] - pa[1]))
                ta = _att.raw_orientation(da, 12)
                tb = _att.raw_orientation(db, 12)
                if ta is not None and tb is not None:
                    a_local.append(_att.axial_difference(ta, tb))
        gaps += g_local
        angs += a_local
        per_scene.append({'scene': '%s%d' % (e['class'], e['object']), 'peaks': npeak,
                          'consec_gaps': len(g_local),
                          'consec_angles': len(a_local)})
    return gaps, angs, per_scene, chain_types


if __name__ == '__main__':
    gaps, angs, per_scene, chain_types = measure()
    g = np.array(gaps)
    a = np.array(angs)
    out = {'calib_cap': CAP,
           'n_scenes': len(per_scene),
           'peaks_total': sum(p['peaks'] for p in per_scene),
           'consec_gaps': int(len(g)),
           'consec_angles': int(len(a)),
           'chain_kind_counts': dict(chain_types),
           'per_scene': per_scene,
           'gap_quantiles': {},
           'angle_quantiles_rad': {},
           'angle_quantiles_deg': {}}
    if len(g):
        for q in (0.5, 0.75, 0.90, 0.95, 0.99):
            out['gap_quantiles'][str(q)] = float(np.quantile(g, q))
        out['gap_mean'] = float(g.mean())
        out['gap_median'] = float(np.median(g))
        out['gap_max'] = float(g.max())
    if len(a):
        for q in (0.5, 0.75, 0.90, 0.95, 0.99):
            v = float(np.quantile(a, q))
            out['angle_quantiles_rad'][str(q)] = v
            out['angle_quantiles_deg'][str(q)] = float(np.degrees(v))
        out['angle_mean'] = float(a.mean())
        out['angle_max_deg'] = float(np.degrees(a.max()))
    with open('results/l3_l4_calib_tracer.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ('n_scenes', 'peaks_total', 'consec_gaps', 'consec_angles')}, indent=2))
    print('gap_quantiles', out.get('gap_quantiles'))
    print('angle_quantiles_deg', out.get('angle_quantiles_deg'))
    print('WROTE results/l3_l4_calib_tracer.json')
