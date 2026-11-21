"""Task 4 PART 0b — Validate the REPAIRED counterpart_map."""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "Topology"), ROOT]
from t4_shift import counterpart_map  # noqa: E402


def _rate(mapping, base_pos):
    return len(mapping) / len(base_pos) if base_pos else 0.0


def _amb_count(info):
    return sum(1 for v in info.values() if v["ambiguous"])


def main() -> int:
    print("=== PART 0b: VALIDATION OF REPAIRED counterpart_map ===", flush=True)
    results = {}
    radius = 8.0
    # T1 rigid shift
    base = {i: (float(i * 10), 0.0, "L") for i in range(10)}
    shift = {("s%d" % i): (float(i * 10) + 2.0, 0.0, "L") for i in range(10)}
    cp, info, stats = counterpart_map(base, shift, 2, 0, radius)
    rate = _rate(cp, base); amb = _amb_count(info)
    print("T1 rigid: rate=%.4f amb=%d conflicts=%d dropped=%d" %
          (rate, amb, stats["conflicts"], stats["dropped"]))
    results["T1"] = {"rate": rate, "ambiguous": amb}
    assert rate == 1.0 and amb == 0
    # T2 duplicate
    base2 = {i: (float(i * 10), 0.0, "L") for i in range(5)}
    shift2 = {("s0"): (2.0, 0.0, "L"), ("s0dup"): (2.0, 0.0, "L")}
    for i in range(1, 5):
        shift2["s%d" % i] = (float(i * 10) + 2.0, 0.0, "L")
    cp2, info2, _ = counterpart_map(base2, shift2, 2, 0, radius)
    amb0 = info2.get(0, {}).get("ambiguous")
    print("T2 duplicate: base0 ambiguous=%s" % amb0)
    results["T2"] = {"base0_ambiguous": amb0}
    assert amb0 is True
    # T3 lone 0.6xradius
    base3 = {0: (0.0, 0.0, "L")}
    shift3 = {("s0"): (2.0 + 0.6 * radius, 0.0, "L")}
    cp3, info3, _ = counterpart_map(base3, shift3, 2, 0, radius)
    amb3 = info3.get(0, {}).get("ambiguous")
    print("T3 lone: has=%s amb=%s" % (0 in cp3, amb3))
    results["T3"] = {"has_counterpart": 0 in cp3, "ambiguous": amb3}
    assert 0 in cp3 and amb3 is False
    # T4 competing claims
    base4 = {0: (0.0, 0.0, "L"), 1: (1.0, 0.0, "L")}
    shift4 = {("s0"): (2.0, 0.0, "L")}
    cp4, info4, st4 = counterpart_map(base4, shift4, 2, 0, radius)
    print("T4 competing: resolved=%s conflicts=%d" % (list(cp4.keys()), st4["conflicts"]))
    results["T4"] = {"resolved": list(cp4.keys())}
    assert len(cp4) == 1 and 0 in cp4
    print("\n=== ALL VALIDATION TESTS PASS ===", flush=True)
    import json
    from guard import RESULTS_DIR, write_json
    write_json(RESULTS_DIR / "t4" / "part0b_validate.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
