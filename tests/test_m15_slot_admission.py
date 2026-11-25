"""Mission 15 item 2: the slots a candidate is scored on MUST include its checkable edges.

The defect this pins: `slots_for_candidate` sorts a node's incident placement edges by
(class name, neighbour id), appends the slots whose residual cannot be formed, and only
THEN truncates to `SLOT_CAP = 8` -- while `TARGET_K = 16`.  A checkable edge that happens
to sort ninth is therefore discarded in favour of an uncheckable slot that carries the
padding value, i.e. the estimator is shown the weakest evidence the node has.

With `PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST` the checkable slots are admitted first.  This
test builds a node with eight uncheckable neighbours whose ids sort FIRST and one
checkable neighbour whose id sorts LAST, and requires:
  * flag off  -> the checkable edge is NOT among the admitted slots (the defect, pinned),
  * flag on   -> it IS, and the uncheckable slots fill the rest,
  * the flag changes nothing else: a node with <= SLOT_CAP slots keeps every slot.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import parameters as P
from matching import placement_llr as L


def _side(n_bad=8, with_good=True, cap=None):
    """(u=0) has n_bad uncheckable neighbours (ids 1..n) and one checkable (id 99).

    `index` maps (src, dst, class) -> PlacementVector.  Only the 0<->99 relation has
    an entry on BOTH sides, so only it can produce a residual.
    """
    from matching.placement import PlacementVector
    vec = PlacementVector(direction=0.3, distance=2.0, log_scale=0.0,
                          source_theta=0.0, source_scale=10.0, tolerance=1.0)
    inc = [(i, True, "CLS") for i in range(1, n_bad + 1)]
    if with_good:
        inc.append((99, True, "CLS"))
    q_idx = {(0, 99, "CLS"): vec}
    r_idx = {(0, 99, "CLS"): vec}
    mapping = {i: i for i in range(1, n_bad + 1)}          # uncheckable on the other side
    if with_good:
        mapping[99] = 99
    return inc, q_idx, r_idx, mapping


class SlotAdmissionTests(unittest.TestCase):
    def setUp(self):
        self._old = P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST

    def tearDown(self):
        P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = self._old

    def test_shipped_order_discards_the_checkable_slot(self):
        P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = False
        inc, q_idx, r_idx, mapping = _side()
        slots = L.slots_for_candidate(0, 0, mapping, {0: inc}, q_idx, r_idx)
        self.assertEqual(len(slots), L.SLOT_CAP)
        self.assertFalse(any(s[2] is not None for s in slots),
                         "shipped order was expected to admit only uncheckable slots")

    def test_checkable_first_admits_it(self):
        P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = True
        inc, q_idx, r_idx, mapping = _side()
        slots = L.slots_for_candidate(0, 0, mapping, {0: inc}, q_idx, r_idx)
        self.assertEqual(len(slots), L.SLOT_CAP)
        good = [s for s in slots if s[2] is not None]
        self.assertEqual(len(good), 1, "exactly one checkable slot exists")
        self.assertEqual(good[0][1], 99, "the checkable slot is the id-99 neighbour")
        self.assertEqual(L.judged_slots(slots), 1)

    def test_a_node_within_the_cap_keeps_every_slot(self):
        P.PLACEMENT_SLOT_ADMIT_CHECKABLE_FIRST = True
        inc, q_idx, r_idx, mapping = _side(n_bad=2)
        slots = L.slots_for_candidate(0, 0, mapping, {0: inc}, q_idx, r_idx)
        self.assertEqual(len(slots), 3)
        self.assertEqual({s[1] for s in slots}, {1, 2, 99})

    def test_judged_counting_is_dark(self):
        old = P.PLACEMENT_SLOT_COUNT_JUDGED
        try:
            P.PLACEMENT_SLOT_COUNT_JUDGED = False
            self.assertFalse(L._count_judged_slots())
            P.PLACEMENT_SLOT_COUNT_JUDGED = True
            self.assertTrue(L._count_judged_slots())
        finally:
            P.PLACEMENT_SLOT_COUNT_JUDGED = old


if __name__ == "__main__":
    unittest.main()
