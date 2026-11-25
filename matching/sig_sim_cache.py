"""Content-keyed signature similarity cache (Mission 12, step 3).

WHY.  `node_sim` is memoised per DIRECTIONAL CALL on the node pair
`(u, v)`, which is the right key inside one pair and the wrong key across a
retrieval run: the same *signature content* recurs constantly -- two
IRREGULAR scenes share most of their quantized bin tuples, and inside one
image every node of a class that rests on the same bins collides.  The
similarity is a pure function of the two signatures (plus the two tolerance
flags, see below), so a cache keyed on the CONTENT of the pair can answer a
question that a cache keyed on node ids must recompute.

WHAT THE KEY COVERS -- the WHOLE signature, not the bins:

* the lens the two nodes were built under (`lens`), because the same tuple of
  floats reached through two lenses is two different readings;
* kind (CANONICAL / OPAQUE);
* for CANONICAL: `class_`, `relation`, the whole `quantized` tuple and the
  whole graded `population` (the population is deliberately EXCLUDED from
  `TypedSignature.__hash__` -- caching on the signature's own hash would
  merge signatures whose similarity differs, which is exactly the bug this
  module must not have);
* for OPAQUE: `type_tag` and `hash_value`;
* floats enter as `float.hex()`, so the key is exact and so a NaN-ish ABSENT
  marker (which compares unequal to itself) still hits its own entry;
* the two tolerance flags (`MATCH_NEIGHBOR_BIN_CREDIT`, `MATCH_POPULATION_EMD`)
  are in the namespace: they change what the comparison returns.

The cached value is `memory.cores._sig_similarity(a, b)` itself -- the same
call the uncached path makes -- so the hit and the miss cannot drift.

The flag `MATCH_SIG_SIM_CACHE` is DARK by default; `joint_mcs` asks for it
and falls through to the shipped call when it is off.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Optional, Tuple

#: (calls, hits, misses) plus the entry count, read by the harness.
_STATS = {"calls": 0, "hits": 0, "misses": 0}
_CACHE: Dict[Hashable, float] = {}
_NAMESPACE: tuple = ()


def _finite(v: Any) -> bool:
    """True when nothing in v is a NaN (NaN would make a content key unequal to itself)."""
    try:
        for x in v:
            if isinstance(x, tuple):
                if not _finite(x):
                    return False
            elif isinstance(x, float) and x != x:
                return False
    except TypeError:
        return False
    return True


def content_key(sig: Any) -> Optional[Hashable]:
    """The full content of one signature, or None when it carries none.

    Raw tuples, not hex strings: the key IS the shipped comparison's input, the
    dict resolves a hash collision by `==`, and hex-formatting 700 floats per
    call was measurably slower than the similarity it was caching.  Only a
    signature carrying a NaN falls to the hex form (a NaN key would never match
    itself and would leak one entry per call).
    """
    if sig is None:
        return None
    kind = getattr(sig, "kind", "")
    if kind == "CANONICAL":
        q = getattr(sig, "quantized", ()) or ()
        pop = getattr(sig, "population", ()) or ()
        if not _finite(q) or not _finite(pop):
            return (0, str(getattr(sig, "class_", "")),
                    str(getattr(sig, "relation", "")),
                    tuple(x.hex() if isinstance(x, float) else repr(x) for x in q),
                    tuple(tuple(x.hex() if isinstance(x, float) else repr(x)
                                for x in p) for p in pop))
        return (0, str(getattr(sig, "class_", "")),
                str(getattr(sig, "relation", "")), q, pop)
    if kind == "OPAQUE":
        return (1, str(getattr(sig, "type_tag", "")),
                int(getattr(sig, "hash_value", 0)))
    return (2, repr(kind))


#: id(sig) -> (sig, hash(content_key(sig))).  The signature object is kept
#: alive by the memo, so its id cannot be recycled under us, and `is` re-checks
#: it anyway.  Hashing the full content once per signature is what turns the
#: per-call cost into a three-int tuple.
_HASH_MEMO: Dict[int, Tuple[Any, int]] = {}


def _sig_hash(sig: Any) -> int:
    ent = _HASH_MEMO.get(id(sig))
    if ent is not None and ent[0] is sig:
        return ent[1]
    h = hash(content_key(sig))
    _HASH_MEMO[id(sig)] = (sig, h)
    return h


def reset() -> None:
    """Drop every entry and the counters (one cache per measured run)."""
    _CACHE.clear()
    _HASH_MEMO.clear()
    _STATS.update({"calls": 0, "hits": 0, "misses": 0})


def stats() -> Dict[str, float]:
    ent = len(_CACHE)
    calls = _STATS["calls"]
    return {"calls": calls, "hits": _STATS["hits"], "misses": _STATS["misses"],
            "entries": ent,
            "hit_rate": (_STATS["hits"] / calls) if calls else 0.0}


def _namespace() -> tuple:
    """The two tolerance flags, read once per process (they are constants)."""
    global _NAMESPACE
    if not _NAMESPACE:
        import parameters as P
        # [M16] the keypoint class CONSTITUTION changes what a comparison returns
        # without changing either signature, so it must be in the namespace: the
        # content key alone would serve an unregistered reading to a registered run.
        _NAMESPACE = (bool(getattr(P, "MATCH_NEIGHBOR_BIN_CREDIT", False)),
                      bool(getattr(P, "MATCH_POPULATION_EMD", False)),
                      bool(getattr(P, "POPULATION_CODING_ENABLED", False)),
                      bool(getattr(P, "KP_GRADED_SIMILARITY", False)),
                      str(getattr(P, "KP_GRADED_VARIANT", "")),
                      bool(getattr(P, "KP_GRADED_POPULATION", False)),
                      str(getattr(P, "KP_GRADED_POP_VARIANT", "")))
    return _NAMESPACE


def sim(a: Any, b: Any, lens: Any = None) -> float:
    """`_sig_similarity(a, b)`, memoised on (lens, content(a), content(b)).

    The key stores the two content HASHES, so a lookup compares three small
    ints; the value stored is the shipped call's own float.  Two distinct
    contents colliding in 64 bits would return the wrong float -- with ~1.4e7
    distinct keys over a retrieval run that is a p ~ 1e-5 event, and the
    identity harness over the fixed pairs is the proof that it did not happen
    there.
    """
    from memory.cores import _sig_similarity
    _STATS["calls"] += 1
    if a is None or b is None:
        _STATS["misses"] += 1
        return 0.0
    key = (lens, _namespace(), _sig_hash(a), _sig_hash(b))
    v = _CACHE.get(key)
    if v is not None:
        _STATS["hits"] += 1
        return v
    _STATS["misses"] += 1
    v = float(_sig_similarity(a, b))
    if math.isfinite(v):
        _CACHE[key] = v
    return v
