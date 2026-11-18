"""Lossless named attribute features on a physical FRAME signature.

A feature retains the WHOLE quantized tuple and population of its source.  It
is not a bin echo, not a new anonymous axis, and not a change to frame identity.
The source class's existing unit/circular/identity registrations govern its
comparison.  Attachment names are canonical typed strings, never repr/id.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import struct
from core.types import TypedSignature


def serialize_feature_payload(features) -> bytes:
    """Typed, lossless binary64 encoding used only for feature-aware IDs."""
    def pack(value):
        if value is None:
            return b'n'
        if isinstance(value, bool):
            return b'b' + bytes((value,))
        if isinstance(value, int):
            data = str(value).encode('ascii')
            return b'i' + struct.pack('<Q', len(data)) + data
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError('non-finite feature float')
            return b'f' + struct.pack('<d', value)
        if isinstance(value, (str, bytes)):
            data = value.encode('utf-8') if isinstance(value, str) else value
            return (b's' if isinstance(value, str) else b'y') + struct.pack('<Q', len(data)) + data
        if isinstance(value, (tuple, list)):
            return b't' + struct.pack('<Q', len(value)) + b''.join(pack(v) for v in value)
        raise TypeError(f'unsupported feature payload type: {type(value).__name__}')
    return pack(tuple(feature.key() for feature in features))


@dataclass(frozen=True)
class NamedFeature:
    name: str
    source_lens: str
    signature: TypedSignature
    attachment: Tuple[Tuple[str, object], ...] = ()
    def key(self):
        return (self.name, self.source_lens, self.signature.kind,
                self.signature.class_, self.signature.relation,
                tuple(self.signature.quantized),
                tuple(tuple(v) for v in self.signature.population), self.attachment)

def with_features(sig: TypedSignature, features) -> TypedSignature:
    from dataclasses import replace
    old = getattr(sig, "_named_features", ())
    # TypedSignature equality intentionally ignores population, so using a
    # set of NamedFeatures would discard distinct graded measurements.
    by_key = {f.key(): f for f in (*old, *tuple(features))}
    allf = [by_key[k] for k in sorted(by_key)]
    out = replace(sig)
    object.__setattr__(out, "_named_features", tuple(allf))
    return out

def feature_similarity(a: TypedSignature, b: TypedSignature, base: float) -> float:
    if not a.compatible(b):
        return float(base)
    from collections import defaultdict
    fa, fb = defaultdict(list), defaultdict(list)
    for f in getattr(a, "_named_features", ()):
        fa[(f.name, f.source_lens)].append(f)
    for f in getattr(b, "_named_features", ()):
        fb[(f.name, f.source_lens)].append(f)
    if not fa and not fb:
        return base
    vals = []
    for name in sorted(set(fa) | set(fb)):
        left, right = fa.get(name, []), fb.get(name, [])
        if not left or not right:
            vals.append(0.0)
        elif len(left) == len(right) == 1:
            vals.append(left[0].signature.similarity(right[0].signature))
        else:
            # Repeated measurements of one named feature retain their full
            # populations and receive an exact, symmetric assignment. An
            # unmatched feature contributes zero, rather than disappearing.
            from scipy.optimize import linear_sum_assignment
            matrix = [[x.signature.similarity(y.signature) for y in right]
                      for x in left]
            rows, cols = linear_sum_assignment(matrix, maximize=True)
            vals.append(sum(matrix[i][j] for i, j in zip(rows, cols))
                        / max(len(left), len(right)))
    return 0.5 * float(base) + 0.5 * sum(vals) / len(vals)
