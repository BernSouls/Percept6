"""Live front-end profile, separate from historical parameter defaults.

The reentrant lock serializes participating callers while process-global
parameters are temporarily changed. Context-local state also lets algorithms
select the live contract without changing historical experiment controls.
"""
from __future__ import annotations
from contextlib import ContextDecorator
from contextvars import ContextVar
from threading import RLock
import parameters as P

LIVE_FLAGS = {
    "MATCH_EDGE_MASS_CONTAINMENT": True,
    "MATCH_SKIP_INCOMPARABLE_RELATIONS": True,
    "MATCH_KEEP_HIERARCHY_MEMBERS": True,
    "MATCH_TEMPLATE_MEDIAN_EXEMPLAR": True,
    "MATCH_CONSENSUS_COUNT_ABSTENTIONS": True,
    "MATCH_GLOBAL_CONSISTENCY_ENABLED": True,
}
EDGE_BUDGET_PER_FRAME = 512
FAMILY_SEED_EDGES = 2
_lock = RLock()
_active = ContextVar("frontend_contract_active", default=False)


class frontend_contract(ContextDecorator):
    def __init__(self):
        self._saved = []

    def __enter__(self):
        _lock.acquire()
        try:
            saved = {k: getattr(P, k) for k in LIVE_FLAGS}
            token = _active.set(True)
            self._saved.append((saved, token))
            for k, v in LIVE_FLAGS.items():
                setattr(P, k, v)
        except BaseException:
            _lock.release()
            raise
        return self

    def __exit__(self, *exc):
        try:
            saved, token = self._saved.pop()
            for k, v in saved.items():
                setattr(P, k, v)
            _active.reset(token)
        finally:
            _lock.release()
        return False


def frontend_contract_active():
    return _active.get()


def assert_live_contract():
    wrong = {k: getattr(P, k) for k, v in LIVE_FLAGS.items() if getattr(P, k) is not v}
    if not frontend_contract_active() or wrong:
        raise AssertionError(f"front-end contract is not live: {wrong}")
