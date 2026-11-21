"""Alias of the repository-root ``eth80_frontend`` module (export shim).

Reconstructed for the front-end fix session: the original export kept the
front end under ``diagnostics/`` and ``eth80_frontend.frontend_source_digest``
still hashes that path.  All logic lives in the root module.
"""
from eth80_frontend import *  # noqa: F401,F403
from eth80_frontend import (  # noqa: F401
    ImageGraph,
    build_graph,
    graph_for,
    frontend_source_digest,
    switch_suffix,
)
