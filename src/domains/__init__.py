"""Cross-domain benchmark domains (Pillar B).

Importing this package registers all built-in domains into `base.REGISTRY`.
A new domain plugs in by adding a module that constructs and `register`s a
`Domain` (see docs/WALKTHROUGH.md).
"""
from __future__ import annotations

from .base import REGISTRY, Domain, register  # noqa: F401
from . import healthcare, finance, government  # noqa: F401  (self-register on import)

__all__ = ["REGISTRY", "Domain", "register"]
