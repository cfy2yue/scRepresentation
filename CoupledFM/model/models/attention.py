"""
Thin shim — canonical implementation lives in :mod:`utils.models.attention`.

Kept to preserve the model-level attention import surface used by
``velocity_field.py`` and external callers. Do not add logic here; edit
``utils/models/attention.py`` instead.

Backends exposed via :class:`MultiHeadAttention`: ``sdpa`` | ``flash`` |
``linear`` | ``sparse`` (CellNavi-style scatter). See the docstring over in
``utils/models/attention.py`` for semantics.
"""

from model.utils.models.attention import (  # noqa: F401
    MultiHeadAttention,
    FeedForward,
)

__all__ = ["MultiHeadAttention", "FeedForward"]
