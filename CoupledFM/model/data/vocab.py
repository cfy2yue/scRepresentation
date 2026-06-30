"""CoupledFM re-export of the shared core `GeneVocab`."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.utils.data.vocab import GeneVocab

__all__ = ["GeneVocab"]
