"""DEPRECATED: kept as a thin shim for backward compatibility.

Real implementation has moved to ``utils.data.ot_pairer.LatentOTPairer``.
Importing from here will raise a DeprecationWarning. Update callers to
``from utils.data.ot_pairer import LatentOTPairer``.
"""

import warnings

from model.utils.data.ot_pairer import LatentOTPairer  # noqa: F401

warnings.warn(
    "model.fm_ot.LatentOTPairer is deprecated; "
    "use model.utils.data.ot_pairer.LatentOTPairer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LatentOTPairer"]
