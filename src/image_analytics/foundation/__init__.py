"""Foundation-model wrappers (promptable, self-supervised, satellite).
Importing populates the registries (BACKBONES for satellite encoders, MODELS
for SAM). Heavy HF dependencies are lazy-imported inside each wrapper.
"""

from image_analytics.foundation import (  # noqa: F401  (registration side effects)
    prithvi,
    sam,
    satmae,
)
