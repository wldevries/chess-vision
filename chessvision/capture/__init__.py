"""Position-capture tool: step through real games and photograph each position.

A small FastAPI web app (run with ``python -m chessvision.capture``) plays a
game move by move; you set the physical board to match the displayed position,
snap a photo with the tablet camera, and the app stores the image alongside the
ground-truth FEN. This produces labelled real-world photos for evaluation and
domain-randomization work without any manual annotation.
"""

from __future__ import annotations
