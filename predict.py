"""Thin CLI shim so `python predict.py <image>` works.

Delegates to chessvision.cli; see that module for the actual entry point.
"""

from chessvision.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
