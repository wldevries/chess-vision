"""Launch the capture/labelling web app.

    uv run python -m chessvision.capture

Serves the labelling tools over the unified store: corner-label and position-label
(inbox -> corners -> pieces), the session-metadata editor, and Read-position (live FEN).
Capture mode (play a game/puzzle and snap photos) and the Label Studio pipeline are retired.

    # corner/position labelling over data/ (the flat store), with corner-assist + Read mode
    uv run python -m chessvision.capture \
        --keypoint-ckpt runs/keypoint_captures/best.pt --corner-ckpt runs/corners/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from chessvision.capture.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m chessvision.capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 for LAN access)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--keypoint-ckpt",
        type=Path,
        default=None,
        help="piece keypoint checkpoint for Read-position (live FEN) mode "
        "(e.g. runs/keypoint_captures/best.pt); needs --corner-ckpt too, since a Read "
        "auto-detects the board corners before the pieces",
    )
    p.add_argument(
        "--corner-ckpt",
        type=Path,
        default=None,
        help="corner-regressor checkpoint (e.g. runs/corners/best.pt): drives corner-assist "
        "(the 'Predict' button pre-fills the corner-marking handles) and the automatic "
        "corner detection in Read-position mode",
    )
    p.add_argument(
        "--corners-root",
        type=Path,
        default=Path("data"),
        help="unified store root (flat layout): phone photos staged in <root>/source/inbox/ are "
        "labelled in the app and written as normalized images to <root>/store/ with the index at "
        "<root>/labels.jsonl. On by default; pass --no-corners-root to disable.",
    )
    p.add_argument(
        "--no-corners-root",
        dest="corners_root",
        action="store_const",
        const=None,
        help="disable corner-label mode (hide the Corners tab)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device for live inference (default: cuda if available, else cpu)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    print(f"Open http://{args.host}:{args.port}  ·  labelling over {args.corners_root}")
    if args.keypoint_ckpt and args.corner_ckpt:
        print(f"Read-position mode ON · pieces {args.keypoint_ckpt} · corners {args.corner_ckpt}")
    elif args.keypoint_ckpt:
        print("Read-position mode OFF: --keypoint-ckpt also needs --corner-ckpt")
    if args.corner_ckpt:
        print(f"Corner-assist ON · checkpoint {args.corner_ckpt}")

    app = create_app(
        # out_root only backstops the Set/Board metadata when no store is wired; with the
        # flat store the dropdowns read from --corners-root, so this is rarely used.
        args.corners_root or Path("data"),
        keypoint_ckpt=args.keypoint_ckpt,
        corner_ckpt=args.corner_ckpt,
        corners_root=args.corners_root,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
