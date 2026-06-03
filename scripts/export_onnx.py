"""Export the deployable models to ONNX for the static web app (web/models/).

Two models the browser pipeline needs:
  - corners.onnx : the corner lattice localizer (CornerHeatmapNet). forward() already does the
    soft-argmax, so the ONNX emits normalized coords (1, num_corners, 2) directly -- no heatmap
    post-processing in JS. ImageNet normalization is baked into the graph (model.normalize), so
    the JS contract is just float CHW in [0,1] at image_size x image_size.
  - pieces.onnx  : the YOLO-pose piece detector (box + contact keypoint). Raw Ultralytics output
    (NMS done in JS) for broad onnxruntime-web compatibility.

Writes into web/models/ so the files travel with the static app (see deployment discussion).
ONNX is validated by loading it back with onnxruntime and printing the real input/output shapes
-- the JS decode is written against exactly these.

    uv run --group yolo python scripts/export_onnx.py \
        --pieces-ckpt runs/yolo11s_pose/weights/best.pt \
        --corner-ckpt runs/corners/best.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from chessvision.corner_regressor import load_corner_regressor


class _Float32(torch.nn.Module):
    """Wrap the corner net so the ONNX output is float32, not float64. The soft-argmax
    expectation traces to double; ort-web (esp. WebGPU) wants float32, so cast on the way out."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float()


def _strip_float64(path: Path) -> int:
    """Convert ALL float64 in the graph to float32 in-place. The soft-argmax expectation traces to
    float64 (Cast->DOUBLE + a double `linspace` constant), but onnxruntime-web's wasm build ships
    NO float64 kernels ("Could not find an implementation for Cast"). Coords live in [0,1], so float
    is numerically identical. Touches initializers, Constant tensors, Cast `to`, and value_info so
    no float/double type clash survives. Returns the number of edits."""
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    m = onnx.load(str(path))
    g, n = m.graph, 0
    for init in g.initializer:
        if init.data_type == TensorProto.DOUBLE:
            init.CopyFrom(
                numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name)
            )
            n += 1
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.DOUBLE:
                    a.i = TensorProto.FLOAT
                    n += 1
        elif node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.DOUBLE:
                    a.t.CopyFrom(
                        numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32))
                    )
                    n += 1
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(m, str(path))
    return n


def export_corners(ckpt: Path, out: Path, opset: int) -> None:
    model = _Float32(load_corner_regressor(ckpt, device="cpu"))
    size = int(getattr(model.model, "image_size", 512))
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["coords"],  # (1, num_corners, 2) normalized [0,1]
        opset_version=opset,
        dynamic_axes=None,  # fixed 1x3xSxS -- the web app always feeds this size
        dynamo=False,  # legacy TorchScript exporter (no onnxscript dep; stable for this graph)
    )
    n = _strip_float64(out)
    print(f"corners -> {out}  (1x3x{size}x{size}, normalize baked in, {n} float64->float32)")


def export_pieces(ckpt: Path, out: Path, imgsz: int, opset: int, quant: str, data: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    # Raw output (no embedded NMS) for broad ort-web op support; JS does NMS.
    # quant: "fp32" (~39MB) | "fp16" (~20MB, no calibration) | "int8" (~10MB, calibrated on `data`).
    kw = dict(format="onnx", imgsz=imgsz, opset=opset, simplify=True, nms=False)
    if quant == "fp16":
        kw["half"] = True
    elif quant == "int8":
        kw["int8"] = True
        kw["data"] = str(data)  # calibration set for int8 quantization
    path = model.export(**kw)
    shutil.copy2(path, out)
    size_mb = out.stat().st_size / 1e6
    print(f"pieces  -> {out}  (from {path}, imgsz={imgsz}, {quant}, {size_mb:.1f}MB)")


def report(out: Path) -> None:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print(f"  [skip shape report for {out.name}] onnxruntime not installed")
        return
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  {out.name} IN  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  {out.name} OUT {o.name}: {o.shape} {o.type}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pieces-ckpt", type=Path, default=Path("runs/yolo11s_pose/weights/best.pt"))
    p.add_argument("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("web/models"))
    p.add_argument(
        "--imgsz", type=int, default=1280, help="YOLO-pose input size (must match training)"
    )
    p.add_argument("--opset", type=int, default=12)
    p.add_argument(
        "--quant",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        # int8 (calibrated on the small pose set) produced a DEAD model -- max class score 0.0
        # (scripts/diag_pose_onnx.py). fp16 matches fp32 detections exactly at ~20MB, so it's the
        # web default; revisit int8 only with >300 calibration imgs + an accuracy check.
        help="pieces.onnx precision: fp32 ~39MB | fp16 ~20MB (web default) | int8 ~10MB (fragile)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/yolo_pose/data.yaml"),
        help="calibration dataset for --quant int8",
    )
    p.add_argument("--skip-pieces", action="store_true")
    p.add_argument("--skip-corners", action="store_true")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_corners:
        export_corners(args.corner_ckpt, args.out_dir / "corners.onnx", args.opset)
        report(args.out_dir / "corners.onnx")
    if not args.skip_pieces:
        export_pieces(
            args.pieces_ckpt,
            args.out_dir / "pieces.onnx",
            args.imgsz,
            args.opset,
            args.quant,
            args.data,
        )
        report(args.out_dir / "pieces.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
