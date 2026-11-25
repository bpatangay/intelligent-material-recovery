import argparse
from pathlib import Path
from ultralytics import YOLO

# Required so PyTorch resolves the pickled CBAM module definition when loading checkpoints
import models.cbam  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Export trained CBAM-YOLOv8 checkpoint to ONNX.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true", help="Export in FP16 (requires CUDA)")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        simplify=True,
    )
    print(f"ONNX model exported to: {exported_path}")


if __name__ == "__main__":
    main()