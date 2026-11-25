import argparse
import gc
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from models.cbam import CBAM


def attach_cbam_to_neck(yolo_model) -> None:
    """
    Dynamically wraps all multi-scale neck feature maps feeding into the Detect head
    (P3, P4, P5 -> layers 15, 18, 21 in YOLOv8) with CBAM modules.
    """
    layers = yolo_model.model if isinstance(getattr(yolo_model, "model", None), nn.Sequential) else yolo_model.model.model
    detect_head = layers[-1]
    neck_exit_indices = list(detect_head.f)

    for idx in neck_exit_indices:
        block = layers[idx]
        if isinstance(block, nn.Sequential):
            continue
        out_channels = block.cv2.conv.out_channels
        cbam = CBAM(channels=out_channels)

        wrapped = nn.Sequential(block, cbam)
        wrapped.i = block.i
        wrapped.f = block.f
        wrapped.type = block.type
        wrapped.np = sum(p.numel() for p in wrapped.parameters())

        layers[idx] = wrapped

    print(f"Attached CBAM modules to neck layers {neck_exit_indices}.")


class CBAMDetectionTrainer(DetectionTrainer):
    """Custom trainer that keeps neck CBAM modules active across init and resume."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.args.resume:
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            attach_cbam_to_neck(model)
            if weights:
                model.load(weights)
        else:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            attach_cbam_to_neck(model)
        return model


def register_mps_cleanup(model: YOLO, device: str | int) -> None:
    """Frees unified GPU memory between epochs on Apple Silicon MPS."""
    if device == "mps":
        def clear_mps_cache(_trainer):
            gc.collect()
            torch.mps.empty_cache()

        model.add_callback("on_train_epoch_end", clear_mps_cache)
        model.add_callback("on_val_end", clear_mps_cache)


def main():
    parser = argparse.ArgumentParser(description="Fast YOLOv8 + CBAM Training for E-Waste Sorting.")
    parser.add_argument("--data", type=str, default="datasets/ewaste_roboflow/data.yaml")
    parser.add_argument("--cfg", type=str, default="yolov8n.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="yolov8n_cbam_ewaste_fast")
    parser.add_argument("--no_cbam", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    trainer_cls = None if args.no_cbam else CBAMDetectionTrainer

    if args.resume:
        model = YOLO(args.resume)
        register_mps_cleanup(model, device)
        model.train(resume=True, trainer=trainer_cls)
        return

    model = YOLO(args.cfg).load(args.weights) if args.cfg else YOLO(args.weights)
    register_mps_cleanup(model, device)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        cache=True,           # Cache preprocessed images in RAM (eliminates SSD I/O bottleneck)
        deterministic=False,  # Avoids slow MPS fallback kernels in CBAM & loss backward passes
        degrees=0.0,          # Eliminates slow single-threaded CPU affine warps on macOS
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        erasing=0.0,
        close_mosaic=10,
        project="runs/ewaste",
        name=args.name,
        device=device,
        trainer=trainer_cls,
    )


if __name__ == "__main__":
    main()