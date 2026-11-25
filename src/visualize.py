import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

import models.cbam  # noqa: F401
from src.preprocess import PhotometricPreprocessor


def save_preprocessing_comparison(image_path: Path, out_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    bgr_clahe = cv2.cvtColor(cv2.merge((l_clahe, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    processor = PhotometricPreprocessor()
    bgr_final = processor(bgr)

    # Amplify pixel difference to visualize suppressed specular glare
    glare_diff = cv2.cvtColor(cv2.absdiff(bgr_clahe, bgr_final), cv2.COLOR_BGR2GRAY)
    glare_map = cv2.applyColorMap(np.clip(glare_diff * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    panels = [
        (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "(a) Raw Conveyor Frame"),
        (cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB), "(b) L* CLAHE Normalized"),
        (cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB), "(c) Bilateral Filtered"),
        (cv2.cvtColor(glare_map, cv2.COLOR_BGR2RGB), "(d) Suppressed Glare Map"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved preprocessing figure -> {out_path}")


def save_attention_heatmap(image_path: Path, weights_path: Path, out_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    resized = cv2.resize(bgr, (640, 640))
    tensor = (
        torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    model = YOLO(str(weights_path))
    net = model.model.eval()
    device = next(net.parameters()).device
    tensor = tensor.to(device)

    captured_maps = []
    p3_layer = net.model[15]
    has_cbam = isinstance(p3_layer, torch.nn.Sequential) and hasattr(p3_layer[1], "spatial_att")

    def hook_fn(_module, _inp, output):
        captured_maps.append(output.detach().cpu())

    target_module = p3_layer[1].spatial_att.gate if has_cbam else p3_layer
    handle = target_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = net(tensor)
    handle.remove()

    feat = captured_maps[0]
    if has_cbam:
        att = feat.squeeze().numpy()
    else:
        # Compute spatial saliency (channel-wise avg + max pooling) from P3 features
        avg_map = torch.mean(feat, dim=1, keepdim=True)
        max_map, _ = torch.max(feat, dim=1, keepdim=True)
        att = torch.sigmoid(avg_map + max_map).squeeze().numpy()

    att_resized = cv2.resize(att, (640, 640))
    att_uint8 = np.uint8(
        255 * (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min() + 1e-8)
    )
    heatmap = cv2.applyColorMap(att_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(resized, 0.55, heatmap, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Preprocessed Input", fontsize=11, fontweight="bold")
    axes[1].imshow(att_uint8, cmap="jet")
    axes[1].set_title("(b) P3 Spatial Attention Map", fontsize=11, fontweight="bold")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("(c) Attention Overlay", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention overlay -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for preprocessing and CBAM.")
    parser.add_argument("--image", type=Path, required=True, help="Sample conveyor image path")
    parser.add_argument("--weights", type=Path, default=None, help="Trained CBAM weights (best.pt)")
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessing_comparison(args.image, args.out_dir / "fig_preprocessing.png")

    if args.weights and args.weights.exists():
        save_attention_heatmap(args.image, args.weights, args.out_dir / "fig_attention.png")


if __name__ == "__main__":
    main()