import argparse
import json
import random
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.preprocess import PhotometricPreprocessor


def poly_to_mask(polygons: list[list[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 6:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def mask_to_yolo_bbox(mask: np.ndarray, min_px: int) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x2 - x1, y2 - y1
    if bw < min_px or bh < min_px:
        return None
    h, w = mask.shape
    return [
        (x1 + bw / 2.0) / w,
        (y1 + bh / 2.0) / h,
        bw / w,
        bh / h,
    ]


def convert_split(
    coco_json: Path,
    img_dir: Path,
    out_root: Path,
    split: str,
    preprocessor: PhotometricPreprocessor | None = None,
    min_box_px: int = 28,
) -> dict[int, str]:
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    sorted_cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    class_names = {idx: cat["name"] for idx, cat in enumerate(sorted_cats)}

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img = {img["id"]: [] for img in coco["images"]}

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 0 and ann["bbox"][2] >= min_box_px and ann["bbox"][3] >= min_box_px:
            anns_by_img[ann["image_id"]].append(ann)

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    minority_cutouts: list[tuple[int, np.ndarray, np.ndarray]] = []
    train_records = []

    for img_id, meta in tqdm(images_by_id.items(), desc=f"Preparing {split}"):
        src_img_path = img_dir / meta["file_name"]
        if not src_img_path.exists():
            continue

        frame = cv2.imread(str(src_img_path))
        if frame is None:
            continue

        if preprocessor is not None:
            frame = preprocessor(frame)

        h_img, w_img = frame.shape[:2]
        stem = Path(meta["file_name"]).stem
        label_lines = []
        instance_masks = []
        has_metal = False

        for ann in anns_by_img[img_id]:
            cls_idx = cat_to_idx.get(ann["category_id"])
            if cls_idx is None:
                continue

            segs = ann.get("segmentation", [])
            if isinstance(segs, list) and segs:
                inst_mask = poly_to_mask(segs, h_img, w_img)
            else:
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                inst_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                inst_mask[max(0, y) : min(h_img, y + bh), max(0, x) : min(w_img, x + bw)] = 255

            box = mask_to_yolo_bbox(inst_mask, min_box_px)
            if box is None:
                continue

            cx, cy, w, h = box
            label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            instance_masks.append((cls_idx, inst_mask))

            cls_name = class_names[cls_idx]
            if cls_name == "metal":
                has_metal = True

            if split == "train" and cls_name in ("metal", "rigid_plastic"):
                ys, xs = np.where(inst_mask > 0)
                x1, x2 = xs.min(), xs.max() + 1
                y1, y2 = ys.min(), ys.max() + 1
                crop_img = frame[y1:y2, x1:x2].copy()
                crop_mask = cv2.GaussianBlur(inst_mask[y1:y2, x1:x2], (3, 3), 0)
                minority_cutouts.append((cls_idx, crop_img, crop_mask))

        # Skip empty frames with zero sortable objects
        if not label_lines:
            continue

        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), frame)
        with open(out_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as lf:
            lf.write("\n".join(label_lines))

        # Oversample real metal-containing frames 3x in training
        if split == "train" and has_metal:
            for rep in range(2):
                cv2.imwrite(str(out_img_dir / f"{stem}_mrep{rep}.jpg"), frame)
                with open(out_lbl_dir / f"{stem}_mrep{rep}.txt", "w", encoding="utf-8") as lf:
                    lf.write("\n".join(label_lines))

        if split == "train":
            train_records.append((stem, frame, instance_masks))

    # Generate 2,500 seamless polygon-blended balanced training composites
    if split == "train" and minority_cutouts and train_records:
        rng = random.Random(42)
        num_synthetic = 2500
        metal_ids = {k for k, v in class_names.items() if v == "metal"}
        metal_pool = [c for c in minority_cutouts if c[0] in metal_ids]
        plastic_pool = [c for c in minority_cutouts if c[0] not in metal_ids]
        print(f"Generating {num_synthetic} polygon-blended images (metal={len(metal_pool)}, rigid_plastic={len(plastic_pool)})...")

        for idx in tqdm(range(num_synthetic), desc="Polygon copy-paste"):
            stem, base_img, existing_masks = rng.choice(train_records)
            comp = base_img.copy()
            h_img, w_img = comp.shape[:2]
            active_masks = [(c, m.copy()) for c, m in existing_masks]

            for _ in range(rng.randint(3, 6)):
                pool = metal_pool if (metal_pool and rng.random() < 0.75) else plastic_pool
                if not pool:
                    continue
                cls_idx, patch, alpha = rng.choice(pool)
                scale = rng.uniform(0.8, 1.4)
                nh, nw = max(min_box_px, int(patch.shape[0] * scale)), max(min_box_px, int(patch.shape[1] * scale))
                if nh >= h_img // 2 or nw >= w_img // 2:
                    continue

                patch_s = cv2.resize(patch, (nw, nh))
                alpha_s = cv2.resize(alpha, (nw, nh))
                if rng.random() < 0.5:
                    patch_s, alpha_s = cv2.flip(patch_s, 1), cv2.flip(alpha_s, 1)
                if rng.random() < 0.5:
                    patch_s, alpha_s = cv2.flip(patch_s, 0), cv2.flip(alpha_s, 0)

                px1 = rng.randint(0, w_img - nw)
                py1 = rng.randint(0, h_img - nh)
                a_norm = (alpha_s.astype(np.float32) / 255.0)[..., None]
                roi = comp[py1 : py1 + nh, px1 : px1 + nw].astype(np.float32)
                comp[py1 : py1 + nh, px1 : px1 + nw] = np.clip(
                    patch_s.astype(np.float32) * a_norm + roi * (1.0 - a_norm), 0, 255
                ).astype(np.uint8)

                full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                full_mask[py1 : py1 + nh, px1 : px1 + nw] = (alpha_s > 127).astype(np.uint8) * 255
                inv_mask = cv2.bitwise_not(full_mask)
                active_masks = [(c, cv2.bitwise_and(m, inv_mask)) for c, m in active_masks]
                active_masks.append((cls_idx, full_mask))

            syn_lines = []
            for c_idx, m in active_masks:
                box = mask_to_yolo_bbox(m, min_box_px)
                if box is not None:
                    cx, cy, w, h = box
                    syn_lines.append(f"{c_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if syn_lines:
                syn_stem = f"{stem}_polybal_{idx:04d}"
                cv2.imwrite(str(out_img_dir / f"{syn_stem}.jpg"), comp)
                with open(out_lbl_dir / f"{syn_stem}.txt", "w", encoding="utf-8") as lf:
                    lf.write("\n".join(syn_lines))

    return class_names


def main():
    parser = argparse.ArgumentParser(description="Convert ZeroWaste COCO dataset to polygon-balanced YOLO format.")
    parser.add_argument("--raw_dir", type=Path, default=Path("datasets/zerowaste-f"))
    parser.add_argument("--out_dir", type=Path, default=Path("datasets/zerowaste_yolo_p2"))
    parser.add_argument("--yaml_out", type=Path, default=Path("configs/ewaste_data.yaml"))
    parser.add_argument("--min_box_px", type=int, default=28, help="Minimum sortable fragment size in pixels")
    parser.add_argument("--preprocess", action="store_true", help="Apply L*a*b* CLAHE + Bilateral filter")
    args = parser.parse_args()

    preprocessor = PhotometricPreprocessor() if args.preprocess else None
    class_names = {}

    for split in ("train", "val", "test"):
        json_path = args.raw_dir / split / "labels.json"
        img_folder = args.raw_dir / split / "data"
        if json_path.exists() and img_folder.exists():
            class_names = convert_split(json_path, img_folder, args.out_dir, split, preprocessor, args.min_box_px)

    if not class_names:
        raise FileNotFoundError(f"No valid splits found under '{args.raw_dir.resolve()}'.")

    args.yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.yaml_out, "w", encoding="utf-8") as yf:
        yaml.dump(
            {
                "path": str(args.out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": class_names,
            },
            yf,
            sort_keys=False,
        )
    print(f"Polygon-balanced dataset ready at {args.out_dir} | Config saved to {args.yaml_out}")


if __name__ == "__main__":
    main()