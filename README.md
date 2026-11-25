
# Intelligent Material Recovery: Real-Time E-Waste Sorting via Spatial-Spectral Attention Mechanisms

Machine Vision Master's Project — Bhavani Patangay (Kennesaw State University)

## Overview
End-to-end conveyor-belt sorting pipeline combining:
1. **Photometric Normalization:** CIE $L^*a^*b^*$ CLAHE on the luminance ($L^*$) channel + edge-preserving bilateral filtering ($d=5$) for specular glare suppression without chromatic distortion.
2. **Spatial-Spectral Attention:** Convolutional Block Attention Module (CBAM) integrated into the YOLOv8 PANet neck ($P_3, P_4, P_5$) prior to the decoupled detection heads.
3. **Real-Time Edge Inference:** ONNX Runtime (`opset 12`, `CoreMLExecutionProvider`) meeting the `< 30.00 ms` (`> 30.0 FPS`) industrial latency constraint across both benchmarks.

## Verified Results

| Benchmark | Model | Resolution | mAP@0.5 | mAP@0.5:0.95 | End-to-End Latency | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E-Waste Components (21 classes)** | **YOLOv8n + CBAM** | $640 \times 640$ | **95.5%** | **88.3%** | **11.50 ms** | **86.9 FPS** |
| **ZeroWaste-f Clutter (4 classes)** | **YOLOv8s + CBAM (Balanced)** | $800 \times 800$ | **41.1%** *(41.6% Test)* | **26.9%** *(27.9% Test)* | **21.67 ms** | **46.1 FPS** |
| **ZeroWaste-f Clutter (4 classes)** | Photometric YOLOv8n (Baseline) | $640 \times 640$ | 28.7% | 17.6% | 9.54 ms | 104.9 FPS |

### End-to-End Latency Breakdown (200 Frames, `CoreMLExecutionProvider`)
- **YOLOv8n-CBAM ($640 \times 640$, 12.3 MB ONNX):** `2.97 ms` preprocess + `8.54 ms` inference = **11.50 ms (86.9 FPS)**
- **YOLOv8s-CBAM ($800 \times 800$, 43.0 MB ONNX):** `4.08 ms` preprocess + `17.60 ms` inference = **21.67 ms (46.1 FPS)**

## Repository Structure
- `models/cbam.py` — Channel and Spatial Attention (CBAM) modules
- `src/preprocess.py` — CIE $L^*a^*b^*$ CLAHE + Bilateral glare-suppression filter
- `src/prepare_data.py` — ZeroWaste COCO-to-YOLO converter with minority-class copy-paste balancing
- `src/train.py` — Custom `CBAMDetectionTrainer` for YOLOv8 + CBAM training
- `src/export_onnx.py` — ONNX export utility (`opset 12`)
- `src/benchmark.py` — End-to-end frame latency profiler
- `src/visualize.py` — Glare-suppression and CBAM spatial attention visualization scripts
- `figures/` — Evaluation curves, confusion matrices, preprocessing comparisons, and attention maps
