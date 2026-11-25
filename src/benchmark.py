import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from src.preprocess import PhotometricPreprocessor


class ConveyorPipelineProfiler:
    def __init__(self, onnx_path: Path, imgsz: int = 640):
        self.imgsz = imgsz
        self.preprocessor = PhotometricPreprocessor()

        providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.input_dtype = (
            np.float16
            if self.session.get_inputs()[0].type == "tensor(float16)"
            else np.float32
        )

    def step(self, raw_bgr: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        resized = cv2.resize(raw_bgr, (self.imgsz, self.imgsz))
        clean_bgr = self.preprocessor(resized)
        t_prep_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(self.input_dtype) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        _ = self.session.run(None, {self.input_name: tensor})
        t_infer_ms = (time.perf_counter() - t1) * 1000.0

        return t_prep_ms, t_infer_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark end-to-end conveyor sorting latency.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to exported .onnx file")
    parser.add_argument("--image", type=Path, default=None, help="Optional real test frame")
    parser.add_argument("--frames", type=int, default=200, help="Number of timed iterations")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    profiler = ConveyorPipelineProfiler(args.onnx, imgsz=args.imgsz)

    if args.image and args.image.exists():
        sample_frame = cv2.imread(str(args.image))
    else:
        sample_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Warm-up passes to stabilize CUDA kernels and memory allocations
    for _ in range(15):
        profiler.step(sample_frame)

    prep_latencies, infer_latencies = [], []
    for _ in range(args.frames):
        t_prep, t_inf = profiler.step(sample_frame)
        prep_latencies.append(t_prep)
        infer_latencies.append(t_inf)

    avg_prep = float(np.mean(prep_latencies))
    avg_inf = float(np.mean(infer_latencies))
    avg_total = avg_prep + avg_inf
    fps = 1000.0 / avg_total

    print(f"\nExecution Provider : {profiler.active_provider}")
    print(f"Preprocess Latency : {avg_prep:.2f} ms (Resize + L*a*b* CLAHE + Bilateral)")
    print(f"Inference Latency  : {avg_inf:.2f} ms (YOLOv8n-CBAM ONNX)")
    print(f"End-to-End Latency : {avg_total:.2f} ms (Target: < 30.00 ms)")
    print(f"Effective Rate     : {fps:.1f} FPS (Target: > 30.0 FPS)")


if __name__ == "__main__":
    main()