# ResNet50 Vulkan Inference Benchmark — AMD RX 580 + Ubuntu

Benchmark of ResNet50 image classification inference on an **AMD Radeon RX 580** using **Vulkan (RADV)** acceleration, comparing the ncnn and ONNX Runtime backends on Ubuntu Linux.

## TL;DR

| Backend | Avg. inference time (100 runs) | Speedup vs. ncnn CPU |
|---|---|---|
| **ncnn (Vulkan GPU)** | **16.62 ms** | **10.8x** |
| ncnn (CPU) | 179.93 ms | 1.0x |
| ONNX Runtime (Vulkan EP) | N/A — EP not compiled into the PyPI build, fell back to CPU | — |
| ONNX Runtime (CPU) | 156.37 ms | 1.15x |

**Vulkan acceleration via ncnn delivers a 10.8x speedup on the RX 580.**

## Environment

| Component | Detail |
|---|---|
| CPU | Intel Xeon X5680 @ 3.33 GHz (12 threads) |
| GPU | AMD Radeon RX 580 Series (Polaris 10), discrete |
| Vulkan driver | Mesa RADV, driver 26.0.3 (Vulkan 1.4) |
| OS | Ubuntu 26.04 LTS (Linux) |
| Python | 3.14 |
| ncnn | 1.0.20260526 (python binding) |
| ONNX Runtime | 1.28.0 |
| PyTorch / TorchVision | 2.13.0+cu130 / 0.28.0 |
| Model | ResNet50 (ImageNet-1K V2 weights) |

## Repository contents

- `export_resnet50.py` — exports the PyTorch ResNet50 model to `resnet50.onnx` (opset 14, dynamic batch).
- `resnet50.onnx` / `resnet50.onnx.data` — the exported ONNX model (weights in external data).
- `resnet50.pnnx.param` / `resnet50.pnnx.bin` / `resnet50.pnnx.onnx` / `resnet50.pnnxsim.onnx` — pnnx conversion outputs.
- `resnet50.ncnn.param` / `resnet50.ncnn.bin` — final ncnn model (graph + weights) used for inference.
- `resnet50_ncnn_vulkan.py` — ncnn inference script with Vulkan acceleration (`--cpu` for CPU comparison).
- `resnet50_onnx_vulkan.py` — ONNX Runtime inference via `VulkanExecutionProvider`.
- `resnet50_ncnn.py` — minimal ncnn smoke-test inference.
- `imagenet_classes.txt` — ImageNet class labels.
- `sample.jpg` — test image.

## Setup

```bash
# install ncnn with Vulkan support (python binding)
pip install ncnn

# ONNX Runtime (CPU build; Vulkan EP requires a special build — see notes)
pip install onnxruntime
pip install torch torchvision pillow numpy
```

### Exporting the model (optional)

```bash
python export_resnet50.py          # -> resnet50.onnx (+ external data)
# convert ONNX -> ncnn using pnnx (from ncnn tools)
pnnx resnet50.onnx                # -> resnet50.ncnn.param / resnet50.ncnn.bin
```

## Usage

```bash
# ncnn + Vulkan GPU (100 runs, default)
python resnet50_ncnn_vulkan.py --runs 100

# ncnn Vulkan vs. CPU comparison
python resnet50_ncnn_vulkan.py --runs 100 --cpu

# ONNX Runtime (attempts Vulkan EP, falls back to CPU)
python resnet50_onnx_vulkan.py
```

## Results

Benchmark performed on `sample.jpg`, batch size 1, input 224x224, after 5 warmup runs.

### ncnn

- **Vulkan:** 16.62 ms avg (10.8x faster than CPU)
- **CPU:** 179.93 ms avg
- **Top-5 prediction:** potter's wheel (739), oil filter (686), milk can (653), hard disc (592), chain saw (491)

### ONNX Runtime

- `VulkanExecutionProvider` was **not available** in the stock PyPI build (providers listed: `AzureExecutionProvider`, `CPUExecutionProvider`), so the script silently fell back to the CPU provider.
- **CPU:** 156.37 ms avg, predicted class 282.

## Analysis

- ncnn's Vulkan backend runs the model on the RX 580 via Mesa RADV with FP16 storage/packing enabled, which is the main driver of the 10.8x speedup.
- The Xeon X5680 (2010-era Westmere, no AVX2/AVX512) makes the CPU baseline particularly slow, further highlighting the GPU gain.
- The ONNX Runtime Vulkan path could not be compared because the PyPI wheel does not ship the Vulkan execution provider. To benchmark it, you would need to build/install ONNX Runtime from source with `--use_vulkan`.

## Notes / Limitations

- The RX 580 (Polaris) has no native FP16 compute, so FP16 packing gives memory-bandwidth savings rather than full arithmetic acceleration.
- Large files over GitHub's 100 MB per-file limit (`resnet50.onnx.data`, `resnet50.pnnx.bin`) are excluded from this repository.
