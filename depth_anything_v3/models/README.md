# Depth Anything V3 Models

This directory should contain the Depth Anything V3 ONNX files or prebuilt TensorRT engines.

## Required Files

Place your Depth Anything V3 ONNX or engine file here (update the config path accordingly), for example:
- `DA3METRIC-LARGE.onnx`
- `DA3METRIC-LARGE.fp16-batch1.engine`

## Model Format

Expected model specifications (matches the export script in `onnx/export.py`):
- **Input**: RGB image, shape `[1, 3, 280, 504]` (batch, channels, height, width), type float32, range [0, 1]
- **Outputs**:
  - `depth`: metric depth map `[1, 1, 280, 504]`
  - `sky`: sky classification logits `[1, 1, 280, 504]` (lower values = sky)

## Installation

After building the package, this directory will be installed to:
```bash
/path/to/install/share/depth_anything_v3/models/
```

The node will look for models at the path specified in the parameter file.

## Runtime Comparison

Use Polygraphy to compare ONNX Runtime and TensorRT performance:
```bash
polygraphy run src/target/depth_anything_v3/models/DA3METRIC-LARGE.onnx \
    --onnxrt --providers cuda \
    --trt --fp16 \
    --iterations 100 \
    --warm-up 10
```