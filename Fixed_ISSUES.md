# Fixed Issues

A record of the bugs, performance problems and robustness gaps found and fixed in
`ros2-depth-anything-v3-trt`. Each entry describes the symptom, the root cause and
the fix.

---

## 1. Invalid default `BatchConfig` (`{1, 0, 1}`)

**Files:** `depth_anything_v3/include/depth_anything_v3/tensorrt_depth_anything.hpp`

**Symptom:** When the caller did not pass an explicit batch configuration, the
constructor received `{1, 0, 1}` (min = 1, opt = 0, max = 1). An optimum batch size
of 0 is invalid for a TensorRT build profile and could break engine generation.

**Root cause:** The default argument used `{1, 0, 1}` instead of `{1, 1, 1}`.

**Fix:** Changed the constructor default to `batch_config = {1, 1, 1}`. The node
also passes `{1, 1, 1}` explicitly.

---

## 2. Hardcoded / fragile TensorRT IO tensor indices

**Files:** `tensorrt_depth_anything.hpp`, `tensorrt_depth_anything.cpp`

**Symptom:** Output buffers were allocated and read at hardcoded indices
(e.g. `depth_output_index = 1`, `sky_output_index = 2`). The input shape was
re-read from index 0 on every frame and the tensor list was walked per-frame.
Any engine whose IO ordering differed would silently produce wrong results or
read out of bounds.

**Root cause:** Dependence on the order in which a specific ONNX→TRT converter
happened to list tensors.

**Fix:** Added `resolveTensors()` which walks the engine's IO tensors once and
resolves them **by name and mode** (`input` via `TensorIOMode::kINPUT`, outputs
`depth` and `sky` by name), with sensible fallbacks. Resolved names and indices
are cached in `input_index_`, `depth_index_`, `sky_index_`, and the mapped
buffers (`bound_tensor_addresses_`) are bound to the execution context a single
time via `setTensorAddress()`. `maybeSetInputShape()` applies `batch_size_` only
when the batch axis is actually dynamic (`-1`).

---

## 3. Scope / visibility of `launchResizeColor` and `launchBuildPointCloud`

**Files:** `preprocess_gpu.cu`, `postprocess_gpu.cu`, `tensorrt_depth_anything.hpp`

**Symptom:** The new GPU kernels and launchers were private to the `.cu`
translation unit, so the core class could not call them.

**Root cause:** Kernels declared `static` inside `namespace { ... }` in the `.cu`
files without matching public launchers / header declarations.

**Fix:** Wrapped the kernels in an anonymous namespace (internal linkage) and
added public launchers `launchResizeColor()` (preprocessing) and
`launchBuildPointCloud()` (postprocessing) declared in
`tensorrt_depth_anything.hpp`.

---

## 4. Per-frame host→device copies of two unnecessary tensors

**Files:** `tensorrt_depth_anything.cpp`

**Symptom:** Every frame copied the model-size `depth` and `sky mask` from device
to host (`depth_elem_num_` and `sky_elem_num_` floats), even though only the
full-resolution depth image and the point cloud are published.

**Root cause:** The point cloud was generated on the CPU from those host copies.

**Fix:** Point cloud generation now runs entirely on the GPU
(`buildPointCloudKernel`). Only the published results are copied back: the
full-resolution depth image and the packed point cloud payload. Both copies use
one synchronous stream (`cudaMemcpyAsync` + single `cudaStreamSynchronize`).

---

## 5. Unnecessary per-frame H2D re-upload and blocked copies

**Files:** `tensorrt_depth_anything.cpp`

**Symptom:** The source frame was copied host→device every frame through
`cuMemcpy2DAsync` from unpinned `cv::Mat` memory, which forces a synchronous
staging copy and blocks the pipeline.

**Root cause:** Copying directly from pageable OpenCV memory.

**Fix:** The frame is first staged into a **pinned** host buffer
(`cuda_utils::make_unique_host<...>(..., cudaHostAllocDefault)`) with rows packed
(i.e. `width * 3` bytes per row, dropping any image-step padding), then uploaded
with a single fast `cudaMemcpyAsync`. The pinned buffer is reused across frames.

---

## 6. Pre-processing only initialized buffers once

**Files:** `tensorrt_depth_anything.cpp`

**Symptom:** If the camera resolution changed at runtime (exposure change, camera
reboot), the preprocessing buffers and the `src_width_`/`src_height_` state kept
the stale resolution, producing corrupt or wrong-sized depths.

**Root cause:** Buffer/state init happened only in `initPreprocessBuffer()`, which
was called solely for the first frame.

**Fix:** `preprocess()` now detects a resolution change itself and re-allocates
the pinned staging buffer and recomputes the scale factors `scale_x_`/`scale_y_`.
Postprocessing buffers self-resize through `initPostprocessBuffers()` when the
network output or output size changes.

---

## 7. `sky_depth_cap` parameter declared but never applied

**Files:** `depth_anything_v3_node.cpp`, `tensorrt_depth_anything.hpp`,
`tensorrt_depth_anything.cpp`

**Symptom:** Changing `sky_depth_cap` had no effect on the sky fill-in depth.

**Root cause:** The value was stored in the node but never forwarded to the model
wrapper (no setter existed; the internal value stayed at its default).

**Fix:**
- Added `TensorRTDepthAnything::setSkyDepthCap(float)`.
- The node constructor calls it with the declared parameter.
- `onSetParam` now forwards `sky_depth_cap` changes to the wrapper.

---

## 8. Runtime `onnx_path` / `precision` parameter changes silently ignored

**Files:** `depth_anything_v3_node.cpp`

**Symptom:** Declaring `onnx_path` and `precision` as dynamically changeable
parameters implied they could be swapped at runtime, but nothing re-loaded the
engine. The UI suggested a non-functional capability.

**Root cause:** No validation on the parameter callback.

**Fix:** `onSetParam` rejects changes to `onnx_path` (and `precision`) with a
clear error message so the limitation is explicit instead of silently ignored;
engine reloads require a node restart.

---

## 9. Debug callbacks ran unconditionally and duplicated work

**Files:** `depth_anything_v3_node.cpp`

**Symptom:** The debug image/camera-info subscribers stayed active even when
`enable_debug` was false, and the RGB debug path pushed an extra copy of the
frame into the pipeline.

**Root cause:** Debug subscriptions were created unconditionally.

**Fix:** Debug subscriptions are only created (and the debug callbacks only
likely run) when `enable_debug` is true.

---

## 10. Per-frame heap binding walk (`setTensorAddress` loop) and FPS recomputation

**Files:** `tensorrt_depth_anything.cpp`, `depth_anything_v3_node.{hpp,cpp}`

**Symptom:** Every inference re-enumerated/named tensors and re-set all tensor
addresses; the node computed FPS over the full run-time window stuffed into a
vector that was re-sorted/allocated each frame.

**Root cause:** No caching; hot-loop bookkeeping errors.

**Fix:**
- Tensor addresses are bound once at setup; the binding walk only re-runs on
  `infer()` with the cached name→pointer map.
- FPS tracking uses a fixed data-member `std::deque<double>` of recent inference
  times (`inference_times_`) instead of an ever-growing vector.

---

## 11. Redundant re-initialization guard (`is_initialized_`)

**Files:** `depth_anything_v3_node.{hpp,cpp}`

**Symptom:** A manual `is_initialized_` flag plus `initPreprocessBuffer(...)`
clamp-and-resize dance guarded the first frame of the node.

**Root cause:** Left over from when preprocessing had to be warmed up manually
with fixed `width`/`height`.

**Fix:** Removed the flag and the clamp/resize logic. The wrapper now self-manages
its buffer sizes and scale; the node just calls `doInference()`.

---

## 12. Point-cloud `rgb` ordering and payload layout

**Files:** `postprocess_gpu.cu`, `tensorrt_depth_anything.cpp`

**Symptom:** CPU implementation wrote pre-packed colors; the GPU path had to
match the exact `PointCloud2` field layout produced by `setPointCloud2FieldsByString`.

**Root cause:** The modifier builds `xyz` (offsets 0,4,8) + `r,g,b` (offsets
12,13,14), `point_step = 15`; for XYZ-only `point_step = 12`.

**Fix:** `buildPointCloudKernel` writes `xyz` at offsets 0/4/8 (NaN for
sky/invalid/depth ≤ 0) and `r@12, g@13, b@14` (swapping BGR→RGB), with a dynamic
12- or 15-byte stride selected by `with_color`. The message is sized with
`point_cloud_.data.resize(total_bytes)` and `is_dense = false`.

---

## 13. Point-cloud downsample edge stitching

**Files:** `postprocess_gpu.cu`

**Symptom:** Downsampling could index past the last row/column of the depth map.

**Root cause:** `ox * downsample` for the last output point reaches `width`
(or beyond) when `width % downsample != 0`.

**Fix:** Source sampling is clamped with `min(ox * downsample, width - 1)` and
`min(oy * downsample, height - 1)`.

---

## 14. GPU color resize (RGB point cloud) missing

**Files:** `preprocess_gpu.cu`, `tensorrt_depth_anything.cpp`

**Symptom:** No device-side path to colorize the point cloud; the GPU point-cloud
path could not sample colors.

**Root cause:** The `preprocess_gpu.cu` file had no color resize kernel.

**Fix:** Added `resizeBgrBilinear` (float interpolation, myBilinearTexFilter-like
clamping, BGR output) and its public launcher `launchResizeColor`. `buildPointCloud`
allocates a model-sized `color_model_d_` buffer and resizes the (already uploaded)
source frame into it before the point-cloud kernel runs.

---

## 15. `tensorrt_common.cpp` missing `<chrono>` include

**Files:** `tensorrt_common.cpp`

**Symptom:** Build failures where `std::chrono` symbols were unavailable
(transitive include removed).

**Root cause:** The file used `std::chrono` timing helpers without including
`<chrono>`.

**Fix:** Added `#include <chrono>`.

---

## 16. Irrelevant network-info logging on cached-engine startup

**Files:** `tensorrt_common.cpp`

**Symptom:** Loading a pre-built engine dumped a full network-info print even
though the engine was already optimized.

**Root cause:** `printNetworkInfo()` was called on every init, cached or not.

**Fix:** `printNetworkInfo()` is now emitted only when an engine is actually
**built** (not when a cached engine is merely loaded), so warm startup logs stay
clean.

---

## 17. `generate_engines.sh` used non-portable relative paths

**Files:** `generate_engines.sh`

**Symptom:** The script resolved the root dir with `$(cd .. && pwd)` and could be
broken if run from the `depth_anything_v3` directory; built workspace location
was hardcoded, and the install dir was assumed to exist.

**Root cause:** Path assumptions baked into the script.

**Fix:** The script now
- resolves its root from its own location (`SCRIPT_DIR`),
- honors a `WS_INSTALL_DIR` environment variable (defaulting to the normal
  `install/`), 
- sources the built workspace's `setup.bash` when present,
- validates the install dir exists and prints actionable errors otherwise.

---

## 18. Stale README / model documentation

**Files:** `README.md`, `depth_anything_v3/models/README.md`

**Symptom:** Docs described a CPU postprocessing pipeline, a `confidence` model
output, and the wrong input/output shape (`[1,3,388,504]`).

**Root cause:** Documentation lagged behind the model export and the GPU port.

**Fix:**
- `models/README.md` now documents the actual exported graph: input
  `[1, 3, 280, 504]`, outputs `depth` `[1,1,280,504]` and `sky` `[1,1,280,504]`.
- `README.md` architecture section now describes the all-GPU pipeline
  (GPU preprocessing with pinned async upload, GPU postprocessing and point
  cloud, single host sync per frame).

---

## 19. CMake language-setup order

**Files:** `depth_anything_v3/CMakeLists.txt`

**Symptom:** `find_package(CUDA)` was called before the CUDA language was enabled,
which is fragile across CMake versions.

**Root cause:** Ordering of `enable_language(CUDA)` / `find_package(CUDA)`.

**Fix:** `enable_language(CUDA)` is now issued before `find_package(CUDA)`.

---

## Not yet validated

The fixes above were implemented and cross-reviewed for API/signature consistency,
but **no compile/run test was possible in the development environment** (no ROS 2,
CUDA, or TensorRT toolchain). Before merging, run:

```bash
colcon build --symlink-install
bash generate_engines.sh
ros2 launch depth_anything_v3 depth_anything_v3_ros2.launch.xml
```