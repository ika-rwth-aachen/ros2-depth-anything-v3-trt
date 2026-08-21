// Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "cuda_utils/cuda_check_error.hpp"

namespace depth_anything_v3
{
namespace
{
constexpr int kBlockSize = 256;
constexpr int kHistogramBins = 4096;
constexpr uint32_t kMaxSample = 100000;
constexpr size_t kScratchAlignment = 256;

// Device scratch for one postprocess call, carved out of a single buffer.
struct Scratch
{
  uint32_t * valid;        // 1 for the pixels that feed the percentile
  uint32_t * rank;         // exclusive prefix sum of valid
  uint32_t * histogram;
  uint32_t * range_lo;     // depth range of the sampled pixels, as float bits
  uint32_t * range_hi;
  uint32_t * sample_size;  // number of pixels the percentile is taken over
  float * fill;            // depth written to the sky pixels
  void * scan;
  size_t scan_bytes;
  size_t total_bytes;
};

// With base == nullptr this only computes total_bytes.
Scratch layoutScratch(int num_pixels, void * base)
{
  Scratch scratch{};
  CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
    nullptr, scratch.scan_bytes, static_cast<uint32_t *>(nullptr),
    static_cast<uint32_t *>(nullptr), num_pixels));

  auto * bytes = static_cast<uint8_t *>(base);
  size_t offset = 0;
  const auto take = [&](size_t size) -> void * {
    void * ptr = bytes != nullptr ? bytes + offset : nullptr;
    offset += (size + kScratchAlignment - 1) / kScratchAlignment * kScratchAlignment;
    return ptr;
  };

  const size_t counts_bytes = static_cast<size_t>(num_pixels) * sizeof(uint32_t);
  scratch.valid = static_cast<uint32_t *>(take(counts_bytes));
  scratch.rank = static_cast<uint32_t *>(take(counts_bytes));
  scratch.histogram = static_cast<uint32_t *>(take(kHistogramBins * sizeof(uint32_t)));
  scratch.range_lo = static_cast<uint32_t *>(take(sizeof(uint32_t)));
  scratch.range_hi = static_cast<uint32_t *>(take(sizeof(uint32_t)));
  scratch.sample_size = static_cast<uint32_t *>(take(sizeof(uint32_t)));
  scratch.fill = static_cast<float *>(take(sizeof(float)));
  scratch.scan = take(scratch.scan_bytes);
  scratch.total_bytes = offset;
  return scratch;
}

// The mask selects the non-sky pixels, which are the ones kept in the point
// cloud. The percentile additionally skips non-finite and non-positive depths.
__global__ void scaleAndMask(
  const float * __restrict__ raw, const float * __restrict__ sky,
  float sky_threshold, float focal_scale, float * __restrict__ depth,
  uint8_t * __restrict__ mask, uint32_t * __restrict__ valid, int num_pixels)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pixels) return;

  float scaled = raw[i];
  if (scaled <= 0.0f) scaled = 0.0f;
  scaled *= focal_scale;

  const bool non_sky = sky[i] < sky_threshold;
  depth[i] = scaled;
  mask[i] = non_sky ? 1u : 0u;
  valid[i] = non_sky && isfinite(scaled) && scaled > 0.0f ? 1u : 0u;
}

// The sample is the first kMaxSample valid pixels in row-major order, not a
// subsample spread over the frame; the sky fill value is calibrated against it.
__global__ void countSample(
  const uint32_t * __restrict__ rank, const uint32_t * __restrict__ valid,
  uint32_t * __restrict__ sample_size, int num_pixels)
{
  const uint32_t total = rank[num_pixels - 1] + valid[num_pixels - 1];
  *sample_size = total < kMaxSample ? total : kMaxSample;
}

// Depths are non-negative, so their IEEE bit patterns compare in the same order
// and integer atomics can be used. Reducing within the block first keeps the
// number of atomics on the two global words down to one per block.
__global__ void reduceRange(
  const float * __restrict__ depth, const uint32_t * __restrict__ valid,
  const uint32_t * __restrict__ rank, const uint32_t * __restrict__ sample_size,
  uint32_t * __restrict__ range_lo, uint32_t * __restrict__ range_hi, int num_pixels)
{
  __shared__ uint32_t block_lo;
  __shared__ uint32_t block_hi;
  if (threadIdx.x == 0) {
    block_lo = 0xFFFFFFFFu;
    block_hi = 0u;
  }
  __syncthreads();

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pixels && valid[i] != 0u && rank[i] < *sample_size) {
    const uint32_t bits = __float_as_uint(depth[i]);
    atomicMin(&block_lo, bits);
    atomicMax(&block_hi, bits);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMin(range_lo, block_lo);
    atomicMax(range_hi, block_hi);
  }
}

__global__ void histogramSample(
  const float * __restrict__ depth, const uint32_t * __restrict__ valid,
  const uint32_t * __restrict__ rank, const uint32_t * __restrict__ sample_size,
  const uint32_t * __restrict__ range_lo, const uint32_t * __restrict__ range_hi,
  uint32_t * __restrict__ histogram, int num_pixels)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pixels || valid[i] == 0u || rank[i] >= *sample_size) return;

  const float lo = __uint_as_float(*range_lo);
  const float hi = __uint_as_float(*range_hi);
  if (hi <= lo) return;

  const float bin_scale = (kHistogramBins - 1) / (hi - lo);
  atomicAdd(&histogram[static_cast<int>((depth[i] - lo) * bin_scale)], 1u);
}

// One thread walks the bins so the fill value stays on the device and the
// pipeline never has to synchronise in the middle of the postprocess.
__global__ void pickFillValue(
  const uint32_t * __restrict__ histogram, const uint32_t * __restrict__ sample_size,
  const uint32_t * __restrict__ range_lo, const uint32_t * __restrict__ range_hi,
  float sky_depth_cap, float * __restrict__ fill)
{
  const uint32_t size = *sample_size;
  if (size == 0u) return;

  const float lo = __uint_as_float(*range_lo);
  const float hi = __uint_as_float(*range_hi);
  if (hi <= lo) {
    *fill = fminf(lo, sky_depth_cap);
    return;
  }

  const uint32_t target = static_cast<uint32_t>(0.99 * (size - 1));
  uint32_t cumulative = 0u;
  int bin = 0;
  for (; bin < kHistogramBins; ++bin) {
    cumulative += histogram[bin];
    if (cumulative > target) break;
  }
  const float bin_scale = (kHistogramBins - 1) / (hi - lo);
  *fill = fminf(lo + static_cast<float>(bin) / bin_scale, sky_depth_cap);
}

// With no valid pixels there is no fill value and the sky is left as it is.
__global__ void fillSky(
  float * __restrict__ depth, const uint8_t * __restrict__ mask,
  const uint32_t * __restrict__ sample_size, const float * __restrict__ fill, int num_pixels)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pixels || *sample_size == 0u || mask[i] != 0u) return;
  depth[i] = *fill;
}

__device__ __forceinline__ void cubicCoeffs(float t, float * c)
{
  const float A = -0.75f;
  c[0] = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A;
  c[1] = ((A + 2) * t - (A + 3)) * t * t + 1;
  c[2] = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1;
  c[3] = 1.0f - c[0] - c[1] - c[2];
}

// Float bicubic upscale. There is no saturating cast here: cv::resize on CV_32F
// keeps the interpolated value, overshoot included.
__global__ void upscaleCubic(
  const float * __restrict__ src, int src_width, int src_height,
  float * __restrict__ dst, int dst_width, int dst_height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_width || y >= dst_height) return;

  const float fx = (x + 0.5f) * src_width / dst_width - 0.5f;
  const float fy = (y + 0.5f) * src_height / dst_height - 0.5f;
  const int ix = __float2int_rd(fx);
  const int iy = __float2int_rd(fy);

  float cx[4], cy[4];
  cubicCoeffs(fx - ix, cx);
  cubicCoeffs(fy - iy, cy);

  float acc = 0.0f;
  for (int m = 0; m < 4; ++m) {
    const int sy = min(max(iy - 1 + m, 0), src_height - 1);
    float row = 0.0f;
    for (int k = 0; k < 4; ++k) {
      const int sx = min(max(ix - 1 + k, 0), src_width - 1);
      row += cx[k] * src[sy * src_width + sx];
    }
    acc += cy[m] * row;
  }
  dst[y * dst_width + x] = acc;
}
}  // namespace

size_t postprocessScratchBytes(int num_pixels)
{
  return layoutScratch(num_pixels, nullptr).total_bytes;
}

void launchPostprocess(
  const float * depth_raw, const float * sky_raw, int width, int height,
  float focal_scale, float sky_threshold, float sky_depth_cap,
  int out_width, int out_height,
  float * depth_out, uint8_t * mask_out, float * depth_full_out,
  void * scratch_buffer, size_t scratch_bytes, cudaStream_t stream)
{
  const int num_pixels = width * height;
  const Scratch scratch = layoutScratch(num_pixels, scratch_buffer);
  if (scratch_bytes < scratch.total_bytes) {
    throw std::runtime_error("postprocess scratch buffer is too small");
  }
  const int blocks = (num_pixels + kBlockSize - 1) / kBlockSize;

  scaleAndMask<<<blocks, kBlockSize, 0, stream>>>(
    depth_raw, sky_raw, sky_threshold, focal_scale,
    depth_out, mask_out, scratch.valid, num_pixels);

  // rank[i] is the number of valid pixels before i, so taking the first N in
  // row-major order is just rank < N.
  size_t scan_bytes = scratch.scan_bytes;
  CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
    scratch.scan, scan_bytes, scratch.valid, scratch.rank, num_pixels, stream));
  countSample<<<1, 1, 0, stream>>>(scratch.rank, scratch.valid, scratch.sample_size, num_pixels);

  CHECK_CUDA_ERROR(cudaMemsetAsync(scratch.range_lo, 0xFF, sizeof(uint32_t), stream));
  CHECK_CUDA_ERROR(cudaMemsetAsync(scratch.range_hi, 0x00, sizeof(uint32_t), stream));
  CHECK_CUDA_ERROR(
    cudaMemsetAsync(scratch.histogram, 0, kHistogramBins * sizeof(uint32_t), stream));

  reduceRange<<<blocks, kBlockSize, 0, stream>>>(
    depth_out, scratch.valid, scratch.rank, scratch.sample_size,
    scratch.range_lo, scratch.range_hi, num_pixels);
  histogramSample<<<blocks, kBlockSize, 0, stream>>>(
    depth_out, scratch.valid, scratch.rank, scratch.sample_size,
    scratch.range_lo, scratch.range_hi, scratch.histogram, num_pixels);
  pickFillValue<<<1, 1, 0, stream>>>(
    scratch.histogram, scratch.sample_size, scratch.range_lo, scratch.range_hi,
    sky_depth_cap, scratch.fill);
  fillSky<<<blocks, kBlockSize, 0, stream>>>(
    depth_out, mask_out, scratch.sample_size, scratch.fill, num_pixels);

  const dim3 block(16, 16);
  const dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
  upscaleCubic<<<grid, block, 0, stream>>>(
    depth_out, width, height, depth_full_out, out_width, out_height);

  CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace depth_anything_v3
