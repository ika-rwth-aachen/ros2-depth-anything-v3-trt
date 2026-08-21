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

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_utils/cuda_check_error.hpp"

namespace depth_anything_v3
{
namespace
{
__constant__ float c_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float c_inv_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

// Cubic convolution weights with A = -0.75, as used by cv::INTER_CUBIC. A
// bilinear substitute is much cheaper but moves the published depth by tens of
// metres, so the filter is reproduced rather than approximated.
__device__ __forceinline__ void cubicCoeffs(float t, float * c)
{
  const float A = -0.75f;
  c[0] = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A;
  c[1] = ((A + 2) * t - (A + 3)) * t * t + 1;
  c[2] = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1;
  c[3] = 1.0f - c[0] - c[1] - c[2];
}

// Resize, BGR to RGB, ImageNet normalisation and NCHW packing in one pass.
__global__ void resizeCubicNormalizeNCHW(
  const uchar3 * __restrict__ src, int src_width, int src_height,
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

  float acc[3] = {0.0f, 0.0f, 0.0f};
  for (int m = 0; m < 4; ++m) {
    const int sy = min(max(iy - 1 + m, 0), src_height - 1);
    float row[3] = {0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 4; ++k) {
      const int sx = min(max(ix - 1 + k, 0), src_width - 1);
      const uchar3 pixel = src[sy * src_width + sx];
      row[0] += cx[k] * pixel.z;
      row[1] += cx[k] * pixel.y;
      row[2] += cx[k] * pixel.x;
    }
    for (int c = 0; c < 3; ++c) acc[c] += cy[m] * row[c];
  }

  const int plane_size = dst_width * dst_height;
  const int i = y * dst_width + x;
  for (int c = 0; c < 3; ++c) {
    // cv::resize rounds and saturates back to 8U before the normalisation.
    const float value = fminf(fmaxf(rintf(acc[c]), 0.0f), 255.0f);
    dst[c * plane_size + i] = (value * (1.0f / 255.0f) - c_mean[c]) * c_inv_std[c];
  }
}
}  // namespace

void launchPreprocess(
  const uint8_t * src_bgr, int src_width, int src_height,
  float * dst_nchw, int dst_width, int dst_height, cudaStream_t stream)
{
  const dim3 block(16, 16);
  const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
  resizeCubicNormalizeNCHW<<<grid, block, 0, stream>>>(
    reinterpret_cast<const uchar3 *>(src_bgr), src_width, src_height,
    dst_nchw, dst_width, dst_height);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace depth_anything_v3
