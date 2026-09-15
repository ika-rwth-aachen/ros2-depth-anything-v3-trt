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

#ifndef DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_

#include <cstdint>

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <utility>
#include <vector>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

namespace depth_anything_v3
{
using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

// Defined in preprocess_gpu.cu
void launchPreprocess(
  const uint8_t * src_bgr, int src_width, int src_height,
  float * dst_nchw, int dst_width, int dst_height, cudaStream_t stream);

// Defined in preprocess_gpu.cu: write a packed (HWC, BGR8) image resized to
// (dst_width x dst_height) so it can be used for point cloud colouring.
void launchResizeColor(
  const uint8_t * src_bgr, int src_width, int src_height,
  uint8_t * dst_bgr, int dst_width, int dst_height, cudaStream_t stream);

// Defined in postprocess_gpu.cu
size_t postprocessScratchBytes(int num_pixels);
void launchPostprocess(
  const float * depth_raw, const float * sky_raw, int width, int height,
  float focal_scale, float sky_threshold, float sky_depth_cap,
  int out_width, int out_height,
  float * depth_out, uint8_t * mask_out, float * depth_full_out,
  void * scratch_buffer, size_t scratch_bytes, cudaStream_t stream);

// Defined in postprocess_gpu.cu: build the packed (xyz) or (xyz,rgb)
// PointCloud2 payload straight on the device from the scaled depth map.
// out_packed has point_step 12 (xyz) or 15 (xyz + r,g,b) bytes.
void launchBuildPointCloud(
  const float * depth, const uint8_t * mask, const uint8_t * color_bgr,
  int width, int height, float fx, float fy, float cx, float cy,
  int downsample, bool with_color, uint8_t * out_packed,
  int out_width, int out_height, cudaStream_t stream);

/**
 * @class TensorRTDepthAnything
 * @brief TensorRT DepthAnythingV3 wrapper for depth estimation and point cloud generation
 */
class TensorRTDepthAnything
{
public:
  /**
   * @brief Construct TensorRTDepthAnything.
   * @param[in] model_path ONNX model_path
   * @param[in] precision precision for inference
   * @param[in] build_config configuration including precision, calibration method, etc.
   * @param[in] use_gpu_preprocess whether use cuda gpu for preprocessing
   * @param[in] calibration_image_list_file path for calibration files
   * @param[in] batch_config configuration for batched execution
   * @param[in] max_workspace_size maximum workspace for building TensorRT engine
   */
  TensorRTDepthAnything(
    const std::string & model_path, const std::string & precision,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30));

  /**
   * @brief run inference including pre-process and post-process
   * @param[in] images batched images
   * @param[in] camera_info camera calibration info for point cloud generation
   * @param[in] downsample_factor only publish every Nth point (1 = no downsampling)
   * @param[in] colorize_pointcloud whether to colorize point cloud with RGB
   */
  bool doInference(
    const std::vector<cv::Mat> & images, const sensor_msgs::msg::CameraInfo & camera_info,
    int downsample_factor = 1, bool colorize_pointcloud = false);

  void initPreprocessBuffer(int width, int height);

  /**
   * @brief output TensorRT profiles for each layer
   */
  void printProfiling(void);

  /**
   * @brief Get the depth image result
   * @return depth image as cv::Mat (const reference)
   */
  const cv::Mat & getDepthImage() const;

  /**
   * @brief Get the point cloud result
   * @return point cloud as ROS2 PointCloud2 message (const reference)
   */
  const sensor_msgs::msg::PointCloud2 & getPointCloud() const;

  void setSkyThreshold(float threshold) { sky_threshold_ = threshold; }
  void setSkyDepthCap(float cap) { sky_depth_cap_ = cap; }

  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

  // Input/output buffers
  CudaUniquePtr<float[]> input_d_;
  size_t input_elem_num_{};

  // Output buffer for predicted depth
  CudaUniquePtr<float[]> depth_d_;
  size_t depth_elem_num_{};
  // Output buffer for predicted sky
  CudaUniquePtr<float[]> sky_d_;
  size_t sky_elem_num_{};

  StreamUniquePtr stream_{makeCudaStream()};

  int batch_size_;

  // preprocessing parameters
  bool use_gpu_preprocess_;
  CudaUniquePtrHost<unsigned char[]> image_buf_h_;
  CudaUniquePtr<unsigned char[]> image_buf_d_;
  size_t image_buf_bytes_{0};

  // postprocessing buffers, allocated for the current input/output size
  CudaUniquePtr<float[]> depth_scaled_d_;
  CudaUniquePtr<float[]> depth_full_d_;
  CudaUniquePtr<uint8_t[]> sky_mask_d_;
  CudaUniquePtr<uint8_t[]> post_scratch_d_;
  size_t post_scratch_bytes_{0};
  int post_width_{0};
  int post_height_{0};
  int post_out_width_{0};
  int post_out_height_{0};

  // point cloud buffers, allocated for the current output size
  CudaUniquePtr<uint8_t[]> color_model_d_;
  CudaUniquePtr<uint8_t[]> point_cloud_d_;
  CudaUniquePtrHost<uint8_t[]> point_cloud_h_;
  size_t point_capacity_bytes_{0};

  int src_width_;
  int src_height_;
  int input_width_ = 504;
  int input_height_ = 280;

  // Results
  cv::Mat depth_image_;
  double scale_x_{1.0};
  double scale_y_{1.0};
  float sky_threshold_{0.3f};
  float sky_depth_cap_{200.0f};
  bool colorize_point_cloud_{false};
  sensor_msgs::msg::PointCloud2 point_cloud_;

private:
  /**
   * @brief Resolve the engine's IO tensors by name/mode once and pre-allocate
   * all input/output buffers so inference does no per-frame bookkeeping.
   */
  void resolveTensors();

  /**
   * @brief Make sure the engine input shape reflects batch_size_ for engines
   * with a dynamic batch dimension.
   */
  void maybeSetInputShape();

  /**
   * @brief run preprocess including resizing, NHWC2NCHW and toFloat on GPU
   * @param[in] images batching images
   */
  void preprocess(const std::vector<cv::Mat> & images);

  /**
   * @brief perform TensorRT inference
   */
  bool infer();

  /**
   * @brief postprocess inference results to generate depth and point cloud
   * @param[in] camera_info camera calibration for point cloud generation
   * @param[in] downsample_factor downsampling factor for point cloud
   */
  void postprocess(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor);

  /**
   * @brief (re)allocate the postprocessing buffers when the resolution changes
   * @param[in] width network output width
   * @param[in] height network output height
   * @param[in] out_width published depth image width
   * @param[in] out_height published depth image height
   */
  void initPostprocessBuffers(int width, int height, int out_width, int out_height);

  /**
   * @brief Generate the PointCloud2 message on the GPU from the scaled depth
   * and sky mask, colouring it from the (already uploaded) source frame.
   * @param[in] camera_info camera calibration parameters
   * @param[in] downsample_factor only publish every Nth point
   */
  void buildPointCloud(
    const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor);

  // Resolved tensor names and indices (set once in resolveTensors)
  std::string input_tensor_name_;
  std::string depth_tensor_name_;
  std::string sky_tensor_name_;
  int input_index_{-1};
  int depth_index_{-1};
  int sky_index_{-1};
  std::vector<std::pair<std::string, void *>> bound_tensor_addresses_;
  std::vector<CudaUniquePtr<float[]>> extra_output_buffers_;
  std::vector<std::string> extra_tensor_names_;
};

}  // namespace depth_anything_v3

#endif  // DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_