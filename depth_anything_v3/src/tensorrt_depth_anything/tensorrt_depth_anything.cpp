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

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/rclcpp.hpp>

#include "depth_anything_v3/tensorrt_depth_anything.hpp"
#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"

namespace
{
namespace fs = std::filesystem;

static size_t volumeFromDims(const nvinfer1::Dims & dims, int batch_size)
{
  return std::accumulate(dims.d, dims.d + dims.nbDims, size_t{1},
    [batch_size](size_t acc, int dim) {
      return acc * (dim == -1 ? static_cast<size_t>(batch_size) : static_cast<size_t>(dim));
    });
}

// Simple depth to point cloud conversion using camera intrinsics
static void depthImageToPointCloud(
  const cv::Mat & depth_image,
  const sensor_msgs::msg::CameraInfo & camera_info,
  sensor_msgs::msg::PointCloud2 & cloud_msg,
  const std::string & frame_id,
  int downsample_factor = 1,
  const cv::Mat & rgb_image = cv::Mat(),
  const cv::Mat & non_sky_mask = cv::Mat())
{
  cloud_msg.header.frame_id = frame_id;
  
  const int downsampled_height = (depth_image.rows + downsample_factor - 1) / downsample_factor;
  const int downsampled_width = (depth_image.cols + downsample_factor - 1) / downsample_factor;
  
  cloud_msg.height = downsampled_height;
  cloud_msg.width = downsampled_width;
  cloud_msg.is_dense = false;
  cloud_msg.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(cloud_msg);
  const bool has_color = !rgb_image.empty();
  if (has_color) {
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  } else {
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
  }

  const double fx = camera_info.k[0];
  const double fy = camera_info.k[4]; 
  const double cx = camera_info.k[2];
  const double cy = camera_info.k[5];

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<uint8_t>> iter_r, iter_g, iter_b;
  if (has_color) {
    iter_r = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "r");
    iter_g = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "g");
    iter_b = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "b");
  }

  const float bad_point = std::numeric_limits<float>::quiet_NaN();

  const bool use_mask = !non_sky_mask.empty();

  for (int v = 0; v < depth_image.rows; v += downsample_factor) {
    for (int u = 0; u < depth_image.cols; u += downsample_factor) {
      if (use_mask && non_sky_mask.at<uint8_t>(v, u) == 0) {
        *iter_x = *iter_y = *iter_z = bad_point;
        if (has_color) { **iter_r = **iter_g = **iter_b = 0; }
        ++iter_x; ++iter_y; ++iter_z;
        if (has_color) { ++(*iter_r); ++(*iter_g); ++(*iter_b); }
        continue;
      }

      const float depth = depth_image.at<float>(v, u);
      
      if (depth <= 0.0f || !std::isfinite(depth)) {
        *iter_x = *iter_y = *iter_z = bad_point;
        if (has_color) { **iter_r = **iter_g = **iter_b = 0; }
      } else {
        *iter_x = static_cast<float>((u - cx) * depth / fx);
        *iter_y = static_cast<float>((v - cy) * depth / fy);
        *iter_z = depth;
        if (has_color) {
          const cv::Vec3b rgb = rgb_image.at<cv::Vec3b>(v, u);
          **iter_r = rgb[2];
          **iter_g = rgb[1];
          **iter_b = rgb[0];
        }
      }
      ++iter_x; ++iter_y; ++iter_z;
      if (has_color) { ++(*iter_r); ++(*iter_g); ++(*iter_b); }
    }
  }
}

} // anonymous namespace

namespace depth_anything_v3
{

TensorRTDepthAnything::TensorRTDepthAnything(
  const std::string & model_path, const std::string & precision,
  tensorrt_common::BuildConfig build_config, const bool use_gpu_preprocess,
  std::string /* calibration_image_list_path */, const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size)
: batch_size_(batch_config[2]), use_gpu_preprocess_(use_gpu_preprocess)
{
  src_width_ = -1;
  src_height_ = -1;

  if (!fs::exists(model_path)) {
    throw std::runtime_error("Model file does not exist: " + model_path);
  }

  // Initialize TensorRT common
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
    model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
  trt_common_->setup();

  auto * engine = trt_common_->getEngine();
  depth_elem_num_ = 0;
  sky_elem_num_ = 0;
  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const auto dims = trt_common_->getBindingDimensions(i);
    if (name && std::string(name) == "depth") {
      depth_elem_num_ = volumeFromDims(dims, batch_size_);
    } else if (name && std::string(name) == "sky") {
      sky_elem_num_ = volumeFromDims(dims, batch_size_);
    }
  }
  if (depth_elem_num_ == 0) {
    // Fallback to the first output binding if names are unavailable
    const auto dims = trt_common_->getBindingDimensions(1);
    depth_elem_num_ = volumeFromDims(dims, batch_size_);
  }
  if (sky_elem_num_ == 0) {
    throw std::runtime_error("Expected TensorRT engine to expose 'sky' output tensor, but none was found");
  }

  // Allocate GPU memory for outputs
  depth_d_ = cuda_utils::make_unique<float[]>(depth_elem_num_);
  sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);

  // Get input dimensions
  const auto input_dims = trt_common_->getBindingDimensions(0);
  const int input_channels = input_dims.d[1];
  input_height_ = input_dims.d[2]; 
  input_width_ = input_dims.d[3];
  
  // Allocate input memory
  const size_t input_elem_num = batch_size_ * input_channels * input_height_ * input_width_;
  input_d_ = cuda_utils::make_unique<float[]>(input_elem_num);

}

void TensorRTDepthAnything::initPreprocessBuffer(int width, int height)
{
  src_width_ = width;
  src_height_ = height;
  scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
  scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);
  
  if (use_gpu_preprocess_) {
    const size_t image_size = src_width_ * src_height_ * 3; // RGB
    image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
      image_size * batch_size_, cudaHostAllocDefault);
    image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(image_size * batch_size_);
  }
}

void TensorRTDepthAnything::initPostprocessBuffers(
  int width, int height, int out_width, int out_height)
{
  if (width == post_width_ && height == post_height_ &&
    out_width == post_out_width_ && out_height == post_out_height_)
  {
    return;
  }

  const size_t plane_size = static_cast<size_t>(width) * height;
  depth_scaled_d_ = cuda_utils::make_unique<float[]>(plane_size);
  depth_full_d_ = cuda_utils::make_unique<float[]>(static_cast<size_t>(out_width) * out_height);
  sky_mask_d_ = cuda_utils::make_unique<uint8_t[]>(plane_size);
  post_scratch_bytes_ = postprocessScratchBytes(static_cast<int>(plane_size));
  post_scratch_d_ = cuda_utils::make_unique<uint8_t[]>(post_scratch_bytes_);

  post_width_ = width;
  post_height_ = height;
  post_out_width_ = out_width;
  post_out_height_ = out_height;
}

bool TensorRTDepthAnything::doInference(
  const std::vector<cv::Mat> & images, 
  const sensor_msgs::msg::CameraInfo & camera_info,
  int downsample_factor,
  bool colorize_pointcloud)
{
  if (images.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("TensorRTDepthAnything"), "No images provided for inference.");
    return false;
  }

  if (images.size() != static_cast<size_t>(batch_size_)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TensorRTDepthAnything"),
      "Batch size mismatch. Expected: %d, got: %zu", batch_size_, images.size());
    return false;
  }

  // Preprocess
  preprocess(images);

  // Run inference
  if (!infer()) {
    return false;
  }

  // Postprocess with downsampling
  cv::Mat rgb_for_pointcloud = colorize_pointcloud ? images[0] : cv::Mat();
  postprocess(camera_info, downsample_factor, rgb_for_pointcloud);
  
  return true;
}

void TensorRTDepthAnything::preprocess(const std::vector<cv::Mat> & images)
{
  auto input_dims = trt_common_->getBindingDimensions(0);
  if (input_dims.d[0] == -1) {
    input_dims.d[0] = batch_size_;
  }
  trt_common_->setBindingDimensions(0, input_dims);

  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
  scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
  scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);

  // Upload the frame and let one kernel write the normalised NCHW tensor
  // straight into the engine's input buffer.
  const cv::Mat & src_image = images[0];
  const size_t row_bytes = static_cast<size_t>(src_image.cols) * 3;
  const size_t src_bytes = row_bytes * src_image.rows;
  if (!image_buf_d_ || src_bytes != image_buf_bytes_) {
    image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(src_bytes);
    image_buf_bytes_ = src_bytes;
  }
  CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
    image_buf_d_.get(), row_bytes, src_image.data, src_image.step,
    row_bytes, src_image.rows, cudaMemcpyHostToDevice, *stream_));
  launchPreprocess(
    image_buf_d_.get(), src_image.cols, src_image.rows,
    input_d_.get(), input_width_, input_height_, *stream_);

  auto * engine = trt_common_->getEngine();
  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const auto dims = trt_common_->getBindingDimensions(i);
    const size_t required_output_elems = volumeFromDims(dims, batch_size_);
    if (name && std::string(name) == "depth") {
      if (required_output_elems != depth_elem_num_) {
        depth_elem_num_ = required_output_elems;
        depth_d_ = cuda_utils::make_unique<float[]>(depth_elem_num_);
      }
    } else if (name && std::string(name) == "sky") {
      if (required_output_elems != sky_elem_num_) {
        sky_elem_num_ = required_output_elems;
        sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);
      }
    }
  }
}

bool TensorRTDepthAnything::infer()
{
  auto * context = trt_common_->getContext();
  auto * engine = trt_common_->getEngine();

  extra_output_buffers_.clear();

  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const std::string tensor_name = name ? std::string(name) : std::string();
    const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

    void * buffer_ptr = nullptr;
    if (is_input || tensor_name.find("input") != std::string::npos ||
        tensor_name.find("image") != std::string::npos) {
      buffer_ptr = input_d_.get();
    } else if (tensor_name == "depth" || tensor_name.find("depth") != std::string::npos) {
      buffer_ptr = depth_d_.get();
    } else if (tensor_name == "sky" || tensor_name.find("sky") != std::string::npos) {
      if (!sky_d_ && sky_elem_num_ > 0) {
        sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);
      }
      buffer_ptr = sky_d_ ? static_cast<void *>(sky_d_.get()) : static_cast<void *>(depth_d_.get());
    } else {
      const auto dims = trt_common_->getBindingDimensions(i);
      const size_t elem_count = volumeFromDims(dims, batch_size_);
      extra_output_buffers_.push_back(cuda_utils::make_unique<float[]>(elem_count));
      buffer_ptr = extra_output_buffers_.back().get();
    }

    context->setTensorAddress(name, buffer_ptr);
  }

  if (!trt_common_->enqueueV3(*stream_)) {
    return false;
  }

  return true;
}

void TensorRTDepthAnything::postprocess(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor, const cv::Mat & rgb_image)
{
  const auto output_dims = trt_common_->getBindingDimensions(1);
  const int height = output_dims.nbDims > 2 ? output_dims.d[2] : input_height_;
  const int width = output_dims.nbDims > 3 ? output_dims.d[3] : input_width_;

  const size_t plane_size = static_cast<size_t>(height) * width;
  const size_t full_size = static_cast<size_t>(src_width_) * src_height_;

  // Use original intrinsics for metric conversion per spec.
  const double fx = camera_info.k[0] * scale_x_;
  const double fy = camera_info.k[4] * scale_y_;
  const double focal_pixels = 0.5 * (fx + fy);
  const double focal_scale = focal_pixels > 0.0 ? focal_pixels / 300.0 : 1.0;

  // Scaling, the sky mask, the sky fill and the upscale to camera resolution all
  // run on the depth map the engine produced; only the published results are
  // copied back.
  initPostprocessBuffers(width, height, src_width_, src_height_);
  launchPostprocess(
    depth_d_.get(), sky_d_.get(), width, height,
    static_cast<float>(focal_scale), sky_threshold_, sky_depth_cap_,
    src_width_, src_height_,
    depth_scaled_d_.get(), sky_mask_d_.get(), depth_full_d_.get(),
    post_scratch_d_.get(), post_scratch_bytes_, *stream_);

  model_depth_.create(height, width, CV_32FC1);
  sky_mask_.create(height, width, CV_8UC1);
  depth_image_.create(src_height_, src_width_, CV_32FC1);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    model_depth_.data, depth_scaled_d_.get(), plane_size * sizeof(float),
    cudaMemcpyDeviceToHost, *stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    sky_mask_.data, sky_mask_d_.get(), plane_size, cudaMemcpyDeviceToHost, *stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    depth_image_.data, depth_full_d_.get(), full_size * sizeof(float),
    cudaMemcpyDeviceToHost, *stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));

  cv::Mat colorized = rgb_image;
  if (!colorized.empty() &&
      (colorized.rows != depth_image_.rows || colorized.cols != depth_image_.cols)) {
    cv::resize(colorized, colorized, depth_image_.size(), 0, 0, cv::INTER_LINEAR);
  }

  buildPointCloud(camera_info, downsample_factor, colorized);
}


void TensorRTDepthAnything::buildPointCloud(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor,
  const cv::Mat & rgb_image)
{

  // Resize color to match model output if provided.
  cv::Mat color;
  if (!rgb_image.empty()) {
    if (rgb_image.type() == CV_8UC3) {
      color = rgb_image;
    } else {
      rgb_image.convertTo(color, CV_8UC3);
    }
    if (color.rows != model_depth_.rows || color.cols != model_depth_.cols) {
      cv::resize(color, color, model_depth_.size(), 0, 0, cv::INTER_LINEAR);
    }
  }

  // Scale intrinsics to model output resolution.
  sensor_msgs::msg::CameraInfo cam_scaled = camera_info;
  cam_scaled.k[0] = camera_info.k[0] * scale_x_;
  cam_scaled.k[4] = camera_info.k[4] * scale_y_;
  cam_scaled.k[2] = camera_info.k[2] * scale_x_;
  cam_scaled.k[5] = camera_info.k[5] * scale_y_;
  cam_scaled.width = model_depth_.cols;
  cam_scaled.height = model_depth_.rows;

  const std::string frame_id =
    camera_info.header.frame_id.empty() ? "camera_link" : camera_info.header.frame_id;

  depthImageToPointCloud(
    model_depth_, cam_scaled, point_cloud_, frame_id, downsample_factor, color, sky_mask_);

  // Preserve original timestamp
  point_cloud_.header.stamp = camera_info.header.stamp;
}
const cv::Mat& TensorRTDepthAnything::getDepthImage() const
{
  return depth_image_;
}

const sensor_msgs::msg::PointCloud2& TensorRTDepthAnything::getPointCloud() const
{
  return point_cloud_;
}

void TensorRTDepthAnything::printProfiling()
{
  trt_common_->printProfiling();
}

} // namespace depth_anything_v3
