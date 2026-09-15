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
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <cstring>

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
}  // anonymous namespace

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
  if (!trt_common_->isInitialized()) {
    throw std::runtime_error("Failed to initialize TensorRT engine from: " + model_path);
  }

  // Resolve the IO tensor list and pre-allocate every input/output buffer once.
  resolveTensors();
}

void TensorRTDepthAnything::resolveTensors()
{
  auto * engine = trt_common_->getEngine();
  const int32_t nb = trt_common_->getNbIOTensors();

  input_index_ = -1;
  depth_index_ = -1;
  sky_index_ = -1;

  for (int32_t i = 0; i < nb; ++i) {
    const char * raw_name = engine->getIOTensorName(i);
    if (raw_name == nullptr) {
      continue;
    }
    const std::string name(raw_name);
    if (engine->getTensorIOMode(raw_name) == nvinfer1::TensorIOMode::kINPUT) {
      input_index_ = i;
      input_tensor_name_ = name;
    } else if (name == "depth") {
      depth_index_ = i;
      depth_tensor_name_ = name;
    } else if (name == "sky") {
      sky_index_ = i;
      sky_tensor_name_ = name;
    }
  }

  // Fallbacks for engines whose output tensors are not named "depth"/"sky".
  if (input_index_ < 0) {
    input_index_ = 0;
    input_tensor_name_ = engine->getIOTensorName(input_index_);
  }
  if (depth_index_ < 0) {
    for (int32_t i = 0; i < nb; ++i) {
      if (i != input_index_) {
        depth_index_ = i;
        depth_tensor_name_ = engine->getIOTensorName(i);
        break;
      }
    }
  }
  if (sky_index_ < 0) {
    throw std::runtime_error(
      "Expected TensorRT engine to expose 'sky' output tensor, but none was found");
  }
  if (depth_index_ < 0 || input_index_ < 0) {
    throw std::runtime_error("Failed to resolve TensorRT engine input/output tensors");
  }

  const auto input_dims = trt_common_->getBindingDimensions(input_index_);
  input_height_ = input_dims.nbDims > 2 ? input_dims.d[2] : 0;
  input_width_ = input_dims.nbDims > 3 ? input_dims.d[3] : 0;

  input_elem_num_ = volumeFromDims(input_dims, batch_size_);
  depth_elem_num_ = volumeFromDims(trt_common_->getBindingDimensions(depth_index_), batch_size_);
  sky_elem_num_ = volumeFromDims(trt_common_->getBindingDimensions(sky_index_), batch_size_);

  input_d_ = cuda_utils::make_unique<float[]>(input_elem_num_);
  depth_d_ = cuda_utils::make_unique<float[]>(depth_elem_num_);
  sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);

  extra_output_buffers_.clear();
  extra_tensor_names_.clear();
  for (int32_t i = 0; i < nb; ++i) {
    if (i == input_index_ || i == depth_index_ || i == sky_index_) {
      continue;
    }
    const char * raw_name = engine->getIOTensorName(i);
    extra_tensor_names_.emplace_back(raw_name != nullptr ? raw_name : std::string());
    const size_t elem_count =
      volumeFromDims(trt_common_->getBindingDimensions(i), batch_size_);
    extra_output_buffers_.push_back(cuda_utils::make_unique<float[]>(elem_count));
  }

  bound_tensor_addresses_.clear();
  bound_tensor_addresses_.emplace_back(input_tensor_name_, static_cast<void *>(input_d_.get()));
  bound_tensor_addresses_.emplace_back(depth_tensor_name_, static_cast<void *>(depth_d_.get()));
  bound_tensor_addresses_.emplace_back(sky_tensor_name_, static_cast<void *>(sky_d_.get()));
  for (size_t j = 0; j < extra_tensor_names_.size(); ++j) {
    bound_tensor_addresses_.emplace_back(
      extra_tensor_names_[j], static_cast<void *>(extra_output_buffers_[j].get()));
  }

  // Addresses are stable for the lifetime of the engine (fixed shapes), so bind
  // them once instead of re-evaluating the tensor list on every frame.
  auto * context = trt_common_->getContext();
  for (const auto & [name, ptr] : bound_tensor_addresses_) {
    if (!name.empty()) {
      context->setTensorAddress(name.c_str(), ptr);
    }
  }
}

void TensorRTDepthAnything::maybeSetInputShape()
{
  if (input_index_ < 0) {
    return;
  }
  const auto current = trt_common_->getBindingDimensions(input_index_);
  if (current.nbDims > 0 && current.d[0] == -1) {
    auto dims = current;
    dims.d[0] = batch_size_;
    trt_common_->setBindingDimensions(input_index_, dims);
  }
}

void TensorRTDepthAnything::initPreprocessBuffer(int width, int height)
{
  if (width <= 0 || height <= 0) {
    return;
  }
  src_width_ = width;
  src_height_ = height;
  scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
  scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);

  const size_t image_size = static_cast<size_t>(src_width_) * src_height_ * 3;  // RGB
  image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(image_size, cudaHostAllocDefault);
  image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(image_size);
  image_buf_bytes_ = image_size;
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

  colorize_point_cloud_ = colorize_pointcloud;

  // Preprocess
  preprocess(images);

  // Run inference
  if (!infer()) {
    return false;
  }

  // Postprocess (depth image + point cloud)
  postprocess(camera_info, std::max(1, downsample_factor));

  return true;
}

void TensorRTDepthAnything::preprocess(const std::vector<cv::Mat> & images)
{
  const cv::Mat & src_image = images[0];
  const int width = src_image.cols;
  const int height = src_image.rows;
  if (width <= 0 || height <= 0 || src_image.data == nullptr) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TensorRTDepthAnything"), "Invalid source image dimensions.");
    return;
  }

  // Re-initialise buffers when the camera resolution changes (exposure changes,
  // camera reboot, ...). Previously this only ever happened on the first frame.
  if (src_width_ != width || src_height_ != height) {
    src_width_ = width;
    src_height_ = height;
    scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
    scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);
  }

  const size_t row_bytes = static_cast<size_t>(width) * 3;
  const size_t src_bytes = row_bytes * static_cast<size_t>(height);

  // Stage the frame in a pinned buffer once and do a single fast asynchronous
  // upload. cuMemcpy2DAsync from unpinned cv::Mat memory would block on the
  // host-sized copy; pinning lets the copy overlap with GPU work.
  if (!image_buf_h_ || src_bytes != image_buf_bytes_) {
    image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(src_bytes, cudaHostAllocDefault);
    image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(src_bytes);
    image_buf_bytes_ = src_bytes;
  }

  const unsigned char * src_data = src_image.data;
  const size_t src_step = src_image.step;
  unsigned char * pinned = image_buf_h_.get();
  if (src_step == row_bytes) {
    std::memcpy(pinned, src_data, src_bytes);
  } else {
    // A ROS image may declare a step larger than width * channels. The GPU
    // kernels expect packed rows, so drop the padding here.
    for (int r = 0; r < height; ++r) {
      std::memcpy(pinned + r * row_bytes, src_data + r * src_step, row_bytes);
    }
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    image_buf_d_.get(), pinned, src_bytes, cudaMemcpyHostToDevice, *stream_));

  maybeSetInputShape();
  launchPreprocess(
    image_buf_d_.get(), width, height,
    input_d_.get(), input_width_, input_height_, *stream_);
}

bool TensorRTDepthAnything::infer()
{
  maybeSetInputShape();
  auto * context = trt_common_->getContext();
  for (const auto & [name, ptr] : bound_tensor_addresses_) {
    if (!name.empty()) {
      context->setTensorAddress(name.c_str(), ptr);
    }
  }

  if (!trt_common_->enqueueV3(*stream_)) {
    return false;
  }

  return true;
}

void TensorRTDepthAnything::postprocess(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor)
{
  if (depth_index_ < 0 || src_width_ <= 0 || src_height_ <= 0) {
    return;
  }

  const auto output_dims = trt_common_->getBindingDimensions(depth_index_);
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

  // Copy the full-resolution depth image (published) back to the host.
  depth_image_.create(src_height_, src_width_, CV_32FC1);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    depth_image_.data, depth_full_d_.get(), full_size * sizeof(float),
    cudaMemcpyDeviceToHost, *stream_));

  // The point cloud is generated entirely on the device from the model-size
  // depth and sky mask; no model-size host copies are needed.
  buildPointCloud(camera_info, downsample_factor);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));

  // Hand the packed points over to the ROS message.
  if (point_cloud_h_ && !point_cloud_.data.empty()) {
    std::memcpy(point_cloud_.data.data(), point_cloud_h_.get(), point_cloud_.data.size());
  }
}

void TensorRTDepthAnything::buildPointCloud(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor)
{
  if (post_width_ <= 0 || post_height_ <= 0 || src_width_ <= 0 || src_height_ <= 0) {
    return;
  }

  const int out_width = (post_width_ + downsample_factor - 1) / downsample_factor;
  const int out_height = (post_height_ + downsample_factor - 1) / downsample_factor;
  const size_t stride = colorize_point_cloud_ ? 15u : 12u;
  const size_t total_bytes = static_cast<size_t>(out_width) * out_height * stride;

  if (!point_cloud_d_ || point_capacity_bytes_ < total_bytes) {
    point_cloud_d_ = cuda_utils::make_unique<uint8_t[]>(total_bytes);
    point_cloud_h_ =
      cuda_utils::make_unique_host<uint8_t[]>(total_bytes, cudaHostAllocDefault);
    point_capacity_bytes_ = total_bytes;
  }

  const size_t model_pixels = static_cast<size_t>(post_width_) * post_height_;
  if (colorize_point_cloud_) {
    if (!color_model_d_) {
      color_model_d_ = cuda_utils::make_unique<uint8_t[]>(model_pixels * 3);
    }
    launchResizeColor(
      image_buf_d_.get(), src_width_, src_height_,
      color_model_d_.get(), post_width_, post_height_, *stream_);
  }

  const double fx = camera_info.k[0] * scale_x_;
  const double fy = camera_info.k[4] * scale_y_;
  const double cx = camera_info.k[2] * scale_x_;
  const double cy = camera_info.k[5] * scale_y_;

  launchBuildPointCloud(
    depth_scaled_d_.get(), sky_mask_d_.get(),
    colorize_point_cloud_ ? color_model_d_.get() : nullptr,
    post_width_, post_height_,
    static_cast<float>(fx), static_cast<float>(fy),
    static_cast<float>(cx), static_cast<float>(cy),
    downsample_factor, colorize_point_cloud_,
    point_cloud_d_.get(), out_width, out_height, *stream_);

  point_cloud_.header.frame_id =
    camera_info.header.frame_id.empty() ? "camera_link" : camera_info.header.frame_id;
  point_cloud_.height = out_height;
  point_cloud_.width = out_width;
  point_cloud_.is_dense = false;
  point_cloud_.is_bigendian = false;
  {
    sensor_msgs::PointCloud2Modifier modifier(point_cloud_);
    if (colorize_point_cloud_) {
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    } else {
      modifier.setPointCloud2FieldsByString(1, "xyz");
    }
  }
  point_cloud_.data.resize(total_bytes);

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    point_cloud_h_.get(), point_cloud_d_.get(), total_bytes,
    cudaMemcpyDeviceToHost, *stream_));

  point_cloud_.header.stamp = camera_info.header.stamp;
}

const cv::Mat & TensorRTDepthAnything::getDepthImage() const
{
  return depth_image_;
}

const sensor_msgs::msg::PointCloud2 & TensorRTDepthAnything::getPointCloud() const
{
  return point_cloud_;
}

void TensorRTDepthAnything::printProfiling()
{
  trt_common_->printProfiling();
}

}  // namespace depth_anything_v3