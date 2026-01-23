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

#include "depth_anything_v3/depth_anything_v3_node.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <unordered_map>

namespace
{
template <class T>
bool update_param(
  const std::vector<rclcpp::Parameter> & params, const std::string & name, T & value)
{
  const auto itr = std::find_if(
    params.cbegin(), params.cend(),
    [&name](const rclcpp::Parameter & p) { return p.get_name() == name; });

  // Not found
  if (itr == params.cend()) {
    return false;
  }

  value = itr->template get_value<T>();
  return true;
}
} // namespace

namespace depth_anything_v3
{
using namespace std::literals;

DepthAnythingV3Node::DepthAnythingV3Node(const rclcpp::NodeOptions & node_options)
: Node("depth_anything_v3", node_options)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  
  // Parameter
  set_param_res_ =
    this->add_on_set_parameters_callback(std::bind(&DepthAnythingV3Node::onSetParam, this, _1));
  
  node_param_.onnx_path = declare_parameter<std::string>(
    "onnx_path", "models/DA3METRIC-LARGE.fp16-batch1.engine");
  node_param_.precision = declare_parameter<std::string>("precision", "fp16");
  
  // Debug parameters
  node_param_.enable_debug = declare_parameter<bool>("enable_debug", false);
  node_param_.debug_colormap = declare_parameter<std::string>("debug_colormap", "JET");
  node_param_.debug_filepath = declare_parameter<std::string>(
    "debug_filepath", "/tmp/depth_anything_v3_debug/");
  node_param_.write_colormap = declare_parameter<bool>("write_colormap", true);
  node_param_.debug_colormap_min_depth = declare_parameter<double>("debug_colormap_min_depth", 2.0);
  node_param_.debug_colormap_max_depth = declare_parameter<double>("debug_colormap_max_depth", 100.0);
  node_param_.sky_threshold = declare_parameter<double>("sky_threshold", 0.3);
  node_param_.sky_depth_cap = declare_parameter<double>("sky_depth_cap", 200.0);
  
  // Point cloud parameters
  node_param_.point_cloud_downsample_factor = declare_parameter<int>("point_cloud_downsample_factor", 10);
  node_param_.colorize_point_cloud = declare_parameter<bool>("colorize_point_cloud", true);
  RCLCPP_INFO(get_logger(), "Point cloud downsampling factor: %d (publishing every %dth point)", 
    node_param_.point_cloud_downsample_factor, node_param_.point_cloud_downsample_factor);

  RCLCPP_INFO(get_logger(), "Using model file: %s", node_param_.onnx_path.c_str());

  // Synchronized subscribers for compressed image and camera_info
  // Use SensorDataQoS to match typical image publishers (Best Effort reliability)
  sub_compressed_image_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
    this, "~/input/image", rclcpp::SensorDataQoS().get_rmw_qos_profile());
  sub_camera_info_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
    this, "~/input/camera_info", rclcpp::SensorDataQoS().get_rmw_qos_profile());
  
  // Use approximate time synchronizer with 100ms tolerance
  sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
    ApproxSyncPolicy(10), *sub_compressed_image_, *sub_camera_info_);
  sync_->registerCallback(std::bind(&DepthAnythingV3Node::onCompressedImageCameraInfo, this, _1, _2));
  
  RCLCPP_INFO(get_logger(), "Using ApproximateTime synchronizer with queue size 10");

  // Debug subscribers to check if individual topics are arriving
  debug_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
    "~/input/image", rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onCompressedImageDebug, this, std::placeholders::_1));
  debug_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "~/input/camera_info", rclcpp::SensorDataQoS(),
    std::bind(&DepthAnythingV3Node::onCameraInfoDebug, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Depth Anything V3 TensorRT node initialized successfully");
  RCLCPP_INFO(get_logger(), "Waiting for synchronized messages on:");
  RCLCPP_INFO(get_logger(), "  - Image topic: ~/input/image");
  RCLCPP_INFO(get_logger(), "  - Camera info topic: ~/input/camera_info");

  // Publishers
  pub_depth_image_ = create_publisher<sensor_msgs::msg::Image>("~/output/depth_image", 1);
  pub_point_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/output/point_cloud", 1);
  
  if (node_param_.enable_debug) {
    pub_depth_image_debug_ = create_publisher<sensor_msgs::msg::Image>(
      "~/output/depth_image_debug", 1);
  }

  // Init TensorRT model
  std::string calibType = "MinMax";
  int dla = -1;
  bool first = false;
  bool last = false;
  bool prof = false;
  double clip = 0.0;
  tensorrt_common::BuildConfig build_config(calibType, dla, first, last, prof, clip);

  int batch = 1;
  tensorrt_common::BatchConfig batch_config{1, batch / 2, batch};

  bool use_gpu_preprocess = false;
  std::string calibration_images = "calibration_images.txt";
  const size_t workspace_size = (1 << 30);

  tensorrt_depth_anything_ = std::make_shared<TensorRTDepthAnything>(
    node_param_.onnx_path, node_param_.precision, build_config, use_gpu_preprocess,
    calibration_images, batch_config, workspace_size);
  tensorrt_depth_anything_->setSkyThreshold(static_cast<float>(node_param_.sky_threshold));
    
  RCLCPP_INFO(get_logger(), "Finished initializing Depth Anything V3 TensorRT model");
}

void DepthAnythingV3Node::onCompressedImageCameraInfo(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
{

  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  const auto width = in_image_ptr->image.cols;
  const auto height = in_image_ptr->image.rows;

  if (!is_initialized_) {
    RCLCPP_INFO(get_logger(), "Initializing TensorRT preprocessing buffer for %dx%d images", width, height);
    tensorrt_depth_anything_->initPreprocessBuffer(width, height);
    is_initialized_ = true;
    RCLCPP_INFO(get_logger(), "TensorRT preprocessing buffer initialized");
  }

  std::vector<cv::Mat> input_images;
  input_images.push_back(in_image_ptr->image);
  
  auto start = std::chrono::high_resolution_clock::now();
  bool success = tensorrt_depth_anything_->doInference(input_images, *camera_info_msg, node_param_.point_cloud_downsample_factor, node_param_.colorize_point_cloud);
  auto end = std::chrono::high_resolution_clock::now();
  const double inference_time_sec = std::chrono::duration<double>(end - start).count();
  
  if (!success) {
    RCLCPP_ERROR(get_logger(), "Depth Anything V3 inference FAILED!");
    return;
  }

  RCLCPP_DEBUG(this->get_logger(), "Inference completed in %.3f ms", inference_time_sec * 1000.0);

  // Get depth image result
  const cv::Mat& depth_image = tensorrt_depth_anything_->getDepthImage();
  
  // Publish depth image (32FC1 format for accurate depth values)
  cv_bridge::CvImage cv_img_depth;
  cv_img_depth.image = depth_image;
  cv_img_depth.encoding = "32FC1";
  cv_img_depth.header = image_msg->header;
  pub_depth_image_->publish(*cv_img_depth.toImageMsg());

  // Publish point cloud
  sensor_msgs::msg::PointCloud2 point_cloud = tensorrt_depth_anything_->getPointCloud();
  point_cloud.header = image_msg->header;
  pub_point_cloud_->publish(point_cloud);

  // Publish debug depth image if enabled
  if (node_param_.enable_debug && pub_depth_image_debug_) {
    // Normalize depth to [0,255] using configured min/max before visualization
    cv::Mat depth_norm;
    const double min_depth = node_param_.debug_colormap_min_depth;
    const double max_depth = node_param_.debug_colormap_max_depth;
    
    // Clamp and normalize in one step
    cv::Mat clamped;
    cv::max(depth_image, min_depth, clamped);
    cv::min(clamped, max_depth, clamped);
    clamped.convertTo(depth_norm, CV_8UC1, 255.0 / (max_depth - min_depth), -255.0 * min_depth / (max_depth - min_depth));
    
    cv::Mat depth_vis_8u;
    cv::applyColorMap(depth_norm, depth_vis_8u, getColorMapType(node_param_.debug_colormap));
    
    // Add FPS text overlay using rolling average
    static std::vector<double> inference_times;
    inference_times.push_back(inference_time_sec);
    if (inference_times.size() > 20) {
      inference_times.erase(inference_times.begin());
    }

    const double mean_inference_time = std::accumulate(
      inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
    const int fps = static_cast<int>(1.0 / mean_inference_time);
    
    // Extract just the filename from the ONNX path
    const std::string model_name = std::filesystem::path(node_param_.onnx_path).filename().string();
    const std::string fps_text = "Depth Anything V3 - " + model_name + " - FPS: " + std::to_string(fps);
    
    constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double font_scale = 1.0;
    constexpr int thickness = 2;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(fps_text, font_face, font_scale, thickness, &baseline);
    const cv::Point text_org((depth_vis_8u.cols - text_size.width) / 2, depth_vis_8u.rows - 10);
    cv::putText(depth_vis_8u, fps_text, text_org, font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
    
    // Write to file if enabled
    if (node_param_.write_colormap) {
      std::filesystem::create_directories(node_param_.debug_filepath);
      char timestamp_buf[32];
      std::snprintf(timestamp_buf, sizeof(timestamp_buf), "%d%09d", 
                    image_msg->header.stamp.sec, image_msg->header.stamp.nanosec);
      const std::string filename = node_param_.debug_filepath + "depth_image_" + timestamp_buf + ".jpg";
      cv::imwrite(filename, depth_vis_8u);
    }
    
    // Publish debug image
    cv_bridge::CvImage cv_img_debug;
    cv_img_debug.image = depth_vis_8u;
    cv_img_debug.encoding = "bgr8";
    cv_img_debug.header = image_msg->header;
    pub_depth_image_debug_->publish(*cv_img_debug.toImageMsg());
  }
}

rcl_interfaces::msg::SetParametersResult DepthAnythingV3Node::onSetParam(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  try {
    auto & p = node_param_;
    
    // Update all parameters uniformly
    update_param(params, "onnx_path", p.onnx_path);
    update_param(params, "precision", p.precision);
    update_param(params, "enable_debug", p.enable_debug);
    update_param(params, "debug_colormap", p.debug_colormap);
    update_param(params, "debug_filepath", p.debug_filepath);
    update_param(params, "write_colormap", p.write_colormap);
    update_param(params, "debug_colormap_min_depth", p.debug_colormap_min_depth);
    update_param(params, "debug_colormap_max_depth", p.debug_colormap_max_depth);
    update_param(params, "sky_threshold", p.sky_threshold);
    update_param(params, "sky_depth_cap", p.sky_depth_cap);
    update_param(params, "point_cloud_downsample_factor", p.point_cloud_downsample_factor);
    update_param(params, "colorize_point_cloud", p.colorize_point_cloud);
    
    // Apply runtime-configurable model parameters
    if (tensorrt_depth_anything_) {
      tensorrt_depth_anything_->setSkyThreshold(static_cast<float>(p.sky_threshold));
    }
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
    return result;
  }
  result.successful = true;
  result.reason = "success";
  return result;
}

int DepthAnythingV3Node::getColorMapType(const std::string& colormap_name)
{
  static const std::unordered_map<std::string, int> colormap_types = {
    {"JET", cv::COLORMAP_JET},
    {"HOT", cv::COLORMAP_HOT},
    {"COOL", cv::COLORMAP_COOL},
    {"SPRING", cv::COLORMAP_SPRING},
    {"SUMMER", cv::COLORMAP_SUMMER},
    {"AUTUMN", cv::COLORMAP_AUTUMN},
    {"WINTER", cv::COLORMAP_WINTER},
    {"BONE", cv::COLORMAP_BONE},
    {"GRAY", cv::COLORMAP_BONE},
    {"HSV", cv::COLORMAP_HSV},
    {"PARULA", cv::COLORMAP_PARULA},
    {"PLASMA", cv::COLORMAP_PLASMA},
    {"INFERNO", cv::COLORMAP_INFERNO},
    {"VIRIDIS", cv::COLORMAP_VIRIDIS},
    {"MAGMA", cv::COLORMAP_MAGMA},
    {"CIVIDIS", cv::COLORMAP_CIVIDIS}
  };
  
  auto it = colormap_types.find(colormap_name);
  if (it != colormap_types.end()) {
    return it->second;
  }
  RCLCPP_WARN(get_logger(), "Unknown colormap '%s', using JET as default", colormap_name.c_str());
  return cv::COLORMAP_JET;
}

void DepthAnythingV3Node::onCompressedImageDebug(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
{
  static int image_count = 0;
  image_count++;
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
    "[DEBUG] Received compressed image #%d, size: %zu bytes, format: %s, timestamp: %d.%09d", 
    image_count, msg->data.size(), msg->format.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
}

void DepthAnythingV3Node::onCameraInfoDebug(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  static int camera_info_count = 0;
  camera_info_count++;
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    "[DEBUG] Received camera info #%d, frame: %s, timestamp: %d.%09d", 
    camera_info_count, msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
}

} // namespace depth_anything_v3

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(depth_anything_v3::DepthAnythingV3Node)
