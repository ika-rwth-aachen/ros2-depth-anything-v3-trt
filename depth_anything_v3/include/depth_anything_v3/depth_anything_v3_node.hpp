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

#ifndef DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_
#define DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <image_transport/subscriber_filter.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "depth_anything_v3/tensorrt_depth_anything.hpp"

namespace depth_anything_v3
{
class DepthAnythingV3Node : public rclcpp::Node
{
public:
  explicit DepthAnythingV3Node(const rclcpp::NodeOptions & node_options);

  struct NodeParam
  {
    std::string onnx_path{};
    std::string precision{};
    bool enable_debug{};
    std::string debug_colormap{};
    std::string debug_filepath{};
    bool write_colormap{};
    double debug_colormap_min_depth{};  // Minimum depth value for colormap visualization
    double debug_colormap_max_depth{};  // Maximum depth value for colormap visualization
    double sky_threshold{};             // Threshold for sky classification
    double sky_depth_cap{};             // Cap for sky depth fill-in
    int point_cloud_downsample_factor{};  // Only publish every Nth point (1 = no downsampling)
    bool colorize_point_cloud{};  // Add RGB colors from input image to point cloud
  };

private:
  // Synchronized subscribers for image (via image_transport) and camera_info
  image_transport::SubscriberFilter sub_image_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> sub_camera_info_;
  
  // Use approximate time synchronizer for more flexible timing
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo> ApproxSyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> sync_;
  
  // Debug subscribers (separate from sync)
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr debug_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr debug_camera_info_sub_;

  // Callbacks
  void onImageCameraInfo(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg);
    
  // Debug callbacks for individual topics
  void onImageDebug(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void onCameraInfoDebug(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_point_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_image_debug_;

  // Helper methods
  int getColorMapType(const std::string& colormap_name);

  // Parameter Server
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  rcl_interfaces::msg::SetParametersResult onSetParam(
    const std::vector<rclcpp::Parameter> & params);

  // Parameter
  NodeParam node_param_{};

  // Core
  std::shared_ptr<TensorRTDepthAnything> tensorrt_depth_anything_;
  bool is_initialized_ = false;
};

} // namespace depth_anything_v3

#endif // DEPTH_ANYTHING_V3__DEPTH_ANYTHING_V3_NODE_HPP_
