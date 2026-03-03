# Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(depth_anything_v3)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV 4 REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_ONNXPARSER})

# Common utilities are included in the include/ and src/ directories

# Include directories
include_directories(
  include
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Create TensorRT depth inference library
add_library(tensorrt_depth_anything SHARED
  src/tensorrt_depth_anything/tensorrt_depth_anything.cpp
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
)

target_link_libraries(tensorrt_depth_anything
  ${TENSORRT_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
)

ament_target_dependencies(tensorrt_depth_anything
  rclcpp
  sensor_msgs
  cv_bridge
)

# Create Depth Anything v3 node library
add_library(depth_anything_v3_node SHARED
  src/depth_anything_v3_node/depth_anything_v3_node.cpp
)

target_link_libraries(depth_anything_v3_node
  tensorrt_depth_anything
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(depth_anything_v3_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

# Register component
rclcpp_components_register_nodes(depth_anything_v3_node "depth_anything_v3::DepthAnythingV3Node")

# Create executable
add_executable(depth_anything_v3_main
  src/main.cpp
)

target_link_libraries(depth_anything_v3_main
  depth_anything_v3_node
)

ament_target_dependencies(depth_anything_v3_main
  rclcpp
)

# Create generate_engines executable
add_executable(generate_engines
  src/generate_engines.cpp
  src/tensorrt_common/tensorrt_common.cpp
  src/tensorrt_common/simple_profiler.cpp
)

target_include_directories(generate_engines PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

target_compile_definitions(generate_engines PRIVATE DEPTH_ANYTHING_STANDALONE)

target_link_libraries(generate_engines
  ${TENSORRT_LIBRARY}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
)

# Install targets
install(TARGETS
  tensorrt_depth_anything
  depth_anything_v3_node
  depth_anything_v3_main
  generate_engines
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_libraries(tensorrt_depth_anything depth_anything_v3_node)
ament_export_include_directories(include)
ament_package()
