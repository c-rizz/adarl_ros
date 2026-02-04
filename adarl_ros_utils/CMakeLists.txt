cmake_minimum_required(VERSION 3.0.2)
project(adarl_ros_utils)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -Wall )

# Conditionally set the dependencies list
set(MOVEIT_DEPENDS)
option(WITH_MOVEIT "Compile moveit-related components" TRUE)
if(WITH_MOVEIT)
  message(STATUS "Will compile moveit-related components.")
  list(APPEND MOVEIT_DEPENDS
    rviz_visual_tools
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_perception
  )
endif()

set(ROS_CONTROL_DEPENDS)
option(WITH_ROS_CONTROL "Compile ros_control-related components" TRUE)
if(WITH_ROS_CONTROL)
  message(STATUS "Will compile ros_control-related components.")
  list(APPEND ROS_CONTROL_DEPENDS
    controller_interface
    hardware_interface
    realtime_tools
  )
endif()

set(CATKIN_DEPENDS_ALL)
list(APPEND CATKIN_DEPENDS_ALL
  roscpp
  rospy
  std_msgs
  interactive_markers
  actionlib_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  tf2_kdl
  tf2_msgs
  tf2_py
  kdl_parser
  kdl_conversions
  geometry_msgs
  pluginlib
  ${ROS_CONTROL_DEPENDS}
  ${MOVEIT_DEPENDS}
)

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_DEPENDS_ALL}
)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(orocos_kdl REQUIRED)
find_package(urdfdom_headers REQUIRED)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  LinkState.msg
  LinkStates.msg
)

## Generate services in the 'srv' folder
add_service_files(
	FILES
	GetEePose.srv
	GetJointState.srv
	AddCollisionBox.srv
	ClearCollisionObjects.srv
)

## Generate actions in the 'action' folder
add_action_files(
	FILES
	MoveToEePose.action
	MoveToJointPose.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES adarl_ros_utils
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    ${CATKIN_DEPENDS_ALL}
  DEPENDS
    EIGEN3
	  orocos_kdl
	  urdfdom_headers
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS} ${orocos_kdl_INCLUDE_DIRS} ${urdfdom_headers_INCLUDE_DIRS}
)

link_directories(${orocos_kdl_LIBRARY_DIRS})

if(WITH_MOVEIT)
  add_executable(move_helper src/move_helper.cpp)
  add_dependencies(move_helper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  target_link_libraries(move_helper ${catkin_LIBRARIES} ${Boost_LIBRARIES})
endif()

# Declare a C++ library
add_library(adarl_ros_utils
# src/GravityCompensatedEffortController.cpp
src/ForwardKinematicsComputer.cpp
src/KdlHelper.cpp
# src/CartesianPositionController.cpp
)
if (WITH_ROS_CONTROL)
add_library(adarl_ros_utils
src/LinkStatePublisherController.cpp
)
endif()

add_dependencies(adarl_ros_utils
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(adarl_ros_utils PUBLIC
  ${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES}
)

target_include_directories(adarl_ros_utils SYSTEM PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
)
target_include_directories(adarl_ros_utils PUBLIC
  include
)

## Installation
# Install headers
install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS adarl_ros_utils
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
# install(DIRECTORY launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES adarl_ros_utils_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
