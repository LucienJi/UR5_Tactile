cmake_minimum_required(VERSION 3.0.2)
project(ur5_camera_calib)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    tf
    tf2_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

