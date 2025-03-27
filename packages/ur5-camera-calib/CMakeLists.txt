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

catkin_install_python(PROGRAMS
  scripts/intrinsic_calibration.py
  scripts/major_camera_calibration.py
  scripts/movable_camera_calibration.py
  scripts/launch_cameras.py
  scripts/identify_cameras.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 