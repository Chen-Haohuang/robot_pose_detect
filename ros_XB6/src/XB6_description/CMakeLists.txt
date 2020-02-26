cmake_minimum_required(VERSION 2.8.3)
project(XB6_description)

find_package(catkin REQUIRED COMPONENTS camera_calibration_parsers cv_bridge image_transport roscpp rviz)
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})
catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

add_executable(joint_state_simulator src/joint_state_simulator.cpp )
target_link_libraries(joint_state_simulator ${catkin_LIBRARIES})
