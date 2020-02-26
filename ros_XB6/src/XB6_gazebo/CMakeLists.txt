cmake_minimum_required(VERSION 2.8.3)
project(XB6_gazebo)

find_package(catkin REQUIRED COMPONENTS camera_calibration_parsers cv_bridge image_transport roscpp)
include_directories(
${catkin_INCLUDE_DIRS}
)

catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


add_executable(joints_controller_cmd_publisher src/joints_controller_cmd_publisher.cpp )
add_dependencies(joints_controller_cmd_publisher mastering_ros_demo_pkg_generate_messages_cpp)
target_link_libraries(joints_controller_cmd_publisher ${catkin_LIBRARIES})

add_executable(camera1_depth_image_saver src/camera1_depth_image_saver.cpp )
target_link_libraries(camera1_depth_image_saver ${catkin_LIBRARIES})

add_executable(camera2_depth_image_saver src/camera2_depth_image_saver.cpp )
target_link_libraries(camera2_depth_image_saver ${catkin_LIBRARIES})

add_executable(camera3_depth_image_saver src/camera3_depth_image_saver.cpp )
target_link_libraries(camera3_depth_image_saver ${catkin_LIBRARIES})

add_executable(camera1_rgb_image_saver src/camera1_rgb_image_saver.cpp )
target_link_libraries(camera1_rgb_image_saver ${catkin_LIBRARIES})

add_executable(camera2_rgb_image_saver src/camera2_rgb_image_saver.cpp )
target_link_libraries(camera2_rgb_image_saver ${catkin_LIBRARIES})

add_executable(camera3_rgb_image_saver src/camera3_rgb_image_saver.cpp )
target_link_libraries(camera3_rgb_image_saver ${catkin_LIBRARIES})

add_executable(link_state_saver src/link_state_saver.cpp )
target_link_libraries(link_state_saver ${catkin_LIBRARIES})

add_executable(joint_state_saver src/joint_state_saver.cpp )
target_link_libraries(joint_state_saver ${catkin_LIBRARIES})
