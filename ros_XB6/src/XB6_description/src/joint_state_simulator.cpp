#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>
#include <robot_state_publisher/robot_state_publisher.h>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream> 

int main(int argc, char **argv)
{
  ros::init(argc, argv, "joint_state_simulator");
  ros::NodeHandle nh;

  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 1000);

  ros::Rate loop_rate(1.0);
  // message declarations
  sensor_msgs::JointState joint_state;

  srand(unsigned(time(NULL)));

  while (ros::ok())
  {

    joint_state.header.stamp = ros::Time::now();
    joint_state.name.resize(6);
    joint_state.position.resize(6);
    
    std::ifstream ifs;
    ifs.open("simulate_joint_state.txt", std::ios::in);
    ifs >> joint_state.position[0] >> joint_state.position[1] >> joint_state.position[2] >> joint_state.position[3] >> joint_state.position[4];
    for(int i=0;i<6;i++){
      joint_state.name[i] = "joint"+(std::to_string(i+1));
    }

    ros::spinOnce();

    joint_pub.publish(joint_state);
    //ros::Duration(1.0).sleep();
    loop_rate.sleep();
  }

  return 0;
}
