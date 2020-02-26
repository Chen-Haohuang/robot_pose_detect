#include <ros/ros.h>
#include "sensor_msgs/JointState.h"  //要发布该类型得数据 需要包含该类型文件
#include <random>
#include <ros/console.h>
#include <string>
#include <sstream>
#include <fstream> 
#include <iostream>

#define JOINT_NUM 6


void saveJointState(const sensor_msgs::JointStateConstPtr& msg){
    
	std::ostringstream ostr;
	ostr << msg->position[0] << " " << msg->position[1] << " " << msg->position[2] << " " << msg->position[3] << " " << msg->position[4] << " " << msg->position[5] << std::endl;
	std::string str = ostr.str();

	std::ofstream ofs;
	ofs.open("real_joint_state.txt", std::ios::out);
	ofs << str << std::endl;
	ofs.close();
	ros::Duration(1).sleep();
}

int main(int argc, char **argv)
{
	ros::init(argc,argv,"joint_state_saver");
	ros::NodeHandle nh;	//实例化句柄，初始化node

	ros::Subscriber sub = nh.subscribe( "/XB6/joint_states", 1, saveJointState);

	ros::spin();

	return 0;
}
