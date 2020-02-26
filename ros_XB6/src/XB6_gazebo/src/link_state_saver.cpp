#include <ros/ros.h>
#include "gazebo_msgs/LinkStates.h"  //要发布该类型得数据 需要包含该类型文件
#include <random>
#include <ros/console.h>
#include <string>

#define JOINT_NUM 6

size_t count = 10000;

void saveLinkState(const gazebo_msgs::LinkStates::ConstPtr& linkstates_msg){
    std::string pos_str = std::to_string(count);
    for(int i=4;i<10;i++){
	pos_str += " | " + std::to_string(linkstates_msg->pose[i].position.x) + "," + std::to_string(linkstates_msg->pose[i].position.y) + "," + std::to_string(linkstates_msg->pose[i].position.z-0.05);
    }
    ROS_INFO("%s",pos_str.c_str());
    count ++;
    ros::Duration(20.0).sleep();
}

int main(int argc, char **argv)
{
    ros::init(argc,argv,"link_state_saver");
    ros::NodeHandle nh;	//实例化句柄，初始化node

    ros::Subscriber sub = nh.subscribe( "/gazebo/link_states", 1, saveLinkState);


    //ros::Rate loop_rate(0.2);//定义发布的频率，1HZ

    ros::Duration(10.0).sleep();

    ros::spin();

    return 0;
}
