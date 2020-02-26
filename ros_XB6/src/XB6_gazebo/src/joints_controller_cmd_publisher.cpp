#include <ros/ros.h>
#include "std_msgs/Float64.h"  //要发布该类型得数据 需要包含该类型文件
#include <random>
#include <ros/console.h>

#define randomDouble(a,b) a+(b-a)*(rand()/double(RAND_MAX))
#define randomMove(x) x+((rand()/double(RAND_MAX))-0.5)*0.2
#define JOINT_NUM 6

int main(int argc, char **argv)
{
    ros::init(argc,argv,"joints_controller_cmd_publisher");
    ros::NodeHandle nh[JOINT_NUM] ;	//实例化句柄，初始化node
    std_msgs::Float64 msg[JOINT_NUM]; //创建该类型消息
    
    size_t count_ = 0;
    
    const double joint_position_lower[6] = {-2.9670597284,-1.6580627894,-3.6651914292,-2.9670597284,-2.3561944902,-6.2831853072};
    const double joint_position_upper[6] = {2.9670597284,2.3561944902,1.1519173063,2.9670597284,2.3561944902,6.2831853072};

    for(int i=0;i<JOINT_NUM;i++){
        msg[i].data = 0.0; // 初始化消息
    }
    

    ros::Publisher pub[6];
    for(int i=0;i<JOINT_NUM;i++){
        pub[i] = nh[i].advertise<std_msgs::Float64>( "/XB6/joint"+std::to_string(i+1)+"_position_controller/command", 1);
    }

    //ros::Rate loop_rate(10);//定义发布的频率，1HZ

    srand(unsigned(time(NULL)));

    while (ros::ok())
    {
	if(count_ == 10000){
		ROS_INFO("10000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000");
	}
	else{
		for(int i=0;i<JOINT_NUM;i++){
		    msg[i].data = randomDouble(joint_position_lower[i],joint_position_upper[i]);
		    //msg[i].data = randomMove(msg[i].data);
		}

		//ROS_INFO("%ld | %lf | %lf | %lf | %lf | %lf | %lf", count_, msg[0].data, msg[1].data, msg[2].data, msg[3].data, msg[4].data, msg[5].data);

		ros::spinOnce();

		for(int i=0;i<JOINT_NUM;i++){
		    pub[i].publish(msg[i]); //发布消息
		}
	}
	//count_++;
	ros::Duration(20.0).sleep();
        //loop_rate.sleep();//根据前面的定义的loop_rate
    }
    return 0;
}
