#include <stdio.h>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QHBoxLayout>
#include <QButtonGroup>
#include <QCheckBox>
#include <QSlider>
#include <QComboBox>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream> 

#include <opencv2/video.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/JointState.h>

#include "ros/ros.h"
#include "std_msgs/Bool.h"

#include "eval_joint_state_panel.h"

namespace rviz_eval_panel
{

EvalJointStatesPanel::EvalJointStatesPanel( QWidget* parent )
  : rviz::Panel( parent )
{
  joint1_real_label_ = makeJointStateLabel();
  joint2_real_label_ = makeJointStateLabel();
  joint3_real_label_ = makeJointStateLabel();
  joint4_real_label_ = makeJointStateLabel();
  joint5_real_label_ = makeJointStateLabel();
  joint1_detect_label_ = makeJointStateLabel();
  joint2_detect_label_ = makeJointStateLabel();
  joint3_detect_label_ = makeJointStateLabel();
  joint4_detect_label_1 = makeJointStateLabel();
  joint4_detect_label_2 = makeJointStateLabel();
  joint5_detect_label_1 = makeJointStateLabel();
  joint5_detect_label_2 = makeJointStateLabel();
  joint1_error_label_ = makeJointStateLabel();
  joint2_error_label_ = makeJointStateLabel();
  joint3_error_label_ = makeJointStateLabel();
  joint4_error_label_1 = makeJointStateLabel();
  joint4_error_label_2 = makeJointStateLabel();
  joint5_error_label_1 = makeJointStateLabel();
  joint5_error_label_2 = makeJointStateLabel();

  catch_button_ = new QPushButton( "catch" );
  catch_button_->setToolTip("catch Robot State.");
  detect_button_ = new QPushButton( "detect" );
  detect_button_->setToolTip("Detect Robot State.");
  simu_button_1 = new QPushButton( "simulate1" );
  simu_button_1->setToolTip("simulate Robot State.");
  simu_button_2 = new QPushButton( "simulate2" );
  simu_button_2->setToolTip("simulate Robot State.");
  
  QGridLayout* layout = new QGridLayout();
  layout->setSpacing(10);//设置间距

  layout->addWidget(catch_button_, 0, 0);
  layout->addWidget(detect_button_, 0, 1);
  layout->addWidget(simu_button_1, 0, 2);
  layout->addWidget(simu_button_2, 0, 3);
  
  layout->addWidget( new QLabel( "joint_index" ) , 1, 0);
  layout->addWidget( new QLabel( "joint1" ), 1, 1);
  layout->addWidget( new QLabel( "joint2" ), 1, 2);
  layout->addWidget( new QLabel( "joint3" ), 1, 3);
  layout->addWidget( new QLabel( "joint4" ), 1, 4);
  layout->addWidget( new QLabel( "joint5" ), 1, 5);
  
  layout->addWidget( new QLabel( "real_state" ), 2, 0);
  layout->addWidget(joint1_real_label_, 2, 1);
  layout->addWidget(joint2_real_label_, 2, 2);
  layout->addWidget(joint3_real_label_, 2, 3);
  layout->addWidget(joint4_real_label_, 2, 4);
  layout->addWidget(joint5_real_label_, 2, 5);

  layout->addWidget( new QLabel( "detect_state" ), 3, 0);
  layout->addWidget(joint1_detect_label_, 3, 1);
  layout->addWidget(joint2_detect_label_, 3, 2);
  layout->addWidget(joint3_detect_label_, 3, 3);
  layout->addWidget(joint4_detect_label_1, 3, 4);
  layout->addWidget(joint5_detect_label_1, 3, 5);
  layout->addWidget(joint4_detect_label_2, 4, 4);
  layout->addWidget(joint5_detect_label_2, 4, 5);

  layout->addWidget( new QLabel( "error" ), 5, 0);
  layout->addWidget(joint1_error_label_, 5, 1);
  layout->addWidget(joint2_error_label_, 5, 2);
  layout->addWidget(joint3_error_label_, 5, 3);
  layout->addWidget(joint4_error_label_1, 5, 4);
  layout->addWidget(joint5_error_label_1, 5, 5);
  layout->addWidget(joint4_error_label_2, 6, 4);
  layout->addWidget(joint5_error_label_2, 6, 5);

  //layout->setContentsMargins( 11, 5, 11, 5 );
  setLayout( layout );

  connect( catch_button_, SIGNAL( clicked()), this, SLOT( catchJointState() ));
  connect( detect_button_, SIGNAL( clicked()), this, SLOT( detectJointState() ));
  connect( simu_button_1, SIGNAL( clicked()), this, SLOT( simulate_1_JointState() ));
  connect( simu_button_2, SIGNAL( clicked()), this, SLOT( simulate_2_JointState() ));
}

void EvalJointStatesPanel::onInitialize()
{
  //ros::init("detect_msg_publisher");
  ros::NodeHandle nh1;
  image_transport::ImageTransport it1(nh1);
  detect_imgae1_publisher = it1.advertise("/XB6/detect_image1", 1);
  ros::NodeHandle nh2;
  image_transport::ImageTransport it2(nh2);
  detect_imgae2_publisher = it2.advertise("/XB6/detect_image2", 1);
  ros::NodeHandle nh3;
  image_transport::ImageTransport it3(nh3);
  detect_imgae3_publisher = it3.advertise("/XB6/detect_image3", 1);

  ros::NodeHandle n1;
  detect_command_publisher = n1.advertise<std_msgs::Bool>("/XB6/detect_command_msg", 1);

  for(int i=0;i<5;i++){
    real_joint_state[i] = 0.0;
  }
  for(int i=0;i<7;i++){
    detect_joint_state[i] = 0.0;
  }
}

void EvalJointStatesPanel::save( rviz::Config config ) const
{
  rviz::Panel::save( config );
}

void EvalJointStatesPanel::load( const rviz::Config& config )
{
  rviz::Panel::load( config );
}

QLineEdit* EvalJointStatesPanel::makeJointStateLabel()
{
  QLineEdit* label = new QLineEdit;
  label->setReadOnly( true );
  return label;
}

void EvalJointStatesPanel::simulate_1_JointState()
{
  std::ostringstream ostr;
  ostr << detect_joint_state[0] << " " << detect_joint_state[1] << " " << detect_joint_state[2] << " " << detect_joint_state[3] << " " << detect_joint_state[5] << std::endl;
  std::string str = ostr.str();

  std::ofstream ofs;
  ofs.open("simulate_joint_state.txt", std::ios::out); 
  ofs << str << std::endl;
  ofs.close();
}

void EvalJointStatesPanel::simulate_2_JointState()
{
  std::ostringstream ostr;
  ostr << detect_joint_state[0] << " " << detect_joint_state[1] << " " << detect_joint_state[2] << " " << detect_joint_state[4] << " " << detect_joint_state[6] << std::endl;
  std::string str = ostr.str();

  std::ofstream ofs;
  ofs.open("simulate_joint_state.txt", std::ios::out); 
  ofs << str << std::endl;
  ofs.close();
}

void EvalJointStatesPanel::detectJointState()
{
  std::remove("detect_result.txt");
  std_msgs::Bool msg;
  msg.data = true;
  detect_command_publisher.publish(msg);
  ros::spinOnce();
  std::ifstream ifs;
  do{
    ifs.open("detect_result.txt", std::ios::in);
    sleep(1);
  }while(!ifs.good());
  ifs >> detect_joint_state[0] >> detect_joint_state[1] >> detect_joint_state[2] >> detect_joint_state[3] >> detect_joint_state[4] >> detect_joint_state[5]  >> detect_joint_state[6];
  ifs.close();
  fillJointStateLabel(joint1_detect_label_, detect_joint_state[0]);
  fillJointStateLabel(joint2_detect_label_, detect_joint_state[1]);
  fillJointStateLabel(joint3_detect_label_, detect_joint_state[2]);
  fillJointStateLabel(joint4_detect_label_1, detect_joint_state[3]);
  fillJointStateLabel(joint4_detect_label_2, detect_joint_state[4]);
  fillJointStateLabel(joint5_detect_label_1, detect_joint_state[5]);
  fillJointStateLabel(joint5_detect_label_2, detect_joint_state[6]);

  fillJointStateLabel(joint1_error_label_, detect_joint_state[0] - real_joint_state[0]);
  fillJointStateLabel(joint2_error_label_, detect_joint_state[1] - real_joint_state[1]);
  fillJointStateLabel(joint3_error_label_, detect_joint_state[2] - real_joint_state[2]);
  fillJointStateLabel(joint4_error_label_1, detect_joint_state[3] - real_joint_state[3]);
  fillJointStateLabel(joint4_error_label_2, detect_joint_state[4] - real_joint_state[3]);
  fillJointStateLabel(joint5_error_label_1, detect_joint_state[5] - real_joint_state[4]);
  fillJointStateLabel(joint5_error_label_2, detect_joint_state[6] - real_joint_state[4]);
}

void EvalJointStatesPanel::catchJointState()
{
  if(catch_lock == true) return;
  catch_lock = true;

  std::ifstream ifs;
  ifs.open("real_joint_state.txt", std::ios::in);
  std::string buf;
  ifs >> real_joint_state[0] >> real_joint_state[1] >> real_joint_state[2] >> real_joint_state[3] >> real_joint_state[4];
  ifs.close();

  cv::Mat rgb_img_1= cv::imread("camera1-rgb-0.png", CV_LOAD_IMAGE_COLOR);
  cv::Mat depth_img_1= cv::imread("camera1-depth-0.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat rgb_img_2= cv::imread("camera2-rgb-0.png", CV_LOAD_IMAGE_COLOR);
  cv::Mat depth_img_2= cv::imread("camera2-depth-0.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat rgb_img_3= cv::imread("camera3-rgb-0.png", CV_LOAD_IMAGE_COLOR);
  cv::Mat depth_img_3= cv::imread("camera3-depth-0.png", CV_LOAD_IMAGE_GRAYSCALE);

  fillJointStateLabel(joint1_real_label_, real_joint_state[0]);
  fillJointStateLabel(joint2_real_label_, real_joint_state[1]);
  fillJointStateLabel(joint3_real_label_, real_joint_state[2]);
  fillJointStateLabel(joint4_real_label_, real_joint_state[3]);
  fillJointStateLabel(joint5_real_label_, real_joint_state[4]);

  for(int i=0;i<rgb_img_1.rows;i++){
    for(int j=0;j<rgb_img_1.cols;j++){
      if(depth_img_1.at<uchar>(i,j) == 0){
	rgb_img_1.at<cv::Vec3b>(i,j)[0] = 0;
	rgb_img_1.at<cv::Vec3b>(i,j)[1] = 0;
	rgb_img_1.at<cv::Vec3b>(i,j)[2] = 0;
      }
    }
  }
  sensor_msgs::ImagePtr msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb_img_1).toImageMsg();
  detect_imgae1_publisher.publish(msg1);

  for(int i=0;i<rgb_img_2.rows;i++){
    for(int j=0;j<rgb_img_2.cols;j++){
      if(depth_img_2.at<uchar>(i,j) == 0){
	rgb_img_2.at<cv::Vec3b>(i,j)[0] = 0;
	rgb_img_2.at<cv::Vec3b>(i,j)[1] = 0;
	rgb_img_2.at<cv::Vec3b>(i,j)[2] = 0;
      }
    }
  }
  sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb_img_2).toImageMsg();
  detect_imgae2_publisher.publish(msg2);

  for(int i=0;i<rgb_img_3.rows;i++){
    for(int j=0;j<rgb_img_3.cols;j++){
      if(depth_img_3.at<uchar>(i,j) == 0){
	rgb_img_3.at<cv::Vec3b>(i,j)[0] = 0;
	rgb_img_3.at<cv::Vec3b>(i,j)[1] = 0;
	rgb_img_3.at<cv::Vec3b>(i,j)[2] = 0;
      }
    }
  }
  sensor_msgs::ImagePtr msg3 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb_img_3).toImageMsg();
  detect_imgae3_publisher.publish(msg3);

  ros::spinOnce();
  cv::imwrite("camera1.png", rgb_img_1);
  cv::imwrite("camera2.png", rgb_img_2);
  cv::imwrite("camera3.png", rgb_img_3);
  catch_lock = false;
}

void EvalJointStatesPanel::fillJointStateLabel( QLineEdit* label, double joint_state )
{
  label->setText( QString::number( joint_state, 'f', 2 ));
}


} // end namespace rviz_eval_panel

// 声明此类是一个rviz的插件
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(rviz_eval_panel::EvalJointStatesPanel,rviz::Panel )
// END_TUTORIAL
