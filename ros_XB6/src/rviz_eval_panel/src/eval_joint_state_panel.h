#ifndef RVIZ_EVAL_JOINT_STATE_PANEL_H
#define RVIZ_EVAL_JOINT_STATE_PANEL_H
#include <stdio.h>
#include "ros/ros.h"
#include <rviz/panel.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/JointState.h>

class QLineEdit;
class QComboBox;
class QCheckBox;
class QPushButton;
class QHBoxLayout;
class QWidget;

namespace rviz_eval_panel
{

class VisualizationManager;
class Display;

/** A place to edit properties of all of the Tools.
 */
class EvalJointStatesPanel: public rviz::Panel
{
Q_OBJECT
public:
  EvalJointStatesPanel( QWidget* parent = 0 );
  virtual ~EvalJointStatesPanel() {}

  virtual void onInitialize();

  /** @brief Load configuration data, specifically the PropertyTreeWidget view settings. */
  virtual void load( const rviz::Config& config );

  /** @brief Save configuration data, specifically the PropertyTreeWidget view settings. */
  virtual void save( rviz::Config config ) const;

public Q_SLOTS:
protected Q_SLOTS:
  void catchJointState();
  void detectJointState();
  void simulate_1_JointState();
  void simulate_2_JointState();

protected:
   /** Create, configure, and return a single label for showing a time value. */
  QLineEdit* makeJointStateLabel();

  /** Fill a single time label with the given time value (in seconds). */
  void fillJointStateLabel( QLineEdit* label, double joint_state );

  QLineEdit* joint1_real_label_;
  QLineEdit* joint2_real_label_;
  QLineEdit* joint3_real_label_;
  QLineEdit* joint4_real_label_;
  QLineEdit* joint5_real_label_;
  QLineEdit* joint1_detect_label_;
  QLineEdit* joint2_detect_label_;
  QLineEdit* joint3_detect_label_;
  QLineEdit* joint4_detect_label_1;
  QLineEdit* joint4_detect_label_2;
  QLineEdit* joint5_detect_label_1;
  QLineEdit* joint5_detect_label_2;
  QLineEdit* joint1_error_label_;
  QLineEdit* joint2_error_label_;
  QLineEdit* joint3_error_label_;
  QLineEdit* joint4_error_label_1;
  QLineEdit* joint4_error_label_2;
  QLineEdit* joint5_error_label_1;
  QLineEdit* joint5_error_label_2;
  QPushButton* catch_button_;
  QPushButton* detect_button_;
  QPushButton* simu_button_1;
  QPushButton* simu_button_2;

  QWidget* detect_widget_;
  QWidget* joint_label_widget_;
  QWidget* joint_real_widget_;
  QWidget* joint_detect_widget_;
  QWidget* joint_error_widget_;

  image_transport::Publisher detect_imgae1_publisher;
  image_transport::Publisher detect_imgae2_publisher;
  image_transport::Publisher detect_imgae3_publisher;

  ros::Publisher detect_command_publisher;

  bool catch_lock = false;
  
  double real_joint_state[5];
  double detect_joint_state[7];
};

} // namespace rviz

#endif

