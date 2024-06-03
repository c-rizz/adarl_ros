#ifndef JOINT_MONITOR_HPP_20200528
#define JOINT_MONITOR_HPP_20200528

#include <string>
#include <memory>
#include "gazebo/sensors/sensors.hh"
#include "gazebo/common/common.hh"
#include "utils.hpp"
#include "gazebo_gym_env_plugin/JointInfo.h"
#include "gazebo/physics/physics.hh"


class JointMonitor
{
private:
    /**
     * Helper class that wraps a Gazebo Joint
     */
    class MonitoredJoint
    {
    public:
      std::vector<double> position;
      std::vector<double> velocity;
      std::vector<double> effort;
      double smoothing_alpha;
      std::pair<std::string,std::string> name;
      gazebo::physics::WorldPtr world;

      /**
       * @brief Construct a new Monitored Joint object
       * 
       * @param joint Gazebo joint to monitor
       * @param smoothing_alpha Exponential smoothing factor that will be applied
       */
      MonitoredJoint(std::pair<std::string,std::string> joint, gazebo::physics::WorldPtr world, double smoothingAlpha);


      /**
       * @brief Updates the recoded joint state applying the exponential smoothing.
       * 
       * @return int Zero if successful, nonzero otherwise
       */
      int update();

      /**
       * @brief Resets the recorded joint state to the current joint state.
       * 
       * @return int Zero if successful, nonzero otherwise
       */
      int reset();  
    };

    gazebo::physics::WorldPtr world;
    gazebo::event::ConnectionPtr stepConnection; //connected to gazebo step event
    std::map<std::pair<std::string,std::string>, std::shared_ptr<MonitoredJoint>> monitoredJoints;
    double smoothing_alpha;
    
    void searchJoints();
public:

  JointMonitor(gazebo::physics::WorldPtr world, double smoothingAlpha);

  /**
   * @brief Called by Gazebo at each step, updates the monitored joint states
   */
  void stepCallback();

  void getJointState(std::pair<std::string,std::string> jointName, gazebo_gym_env_plugin::JointInfo& jointState);
};


#endif
