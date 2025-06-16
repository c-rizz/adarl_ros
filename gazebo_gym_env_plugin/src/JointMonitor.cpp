#include "JointMonitor.hpp"

using namespace std;

double getCorrectedForce(gazebo::physics::JointPtr joint, int dof_index)
{
  // Looking into the code GetForce seems to actually return the commanded efforts for the specified DOF. This
  // can be seen in ODEJoint, BulletJoint, DARTJoint and SimbodyJoint. They all return the value of _forceApplied.
  // This actually differs from what is applied to the joint due to the internal velocity damping, which is added to the 
  // joint torque, but is not saved in _forceApplied. In the case of explicit damping this is true because it is aplied directly with
  // SetForceImpl (at least in ODE, https://github.com/gazebosim/gazebo-classic/blob/e4b4d0fb752c7e43e34ab97d0e01a2a3eaca1ed4/gazebo/physics/ode/ODEJoint.cc#L1206).
  // Implicit damping does something more complicated, but by default it is not used.
  // Anyway, this damping usse can be compensated by removing the damping component from GetForce().
  // An alternative may be to use GetForceTorque and do some kind of projection.
  return joint->GetForce(dof_index) - joint->GetDamping(dof_index)*joint->GetVelocity(dof_index);
}

JointMonitor::MonitoredJoint::MonitoredJoint(pair<string,string> joint_name, gazebo::physics::WorldPtr world, double smoothingAlpha)
{
  this->world = world;
  this->smoothing_alpha = smoothingAlpha;
  this->name = joint_name;
  int r = this->reset();
  if (r!=0)
    throw runtime_error("Tied to set up joint monitoring for non-existant joint "+joint_name.first+":"+joint_name.second);
  ROS_INFO_STREAM("Created monitoredJoint with "<<joint_name.first<<":"<<joint_name.second<<" "<<this->position.size()<<" DOF");
}

int JointMonitor::MonitoredJoint::reset()
{
  gazebo::physics::ModelPtr model = world->ModelByName(name.first);
  if (!model)
    return -1;
  gazebo::physics::JointPtr joint = model->GetJoint(name.second);
  if (!joint)
    return -2;

  this->position.clear();
  this->velocity.clear();
  this->effort.clear();
  for(unsigned int i=0;i<joint->DOF();i++)
  {
    this->position.push_back(joint->Position(i));
    this->velocity.push_back(joint->GetVelocity(i));
    this->effort.push_back(getCorrectedForce(joint, i));
  }
  return 0;
}

int JointMonitor::MonitoredJoint::update()
{
  gazebo::physics::ModelPtr model = world->ModelByName(name.first);
  if (!model)
    return -1;
  gazebo::physics::JointPtr joint = model->GetJoint(name.second);
  if (!joint)
    return -2;

  for(unsigned int i=0;i<joint->DOF();i++)
  {
    this->position[i] = this->position[i]*smoothing_alpha + (1-smoothing_alpha)*joint->Position(i);
    this->velocity[i] = this->velocity[i]*smoothing_alpha + (1-smoothing_alpha)*joint->GetVelocity(i);
    this->effort[i] = this->effort[i]*smoothing_alpha + (1-smoothing_alpha)*getCorrectedForce(joint, i);
  }
  return 0;
}





JointMonitor::JointMonitor(gazebo::physics::WorldPtr world, double smoothingAlpha)
{
  this->world = world;
  this->smoothing_alpha = smoothingAlpha;
  searchJoints();
  // This should set stepCallback to be called at each simulation step
  stepConnection = gazebo::event::Events::ConnectWorldUpdateEnd(std::bind(&JointMonitor::stepCallback, this));
  // Could also connect to worldReset and timeReset to reset the averages
}

void JointMonitor::searchJoints()
{
  for(gazebo::physics::ModelPtr model : world->Models())
  {
    for(gazebo::physics::JointPtr joint : model->GetJoints())
    {
      auto name = std::make_pair(model->GetName(),joint->GetName());
      if(this->monitoredJoints.find(name) == this->monitoredJoints.end())
      {
        this->monitoredJoints[name] = std::make_shared<MonitoredJoint>(name, world, this->smoothing_alpha);
        ROS_INFO_STREAM("Found new joint '"<<name.first<<":"<<name.second<<"'");
      }
    }
  }
}

void JointMonitor::stepCallback()
{
//   string jnames = "";
//   for(const auto& [jname, monJoint] : monitoredJoints)
//   {
//     jnames = jnames + jname.first + ":" + jname.second + ",";
//   }
//   ROS_INFO_STREAM("stepCallback "<<world->Iterations()<<" "<<jnames);
  searchJoints();
  vector<pair<string,string>> jointsToRemove;
  for(const auto& [jname, monJoint] : monitoredJoints)
  {
    int r = monJoint->update();
    if(r!=0)
        jointsToRemove.push_back(jname);
  }
  for(const auto& jname : jointsToRemove)
    monitoredJoints.erase(jname);
}


void JointMonitor::getJointState(pair<string,string> jointName, gazebo_gym_env_plugin::JointInfo& jointState)
{
  const auto& monJoint = monitoredJoints.at(jointName);
  jointState.joint_id.model_name = jointName.first;
  jointState.joint_id.joint_name = jointName.second;
  jointState.position = monJoint->position;
  jointState.rate = monJoint->velocity;
  jointState.effort = monJoint->effort;
//   ROS_INFO_STREAM("getJointState(): "<<jointName.first<<":"<<jointName.second<<" : "<<jointState.position.size()<<" DOF");
}