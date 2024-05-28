#include <gazebo/gazebo.hh>
#include "gazebo/physics/physics.hh"
#include "gazebo/sensors/sensors.hh"
#include "gazebo/rendering/rendering.hh"
#include "gazebo/common/common.hh"

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include "gazebo_gym_env_plugin/StepSimulation.h"
#include "gazebo_gym_env_plugin/RenderCameras.h"
#include "gazebo_gym_env_plugin/GetInfo.h"
#include "gazebo_gym_env_plugin/JointInfo.h"
#include "gazebo_gym_env_plugin/JointProperties.h"
#include "gazebo_gym_env_plugin/SetJointProperties.h"
#include "gazebo_gym_env_plugin/JointEffortRequest.h"

#include <boost/algorithm/string.hpp>
#include <thread>
#include <mutex>
#include <chrono>
#include <functional>
#include <string>

#include "utils.hpp"
#include "RenderingHelper.hpp"
#include "JointEffortControl.hpp"

namespace gazebo
{
  /**
   * Gazebo plugin that provides methods necessary for correctly implementing an
   * OpenAI-gym environment.
   */
  class GazeboGymEnvPlugin : public WorldPlugin
  {

  private:

    std::shared_ptr<ros::NodeHandle> nodeHandle;
    std::shared_ptr<std::thread> callbacksThread;
    ros::CallbackQueue callbacksQueue;
    physics::WorldPtr world;
    long stepCounter = 0;


    ros::ServiceServer stepService;
    const std::string stepServiceName = "step";

    ros::ServiceServer observeService;
    const std::string observeServiceName = "observe";

    ros::ServiceServer renderService;
    const std::string renderServiceName = "render";

    ros::ServiceServer infoService;
    const std::string infoServiceName = "get_info";

    ros::ServiceServer setJointPropertiesService;
    const std::string setJointPropertiesServiceName = "set_joint_properties";

    bool keepServingCallbacks = true;


    std::shared_ptr<RenderingHelper> renderingHelper;
    std::shared_ptr<JointEffortControl> jointEffortControl;

    AverageKeeper avgRenderRequestDelay;
    AverageKeeper avgStepRequestDelay;
    AverageKeeper avgSteppingTime;
    AverageKeeper totStepCallbackDuration;





  public:

    virtual ~GazeboGymEnvPlugin()
    {
      //std::cout<<"Destructor!!"<<std::endl;
      keepServingCallbacks = false;
      callbacksThread->join();
    }


    /**
     * Get a list of all the links. Mostly for debug purposes.
     * @param  world World to get the links for.
     * @return The list of the links
     */
    std::vector<gazebo::physics::LinkPtr> getAllLinks(physics::WorldPtr world)
    {
      std::vector<gazebo::physics::LinkPtr> ret;
      auto models = world->Models();
      for(gazebo::physics::ModelPtr model : models)
      {
        auto links = model->GetLinks();
        for(gazebo::physics::LinkPtr link : links)
          ret.push_back(link);
      }
      return ret;
    }

    /**
     * Loads the plugin setting up the necessary things
     * @param _parent         [description]
     * @param sdf::ElementPtr [description]
     */
    void Load(physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM_NAMED("GazeboGymEnvPlugin", "A ROS node for Gazebo has not been initialized, unable to load plugin. "
          << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
        return;
      }

      world = _parent;

      renderingHelper = std::make_shared<RenderingHelper>(world);
      jointEffortControl = std::make_shared<JointEffortControl>(world);

      this->nodeHandle = std::make_shared<ros::NodeHandle>("~/gym_env_interface");
      //ROS_INFO("Got node handle");

      this->nodeHandle->setCallbackQueue( &callbacksQueue);


      //Start thread that will handle the service calls
      callbacksThread = std::make_shared<std::thread>(&GazeboGymEnvPlugin::callbacksThreadMain, this);


      stepService = nodeHandle->advertiseService(stepServiceName, &GazeboGymEnvPlugin::stepServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<stepServiceName);

      renderService = nodeHandle->advertiseService(renderServiceName, &GazeboGymEnvPlugin::renderServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<renderServiceName);

      observeService = nodeHandle->advertiseService(observeServiceName, &GazeboGymEnvPlugin::observeServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<observeServiceName);

      infoService = nodeHandle->advertiseService(infoServiceName, &GazeboGymEnvPlugin::infoServiceCallback,this);
      ROS_INFO_STREAM("Advertised service "<<infoServiceName);

      setJointPropertiesService = nodeHandle->advertiseService(setJointPropertiesServiceName, &GazeboGymEnvPlugin::setJointPropertiesCallback,this);
      ROS_INFO_STREAM("Advertised service "<<setJointPropertiesServiceName);

      ROS_INFO_STREAM("Startup models:");
      auto links = getAllLinks(world);
      for(gazebo::physics::LinkPtr link : links)
        ROS_INFO_STREAM(" - "<<link->GetModel()->GetName()<<" : "<<link->GetName());
      //world->Physics()->SetSeed(20200413);
    }






  private:



    /**
     * Check if the specified joint exists
     * @param  jointId Joint to check
     * @return         true if it exists
     */
    bool doesJointExist(const gazebo_gym_env_plugin::JointId& jointId)
    {
        gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
        if (!model)
          return false;
        gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
        if (!joint)
          return false;
        return true;
    }


    /**
     * Gets Position and speed of a joint
     * @param  jointId   The joint to get the information for
     * @param  ret       The result is returned here
     * @return           Positive in case of success, negative in case of error
     */
    int getJointInfo(const gazebo_gym_env_plugin::JointId& jointId, gazebo_gym_env_plugin::JointInfo& ret)
    {
      //ROS_DEBUG_STREAM("Getting joint info for "<<jointId.model_name<<"."<<jointId.joint_name);
      gazebo::physics::ModelPtr model = world->ModelByName(jointId.model_name);
      if (!model)
        return -1;
      gazebo::physics::JointPtr joint = model->GetJoint(jointId.joint_name);
      if (!joint)
        return -2;

      ret.joint_id = jointId;
      ret.position.clear();
      ret.position.push_back(joint->Position(0));
      ret.rate.clear();
      ret.rate.push_back(joint->GetVelocity(0));

      // About the joint effort there is a bit of confusion. We have:
      //  - physics:Joint::GetForce(int) which the doc says returns "external forces"
      //  - physics::Joint::GetForceTorque(int) which seems to return the full force-torques between the two adjecent links
      // Looking into the code GetForce seems to actually return the commanded efforts for the specified DOF. This
      // can be seen in ODEJoint, BulletJoint, DARTJoint and SimbodyJoint. They all return the value of _forceApplied.
      // This actually differs from what is applied to the joint due to the internal velocity damping, which is added to the 
      // joint torque, but is not saved in _forceApplied. In the case of explicit damping this is true because it is aplied directly with
      // SetForceImpl (at least in ODE, https://github.com/gazebosim/gazebo-classic/blob/e4b4d0fb752c7e43e34ab97d0e01a2a3eaca1ed4/gazebo/physics/ode/ODEJoint.cc#L1206).
      // Implicit damping does something more complicated, but by default it is not used.
      // Anyway, this damping usse can be compensated by removing the damping component from GetForce().
      // An alternative may be to use GetForceTorque and do some kind of projection.

      ret.effort.clear();
      ret.effort.push_back(joint->GetForce(0) - joint->GetDamping(0)*joint->GetVelocity(0));



      return 0;
    }

    /**
     * Gets position and speed of a set of joints
     * @param  jointIds  The joints to get the information for
     * @param  ret       The result is returned here (the returned joints are n the same order as in jointIds)
     * @return           Positive in case of success, negative in case of error
     */
    void getJointsInfo(std::vector<gazebo_gym_env_plugin::JointId> jointIds, gazebo_gym_env_plugin::JointsInfoResponse& ret)
    {
      ret.error_message = "Error in getJointsInfo():";
      for(const gazebo_gym_env_plugin::JointId& jointId : jointIds)
      {
        gazebo_gym_env_plugin::JointInfo jointInfo;
        int r = getJointInfo(jointId, jointInfo);
        if(r<0)
        {
          ret.success=false;
          ret.error_message = ret.error_message + "\n - Could not get info for joint '" + jointId.model_name + "." + jointId.joint_name+"'. Error "+std::to_string(r)+": ";
          if (r == -1)
            ret.error_message = ret.error_message + "model not found.";
          else if (r == -2)
            ret.error_message = ret.error_message + "joint not found.";
          ROS_WARN_STREAM(ret.error_message);
        }
        ret.joints_info.push_back(jointInfo);
      }
      ret.success=true;
      ret.error_message="No Error";
    }

    /**
     * Get pose and linear/angular velocity of a link
     * @param  linkId Identifier for the link to get the information for
     * @param  ret    The result is returned here
     * @return        0 if successfull
     */
    int getLinkInfo(const gazebo_gym_env_plugin::LinkId& linkId, gazebo_gym_env_plugin::LinkInfo& ret)
    {
      //ROS_DEBUG_STREAM("Getting link info for "<<linkId.model_name<<"."<<linkId.link_name);
      gazebo::physics::ModelPtr model = world->ModelByName(linkId.model_name);
      if (!model)
        return -1;
      gazebo::physics::LinkPtr link = model->GetLink(linkId.link_name);
      if (!link)
        return -2;

      ret.link_id = linkId;
      ret.pose.position.x = link->WorldPose().Pos().X();
      ret.pose.position.y = link->WorldPose().Pos().Y();
      ret.pose.position.z = link->WorldPose().Pos().Z();
      ret.pose.orientation.x = link->WorldPose().Rot().X();
      ret.pose.orientation.y = link->WorldPose().Rot().Y();
      ret.pose.orientation.z = link->WorldPose().Rot().Z();
      ret.pose.orientation.w = link->WorldPose().Rot().W();
      ret.twist.linear.x = link->WorldLinearVel().X();
      ret.twist.linear.y = link->WorldLinearVel().Y();
      ret.twist.linear.z = link->WorldLinearVel().Z();
      ret.twist.angular.x = link->WorldAngularVel().X();
      ret.twist.angular.y = link->WorldAngularVel().Y();
      ret.twist.angular.z = link->WorldAngularVel().Z();
      ret.reference_frame = "world";

      return 0;
    }

    /**
     * Get pose and linear/angular velocity of a set of links
     * @param  linkIds Identifiers for the links to get the information for
     * @param  ret    The result is returned here
     * @return        0 if successfull
     */
    void getLinksInfo(std::vector<gazebo_gym_env_plugin::LinkId> linkIds, gazebo_gym_env_plugin::LinksInfoResponse& ret)
    {
      ret.error_message = "Error in getLinksInfo():";
      for(const gazebo_gym_env_plugin::LinkId& linkId : linkIds)
      {
        gazebo_gym_env_plugin::LinkInfo linkInfo;
        int r = getLinkInfo(linkId, linkInfo);
        if(r<0)
        {
          ret.success=false;
          ret.error_message = ret.error_message + "\n - Could not get info for link '" + linkId.model_name + "." + linkId.link_name+"'. Error "+std::to_string(r)+": ";
          if (r == -1)
            ret.error_message = ret.error_message + "model not found.";
          else if (r == -2)
            ret.error_message = ret.error_message + "link not found.";
          ROS_WARN_STREAM(ret.error_message);
          // ROS_INFO_STREAM("Available links:");
          // auto links = getAllLinks(world);
          // for(gazebo::physics::LinkPtr link : links)
          //   ROS_INFO_STREAM(" - "<<link->GetModel()->GetName()<<" : "<<link->GetName());
        }
        ret.links_info.push_back(linkInfo);
      }
      ret.success=true;
      ret.error_message="No Error";
    }




















    /**
     * Executed as a thread to handle the ROS service calls
     */
    void callbacksThreadMain()
    {
      //Initialize rendering engine in this thread (necessary for rendeing the camera)
      //rendering::load();
      //rendering::init();
      static const double timeout = 0.0005;
      while(keepServingCallbacks)
      {
        //ROS_INFO("Looping callbacksThreadMain");
        callbacksQueue.callAvailable(ros::WallDuration(timeout));
      }
      //close the rendering engine for this thread
      //rendering::fini();
    }

    /**
     * Handles a call from the step ROS service
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
    bool stepServiceCallback(gazebo_gym_env_plugin::StepSimulation::Request &req, gazebo_gym_env_plugin::StepSimulation::Response &res)
    {
      /*
      By looking at the code in gazebo/physics/World.cc:
        stop makes the simulation loop stop
        pause makes the simulation loop do nothing (but it still loops!)
        world->Running() indicates if the simulation loop is stopped
        world->IsPaused() indicates if the simulation is paused
        Step(int _steps) makes the simulation run even if it is paused, for _steps steps

      So:
        We keep the simulation paused and we make it go forward with Step(int). We cannot suse stop because it
        also stops the sensor updates, which prevents us from using the cameras, we cannot render them ourselves
        because even the Render event gets stopped.
      */
      if(req.iterations !=0 && req.step_duration_nanosecs!=0)
      {
        res.success = false;
        res.error_message = "GazeboGymEnvPlugin: step was requested specifying both iterations and step_duration. Only one can be set at a time. No action taken.";
        ROS_WARN_STREAM(res.error_message.c_str());
        return true;
      }

      if(!world->IsPaused())
      {
        res.success = false;
        res.error_message = "Called step_simulation while the simulation was running. Simulation must be paused. No action taken.";
        ROS_WARN_STREAM(res.error_message.c_str());
        return true;
      }

      double delay_secs = ros::WallTime::now().toSec() - req.request_time;
      avgStepRequestDelay.addValue(delay_secs);
      //ROS_INFO_STREAM("Stepping simulation. Service request delay = "<<delay_secs);



      res.success = false;
      res.iterations_done = 0;
      res.step_duration_nanosecs = 0;
      for(const gazebo_gym_env_plugin::JointEffortRequest& jer : req.joint_effort_requests)
      {
        if(!doesJointExist(jer.joint_id))
        {
          res.error_message = "Requested effort for non-existing joint "+jer.joint_id.model_name+"."+jer.joint_id.joint_name+", aborting step";
          res.response_time = ros::WallTime::now().toSec();
          ROS_WARN_STREAM(res.error_message);
          return true;//must return false only if we cannot send a response
        }
      }
      for(const gazebo_gym_env_plugin::JointId& jid : req.requested_joints)
      {
        if(!doesJointExist(jid))
        {
          res.error_message = "Requested state for non-existing joint "+jid.model_name+"."+jid.joint_name+", aborting step";
          res.response_time = ros::WallTime::now().toSec();
          ROS_WARN_STREAM(res.error_message);
          return true;//must return false only if we cannot send a response
        }
      }

      totStepCallbackDuration.onTaskStart();
      int requestedIterations = -1;
      if(req.step_duration_nanosecs!=0)
        requestedIterations = req.step_duration_nanosecs/(1000000000*world->Physics()->GetMaxStepSize());
      else
        requestedIterations = req.iterations;

      common::Time startTime = world->SimTime();


      for(const gazebo_gym_env_plugin::JointEffortRequest& jer : req.joint_effort_requests)
      {
        jointEffortControl->requestJointEffort(jer);
      }


      int iterationsBefore = world->Iterations();
      if(requestedIterations>0)
      {
        //ROS_INFO("Stepping simulation...");
        avgSteppingTime.onTaskStart();
        world->Step(requestedIterations);
        avgSteppingTime.onTaskEnd();
      }


      jointEffortControl->clearRequestedJointEfforts();




      common::Time endTime = world->SimTime();
      common::Time durationTime = endTime-startTime;

      int iterations_done = world->Iterations() - iterationsBefore;
      res.success = iterations_done == requestedIterations;
      res.error_message = "No error";
      res.iterations_done = iterations_done;
      res.max_step_size = world->Physics()->GetMaxStepSize();
      res.step_duration_nanosecs = int64_t(durationTime.sec) * 1000000000 + durationTime.nsec;
      res.response_time = ros::WallTime::now().toSec();

      stepCounter++;

      if(req.render)
        renderingHelper->renderCameras(req.cameras,res.render_result);

      if(req.requested_joints.size()>0)
        getJointsInfo(req.requested_joints,res.joints_info);

      if(req.requested_links.size()>0)
        getLinksInfo(req.requested_links,res.links_info);

      totStepCallbackDuration.onTaskEnd();
      //Print timing info
      ROS_DEBUG_STREAM("-------------------------------------------------");
      ROS_DEBUG_STREAM("Render request delay:         avg="<<avgRenderRequestDelay.getAverage()*1000<<"ms");
      ROS_DEBUG_STREAM("Render thread call delay:     avg="<<renderingHelper->getAvgRenderThreadDelay()*1000<<"ms");
      ROS_DEBUG_STREAM("Render duration:              avg="<<renderingHelper->getAvgRenderTime()*1000<<"ms");
      ROS_DEBUG_STREAM("Image fill duration:          avg="<<renderingHelper->getAvgFillTime()*1000<<"ms");
      ROS_DEBUG_STREAM("Total Render duration:        avg="<<renderingHelper->getAvgTotalRenderTime()*1000<<"ms");
      ROS_DEBUG_STREAM("Step request delay:           avg="<<avgStepRequestDelay.getAverage()*1000<<"ms");
      ROS_DEBUG_STREAM("Physics step wall duration:   avg="<<avgSteppingTime.getAverage()*1000<<"ms");
      ROS_DEBUG_STREAM("Tot call duration:            avg="<<totStepCallbackDuration.getAverage()*1000<<"ms");

      return true;//Must be false only in case we cannot send a response
    }


    /**
     * Handles a ROS render service call
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
    bool renderServiceCallback(gazebo_gym_env_plugin::RenderCameras::Request &req, gazebo_gym_env_plugin::RenderCameras::Response &res)
    {
      double delay_secs = ros::WallTime::now().toSec() - req.request_time;
      avgRenderRequestDelay.addValue(delay_secs);
      //ROS_INFO_STREAM("Rendering cameras. Service request delay = "<<delay_secs);

      renderingHelper->renderCameras(req.cameras,res.render_result);

      res.response_time = ros::WallTime::now().toSec();
      return true;//Must be false only in case we cannot send a response
    }

    /**
     * Handles a call from the observe ROS service
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
    bool observeServiceCallback(gazebo_gym_env_plugin::StepSimulation::Request &req, gazebo_gym_env_plugin::StepSimulation::Response &res)
    {
      bool wasPaused = world->IsPaused();
      world->SetPaused(true);



      res.success = false;
      res.iterations_done = 0;
      res.step_duration_nanosecs = 0;
      if (req.joint_effort_requests.size()!=0)
      {
        res.error_message = "Requested effort on observe service. joint_effort_requests must be empty";
        res.response_time = ros::WallTime::now().toSec();
        ROS_WARN_STREAM(res.error_message);
        return true;//must return false only if we cannot send a response
      }
      if (req.iterations!=0)
      {
        res.error_message = "Requested iterations on observe service. iterations must be 0";
        res.response_time = ros::WallTime::now().toSec();
        ROS_WARN_STREAM(res.error_message);
        return true;//must return false only if we cannot send a response
      }
      if (req.step_duration_nanosecs!=0)
      {
        res.error_message = "Requested step_duration_nanosecs on observe service. step_duration_nanosecs must be 0";
        res.response_time = ros::WallTime::now().toSec();
        ROS_WARN_STREAM(res.error_message);
        return true;//must return false only if we cannot send a response
      }


      for(const gazebo_gym_env_plugin::JointId& jid : req.requested_joints)
      {
        if(!doesJointExist(jid))
        {
          res.error_message = "Requested state for non-existing joint "+jid.model_name+"."+jid.joint_name+", aborting step";
          res.response_time = ros::WallTime::now().toSec();
          ROS_WARN_STREAM(res.error_message);
          return true;//must return false only if we cannot send a response
        }
      }



      res.success = true;
      res.error_message = "No error";
      res.iterations_done = 0;
      res.step_duration_nanosecs = 0;
      res.response_time = ros::WallTime::now().toSec();


      if(req.render)
        renderingHelper->renderCameras(req.cameras,res.render_result);

      if(req.requested_joints.size()>0)
        getJointsInfo(req.requested_joints,res.joints_info);

      if(req.requested_links.size()>0)
        getLinksInfo(req.requested_links,res.links_info);



      world->SetPaused(wasPaused);
      return true;
    }


    /**
     * Handles a call from the get_info ROS service
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
     */
    bool infoServiceCallback(gazebo_gym_env_plugin::GetInfo::Request &req, gazebo_gym_env_plugin::GetInfo::Response &res)
    {
      res.is_paused = world->IsPaused();
      return true;
    }

    gazebo_gym_env_plugin::JointProperties getJointProperties(gazebo_gym_env_plugin::JointId joint_id)
    {
      gazebo::physics::ModelPtr model = world->ModelByName(joint_id.model_name);
      if (!model)
        throw std::runtime_error(std::string(__func__)+": Model '"+joint_id.model_name+"' not found.");
      gazebo::physics::JointPtr joint = model->GetJoint(joint_id.joint_name);
      if (!joint)
        throw std::runtime_error(std::string(__func__)+": Model '"+joint_id.model_name+"' not found.");

      gazebo_gym_env_plugin::JointProperties jp;
      jp.joint_id = joint_id;
      jp.degrees_of_freedom = joint->DOF();
      for(unsigned int i=0;i<joint->DOF();i++) jp.position_limit_low.push_back(joint->GetParam("lo_stop",i));
      for(unsigned int i=0;i<joint->DOF();i++) jp.position_limit_high.push_back(joint->GetParam("hi_stop",i));
      return jp;
    }

    void setJointProperties(const gazebo_gym_env_plugin::JointProperties& joint_properties)
    {
      gazebo::physics::ModelPtr model = world->ModelByName(joint_properties.joint_id.model_name);
      if (!model)
        throw std::runtime_error(std::string(__func__)+": Model '"+joint_properties.joint_id.model_name+"' not found.");
      gazebo::physics::JointPtr joint = model->GetJoint(joint_properties.joint_id.joint_name);
      if (!joint)
        throw std::runtime_error(std::string(__func__)+": Joint '"+joint_properties.joint_id.joint_name+"' not found in model '"+joint_properties.joint_id.model_name+"'.");

      /*
       if (joint_properties.degrees_of_freedom != joint->DOF())
         return "Joint degrees_of_freedom cannot be changed (requested "\
                 +std::to_string(joint_properties.degrees_of_freedom)+" but current is "+std::to_string(joint.DOF())+")";
      */

      if(joint_properties.position_limit_low.size()>0)
      {
        if(joint_properties.position_limit_low.size() != joint->DOF())
          throw std::runtime_error(std::string(__func__)+": Requested position_limit_low has size "+std::to_string(joint_properties.position_limit_low.size())+\
                    " but joint has "+std::to_string(joint->DOF())+" degrees of freedom.");
      }
      if(joint_properties.position_limit_high.size()>0)
      {
        if(joint_properties.position_limit_high.size() != joint->DOF())
          throw std::runtime_error(std::string(__func__)+": Requested position_limit_high has size "+std::to_string(joint_properties.position_limit_high.size())+\
                  " but joint has "+std::to_string(joint->DOF())+" degrees of freedom.");
      }

      for(unsigned int i=0;i<joint_properties.position_limit_low.size();i++)
        joint->SetParam("lo_stop",i,joint_properties.position_limit_low[i]);
      for(unsigned int i=0;i<joint_properties.position_limit_high.size();i++)
        joint->SetParam("hi_stop",i,joint_properties.position_limit_high[i]);
      // 
      // for(unsigned int i=0;i< req.ode_joint_config.damping.size();i++)
      //   joint->SetDamping(i,req.ode_joint_config.damping[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.erp.size();i++)
      //   joint->SetParam("erp",i,req.ode_joint_config.erp[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.cfm.size();i++)
      //   joint->SetParam("cfm",i,req.ode_joint_config.cfm[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.stop_erp.size();i++)
      //   joint->SetParam("stop_erp",i,req.ode_joint_config.stop_erp[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.stop_cfm.size();i++)
      //   joint->SetParam("stop_cfm",i,req.ode_joint_config.stop_cfm[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.fudge_factor.size();i++)
      //   joint->SetParam("fudge_factor",i,req.ode_joint_config.fudge_factor[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.fmax.size();i++)
      //   joint->SetParam("fmax",i,req.ode_joint_config.fmax[i]);
      // for(unsigned int i=0;i< req.ode_joint_config.vel.size();i++)
      //   joint->SetParam("vel",i,req.ode_joint_config.vel[i]);
    }

    /**
     * Set the properties of a joint (e.g. joint limits)
     * @param  req [description]
     * @param  res [description]
     * @return     [description]
    */
    bool setJointPropertiesCallback(gazebo_gym_env_plugin::SetJointProperties::Request &req,
                            gazebo_gym_env_plugin::SetJointProperties::Response &res)
    {
      res.success = true;
      res.error_message = "";
      for(gazebo_gym_env_plugin::JointProperties jp : req.joint_properties)
      {
        gazebo::physics::ModelPtr model = world->ModelByName(jp.joint_id.model_name);
        if (!model)
        {
          res.success = false;
          res.error_message = "Model "+jp.joint_id.model_name+" not found.";
          return true;
        }
        gazebo::physics::JointPtr joint = model->GetJoint(jp.joint_id.joint_name);
        if (!joint)
        {
          res.success = false;
          res.error_message = "Joint "+jp.joint_id.joint_name+" not found in model "+jp.joint_id.model_name;
          return true;
        }
      }


      for(gazebo_gym_env_plugin::JointProperties jp : req.joint_properties)
      {
        try
        {
          setJointProperties(jp);
        }
        catch (const std::runtime_error& e)
        {
          res.success = false;
          if(res.error_message.size()>0)
            res.error_message += "\n";
          res.error_message += e.what();
        }
      }
      
      for(gazebo_gym_env_plugin::JointProperties jp : req.joint_properties)
      {
        try
        {
          res.resulting_joint_properties.push_back(getJointProperties(jp.joint_id));
        }
        catch (const std::runtime_error& e)
        {
          res.success = false;
          if(res.error_message.size()>0)
            res.error_message += "\n";
          res.error_message += e.what();
        }
      }
      
      return true;
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_WORLD_PLUGIN(GazeboGymEnvPlugin)
}
