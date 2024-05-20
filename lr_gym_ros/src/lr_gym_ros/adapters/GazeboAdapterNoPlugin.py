#!/usr/bin/env python3

from __future__ import annotations
import traceback
from typing import List, Tuple, Dict, Any, Union, Optional
import time
import gazebo_msgs
import gazebo_msgs.msg
import gazebo_msgs.srv
import rosgraph_msgs.msg

import rospy
from std_srvs.srv import Empty

from lr_gym_ros.adapters.RosAdapter import RosAdapter
from lr_gym.adapters.BaseJointEffortAdapter import BaseJointEffortAdapter
from lr_gym.adapters.BaseSimulationAdapter import BaseSimulationAdapter 
from lr_gym.utils.utils import JointState, LinkState, RequestFailError
import os
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.utils import Pose, buildRos1PoseStamped
from lr_gym_ros.utils.gazebo_models_manager import delete_model, spawn_model
import rospkg
import lr_gym.utils
import lr_gym.utils.utils
from lr_gym.adapters.BaseAdapter import JointName, LinkName

class GazeboAdapterNoPlugin(RosAdapter, BaseJointEffortAdapter, BaseSimulationAdapter):
    """This class allows to control the execution of a Gazebo simulation.

    It only uses the default gazebo plugins which are usually included in the installation.
    Because of this the duration of the simulation steps may not be accurate and simulation
    speed is low due to communication overhead.
    """

    def __init__(   self,
                    usePersistentConnections : bool = False,
                    stepLength_sec : float = 0.001,
                    rosMasterUri : Union[str, None] = None):
        """Initialize the Gazebo controller.

        Parameters
        ----------
        usePersistentConnections : bool
            Controls wheter to use persistent connections for the gazebo services.
            IMPORTANT: enabling this seems to create problems with the synchronization
            of the service calls. This breaks the pause/unpause/reset order and
            leads to deadlocks
            In theory it should have been fine as long as there are no connection
            problems and gazebo does not restart.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__()

        self._stepLength_sec = stepLength_sec
        self._lastPausedTime = 0
        self._episodeCountedSimDuration = 0
        self._totalCountedSimDuration = 0
        self._episode_steps_taken = 0

        self._lastStepRendered = None
        self._lastRenderResult = None
        self._usePersistentConnections = usePersistentConnections

        self._rosMasterUri = rosMasterUri

    def _makeRosConnections(self):
        serviceNames = {"applyJointEffort" : "/gazebo/apply_joint_effort",
                        "clearJointEffort" : "/gazebo/clear_joint_forces",
                        "getJointProperties" : "/gazebo/get_joint_properties",
                        "getLinkState" : "/gazebo/get_link_state",
                        "pause" : "/gazebo/pause_physics",
                        "unpause" : "/gazebo/unpause_physics",
                        "get_physics_properties" : "/gazebo/get_physics_properties",
                        "reset" : "/gazebo/reset_simulation",
                        "setLinkState" : "/gazebo/set_link_state",
                        "setJointState" : "/gazebo/set_model_configuration",
                        "setLightProperties" : "/gazebo/set_light_properties"}

        timeout_secs = 30.0
        for serviceName in serviceNames.values():
            try:
                rospy.loginfo("waiting for service "+serviceName+" ...")
                rospy.wait_for_service(serviceName)
                rospy.loginfo("got service "+serviceName)
            except rospy.ROSInterruptException as e:
                rospy.logfatal("Interrupeted while waiting for service "+serviceName+". Exception = "+str(e))
                raise
            except rospy.ROSException as e:
                rospy.logfatal("Failed to wait for service "+serviceName+". Timeouts were "+str(timeout_secs)+"s. Exception = "+str(e))
                raise

        self._applyJointEffortService   = rospy.ServiceProxy(serviceNames["applyJointEffort"], gazebo_msgs.srv.ApplyJointEffort, persistent=self._usePersistentConnections)
        self._clearJointEffortService   = rospy.ServiceProxy(serviceNames["clearJointEffort"], gazebo_msgs.srv.JointRequest, persistent=self._usePersistentConnections)
        self._getJointPropertiesService = rospy.ServiceProxy(serviceNames["getJointProperties"], gazebo_msgs.srv.GetJointProperties, persistent=self._usePersistentConnections)
        self._getLinkStateService       = rospy.ServiceProxy(serviceNames["getLinkState"], gazebo_msgs.srv.GetLinkState, persistent=self._usePersistentConnections)
        self._pauseGazeboService        = rospy.ServiceProxy(serviceNames["pause"], Empty, persistent=self._usePersistentConnections)
        self._unpauseGazeboService      = rospy.ServiceProxy(serviceNames["unpause"], Empty, persistent=self._usePersistentConnections)
        self._getPhysicsProperties      = rospy.ServiceProxy(serviceNames["get_physics_properties"], gazebo_msgs.srv.GetPhysicsProperties, persistent=self._usePersistentConnections)
        self._resetGazeboService        = rospy.ServiceProxy(serviceNames["reset"], Empty, persistent=self._usePersistentConnections)
        self._setLinkStateService       = rospy.ServiceProxy(serviceNames["setLinkState"], gazebo_msgs.srv.SetLinkState, persistent=self._usePersistentConnections)
        self._setJointStateService      = rospy.ServiceProxy(serviceNames["setJointState"], gazebo_msgs.srv.SetModelConfiguration, persistent=self._usePersistentConnections)
        self._setLightPropertiesService = rospy.ServiceProxy(serviceNames["setLightProperties"], gazebo_msgs.srv.SetLightProperties, persistent=self._usePersistentConnections)

        #self._setGazeboPhysics = rospy.ServiceProxy(self._setGazeboPhysics, SetPhysicsProperties, persistent=self._usePersistentConnections)

        # Crete a publisher to manually send clock messages (used in reset, very ugly, sorry)
        self._clockPublisher = rospy.Publisher("/clock", rosgraph_msgs.msg.Clock, queue_size=1)


    def startup(self):
        """Start up the controller. This must be called after set_monitored_cameras, set_monitored_links and set_monitored_joints."""

        super().startup()

        self._makeRosConnections()


        rospy.loginfo("ROS time is "+str(rospy.get_time())+" pid = "+str(os.getpid()))
        self.pauseSimulation()
        self.resetWorld()

    def _callService(self,serviceProxy : rospy.ServiceProxy) -> bool:
        """Call the provided service. It retries in case of failure and handles exceptions. Returns false if the call failed.

        Parameters
        ----------
        serviceProxy : rospy.ServiceProxy
            ServiceProxy for the service to be called

        Returns
        -------
        bool
            True if the service was called, false otherwise

        """
        done = False
        counter = 0
        maxRetry = 10
        while not done and not rospy.is_shutdown():
            if counter < maxRetry:
                try:
                    serviceProxy.call()
                    done = True
                except rospy.ServiceException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed: "+lr_gym.utils.utils.exc_to_str(e))
                except rospy.ROSInterruptException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call interrupted: "+lr_gym.utils.utils.exc_to_str(e))
                    counter+=maxRetry #don't retry
                except rospy.ROSSerializationException as e:
                    rospy.logerr("Service "+serviceProxy.resolved_name+", call failed to serialize: "+lr_gym.utils.utils.exc_to_str(e))
                counter += 1
            else:
                rospy.logerr("Failed to call service")
                break
        return done

    def pauseSimulation(self) -> bool:
        """Pause the simulation.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        ret = self._callService(self._pauseGazeboService)
        #rospy.loginfo("paused sim")
        self._lastPausedTime = rospy.get_time()
        return ret

    def unpauseSimulation(self) -> bool:
        """Unpause the simulation.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        t = rospy.get_time()
        if self._lastPausedTime>t:
            rospy.logwarn("Simulation time increased since last pause! (time diff = "+str(t-self._lastPausedTime)+"s)")
        ret = self._callService(self._unpauseGazeboService)
        #rospy.loginfo("unpaused sim")
        return ret

    def isPaused(self):
        return self._getPhysicsProperties.call().pause
    
    def resetWorld(self) -> bool:
        """Reset the world to its initial state.

        Returns
        -------
        bool
            True if the simulation was paused, false in case of failure

        """
        self.pauseSimulation()
        totalEpSimDuration = self.getEnvTimeFromStartup()

        # ret = self._callService(self._resetGazeboService)


        totalSimTimeError = totalEpSimDuration - self._episodeCountedSimDuration
        if abs(totalSimTimeError)>=0.1:
            rospy.logwarn("Episode error in simulation time keeping = "+str(totalSimTimeError)+"s (This is just an upper bound, may actually be fine)")

        self._episodeCountedSimDuration = 0
        self._episode_steps_taken = 0


        #rospy.loginfo("resetted sim")
        return True


    def step(self) -> float:
        """Run the simulation for the specified time.

        It unpauses and the simulation, sleeps and then pauses it back. It may not be precise.

        Parameters
        ----------
        runTime_secs : float
            Time to run the simulation for, in seconds

        Returns
        -------
        None


        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        t0_ = self.getEnvTimeFromStartup()
        self.freerun(self._stepLength_sec)
        elapsed_time = self.getEnvTimeFromStartup() - t0_
        self._episodeCountedSimDuration += elapsed_time
        self._totalCountedSimDuration += elapsed_time
        self._episode_steps_taken += 1
        return elapsed_time




    def setJointsEffortCommand(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        for command in jointTorques:
            jointName = command[1]
            torque = command[2]
            duration_secs = self._stepLength_sec
            secs = int(duration_secs)
            nsecs = int((duration_secs - secs) * 1000000000)

            request = gazebo_msgs.srv.ApplyJointEffortRequest()
            request.joint_name = jointName
            request.effort = torque
            request.duration.secs = secs
            request.duration.nsecs = nsecs
            # ggLog.info(f"Calling _applyJointEffortService")
            res = self._applyJointEffortService.call(request)
            # ggLog.info(f"Called _applyJointEffortService")
            if not res.success:
                ggLog.error("Failed applying effort for joint "+jointName+": "+res.status_message)


    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        #ggLog.info("GazeboAdapterNoPlugin.getJointsState() called")
        gottenJoints = {}
        missingJoints = []
        for joint in requestedJoints:
            jointName = joint[1]
            modelName = joint[0]

            jointProp = gazebo_msgs.srv.GetJointPropertiesResponse()
            gotit = False
            tries = 0
            while not gotit and tries <10:
                # ggLog.info(f"Calling _getJointPropertiesService")
                jointProp = self._getJointPropertiesService.call(jointName) ## TODO: this ignores the model name!
                # ggLog.info(f"Called _getJointPropertiesService")
                #ggLog.info("Got joint prop for "+jointName+" = "+str(jointProp))
                gotit = jointProp.success
                tries+=1
            if gotit:
                jointState = JointState(list(jointProp.position), list(jointProp.rate), [0]) #NOTE: effort is not returned by the gazeoo service
                gottenJoints[(modelName,jointName)] = jointState
            else:
                missingJoints.append(joint)
                # err = "GazeboAdapterNoPlugin: Failed to get state for joint '"+str(jointName)+"' of model '"+str(modelName)+"'"
                # ggLog.error(err)
                # raise RuntimeError(err)

        if len(missingJoints)>0:
            err = f"Failed to get state for joints {missingJoints}. requested {requestedJoints}"
            # rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenJoints)


        return gottenJoints



    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        gottenLinks = {}
        missingLinks = []
        for link in requestedLinks:
            linkName = link[0]+"::"+link[1]
            # ggLog.info(f"Calling _getLinkStateService")
            resp = self._getLinkStateService.call(link_name=linkName)
            # ggLog.info(f"Called _getLinkStateService")

            if resp.success:
                linkState = LinkState(  position_xyz = (resp.link_state.pose.position.x, resp.link_state.pose.position.y, resp.link_state.pose.position.z),
                                        orientation_xyzw = (resp.link_state.pose.orientation.x, resp.link_state.pose.orientation.y, resp.link_state.pose.orientation.z, resp.link_state.pose.orientation.w),
                                        pos_velocity_xyz = (resp.link_state.twist.linear.x, resp.link_state.twist.linear.y, resp.link_state.twist.linear.z),
                                        ang_velocity_xyz = (resp.link_state.twist.angular.x, resp.link_state.twist.angular.y, resp.link_state.twist.angular.z))

                gottenLinks[link] = linkState
            else:
                # err = f"Failed to get Link state for link {linkName}: resp = {resp}"
                # ggLog.warn(err)
                # world_props = rospy.ServiceProxy("/gazebo/get_world_properties", gazebo_msgs.srv.GetWorldProperties)()
                # ggLog.error(f"World properties are: {world_props}")
                # model_props = rospy.ServiceProxy("/gazebo/get_model_properties", gazebo_msgs.srv.GetModelProperties)(model_name=link[0])
                # ggLog.error(f"Model '{link[0]}' properties are: {model_props}")
                missingLinks.append(link)       
        
        if len(missingLinks)>0:
            err = f"Failed to get state for links {missingLinks}. requested {requestedLinks}"
            # rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenLinks)
       
        return gottenLinks


    def setRosMasterUri(self, rosMasterUri : str):
        self._rosMasterUri = rosMasterUri

    # def spawnModel(self, xacro_file_path : str,
    #                     pose : Pose = Pose(0,0,0,0,0,0,1), 
    #                     args : Dict[str,str] = {}, 
    #                     model_name = "model", 
    #                     robot_namespace = "", 
    #                     reference_frame = "world",
    #                     format = "urdf"):
    #     """Spawn a model in the environment, arguments depend on the type of BaseSimulationAdapter
    #     """
    #     spawn_model(xacro_file_path = xacro_file_path,
    #                     pose = pose, 
    #                     args = args, 
    #                     model_name = model_name, 
    #                     robot_namespace = robot_namespace, 
    #                     reference_frame = reference_frame,
    #                     format = format)


    def deleteModel(self, model : str):
        """Delete a model from the environment"""
        delete_model(model_name=model)


    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        model_configs = {}
        for model_joint_names, joint_state in jointStates.items():
            if model_joint_names[0] not in model_configs:
                model_configs[model_joint_names[0]] = []
            model_configs[model_joint_names[0]].append((model_joint_names[1], joint_state))

        for model_name, joint_confs in model_configs.items():
            req = gazebo_msgs.srv.SetModelConfigurationRequest()
            req.model_name = model_name
            req.joint_names = []
            req.joint_positions = [] # Only uses first joint position
            for jc in joint_confs:
                req.joint_names.append(jc[0])
                if len(jc[1].position) > 1:
                    ggLog.warn(f"GazeboAdapter only supports setting state for 1-D joints")
                if jc[1].rate is not None:
                    ggLog.warn(f"GazeboAdapter does not support setting joint state rate directly")
                if jc[1].effort is not None:
                    ggLog.warn(f"GazeboAdapter does not support setting joint state effort directly")
                req.joint_positions.append(jc[1].position[0])
                resp = self._setJointStateService(req)
                
                if not resp.success:
                    ggLog.error(f"Failed setting joint state for model {req.model_name}, joints = {req.joint_names}, positions = {req.joint_positions}, error = "+resp.status_message)
    

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        
        ret = {}
        for item in linksStates.items():
            linkName  = item[0][1]
            modelName = item[0][0]
            linkState = item[1]
            req = gazebo_msgs.srv.SetLinkStateRequest()
            req.link_state = gazebo_msgs.msg.LinkState()
            req.link_state.link_name = modelName+"::"+linkName
            req.link_state.reference_frame = "world"
            req.link_state.pose = buildRos1PoseStamped(linkState.pose.position, linkState.pose.orientation_xyzw, None).pose
            req.link_state.twist.linear.x = linkState.pos_velocity_xyz[0].item()
            req.link_state.twist.linear.y = linkState.pos_velocity_xyz[1].item()
            req.link_state.twist.linear.z = linkState.pos_velocity_xyz[2].item()
            req.link_state.twist.angular.x = linkState.ang_velocity_xyz[0].item()
            req.link_state.twist.angular.y = linkState.ang_velocity_xyz[1].item()
            req.link_state.twist.angular.z = linkState.ang_velocity_xyz[2].item()

            #print(req)
            #print(type(req))
            resp = self._setLinkStateService(req)
            
            if not resp.success:
                ggLog.error("Failed setting link state for link "+modelName+"::"+linkName+": "+resp.status_message)
            # else:
            #     ggLog.info("Successfully set Linkstate for link "+modelName+"::"+linkName)
        return ret

    def freerun(self, duration_sec : float):
        wasPaused = self.isPaused()
        if wasPaused:
            self.unpauseSimulation()
        rospy.sleep(duration_sec)
        if wasPaused:
            self.pauseSimulation()

    
    def setupLight(self, gz_req : gazebo_msgs.srv.SetLightPropertiesRequest):
        # ggLog.info(f"Calling _setLightPropertiesService")
        res = self._setLightPropertiesService.call(gz_req)
        # ggLog.info(f"Called _setLightPropertiesService")
        if not res.success:
            ggLog.error(f"GazeboAdapterNoPlugin: failed to setup Light.\n req = {gz_req}\n res={res}")
            return False
        return True

    
    def build_scenario(self, launch_file_pkg_and_path : Tuple[str,str],
                             launch_file_args : Dict[str,str]):
        os.environ["IGN_IP"] = "127.0.0.1" # to avoid "Exception sending a multicast message:Network is unreachable" errors when changing network things
        super().build_scenario(launch_file_pkg_and_path=launch_file_pkg_and_path, launch_file_args=launch_file_args)
        self.setRosMasterUri(self._mmRosLauncher.getRosMasterUri())

    
    def spawn_model(self,
                    model_file : Optional[Union[str,Tuple[str,str]]],
                    model_name : str,
                    pose : Pose,
                    model_kwargs : Dict[Any,Any] = {},
                    model_format = None,
                    model_definition_string : Optional[str] = None):
        if isinstance(model_file, str):
            path = model_file
        elif isinstance(model_file, tuple):        
            path = rospkg.RosPack().get_path(model_file[0])+model_file[1]
        else:
            raise AttributeError("model_definition should be either a tuple (pkg, path) or a string (path)")

        if model_format is None and model_file is not None:
            filename_split = path.split(".")
            ext = filename_split[-1]
            if ext == "urdf":
                model_format = "urdf"
            elif ext == "sdf":
                model_format = "sdf"
            elif ext == "xacro":
                ext = filename_split[-2]
                if ext == "urdf":
                    model_format = "urdf"
                elif ext == "sdf":
                    model_format = "sdf"
        if model_format is None:
            raise RuntimeError(f"Model definition format was not specified and could not determine it automatically. model_file = {model_file}")
        
        spawn_model(xacro_file_path=path,
                    pose=pose,
                    model_name=model_name,
                    args=model_kwargs,
                    format=model_format,
                    xacro_string=model_definition_string)
        return model_name

    def delete_model(self, model_name : str):
        """Delete a model from the environment"""
        delete_model(model_name=model_name)

