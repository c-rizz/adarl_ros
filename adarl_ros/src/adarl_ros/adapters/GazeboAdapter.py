#!/usr/bin/env python3

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Union, Mapping

from torch._tensor import Tensor

import gazebo_gym_env_plugin.msg
import gazebo_gym_env_plugin.srv
import adarl.utils.dbg.ggLog as ggLog
import rospy
import sensor_msgs
import sensor_msgs.msg
from adarl.adapters.BaseJointEffortAdapter import BaseJointEffortAdapter
from adarl.utils.utils import JointState, LinkState
from adarl_ros.adapters.GazeboAdapterNoPlugin import GazeboAdapterNoPlugin
import numpy as np
import adarl.utils.utils
from typing_extensions import override
from adarl.adapters.BaseAdapter import JointName, LinkName

class GazeboAdapter(GazeboAdapterNoPlugin):
    """This class allows to control the execution of a Gazebo simulation.

    It makes use of the adarl_ros_env gazebo plugin to perform simulation stepping and rendering.
    """

    class _SimState:
        stepNumber = -1
        jointsState = {} # key = (model_name, joint_name), value=gazebo_gym_env_plugin.JointInfo
        linksState = {}
        cameraRenders = {} # key = camera_name, value = (sensor_msgs.Image, sensor_msgs.CameraInfo)

    def __init__(   self,
                    usePersistentConnections : bool = False,
                    stepLength_sec : float = 0.001,
                    rosMasterUri : Union[str,None] = None):
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
        fastRendering : bool
            Performs the rendering at each step(), and caches it for future render()
            call. This avoids some overhead.
            Enable this only if you really need the rendering at each step, otherwise,
            performing the rendering at each step would be just a waste of resources.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """

        super().__init__(stepLength_sec=stepLength_sec, rosMasterUri = rosMasterUri)
        self._usePersistentConnections = usePersistentConnections
        self._simulationState = GazeboAdapter._SimState()
        self._jointEffortsToRequest = []
        self._totalCountedSimDuration_pico = 0

    def _makeRosConnections(self):
        super()._makeRosConnections()

        serviceNames = {"step" : "/gazebo/gym_env_interface/step",
                        "render" : "/gazebo/gym_env_interface/render",
                        "get_info" : "/gazebo/gym_env_interface/get_info",
                        "setJointProperties" : "/gazebo/gym_env_interface/set_joint_properties"}

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

        self._stepGazeboService   = rospy.ServiceProxy(serviceNames["step"], gazebo_gym_env_plugin.srv.StepSimulation, persistent=self._usePersistentConnections)
        self._renderGazeboService   = rospy.ServiceProxy(serviceNames["render"], gazebo_gym_env_plugin.srv.RenderCameras, persistent=self._usePersistentConnections)
        self._infoGazeboService   = rospy.ServiceProxy(serviceNames["get_info"], gazebo_gym_env_plugin.srv.GetInfo, persistent=self._usePersistentConnections)
        self._setJointPropertiesService = rospy.ServiceProxy(serviceNames["setJointProperties"], gazebo_gym_env_plugin.srv.SetJointProperties, persistent=self._usePersistentConnections)


    def isPaused(self):
        # ggLog.info(f"Calling _infoGazeboService")
        ret = self._infoGazeboService.call().is_paused
        # ggLog.info(f"Called _infoGazeboService")
        return ret

    def _step_sim(self, duration_pico : int):
        # ggLog.info(f"_step_sim({duration_pico})")
        request = gazebo_gym_env_plugin.srv.StepSimulationRequest()
        request.step_duration_picosecs = duration_pico
        request.request_time = time.time()
        #ggLog.info("self._camerasToObserve = "+str(self._camerasToObserve))
        if len(self._monitored_cameras)>0:
            #ggLog.info("Performing rendering within step")
            request.render = True
            request.cameras = self._monitored_cameras
        if len(self._monitored_joints)>0:
            request.requested_joints = []
            for j in self._monitored_joints:
                jointId = gazebo_gym_env_plugin.msg.JointId()
                jointId.joint_name = j[1]
                jointId.model_name = j[0]
                request.requested_joints.append(jointId)
        if len(self._monitored_links)>0:
            request.requested_links = []
            for l in self._monitored_links:
                linkId = gazebo_gym_env_plugin.msg.LinkId()
                linkId.link_name  = l[1]
                linkId.model_name = l[0]
                request.requested_links.append(linkId)

        request.joint_effort_requests = self._jointEffortsToRequest
        #print("Step request = "+str(request))

        servicecalltries = 0
        while True:
            try:
                # ggLog.info(f"Calling _stepGazeboService")
                response = self._stepGazeboService.call(request)
                # ggLog.info(f"Called _stepGazeboService")
                break
            except rospy.service.ServiceException as e:
                if servicecalltries > 20:
                    ggLog.error("Gazebo step service failed too many times.")
                    raise e
                else:
                    ggLog.error(f"Gazebo step service call failed {servicecalltries} times. Exception:\n"+str(e)+"\n retrying")
                    time.sleep(1)
            servicecalltries += 1
        self._episode_steps_taken +=1
        step_duration_sec = request.step_duration_picosecs / 1e12
        self._episodeCountedSimDuration += step_duration_sec
        self._totalCountedSimDuration += step_duration_sec
        self._totalCountedSimDuration_pico += request.step_duration_picosecs 
        self._simulationState.stepNumber = self._episode_steps_taken

        # print("Step response = "+str(response))
        #rospy.loginfo("Transfer time of stepping response = "+str(time.time()-response.response_time))


        if len(self._monitored_cameras)>0:
            if not response.render_result.success:
                ggLog.warn("Error getting renderings: "+response.render_result.error_message)
            for i in range(len(response.render_result.camera_names)):
                #ggLog.info("got image for camera "+response.render_result.camera_names[i])
                self._simulationState.cameraRenders[response.render_result.camera_names[i]] = (response.render_result.images[i],response.render_result.camera_infos[i])

        if len(self._monitored_joints)>0:
            if not response.joints_info.success:
                ggLog.warn("Error getting joint information: "+response.joints_info.error_message)
            for ji in response.joints_info.joints_info:
                # ggLog.info(f"Got joint info {(ji.joint_id.model_name,ji.joint_id.joint_name)} = {ji}")
                self._simulationState.jointsState[(ji.joint_id.model_name,ji.joint_id.joint_name)] = ji

        if len(self._monitored_links)>0:
            if not response.links_info.success:
                ggLog.warn("Error getting link information: "+response.joints_info.error_message)
            for li in response.links_info.links_info:
                self._simulationState.linksState[(li.link_id.model_name,li.link_id.link_name)] = li

        #print("Step done, joint state = "+str(self._simulationState.jointsState))
        if not response.success:
            ggLog.error("Simulation stepping failed")
        # ggLog.info(f"Stepped of {request.step_duration_nanosecs / 1e9:.10f} (requested {duration_sec:.10f})s")

    @override
    def run(self, duration_sec : float):
        self._step_sim(duration_pico=int(duration_sec * 1e12))
    
    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._totalCountedSimDuration_pico /1e12
    
    def _performRender(self, requestedCameras : List[str]):
        # ggLog.info("Rendering cameras "+str(requestedCameras))
        req = gazebo_gym_env_plugin.srv.RenderCamerasRequest()
        req.cameras=requestedCameras
        req.request_time = time.time()
        #t0 = time.time()
        # ggLog.info(f"Calling _renderGazeboService")
        res = self._renderGazeboService.call(req)
        # ggLog.info(f"Calling _renderGazeboService")
        #t1 = time.time()
        #rospy.loginfo("Transfer time of rendering response = "+str(time.time()-res.response_time))

        if not res.render_result.success:
            ggLog.error("Error rendering cameras: "+res.render_result.error_message)

        renders = {}
        for i in range(len(res.render_result.camera_names)):
            renders[res.render_result.camera_names[i]] = (res.render_result.images[i],res.render_result.camera_infos[i])


        return renders

    @override
    def getRenderings(self, requestedCameras : List[str]) -> dict[str, tuple[np.ndarray, float]]:
        # ggLog.info("GazebController.getRenderings")
        for name in requestedCameras:
            if name not in self._camerasToObserve:
                # print(f"Requested rendering camera {name} which was not set with set_monitored_cameras (cameras are {self._camerasToObserve})")
                raise RuntimeError(f"Requested rendering camera {name} which was not set with set_monitored_cameras (cameras are {self._camerasToObserve})")

        if self._simulationState.stepNumber!=self._episode_steps_taken: #If no step has ever been done
            # ggLog.info("Manually rendering images for "+str(requestedCameras))
            cameraRenders = self._performRender(requestedCameras)
        else:
            # ggLog.info("Using available renders for "+str(requestedCameras)+" step = "+str(self._simulationState.stepNumber))
            cameraRenders = self._simulationState.cameraRenders

        ret = {}
        for name in requestedCameras:
            rosimg = cameraRenders[name][0]
            ret[name] = (adarl.utils.utils.ros1_image_to_numpy(rosimg), rosimg.header.stamp.to_sec())
        return ret

    @override
    def getJointsState(self, requestedJoints : List[tuple[str,str]]) -> dict[tuple[str,str],JointState]:
        if self._simulationState.stepNumber!=self._episode_steps_taken: #If no step has ever been done
            return super().getJointsState(requestedJoints)

        ret = {}
        for rj in requestedJoints:
            jointInfo = self._simulationState.jointsState[rj]
            jointState = JointState(list(jointInfo.position),
                                    list(jointInfo.rate),
                                    list(jointInfo.effort))
            ret[rj] = jointState
        return ret

    @override
    def getLinksState(self, requestedLinks : List[tuple[str,str]]) -> dict[tuple[str,str],LinkState]:

        if self._simulationState.stepNumber!=self._episode_steps_taken: #If no step has ever been done
            return super().getLinksState(requestedLinks)

        ret = {}
        for rl in requestedLinks:
            linkInfo = self._simulationState.linksState[rl]

            linkState = LinkState(  position_xyz = (linkInfo.pose.position.x, linkInfo.pose.position.y, linkInfo.pose.position.z),
                                    orientation_xyzw = (linkInfo.pose.orientation.x, linkInfo.pose.orientation.y, linkInfo.pose.orientation.z, linkInfo.pose.orientation.w),
                                    pos_com_velocity_xyz = (linkInfo.twist.linear.x, linkInfo.twist.linear.y, linkInfo.twist.linear.z),
                                    ang_velocity_xyz = (linkInfo.twist.angular.x, linkInfo.twist.angular.y, linkInfo.twist.angular.z))
            ret[rl] = linkState
        return ret

    @override
    def setJointsEffortCommand(self, jointTorques : List[tuple[str,str,float]]) -> None:
        self._jointEffortsToRequest = []
        for jt in jointTorques:
            jer = gazebo_gym_env_plugin.msg.JointEffortRequest()
            jer.joint_id.model_name = jt[0]
            jer.joint_id.joint_name = jt[1]
            jer.effort = jt[2]
            self._jointEffortsToRequest.append(jer)



    def setJointsStateDirect(self, jointStates : dict[tuple[str,str],JointState]):
        r = super().setJointsStateDirect(jointStates=jointStates)
        self._step_sim(0) # update simulation state
        return r
    
    @override
    def setLinksStateDirect(self, linksStates : dict[tuple[str,str],LinkState]):
        r = super().setLinksStateDirect(linksStates=linksStates)
        self._step_sim(0) # update simulation state
        return r
    
    def set_sim_joint_limits(self, joint_limits_minmax : Mapping[JointName, tuple[float | None,float  | None]]):
        if len(joint_limits_minmax)==0:
            return
        for (model_name, joint_name), (min_pos, max_pos) in joint_limits_minmax.items():
            msg = gazebo_gym_env_plugin.srv.SetJointPropertiesRequest()
            jp = gazebo_gym_env_plugin.msg.JointProperties()
            jp.joint_id.model_name = model_name
            jp.joint_id.joint_name = joint_name
            if max_pos is not None:
                jp.position_limit_high = [max_pos]
            if min_pos is not None:
                jp.position_limit_low = [min_pos]
            msg.joint_properties.append(jp)
        res = self._setJointPropertiesService.call(msg)
        if not res.success:
            raise RuntimeError(f"Failed to set joint limits: {res}")
        
    def get_sim_joint_limits(self, joint_names : list[JointName]) -> dict[JointName, tuple[float,float]]:
        if len(joint_names) == 0:
            return {}
        for (model_name, joint_name) in joint_names:
            msg = gazebo_gym_env_plugin.srv.SetJointPropertiesRequest()
            jp = gazebo_gym_env_plugin.msg.JointProperties()
            jp.joint_id.model_name = model_name
            jp.joint_id.joint_name = joint_name
            jp.position_limit_high = []
            jp.position_limit_low = []
            msg.joint_properties.append(jp)
        res = self._setJointPropertiesService.call(msg)
        if not res.success:
            raise RuntimeError(f"Failed to get joint limits: {res}")
        
        for jp in res.resulting_joint_properties:
            if jp.degrees_of_freedom != 1:
                raise RuntimeError(f"Only 1-dof joints are supported, {jp.joint_id} has {jp.degrees_of_freedom} DOF")

        return {(jp.joint_id.model_name, jp.joint_id.joint_name): (jp.position_limit_low[0], jp.position_limit_high[0])
                 for jp in res.resulting_joint_properties}

    @override
    def get_joints_state_step_stats(self) -> Tensor:
        raise NotImplementedError()