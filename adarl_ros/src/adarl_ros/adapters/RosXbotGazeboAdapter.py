#!/usr/bin/env python3
import os
import time
from threading import Lock
from typing import Dict, List, Tuple, Union, Optional, Any

import adarl.utils.beep
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import rospkg
import rospy
import sensor_msgs.msg
from adarl_ros.adapters.RosAdapter import RosAdapter
from adarl.utils.utils import JointState, LinkState, RequestFailError, Pose, build_pose, MoveFailError
import numpy as np

from xbot_interface import config_options as opt
from xbot_interface import xbot_interface as xbot
from urdf_parser_py.urdf import URDF

from std_srvs.srv import SetBool
from xbot_msgs.srv import PluginStatus, SetControlMask
from xbot_msgs.msg import JointDeviceInfo
import torch as th
from adarl_ros.adapters.GazeboAdapter import GazeboAdapter
from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter
from typing_extensions import override
from cartesian_interface.affine3 import Affine3 # needed by xbot_interface as it doesn't import it correctly
import traceback

from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from adarl_ros.adapters.GazeboAdapter import GazeboAdapter
from adarl_ros.adapters.GazeboAdapterNoPlugin import GazeboAdapterNoPlugin
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
from adarl.adapters.BaseJointEffortAdapter import BaseJointEffortAdapter
from adarl_ros.adapters.RosAdapter import RosAdapter
from adarl.adapters.BaseAdapter import BaseAdapter
import time


class RosXbotGazeboAdapter(RosXbotAdapter, BaseSimulationAdapter):

    def __init__(self,  model_name : str,
                        stepLength_sec : float,
                        forced_ros_master_uri : Union[str, None] = None,
                        maxObsDelay = float("+inf"),
                        blocking_observation = False,
                        is_floating_base : bool = True,
                        reference_frame : str = "world",
                        torch_device : th.device = th.device("cpu"),
                        fallback_cmd_stiffness : float = 200.0,
                        fallback_cmd_damping : float= 100.0,
                        allow_fallback : bool = True,
                        jpos_cmd_max_vel = {},
                        jpos_cmd_max_vel_default = 0.0,
                        jpos_cmd_max_acc = {},
                        jpos_cmd_max_acc_default = 0.0):
        # Could probably use multiple inheritance, but would need to structure all the variuso classes accordingly,
        # and it would just be confusing. So instead I use a separate wrapped gazebo adapter.
        super().__init__(model_name = model_name,
                        stepLength_sec = stepLength_sec,
                        forced_ros_master_uri = forced_ros_master_uri,
                        maxObsDelay = maxObsDelay,
                        blocking_observation = blocking_observation,
                        is_floating_base = is_floating_base,
                        reference_frame = reference_frame,
                        torch_device = torch_device,
                        fallback_cmd_stiffness = fallback_cmd_stiffness,
                        fallback_cmd_damping = fallback_cmd_damping,
                        allow_fallback = allow_fallback,
                        jpos_cmd_max_vel = jpos_cmd_max_vel,
                        jpos_cmd_max_vel_default = jpos_cmd_max_vel_default,
                        jpos_cmd_max_acc = jpos_cmd_max_acc,
                        jpos_cmd_max_acc_default = jpos_cmd_max_acc_default)
        self._gazeboAdapter = GazeboAdapter(usePersistentConnections=False,
                                            stepLength_sec=stepLength_sec,
                                            rosMasterUri=forced_ros_master_uri)

    @override
    def getEnvTimeFromStartup(self) -> float:
        return self._gazeboAdapter.getEnvTimeFromStartup()
    
    @override
    def getEnvTimeFromReset(self) -> float:
        return self._gazeboAdapter.getEnvTimeFromReset()
    
    @override
    def set_monitored_joints(self, jointsToObserve : List[Tuple[str,str]]):
       super().set_monitored_joints(jointsToObserve=jointsToObserve)
       self._gazeboAdapter.set_monitored_joints(jointsToObserve=jointsToObserve)

    @override
    def set_monitored_links(self, linksToObserve : List[Tuple[str,str]]):
       super().set_monitored_links(linksToObserve=linksToObserve)
       self._gazeboAdapter.set_monitored_links(linksToObserve=linksToObserve)

    @override
    def set_monitored_cameras(self, camerasToRender : List[str] = []):
       super().set_monitored_cameras(camerasToRender=camerasToRender)
       self._gazeboAdapter.set_monitored_cameras(camerasToRender=camerasToRender)

    @override
    def run(self, duration_sec : float):
        self._apply_controls()
        self._gazeboAdapter.run(duration_sec)

    @override
    def startup(self):
        super().startup()
        self._gazeboAdapter._makeRosConnections()
        rospy.loginfo("ROS time is "+str(rospy.get_time())+" pid = "+str(os.getpid()))
        # self._gazeboAdapter.unpauseSimulation()
        # ggLog.info(f"unpaused")
        self.resetWorld()
        # ggLog.info(f"resetted")
        # self._gazeboAdapter.pauseSimulation()
        # ggLog.info(f"paused")

    @override
    def resetWorld(self):
        super(RosXbotAdapter, self).resetWorld() # skip the reset of RosXbotAdapter, we'll do it ourselves
        self._gazeboAdapter.resetWorld()

        # switched_on = self._switch_control(True)
        # if not switched_on:
        #     raise RuntimeError(f"Failed to switch on control.")
        self.clear_commands()

        self._last_jdi_time = -1
        self._last_joint_device_info = None
        req_mask = 255
        mask = self._setup_joint_control(control_mask = req_mask)
        if mask != req_mask:
            raise RuntimeError(f"Failed to set control mask, wanted {req_mask}, got {mask}")
        switched_on = self._switch_control(True, timeout_s = 300.0)
        if not switched_on:
            raise RuntimeError(f"Failed to switch on control.")


    
    def _setup_joint_control(self, control_mask : int, timeout_s = 300.0) -> int:
        mask = -1
        def check_done():
            # This will be run at the end of the async run. Either if it times out
            # or if it completes (successfully or not)
            nonlocal mask
            # ggLog.info(f"callback: mask = {mask}")
            if mask != control_mask:
                raise RuntimeError(f"Failed to switch control to {mask}.")
        self.run_async(duration_sec=timeout_s, on_finish_callback=check_done)
        # self._gazeboAdapter.unpauseSimulation()
        mask = super()._setup_joint_control(control_mask, timeout_s=timeout_s)
        # ggLog.info(f"_setup_joint_control returned {control_mask}")
        # ggLog.info(f"control_mask = {control_mask}")
        # self._gazeboAdapter.pauseSimulation()
        self.stop_run_async()
        self.wait_run_async() # check_done will always be run before this
        # check_done()
        return mask
        
    def _switch_control(self, switch_on: bool, timeout_s = 300.0) -> bool:
        # The switch service works only if the simulation is running
        switched_on = not switch_on
        def check_done():
            # This will be run at the end of the async run. Either if it times out
            # or if it completes (successfully or not)
            nonlocal switched_on
            # ggLog.info(f"callback: switched_on = {switched_on}")
            if switched_on != switch_on:
                raise RuntimeError(f"Failed to switch control to {switch_on}.")
        self.run_async(duration_sec=timeout_s, on_finish_callback=check_done)
        # self._gazeboAdapter.unpauseSimulation()
        switched_on = super()._switch_control(switch_on, timeout_s=timeout_s)
        # ggLog.info(f"switch_control returned {switched_on}")
        # ggLog.info(f"switched_on = {switched_on}")
        # self._gazeboAdapter.pauseSimulation()
        self.stop_run_async()
        self.wait_run_async() # check_done will always be run before this
        # check_done()
        return switched_on

    @override
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        # js = super().getJointsState(requestedJoints=requestedJoints)
        # return js
        try:
            js = self._gazeboAdapter.getJointsState(requestedJoints=requestedJoints)
        except RequestFailError as e:
            ggLog.warn(f"Failed to read joints from gazebo, using xbot/ros")
            missing_jonts = [jr for jr in requestedJoints if jr not in e.partialResult]
            js = super().getJointsState(requestedJoints=missing_jonts)
            js.update(e.partialResult)
        return js


    @override
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        try:
            ls =  self._gazeboAdapter.getLinksState(requestedLinks=requestedLinks) # WARNING! These may not be the same frames as the urdf ones!
        except RequestFailError as e:
            ggLog.warn(f"Failed to read links from gazebo, using xbot/ros")
            missing_links = [rl for rl in requestedLinks if rl not in e.partialResult]
            ls = super().getLinksState(requestedLinks=missing_links)
            ls.update(e.partialResult)
        return ls

    @override
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
        try:
            r = super().getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from ros")
        except:
            r = self._gazeboAdapter.getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from gazebo plugin")
        return r
    
    @override
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        controlled_joints =   {jn:js for jn,js in jointStates.items() if jn in self.get_controlled_joints()}
        uncontrolled_joints = {jn:js for jn,js in jointStates.items() if jn not in self.get_controlled_joints()}

        self._gazeboAdapter.setJointsStateDirect(jointStates=uncontrolled_joints)
        e = 0.001
        prev_lims = self._gazeboAdapter.get_sim_joint_limits(list(uncontrolled_joints.keys()))
        self._gazeboAdapter.set_sim_joint_limits(joint_limits_minmax={jn:(js.position.item()-e, js.position.item()+e) for jn, js in uncontrolled_joints.items()})

        # self.run(1.0)
        retry = 5
        for i in range(retry):
            try:
                super().moveToJointPoseSync(jointPositions={jn:js.position.item() for jn,js in controlled_joints.items()},
                                            velocity_scaling=1.0,
                                            acceleration_scaling=1.0,
                                            joint_position_tolerance=0.05)
                break
            except MoveFailError as e:
                ggLog.warn(f"Failed to move to initial pose, will retry {retry-1} times. Exception: \n{e}")
        self._gazeboAdapter.set_sim_joint_limits(joint_limits_minmax=prev_lims)

        # # wasPaused = self._gazeboAdapter.isPaused()
        # # if wasPaused:
        # #     self._gazeboAdapter.unpauseSimulation()
        # self._switch_control(False)
        # self._setup_joint_control(0)
        # self._gazeboAdapter.setJointsStateDirect(jointStates = jointStates)
        # self.apply_joint_impedances(joint_impedances_pvesd=
        #                             [(jn,(js.position.item(), 0.0, 0.0, self._fallback_cmd_stiffness, self._fallback_cmd_damping))
        #                               for jn,js in jointStates.items() if jn in self._xbotjname_to_jid])
        # # time.sleep(10)
        # self._setup_joint_control(255)
        # self._switch_control(True)
        
        # # if wasPaused:
        # #     self._gazeboAdapter.pauseSimulation()
    
    @override
    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        # wasPaused = self._gazeboAdapter.isPaused()
        # if wasPaused:
        #     self._gazeboAdapter.unpauseSimulation()
        self._gazeboAdapter.setLinksStateDirect(linksStates = linksStates)
        # if wasPaused:
            # self._gazeboAdapter.pauseSimulation()

    @override
    def delete_model(self, model_name : str):
        """Delete a model from the environment"""
        self._gazeboAdapter.delete_model(model_name = model_name)
    
    @override
    def setupLight(self, *args, **kwargs):
        self._gazeboAdapter.setupLight(*args, **kwargs)
        

    @override
    def spawn_model(self,   model_name : str,
                            model_definition_string : Optional[str] = None,
                            model_format : Optional[str] = None,
                            model_file : Optional[str] = None,
                            pose : Pose = build_pose(0,0,0,0,0,0,1),
                            model_kwargs : Dict[Any,Any] = {}) -> str:
        """Spawn a model in the simulation

        Parameters
        ----------
        model_definition_string : str
            Model definition specified in as a string. e.g. an SDF definition
        model_format : str
            Format of the model definition. E.g. 'sdf' or 'urdf'
        model_file : _type_
            File to load the model definition from
        model_name : str
            Name to give to the spawned model
        pose : Pose
            Pose to spawn the model at
        model_kwargs : Dict[Any,Any]
            Arguments to use in interpreting the model definition

        Returns
        -------
        str
            The model name
        """
        return self._gazeboAdapter.spawn_model(model_file = model_file,
                                                pose=pose,
                                                model_name=model_name,
                                                model_kwargs=model_kwargs,
                                                model_format=model_format,
                                                model_definition_string=model_definition_string)
