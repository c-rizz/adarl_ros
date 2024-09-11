#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union, overload

from adarl.utils.utils import JointState, LinkState, Pose, build_pose, buildQuaternion
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter

from xbot2_mujoco.PyXbotMjSimEnv import XBotMjSimEnv
from xbot2_mujoco.PyXbotMjSimEnv import LoadingUtils

import numpy as np

import time
import torch as th

class XbotMjAdapter(BaseSimulationAdapter, RosXbotAdapter
    ):

    """This class allows to control the execution of a Mujoco+XBot2 simulation and command the robot thorugh XBot2 ROS topic interface.

    """

    def __init__(self,
        model_name : str,
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
        jpos_cmd_max_acc_default = 0.0,
        enable_filters = True):

        """Initialize the Simulator.

        """
        super().__init__(model_name=model_name,
                        stepLength_sec=stepLength_sec,
                        forced_ros_master_uri=forced_ros_master_uri,
                        maxObsDelay=maxObsDelay,
                        blocking_observation=blocking_observation,
                        is_floating_base=is_floating_base,
                        reference_frame=reference_frame,
                        torch_device=torch_device,
                        fallback_cmd_stiffness=fallback_cmd_stiffness,
                        fallback_cmd_damping=fallback_cmd_damping,
                        allow_fallback=allow_fallback,
                        jpos_cmd_max_vel=jpos_cmd_max_vel,
                        jpos_cmd_max_vel_default=jpos_cmd_max_vel_default,
                        jpos_cmd_max_acc=jpos_cmd_max_acc,
                        jpos_cmd_max_acc_default=jpos_cmd_max_acc_default,
                        enable_filters=enable_filters)
        
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        raise NotImplementedError()

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        raise NotImplementedError()

    def setupLight(self):
        raise NotImplementedError()
    
    def spawn_model(self,   
        model_name : str,
        model_definition_string : Optional[str] = None,
        model_format : Optional[str] = None,
        model_file : Optional[str] = None,
        pose : Pose = build_pose(0,0,0,0,0,0,1),
        model_kwargs : Dict[Any,Any] = {}) -> str:
        raise NotImplementedError()
    
    def delete_model(self, model_name : str):
        raise NotImplementedError()
    
    def getEnvTimeFromStartup(self) -> float:
        raise NotImplementedError()
    
    def getEnvTimeFromReset(self) -> float:
        raise NotImplementedError()
    
    def set_monitored_joints(self, jointsToObserve : List[Tuple[str,str]]):
        raise NotImplementedError()
    
    def set_monitored_links(self, linksToObserve : List[Tuple[str,str]]):
        raise NotImplementedError()
    
    def set_monitored_cameras(self, camerasToRender : List[str] = []):
        raise NotImplementedError()
    
    def run(self, duration_sec : float):
        raise NotImplementedError()
    
    def startup(self):
        raise NotImplementedError()
    
    def resetWorld(self):
        raise NotImplementedError()
    
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        raise NotImplementedError()
    
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        raise NotImplementedError()
    
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
        raise NotImplementedError()
    
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        raise NotImplementedError()
    
    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        raise NotImplementedError()
    
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