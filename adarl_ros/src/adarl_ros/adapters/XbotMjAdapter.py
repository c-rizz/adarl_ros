#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union, overload

from adarl.utils.utils import JointState, LinkState, Pose, build_pose, buildQuaternion
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
from adarl_ros.adapters.RosXbotAdapter import RosXbotAdapter

from xbot2_mujoco.PyXbotMjSimEnv import XBotMjSimEnv

import numpy as np

import time
import torch as th
import rospy
import os
from typing_extensions import override

class XbotMjAdapter(RosXbotAdapter, BaseSimulationAdapter
    ):

    """This class allows to control the execution of a Mujoco+XBot2 simulation and command the robot thorugh XBot2 ROS topic interface.

    """

    def __init__(self,
        model_fpath: str,
        model_name : str,
        stepLength_sec : float,
        xbot2_config_path: str = None,
        headless: bool = False,
        init_steps: int = 0,
        timeout_ms: int = 1000,
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
        self._model_fpath=model_fpath
        self._xbot2_config_path=xbot2_config_path
        self._headless=headless
        self._init_steps=init_steps
        self._timeout_ms=timeout_ms
        self._closed=False

        self._xmj_sim_env=None
        self._abs_sim_timer=0
        self._sim_time=0
        sim_ok=self._init_simulation() # after this, all data from sim is available
        if not sim_ok:
            msg="Failed to initialize simulation!!"
            ggLog.error(f"{__class__}: {msg}")
            raise RuntimeError(msg)

        if not (stepLength_sec==self._xmj_env.physics_dt):
            msg=f"stepLength_sec {stepLength_sec} is not equal to {self._xmj_env.physics_dt} (physics dt)"
            ggLog.error(f"{__class__}: {msg}")
            raise ValueError(msg)

        self._jimpedance_controlled_joints : list[tuple[str,str]] = []

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

        joints_to_observe = [(model_name, joint) for joint in self._xmj_env_jnt_names]
        self.set_monitored_joints(joints_to_observe)
        self.set_monitored_links([])

    def __del__(self):
        self._close()

    def _close(self):
        if not self._closed:
            if self._xmj_env is not None:
                self._xmj_env.close()
            self._closed=True

    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        raise NotImplementedError()

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        raise NotImplementedError()

    def setupLight(self):
        raise NotImplementedError()
    
    def sim_is_running(self):
        return self._xmj_env.is_running()

    def _init_simulation(self):
        self._xmj_env = XBotMjSimEnv(
            model_fname=self._model_fpath,
            xbot2_config_path=self._xbot2_config_path,
            headless=self._headless,
            manual_stepping=True,
            init_steps=self._init_steps,
            timeout=self._timeout_ms # [ms]
        )

        reset_ok=self._xmj_env.reset()

        pi=np.zeros((3))
        qi=np.zeros((4))
        qi[0] = 1
        pi[2]=self._xmj_env.p[2]
        self._xmj_env.set_pi(pi)
        self._xmj_env.set_qi(qi)
        
        if reset_ok:
            for i in range(0, self._init_steps):
                if not self._xmj_env.step(): 
                    return False
        else:
            return False
        
        pi[2]= self._xmj_env.p[2] # uise pz after init tsteps as
        # # initial spawning height
        self._xmj_env.set_pi(pi)
        reset_ok=self._xmj_env.reset()
        if not reset_ok:
            return False
        self._xmj_env_jnt_names=self._xmj_env.jnt_names()
        self._xmk_evn_n_dofs=self._xmj_env.n_jnts()
        
        return True

    def build_scenario(self, file_path = None, format = "urdf"):
        pass

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
        self._abs_sim_timer+=self.getEnvTimeFromReset()
        return self._abs_sim_timer
    
    def getEnvTimeFromReset(self) -> float:
        return self._sim_time
    
    def set_monitored_links(self, linksToObserve : List[Tuple[str,str]]):
        super().set_monitored_links(linksToObserve=linksToObserve)
    
    def set_monitored_cameras(self, camerasToRender : List[str] = []):
        super().set_monitored_cameras(camerasToRender=camerasToRender)
    
    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[Tuple[str,str]]):
        self._jimpedance_controlled_joints = list(joint_names)
        self._monitored_to_controlled_idxs = th.as_tensor([self._monitored_joints.index(jn) for jn in self._jimpedance_controlled_joints])

    @override
    def set_monitored_joints(self, jointsToObserve: Sequence[tuple[str, str]]):
        ret = super().set_monitored_joints(jointsToObserve=jointsToObserve)
        self._monitored_to_controlled_idxs = th.as_tensor([self._monitored_joints.index(jn) for jn in self._jimpedance_controlled_joints], dtype = th.long)
        return ret
    
    @override
    def clear_commands(self):
        super().clear_commands()
        self._commanded_joint_impedances : dict[float, dict] = {}

    def run(self, duration_sec : float):

        n_sim_steps_to_do=round(duration_sec/self._xmj_env.physics_dt)
        for i in range(n_sim_steps_to_do):
            step_ok=self._xmj_env.step()
            if not step_ok:
                msg=f"Failed to step XMj simulation!"
                ggLog.error(f"{__class__}: {msg}")
                raise ValueError(msg)
            self._sim_time+=self._stepLength_sec
    
    def step(self) -> float:
        # always step on a _xmj_env environment dt
        stime_before=self._sim_time
        self.run(duration_sec=self._stepLength_sec)
        return self._sim_time-stime_before
    
    def startup(self):
        super().startup()
        rospy.loginfo("ROS time is "+str(rospy.get_time())+" pid = "+str(os.getpid()))
        self.resetWorld()

    def xmj_env(self):
        return self._xmj_env
    
    def resetWorld(self):
        super(RosXbotAdapter, self).resetWorld()
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
        reset_ok=self._xmj_env.reset()
        if not reset_ok:
            raise RuntimeError(f"Sim env reset failed!")
        
        self._sim_time=0

    def _setup_joint_control(self, control_mask : int, timeout_s = 300.0) -> int:
        mask = -1
        def check_done():
            # This will be run at the end of the async run. Either if it times out
            # or if it completes (successfully or not)
            nonlocal mask
            # ggLog.info(f"callback: mask = {mask}")
            if mask != control_mask:
                raise RuntimeError(f"Failed to switch control to {mask}.")
        self.run_async(on_finish_callback=check_done)
        # self._gazeboAdapter.unpauseSimulation()
        mask = super()._setup_joint_control(control_mask, timeout_s=timeout_s)
        # ggLog.info(f"_setup_joint_control returned {control_mask}")
        # ggLog.info(f"control_mask = {control_mask}")
        # self._gazeboAdapter.pauseSimulation()
        self.stop_run_async()
        self.wait_run_async(timeout_sec=60) # check_done will always be run before this
        # check_done()
        return mask
    
    def _switch_control(self, switch_on: bool, timeout_s = 300.0) -> bool:
        # The switch service works only if the simulation is running
        switched_on = not switch_on
        def check_done():
            # This will be run at the end of the async run. Either if it times out
            # or if it completes (successfully or not)
            nonlocal switched_on
            if switched_on != switch_on:
                raise RuntimeError(f"Failed to switch control to {switch_on}.")
        self.run_async(on_finish_callback=check_done)
        switched_on = super()._switch_control(switch_on, timeout_s=timeout_s)
        self.stop_run_async()
        self.wait_run_async(timeout_sec = timeout_s) # check_done will always be run before this
        # check_done()
        return switched_on

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        
        jnts_q=self._xmj_env.jnts_q
        jnts_v=self._xmj_env.jnts_v
        jnts_a=self._xmj_env.jnts_a
        jnts_eff=self._xmj_env.jnts_eff

        ret = {}
        for rj in requestedJoints:
            jnt_idx=-1
            try:
                jnt_idx=self._xmj_env_jnt_names.index(rj)    
            except:
                ggLog.info(f"Joint {rj} not found in available joint. Cannot read state.")
                continue        
            jointState = JointState(list(jnts_q[jnt_idx]),
                                    list(jnts_v[jnt_idx]),
                                    list(jnts_eff[jnt_idx]))
            ret[rj] = jointState

        return ret

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        self._xmj_env.p
        self._xmj_env.q
        self._xmj_env.twist
        self._xmj_env.jnts_q

        ret = {}
        for rl in requestedLinks:
            if not (("base_link" in rl) or ("root_link" in rl)):
                ggLog.info(f"getLinksState currently supports reading base link state only!")
            else:
                linkState = LinkState(position_xyz = (self._xmj_env.p[0], self._xmj_env.p[1], self._xmj_env.p[2]),
                    orientation_xyzw = (self._xmj_env.q[1], self._xmj_env.q[2], self._xmj_env.q[3], self._xmj_env.q[0]),
                    pos_velocity_xyz = (self._xmj_env.twist[0], self._xmj_env.twist[1]. self._xmj_env.twist[2]),
                    ang_velocity_xyz = (self._xmj_env.twist[3], self._xmj_env.twist[4], self._xmj_env.twist[5]))
                ret[rl] = linkState
        return ret
    
    def getRenderings(self, requestedCameras : List[str]) -> Dict[str, Tuple[np.ndarray, float]]:
        raise NotImplementedError()
    
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        controlled_joints = {jn:js for jn,js in jointStates.items() if jn in self.get_controlled_joints()}
        uncontrolled_joints = {jn:js for jn,js in jointStates.items() if jn not in self.get_controlled_joints()}

        super().moveToJointPoseSync(jointPositions={jn:js.position.item() for jn,js in controlled_joints.items()},
                                    velocity_scaling=1.0,
                                    acceleration_scaling=1.0,
                                    joint_position_tolerance=0.05)

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        raise NotImplementedError()
    
    def get_joints_state_step_stats(self):
        raise NotImplementedError()