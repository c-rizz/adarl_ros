#!/usr/bin/env python3
from __future__ import annotations
import os
import time
from threading import Lock
from typing import Dict, List, Tuple, Union, Optional, Sequence, Mapping

import adarl.utils.beep
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.utils
import rospkg
import rospy
import std_srvs.srv
from adarl_ros.adapters.RosAdapter import RosAdapter
from adarl.utils.utils import JointState, LinkState, RequestFailError, build_1D_vramp_trajectory, MoveFailError
import numpy as np

from xbot_interface import config_options as opt
from xbot_interface import xbot_interface as xbot
from urdf_parser_py.urdf import URDF

from std_srvs.srv import SetBool
from xbot_msgs.srv import PluginStatus, SetControlMask
from xbot_msgs.msg import JointDeviceInfo
import torch as th
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from typing_extensions import override
from cartesian_interface.affine3 import Affine3 # needed by xbot_interface as it doesn't import it correctly
import traceback
from threading import RLock, Condition
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.BaseAdapter import JointName,  LinkName
import adarl.utils.session





# ------------------------------------------------------------------------------------------
# XBOT helper functions
# ------------------------------------------------------------------------------------------



def build_xbot_cfg(is_floating_base):
    """
    A function to construct the xbotinterface config object from ros
    """
    t0 = time.monotonic()
    while True:
        urdf = rospy.get_param('/xbotcore/robot_description', default=None) # type: ignore
        if urdf is not None:
            break
        if time.monotonic() - t0 > 60:
            raise TimeoutError("Timed out waiting for /xbotcore/robot_description. Is xbot-core running?")
        time.sleep(0.2)
    srdf = rospy.get_param('/xbotcore/robot_description_semantic') # type: ignore
    if not isinstance(urdf, str):
        raise RuntimeError(f"URDF is not a string, it's a {type(urdf)}")
    if not isinstance(srdf, str):
        raise RuntimeError(f"SRDF is not a string, it's a {type(srdf)}")
    cfg = opt.ConfigOptions()
    cfg.set_urdf(urdf)
    cfg.set_srdf(srdf)
    cfg.generate_jidmap()
    cfg.set_bool_parameter('is_model_floating_base', is_floating_base)
    cfg.set_string_parameter('model_type', 'RBDL')
    cfg.set_string_parameter('framework', 'ROS')    
    return cfg


def get_link_names(robot):
    urdf = URDF.from_xml_string(robot.getUrdfString())
    print(urdf.links)
    lnames = [l.name for l in urdf.links]
    return lnames

def get_system_recap_string(robot):
    robot.sense()  # update robot pose
    jnames = robot.getEnabledJointNames()   # get list of joint names
    ret = f"XBot URDF:\n{robot.getUrdfString()}"
    ret += f"\nJoint names = {jnames}"

    lnames = get_link_names(robot)
    ret += f"\nlinks = {lnames}"
    for lname in lnames:
        indent = "\n    "
        n = "\n"
        ret += f"\n{lname} pose: {indent+str(robot.model().getPose(lname)).replace(n, indent)}"

    jpos = robot.getJointPosition()  # get actual joint position
    jref = robot.getPositionReference()  # get actual position reference

    for n, q, qref in zip(jnames, jpos, jref):
        ret += f"\n{n}: {q} vs {qref}"
    return ret



def set_filters(set_enabled : bool, required_filter_hz = 20.0):
    enable_filter_srv_name = "/xbotcore/enable_joint_filter"
    rospy.wait_for_service(enable_filter_srv_name)
    enable_filter_srv = rospy.ServiceProxy(enable_filter_srv_name, SetBool)
    set_filter_srv_name = "/xbotcore/set_filter_profile_fast"
    rospy.wait_for_service(set_filter_srv_name)
    set_filter_mode_srv = rospy.ServiceProxy(set_filter_srv_name, std_srvs.srv.Trigger)
    filter_status = not set_enabled
    filter_hz = None
    while filter_status != set_enabled and filter_hz!=required_filter_hz:
        topic_name = "/xbotcore/joint_device_info"
        jdi = rospy.wait_for_message(topic_name, JointDeviceInfo, timeout = 10)
        if not isinstance(jdi, JointDeviceInfo):
            raise RuntimeError(f"Unexpected type received from {topic_name}, should be JointDeviceInfo but it's {type(jdi)}")
        filter_status = jdi.filter_active
        filter_hz = jdi.filter_cutoff_hz
        if filter_hz != required_filter_hz:
            resp = set_filter_mode_srv() # this service always returns success = False
            # if not resp.success: 
            #     raise RuntimeError(f"Failed to set filter cutoff. Status: {resp}")
        if filter_status != set_enabled:
            # print(("Enabling" if set_enabled else "Disabling")+" filters...")
            resp = enable_filter_srv(set_enabled)
            if not resp.success:
                raise RuntimeError(f"Failed to set filters status: {resp}")
    ggLog.info(f"Filters {'enabled' if filter_status else 'disabled'}. Cutoff = {filter_hz}")




def detect_simulated():
    # Is here some better way to do this?
    # Can I ask xbot?
    topic_names = [t[0] for t in rospy.get_published_topics()]
    if "/gazebo/link_states" in topic_names:
        return True
    else:
        return False














class RosXbotAdapter(RosAdapter, BaseJointImpedanceAdapter, BaseJointPositionAdapter):

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
                        jpos_cmd_max_acc_default = 0.0,
                        enable_filters = True,
                        position_commands_stiffness : float = 100.0,
                        position_commands_damping : float = 10.0,
                        is_simulated : bool | None = False,
                        walltime_factor : float = 1.0):
        super().__init__(stepLength_sec, forced_ros_master_uri, maxObsDelay, blocking_observation, walltime_factor=walltime_factor)
        self._is_floating_base = is_floating_base
        self._model_name = model_name
        self._reference_frame = reference_frame
        self._torch_device = torch_device
        self._commanded_joint_impedances_by_name : dict[tuple[str,str], Tuple[float,float,float,float,float] | th.Tensor]= {}
        self._joint_cmd_fallback_by_jid = {}
        self._fallback_cmd_stiffness = fallback_cmd_stiffness
        self._fallback_cmd_damping = fallback_cmd_damping
        self._allow_fallback = allow_fallback
        self._joint_device_info_mutex = RLock()
        self._joint_device_info_cv = Condition(self._joint_device_info_mutex)
        self._last_joint_device_info : JointDeviceInfo | None = None
        self._position_command_stiffness = position_commands_stiffness
        self._position_command_damping = position_commands_damping
        self._commanded_joint_positions : Dict[Tuple[str,str],Tuple[float,float,float]] = {}
        # joint trajectories are ndarrays listing waypoints of format (time, position, velocuty, acceleration)
        self._commanded_joint_trajs_tpva : Dict[Tuple[str,str],np.ndarray]= {}


        self._jpos_cmd_vel_scaling : Dict[Tuple[str,str],float]= {}
        self._jpos_cmd_acc_scaling : Dict[Tuple[str,str],float]= {}
        self._jpos_cmd_vel_scaling_default = 1.0
        self._jpos_cmd_acc_scaling_default = 1.0
        self._jpos_cmd_max_vel = jpos_cmd_max_vel
        self._jpos_cmd_max_vel_default = jpos_cmd_max_vel_default
        self._jpos_cmd_max_acc = jpos_cmd_max_acc
        self._jpos_cmd_max_acc_default = jpos_cmd_max_acc_default

        self._xbotjname_to_jid : Dict[str, int]
        self._jid_to_xbotjname : Dict[int, str]
        self._last_jdi_time = float("-inf")
        self._enable_filters = enable_filters
        self._jimpedance_controlled_joints : list[tuple[str,str]] = []
        self._control_dt = 0.001 # can I get this from somewhere?
        self._is_simulated = is_simulated if is_simulated is not None else detect_simulated()

    def is_simulated(self):
        return self._is_simulated

    def _joint_device_info_callback(self, msg):
        with self._joint_device_info_mutex:
            self._last_jdi_time = self.getEnvTimeFromStartup()
            self._last_joint_device_info = msg

    @override
    def set_monitored_joints(self, jointsToObserve: List[Tuple[str, str]]):
        self._xbot_joints_to_monitor = jointsToObserve # keep empty the normal jointsToObserve and use this instead
        super().set_monitored_joints(jointsToObserve)

    def startup(self):
        super().startup()

        cfg = build_xbot_cfg(is_floating_base=self._is_floating_base)
        self._robot_interface = xbot.RobotInterface(cfg)
        # ggLog.info(get_system_recap_string(self._robot_interface))
        set_filters(True)
        self._setup_joint_control(control_mask=255)
        self._switch_control(self._enable_filters)
        enabled_joint_names = self._robot_interface.getEnabledJointNames() # this is different from robot.model().getEnabledJointNames()
        self._joints_num = len(enabled_joint_names)
        self._xbotjname_to_jid = {jname : enabled_joint_names.index(jname) for jname in enabled_joint_names}
        self._jid_to_xbotjname = {jid : jname for jname, jid in self._xbotjname_to_jid.items()}

        ggLog.info(f"RosXbotAdapter found joints: {list(self._xbotjname_to_jid.keys())}")

        self._jimpedance_controlled_joints_jids = np.array([self._xbotjname_to_jid[jn] for model_name,jn in self._jimpedance_controlled_joints])
        topic_name = "/xbotcore/joint_device_info"
        self._jdi_subscriber = rospy.Subscriber(topic_name, JointDeviceInfo, self._joint_device_info_callback, queue_size=1)
        ggLog.info(f"Subscribed to {topic_name}")

    def get_xbot_controlled_joints(self) -> list[tuple[str,str]]:
        """Get the names of the joint that XBot is controlling

        Returns
        -------
        list[tuple[str,str]]
            The list of the joints
        """
        return [(self._model_name, xbot_jname) for xbot_jname in self._xbotjname_to_jid.keys()]

    @override
    def set_impedance_controlled_joints(self, joint_names : Sequence[Tuple[str,str]]):
        self._jimpedance_controlled_joints = list(joint_names)

    @override
    def get_impedance_controlled_joints(self) -> list[tuple[str,str]]:
        return self._jimpedance_controlled_joints

    def get_joint_device_info(self, after_env_time : float = float("-inf"), timeout_wall : float = 30.0) -> JointDeviceInfo | None:
        with self._joint_device_info_cv:
            self._joint_device_info_cv.wait_for(lambda: self._last_jdi_time >= after_env_time, timeout=timeout_wall)
            ret = self._last_joint_device_info
        return ret

    @override
    def getJointsState(self, requestedJoints : List[Tuple[str,str]] | None = None) -> Dict[Tuple[str,str],JointState] | th.Tensor:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call set_monitored_joints, and then call startController")

        if requestedJoints is None:
            return_tensor=True
            requestedJoints = self._jointsToObserve
        else:
            return_tensor = False

        self._robot_interface.sense(update_model=False)

        #TODO: get the delay time somehow
        # obsDelay = float("+inf")
        # while obsDelay > self._maxObsAge:
        #     self.run(0.001)
        #     self._robot_interface.sense(update_model=False)
        #     obsDelay = self._robot_interface.getTime() - self._robot_interface.getTimestampRx()            
        # self._jointStateMsgAgeAvg.addValue(obsDelay)

        jpos = self._robot_interface.getJointPosition()
        jvel = self._robot_interface.getJointVelocity()
        jeff = self._robot_interface.getJointEffort()


        ret : dict[tuple[str,str], JointState]= {}
        for full_joint_name in requestedJoints:
            model, jname = full_joint_name
            if model != self._model_name:
                raise RuntimeError(f"Requested joint for model different from the monitored one (asked '{model, jname}', but have '{self._model_name}')")
            jid = self._xbotjname_to_jid[jname]
            ret[full_joint_name] = JointState(position=th.as_tensor(jpos[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                                                rate=th.as_tensor(jvel[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                                                effort=th.as_tensor(jeff[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32))
        if self._torch_device.type == "cuda":
            # sync non_blocking cuda transfers
            th.cuda.synchronize(self._torch_device)
        if return_tensor:
            return th.as_tensor([[ret[n].position,ret[n].rate,ret[n].effort] for n in self._jointsToObserve]).view(size=(len(self._jointsToObserve),3))
        else:
            return ret

    @override
    def getLinksState(self, requestedLinks : List[Tuple[str,str]], use_com_pose = False) -> Dict[Tuple[str,str],LinkState]:
        if not self._listenersStarted:
            raise RuntimeError("called getLinksState without having called startController. The proper way to initialize the controller is to first build the controller, then call set_monitored_links, and then call startController")
        if use_com_pose:
            raise NotImplementedError(f"use_com_frame not supported")
        #print("self._linksToObserve = "+str(self._linksToObserve))
        for l in requestedLinks:
            if l not in self._linksToObserve:
                raise RuntimeError(f"Requested link '{l}' that was not requested in set_monitored_links (only observing {self._linksToObserve})")

        self._robot_interface.sense(update_model=True)
        #TODO: get the delay time somehow
        # obsDelay = float("+inf")
        # while obsDelay > self._maxObsAge:
        #     self.run(0.001)
        #     self._robot_interface.sense(update_model=True)
        #     obsDelay = self._robot_interface.getTime() - self._robot_interface.getTimestampRx()            
        # self._jointStateMsgAgeAvg.addValue(obsDelay)

        model = self._robot_interface.model()

        ret = {}
        for full_link_name in requestedLinks:
            mname, lname = full_link_name
            if mname != self._model_name:
                raise RuntimeError(f"Requested link for model different from the monitored one (asked '{mname, lname}', but have '{self._model_name}')")

            lpose = model.getPose(lname, self._reference_frame)
            ltwist = model.getRelativeVelocityTwist(lname, self._reference_frame)

            ls = LinkState(position_xyz=th.as_tensor(lpose.translation).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                           orientation_xyzw=th.as_tensor(lpose.quaternion).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                           pos_com_velocity_xyz=th.as_tensor(ltwist[:3]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                           ang_velocity_xyz=th.as_tensor(ltwist[3:]).to(device=self._torch_device, non_blocking=True, dtype=th.float32))
            ret[full_link_name] = ls
            
        if self._torch_device.type == "cuda":
            # sync non_blocking cuda transfers
            th.cuda.synchronize(self._torch_device)

        return ret

    @override
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : Mapping[Tuple[str,str],Tuple[float,float,float,float,float]] | th.Tensor,
                                        delay_sec : float = 0) -> None:
        if delay_sec!=0.0:
            raise NotImplementedError("Impedance command delay is not supported")
        
        if isinstance(joint_impedances_pvesd, th.Tensor):
            joint_impedances_pvesd_dict = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, Mapping):
            joint_impedances_pvesd_dict = joint_impedances_pvesd

        
        # ggLog.info(f"Setting impedances: {joint_impedances_pvesd}")
        jdi = self.get_joint_device_info(after_env_time=float("-inf"))
        if jdi is not None and jdi.mask == 0:
            ggLog.warn(f"Commanding impedance, but joint device mask is {jdi.mask}.")
        for full_jname, jcmd in joint_impedances_pvesd_dict.items():
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controlled one (asked '{model_name, jname}', but have '{self._model_name}')")
            self._commanded_joint_impedances_by_name[full_jname] = jcmd

    def _apply_commanded_joint_impedances(self):
        self.apply_joint_impedances(self._commanded_joint_impedances_by_name)

    @override
    def apply_joint_impedances(self, joint_impedances_pvesd : Dict[Tuple[str,str],Tuple[float,float,float,float,float] | th.Tensor]):
        # ggLog.info(f"applying joint impedances {joint_impedances_pvesd}")
        if len (joint_impedances_pvesd)==0:
            return
        
        if isinstance(joint_impedances_pvesd, th.Tensor):
            joint_impedances_pvesd_dict = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, Mapping):
            joint_impedances_pvesd_dict = joint_impedances_pvesd

        commanded_joint_impedances_by_jid = {}
        for full_jname, jcmd in joint_impedances_pvesd_dict.items():
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controleld one (asked '{model_name, jname}', but have '{self._model_name}')")
            jid = self._xbotjname_to_jid[jname]
            commanded_joint_impedances_by_jid[jid] = jcmd
        
        prefs, vrefs, erefs, pgains, vgains = (np.zeros(shape=(self._joints_num,), dtype=np.float64) 
                                               for _ in range(5))

        curr_pos = self._robot_interface.getJointPosition()
        used_fallback = False
        for jid in range(self._joints_num):
            cmd = commanded_joint_impedances_by_jid.get(jid,None)
            if cmd is None and self._allow_fallback:
                used_fallback = True
                ggLog.warn(f"Missing command for joint {self._jid_to_xbotjname[jid]} ({jid}), using fallback.")
                # keeps current position
                cmd = (curr_pos[jid], 0, 0, self._fallback_cmd_stiffness, self._fallback_cmd_damping)
            prefs[jid], vrefs[jid], erefs[jid], pgains[jid], vgains[jid] = cmd
        if used_fallback:
            ggLog.warn(f"Used fallback because only had commands for joints_ids:\n {list(commanded_joint_impedances_by_jid.keys())}")
            ggLog.warn(f"Which correspond to joint names:\n {[jn for jn,ji in joint_impedances_pvesd_dict.items()]}")

        self._robot_interface.setStiffness(pgains)
        self._robot_interface.setDamping(vgains)
        self._robot_interface.setPositionReference(prefs)
        self._robot_interface.setVelocityReference(vrefs)
        self._robot_interface.setEffortReference(erefs)
        self._robot_interface.move()

        self._last_sent_pvesd = np.stack([prefs,vrefs,erefs,pgains,vgains], axis = 1)
        # ggLog.info(f"Sent robot_interface command")


    def _apply_commanded_joint_trajectories(self):
        jimp_cmds_pvesd : Dict[Tuple[str,str],Tuple[float,float,float,float,float]] = {}
        t = self.getEnvTimeFromStartup()
        for jn,traj_tpva in self._commanded_joint_trajs_tpva.items():
            sample_idx = np.searchsorted(traj_tpva[:,0], t) # index of the next trajectory sample (the first with time higher of t)
            if sample_idx>=traj_tpva.shape[0]:
                sample_idx = traj_tpva.shape[0]-1
            time, pos, vel, acc = traj_tpva[sample_idx]
            jimp_cmds_pvesd[jn] = (pos, vel, 0, self._position_command_stiffness, self._position_command_damping)
        self.setJointsImpedanceCommand(joint_impedances_pvesd = jimp_cmds_pvesd)

    def _apply_commanded_joint_positions(self):
        jimp_pvesd_cmds : Dict[Tuple[str,str],Tuple[float,float,float,float,float]] = {}
        js = self.getJointsState(list(self._commanded_joint_positions.keys()))
        for jn,p_ref_velsc_accsc in self._commanded_joint_positions.items():
            # just to compute the duration
            p_ref, velocity_scaling, acceleration_scaling = p_ref_velsc_accsc
            traj_tpva = build_1D_vramp_trajectory(t0 = 0.0,
                                                p0 = js[jn].position.item(),
                                                v0 = js[jn].rate.item(),
                                                pf = p_ref,
                                                ctrl_freq_hz = 10, # we just use the first sample anyway
                                                max_vel=self._jpos_cmd_max_vel.get(jn, self._jpos_cmd_max_vel_default)*velocity_scaling,
                                                max_acc=self._jpos_cmd_max_acc.get(jn, self._jpos_cmd_max_acc_default)*acceleration_scaling)
            t, pos, vel, a = traj_tpva[0]
            jimp_pvesd_cmds[jn] = (pos, vel, 0.0, self._position_command_stiffness, self._position_command_damping)
        self.setJointsImpedanceCommand(jimp_pvesd_cmds)

    def clear_commands(self):
        self._commanded_joint_impedances_by_name = {}
        self._commanded_joint_positions = {}
        self._commanded_joint_trajs_tpva = {}

    def _apply_controls(self):
        self._apply_commanded_joint_trajectories() # trajectories override positions by setting position commands
        self._apply_commanded_joint_positions() # positions override impedances by setting impedance commands
        self._apply_commanded_joint_impedances() 

    @override
    def run(self, duration_sec: float):
        # ggLog.info(f"XbotAdapter.run()")
        self._apply_controls()
        super().run(duration_sec)

    @override
    def initialize_for_step(self):
        pass

    @override
    def step(self) -> float:
        step_duration = super().step()
        self.clear_commands()

        # model = self._robot_interface.model()
        # model.setJointPosition(self._robot_interface.getJointPosition())
        # model.setJointVelocity(self._robot_interface.getJointVelocity())
        # model.setJointAcceleration(self._robot_interface.getJointAcceleration())
        # model.update()
        # ggLog.info(f"q = {model.getJointPosition()}")
        # ggLog.info(f"v = {model.getJointVelocity()}")
        # ggLog.info(f"a = {model.getJointAcceleration()}")
        # ggLog.info(f"tau = {model.computeInverseDynamics()}")


        return step_duration

    def _switch_control(self, switch_on : bool, timeout_s : float = 5.) -> bool:
        # for i in range(20):
        #     time.sleep(1)
        #     print(i)
        # ggLog.info(f"switch_xbotros_control({switch_on})")
        switch_srv_name = "/xbotcore/ros_control/switch"
        state_srv_name = "/xbotcore/ros_control/state"
        rospy.wait_for_service(switch_srv_name, timeout = timeout_s)
        ros_ctrl_switch = rospy.ServiceProxy(switch_srv_name, SetBool)
        # ggLog.info(f"Created service proxy for {switch_srv_name}")
        rospy.wait_for_service(state_srv_name, timeout = timeout_s)
        ros_ctrl_state = rospy.ServiceProxy(state_srv_name, PluginStatus)
        # ggLog.info(f"Created service proxy for {state_srv_name}")

        t0 = time.monotonic()
        switched_on = not switch_on
        status = None
        while switched_on != switch_on:
            if time.monotonic()-t0>timeout_s:
                raise TimeoutError(f"Timed out waiting for xbot ros_control switch. Status = '{status}'")
            try:
                # ggLog.info(f"Calling {state_srv_name}")
                resp = ros_ctrl_state()
            except rospy.ServiceException as e:
                ggLog.warn(f"ros_ctrl_state call failed: {e}")
                raise e
            status = resp.status
            switched_on = resp.status == "Running"
            # ggLog.info(f"ros_control state: {resp}")
            if switched_on != switch_on:
                try:
                    # ggLog.info(f"ros_ctrl_switch({switch_on})")
                    resp = ros_ctrl_switch(switch_on) # DOES NOT WORK IF THE SIMULATION IS PAUSED. A sadly, services have no timeouts (https://github.com/ros/ros_comm/pull/2144)
                except rospy.ServiceException as e:
                    ggLog.info(f"ros_ctrl_switch call failed: {e}")
                    raise e
                # ggLog.info(f"ros_ctrl_switch service responded {resp}")        
            time.sleep(0.5)
        ggLog.info(f"switched ros_control to state: {resp}")
        return switched_on

    def _setup_joint_control(self, control_mask : int, timeout_s = 300.0) -> int:
        control_mask_srv_name = "/xbotcore/joint_master/set_control_mask"
        rospy.wait_for_service(control_mask_srv_name, timeout=timeout_s)
        control_mask_srv = rospy.ServiceProxy(control_mask_srv_name, SetControlMask)
        t0 = time.monotonic()
        current_mask = None
        while current_mask!=control_mask:
            if time.monotonic()-t0>timeout_s:
                raise TimeoutError(f"Timed out waiting for control_mask set. Status = '{current_mask}'")
            topic_name = "/xbotcore/joint_device_info"
            jdi = rospy.wait_for_message(topic_name, JointDeviceInfo, timeout = 10)
            if not isinstance(jdi, JointDeviceInfo):
                raise RuntimeError(f"Unexpected type received from {topic_name}, should be JointDeviceInfo but it's {type(jdi)}")
            current_mask = jdi.mask
            if jdi.mask!=control_mask:
                # ggLog.info(f"Setting control mask to {control_mask}")
                resp = control_mask_srv(ctrl_mask = control_mask)
                if not resp.success:
                    raise RuntimeError(f"Failed to set control mask: {resp}")
        ggLog.info(f"Control mask set to {jdi.mask}")
        return jdi.mask

    @override
    def initialize_for_episode(self):
        super().initialize_for_episode()
        self.clear_commands()
        self._last_jdi_time = float("-inf")
        self._last_joint_device_info = None
        if self.is_simulated():
            req_mask = 255
            mask = self._setup_joint_control(control_mask = req_mask)
            if mask != req_mask:
                raise RuntimeError(f"Failed to set control mask, wanted {req_mask}, got {mask}")
            switched_on = self._switch_control(True)
            if not switched_on:
                raise RuntimeError(f"Failed to switch on control.")

    @override
    def setJointsPositionCommand(self, jointPositions : Dict[Tuple[str,str],float],
                                        velocity_scaling : Optional[float] = None,
                                        acceleration_scaling : Optional[float] = None) -> None:
        if velocity_scaling is None:
            velocity_scaling = 1.0
        if acceleration_scaling is None:
            acceleration_scaling = 1.0
        self._commanded_joint_positions.update({k:(p,velocity_scaling,acceleration_scaling) for k,p in jointPositions.items()})

        # jic_pvesd = [(jn,(pos, 0.0, 0.0, self._position_command_stiffness, self._position_command_damping)) for jn, pos in jointPositions.items()]
        # self.setJointsImpedanceCommand(joint_impedances_pvesd=jic_pvesd)

    def _setJointTrajectoryCommand(self, jointTrajectories_tpva : Dict[Tuple[str,str], np.ndarray]):
        self._commanded_joint_trajs_tpva.update(jointTrajectories_tpva)

    @override
    def moveToJointPoseSync(self,   jointPositions : Dict[Tuple[str,str],float],
                                    velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None,
                                    joint_position_tolerance : float = 0.01,
                                    max_time_s : float = 60,
                                    joint_velocity_termination_threshold = 0.01,
                                    joint_velocity_scaling : dict[Tuple[str,str],float] = {}) -> None:
        self.clear_commands()
        if velocity_scaling is None:
            velocity_scaling = 1.0
        if acceleration_scaling is None:
            acceleration_scaling = 1.0
        js = self.getJointsState(list(jointPositions.keys()))
        last_refs = self.get_current_joint_impedance_command()
        last_refs_dict = {self._jimpedance_controlled_joints[i]:last_refs[i] for i in range(len(self._jimpedance_controlled_joints))}
        max_traj_duration = 0
        joint_trajs = {}
        for jn,p_ref in jointPositions.items():
            # just to compute the duration
            vs = joint_velocity_scaling.get(jn, velocity_scaling)
            traj_tpva = build_1D_vramp_trajectory(  t0 = 0.0,
                                                    p0 = last_refs_dict[jn][0].item(),
                                                    v0 = js[jn].rate.item(),
                                                    pf = p_ref,
                                                    ctrl_freq_hz = 1000.0,
                                                    max_vel=self._jpos_cmd_max_vel.get(jn, self._jpos_cmd_max_vel_default)*vs,
                                                    max_acc=self._jpos_cmd_max_acc.get(jn, self._jpos_cmd_max_acc_default)*acceleration_scaling)
            joint_trajs[jn] = traj_tpva
            traj_duration = traj_tpva[-1][0]
            max_traj_duration = max(max_traj_duration,traj_duration)
        for jn,p_ref in jointPositions.items(): # scale to have all trajectories be the same duration
            joint_traj = joint_trajs[jn]
            traj_duration = joint_traj[-1][0]
            scale = max_traj_duration/traj_duration
            joint_traj[:,0] *= scale # time
            joint_traj[:,2] *= 1/scale # velocity
            joint_traj[:,3] *= 1/(scale**2) # acceleration
        # ggLog.info(f"traj_tpva = \n{traj_tpva}")
        timeout_env = max_traj_duration*2+1
        timeout_wall = timeout_env*20
        if max_traj_duration > max_time_s:
            raise RuntimeError(f"Computed trajectory is excessively long, would last {max_traj_duration}s, max_time is set to {max_time_s}s. \n"
                               f"Joint names           : {[jn for jn,ji in js.items()]}\n"
                               f"Initial joint position: "+str([f"{ji.position.item(): 2.4f}" for jn,ji in js.items()])+"\n"
                               f"Initial joint velocity: "+str([f"{ji.rate.item(): 2.4f}" for jn,ji in js.items()])+"\n"
                               f"Target  joint position: "+str([f"{jp: 2.4f}" for jp in jointPositions.values()])+"\n"
                               f"Durations {[(jn,traj_tpva[-1][0]) for jn,traj_tpva in joint_trajs.items()]}\n"
                               f"Raise the max_time if it is actually ok.")
        # print(f"joint_trajs max_v = {max([max(t[2]) for t in joint_trajs.values() ])}")
        t0 = self.getEnvTimeFromStartup()
        for jn in joint_trajs.keys():
            joint_trajs[jn][:,0] += t0
        self._setJointTrajectoryCommand(jointTrajectories_tpva = joint_trajs)

        # self.setJointsPositionCommand(jointPositions=jointPositions)
        t0_env = self.getEnvTimeFromStartup()
        t0_wall = time.monotonic()
        js = self.getJointsState(list(jointPositions.keys()))
        errors = [ji.position.item() - jointPositions[jn] for jn,ji in js.items()]
        reached_position = all([abs(e) < joint_position_tolerance for e in errors])
        elapsed_env_time = 0.0
        elapsed_wall_time = 0.0
        stopped = False
        while not (reached_position or (stopped and elapsed_env_time>=max_traj_duration)):
            self.run(self._stepLength_sec)
            js = self.getJointsState(list(jointPositions.keys()))
            errors = [ji.position.item() - jointPositions[jn] for jn,ji in js.items()]
            reached_position = all([abs(e) < joint_position_tolerance for e in errors])
            stopped = all([abs(ji.rate.item())<joint_velocity_termination_threshold for ji in js.values()])
            elapsed_env_time = self.getEnvTimeFromStartup() - t0_env
            elapsed_wall_time = time.monotonic() - t0_wall
            if elapsed_env_time > timeout_env:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (env timeout {elapsed_env_time}>{timeout_env})\n"
                                    f"    target = {jointPositions}\n"
                                    f"    joint state = {[ji.position.item() for jn,ji in js.items()]}\n"
                                    f"    errors = {errors}\n"
                                    f"    max_error = {max(errors)}\n"
                                    f"    tolerance = {joint_position_tolerance}")
            if elapsed_wall_time > timeout_wall:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (wall timeout {elapsed_wall_time}>{timeout_wall}) errors = {errors} tolerance = {joint_position_tolerance}")



    def is_safety_triggered(self):
        raise NotImplementedError()
    
    def _get_current_refs_pvesd(self):
        self._robot_interface.sense(update_references=True)
        return np.stack([   self._robot_interface.getPositionReference(),
                            self._robot_interface.getVelocityReference(),
                            self._robot_interface.getEffortReference(),
                            self._robot_interface.getStiffness(),
                            self._robot_interface.getDamping()], axis = 1)

    @override
    def get_current_joint_impedance_command(self) -> th.Tensor:
        ref_j_pvesd = self._get_current_refs_pvesd()
        # pvesd_by_name = {(mn,jn):ref_j_pvesd[self._xbotjname_to_jid[jn]] for mn,jn in self._jimpedance_controlled_joints}
        return th.as_tensor(ref_j_pvesd[self._jimpedance_controlled_joints_jids], device=self._torch_device, dtype=th.float32)
    
    @override
    def get_link_gravity_direction(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        imus = self._robot_interface.getImu()
        # print(f"imus = {imus}")
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        req_links = [ln[1] for ln in requestedLinks] # Remove the model name
        ref_imus : dict[str,str] = {} # What imu to use for which links
        for rl in req_links:
            ref_imus[rl] = rl if rl in imus else list(imus.keys())[0] # use the first available imu (maybe we can do better than this? find a "best" one?)
        link2imu_poses : dict[str,Affine3] = {ln:self._robot_interface.model().getPose(ln,ref_imus[ln]) for ln in req_links}
        orientation_mats = [link2imu_poses[ln].matrix()[:3,:3]*imus[ref_imus[ln]].getOrientation() for ln in req_links]
        # Gravity direction is rotmat*[0,0,-1], which is -1 by the last colunn of rotmat
        gdirs = [-th.as_tensor(m[2,:]) for m in orientation_mats]
        return th.stack(gdirs)
    
    @override
    def get_link_relative_angular_velocity(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        imus = self._robot_interface.getImu()
        # print(f"imus = {imus}")
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        req_links = [ln[1] for ln in requestedLinks] # Remove the model name
        ref_imus : dict[str,str] = {} # What imu to use for which links
        for rl in req_links:
            ref_imus[rl] = rl if rl in imus else list(imus.keys())[0] # use the first available imu (maybe we can do better than this? find a "best" one?)
        link2imu_poses : dict[str,Affine3] = {ln:self._robot_interface.model().getPose(ln,ref_imus[ln]) for ln in req_links}
        angvels = [self._thtens(np.matmul(link2imu_poses[ln].matrix()[:3,:3],imus[ref_imus[ln]].getAngularVelocity())) for ln in req_links]
        # for ln in req_links:
        #     ggLog.info(f"imu angvel = {imus[ref_imus[ln]].getAngularVelocity().transpose()}")
        #     ggLog.info(f"link2imu_poses[ln].matrix()[:3,:3] = {link2imu_poses[ln].matrix()[:3,:3]}")
        # ggLog.info(f"angvels = {angvels}")
        return th.stack(angvels)
    
    def control_period(self):
        return self._control_dt
        
    def _thtens(self, arr: np.ndarray) -> th.Tensor:
        """Convert a numpy array to a torch tensor on the configured device."""
        return th.as_tensor(arr, device=self._torch_device)
    
    def get_local_link_linear_acceleration(self, requestedLinks : Sequence[tuple[str,str]] | None) -> th.Tensor:
        imus = self._robot_interface.getImu()
        # print(f"imus = {imus}")
        if requestedLinks is None:
            requestedLinks = self._monitored_links
        req_links = [ln[1] for ln in requestedLinks] # Remove the model name
        ref_imus : dict[str,str] = {} # What imu to use for which links
        for rl in req_links:
            ref_imus[rl] = rl if rl in imus else list(imus.keys())[0] # use the first available imu (maybe we can do better than this? find a "best" one?)
        imu2link_poses : dict[str,Affine3] = {ln:self._robot_interface.model().getPose(ln,ref_imus[ln]) for ln in req_links}
        accelerations : list[th.Tensor] = []
        for ln in req_links:
            imu2link = imu2link_poses[ln]
            imu2link_rotmat = imu2link.matrix()[:3,:3]
            imu = imus[ref_imus[ln]]
            imu_linacc = imu.getLinearAcceleration()
            if np.allclose(imu2link_rotmat, np.eye(imu2link_rotmat.shape[0]), atol=1e-5):
                acceleration = imu_linacc
            else:
                raise RuntimeError(f"Cannot compute local linear acceleration for link '{ln}' as it is not directly attached to an IMU")
                com_offset_xyz = imu2link.translation()
                imu_angvel = imu.getAngularVelocity()
                imu_angacc # Would need this somehow
                imu_linvel # Would need this somehow
                local_angvel = imu2link_rotmat @ imu_linacc
                local_linvel = imu2link_rotmat @ (imu_linvel - np.cross(com_offset_xyz, imu_angvel))
                acc = imu2link_rotmat @ (imu_linacc - np.cross(com_offset_xyz, imu_angacc))
                correction = np.cross(local_angvel, local_linvel)
                acceeleration = acc + correction
            accelerations.append(self._thtens(acceleration))
        return th.stack(accelerations)