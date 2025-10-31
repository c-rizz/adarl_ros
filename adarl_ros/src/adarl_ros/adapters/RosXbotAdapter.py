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
from xbot_msgs.srv import PluginStatus, SetControlMask, GetStringList, GetParameterInfo, GetParameterInfoRequest
from xbot_msgs.msg import JointDeviceInfo, Statistics2
from sensor_msgs.msg import Imu

import torch as th
from adarl.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from typing_extensions import override
from cartesian_interface.affine3 import Affine3 # needed by xbot_interface as it doesn't import it correctly
import traceback
from threading import RLock, Condition
from adarl.adapters.BaseJointPositionAdapter import BaseJointPositionAdapter
from adarl.adapters.BaseAdapter import JointName,  LinkName
import adarl.utils.session

from transforms3d.quaternions import mat2quat

import math

# ------------------------------------------------------------------------------------------
# XBOT helper functions
# ------------------------------------------------------------------------------------------



def build_xbot_cfg(is_floating_base):
    """
    A function to construct the xbotinterface config object from ros
    """
    t0 = time.monotonic()
    timeout=30
    retry_freq=1.5
    robot_description_name='/xbotcore/robot_description'
    semantic_description_name='/xbotcore/robot_description_semantic'
    while True:
        urdf = rospy.get_param(robot_description_name, default=None) # type: ignore
        if urdf is not None:
            break
        if time.monotonic() - t0 > timeout: # retry for max timeout secs
            raise TimeoutError()
        time.sleep(retry_freq)
        ggLog.warn(f"build_xbot_cfg: could not get robot description parameter at \"{robot_description_name}\"! Trying again...")

    srdf = rospy.get_param(semantic_description_name) # type: ignore
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
    # print(urdf.links)
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

# def set_filters(set_enabled : bool, required_filter_hz = 2.0):
#     enable_filter_srv_name = "/xbotcore/enable_joint_filter"
#     wait_for_ros_service(enable_filter_srv_name)
#     enable_filter_srv = rospy.ServiceProxy(enable_filter_srv_name, SetBool)
#     set_filter_srv_name = "/xbotcore/set_filter_profile_medium"
#     wait_for_ros_service(set_filter_srv_name)
#     set_filter_mode_srv = rospy.ServiceProxy(set_filter_srv_name, std_srvs.srv.Trigger)
#     filter_status = not set_enabled
#     filter_hz = None
#     while filter_status != set_enabled and filter_hz!=required_filter_hz:
        
#         topic_name = "/xbotcore/joint_device_info"
#         jdi = rospy.wait_for_message(topic_name, JointDeviceInfo, timeout = 10)
#         if not isinstance(jdi, JointDeviceInfo):
#             raise RuntimeError(f"Unexpected type received from {topic_name}, should be JointDeviceInfo but it's {type(jdi)}")
#         filter_status = jdi.filter_active
#         filter_hz = jdi.filter_cutoff_hz

#         if filter_hz != required_filter_hz:
#             resp = set_filter_mode_srv() # this service always returns success = False
#             # if not resp.success: 
#             #     raise RuntimeError(f"Failed to set filter cutoff. Status: {resp}")
#         if filter_status != set_enabled:
#             # print(("Enabling" if set_enabled else "Disabling")+" filters...")
#             resp = enable_filter_srv(set_enabled)
#             if not resp.success:
#                 raise RuntimeError(f"Failed to set filters status: {resp}")
#     ggLog.info(f"Filters {'enabled' if filter_status else 'disabled'}. Cutoff = {filter_hz}")

def set_filters(set_enabled : bool, profile_name = "safe"):
    ggLog.info(f"Trying to set filter profile {profile_name}")

    # # # get filter cutoff hz for the selected profile (to be used for checking)
    # profile_freq_getter_srv_name = "/xbotcore/get_parameter_info"
    # filter_profile_srv = rospy.ServiceProxy(profile_freq_getter_srv_name, GetParameterInfo)
    # param_info_msg=GetParameterInfoRequest()
    # param_info_msg.name=f"/xbot/hal/joint_safety/filter_{profile_name}_cutoff_hz"
    # resp = filter_profile_srv(param_info_msg)

    enable_filter_srv_name = "/xbotcore/enable_joint_filter"
    wait_for_ros_service(enable_filter_srv_name, timeout=1.0)
    enable_filter_srv = rospy.ServiceProxy(enable_filter_srv_name, SetBool)
    set_filter_srv_name = f"/xbotcore/set_filter_profile_{profile_name}"
    wait_for_ros_service(set_filter_srv_name, timeout=1.0)
    set_filter_mode_srv = rospy.ServiceProxy(set_filter_srv_name, std_srvs.srv.Trigger)
    filter_status = not set_enabled
    filter_hz = None
    while filter_status != set_enabled:
        
        resp = set_filter_mode_srv() # this service always returns success = False
        # if not resp.success: 
        #     raise RuntimeError(f"Failed to set filter cutoff. Status: {resp}")

        topic_name = "/xbotcore/joint_device_info"
        jdi = rospy.wait_for_message(topic_name, JointDeviceInfo, timeout = 10)
        if not isinstance(jdi, JointDeviceInfo):
            raise RuntimeError(f"Unexpected type received from {topic_name}, should be JointDeviceInfo but it's {type(jdi)}")
        filter_status = jdi.filter_active
        filter_hz = jdi.filter_cutoff_hz

        if filter_status != set_enabled:
            # print(("Enabling" if set_enabled else "Disabling")+" filters...")
            resp = enable_filter_srv(set_enabled)
            if not resp.success:
                raise RuntimeError(f"Failed to set filters status: {resp}")
    ggLog.info(f"Filters {'enabled' if filter_status else 'disabled'}. Cutoff = {filter_hz}")

def wait_for_ros_service(server_name: str, timeout=1.0):
    try:
        rospy.wait_for_service(server_name, timeout=timeout)
    except rospy.exceptions.ROSException:
        ggLog.warn(f"wait for service {server_name} timeouted.")
        False
    return True

def is_simulated():
    # Is here some better way to do this?
    # Can I ask xbot?
    switch_srv_name = "/xbotcore/get_parameter_value"
    hw_type_param_name="/xbot/hal/hw_type"
    service_avail=wait_for_ros_service(switch_srv_name, timeout=1.0)
    if not service_avail:
        return False
    get_param_value_srv = rospy.ServiceProxy(switch_srv_name, GetStringList)
    try:
        res=get_param_value_srv(hw_type_param_name)
        if not res.success:
            raise RuntimeError(f"Failed to get hw type parameter from XBot!")
        
        hw_type=res.response[0]
        
        return hw_type=="sim"
    except:
        ggLog.warn(f"Failed to call sevice proxy for {switch_srv_name}")
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
                        fallback_cmd_damping : float= 60.0,
                        allow_fallback : bool = True,
                        jpos_cmd_max_vel = {},
                        jpos_cmd_max_vel_default = 0.0,
                        jpos_cmd_max_acc = {},
                        jpos_cmd_max_acc_default = 0.0,
                        enable_filters = True,
                        imu_link: str = "imu_link",
                        base_link : str = "imu_link"):
        super().__init__(stepLength_sec, forced_ros_master_uri, maxObsDelay, blocking_observation)
        self._is_floating_base = is_floating_base
        self._imu_link= imu_link
        self._base_link = base_link
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
        self._position_command_stiffness = 100.0
        self._position_command_damping = 50.0
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

        self.impedance_ramp_time=2.0 # [s]
        self._impedance_ramp_tinysleep=0.005

    def _joint_device_info_callback(self, msg):
        with self._joint_device_info_mutex:
            self._last_jdi_time = self.getEnvTimeFromReset()
            self._last_joint_device_info = msg
    
    def _xbot_statistics_callback(self, msg):
        self._xbot_task_stats=msg.task_stats
        
    def _imu_device_callback(self, msg):
        # read imu data from xbot topic
        self._imu_frame = msg.header.frame_id

        orientation = msg.orientation
        self._imu_q_last[:, 0]=orientation.w 
        self._imu_q_last[:, 1]=orientation.x
        self._imu_q_last[:, 2]=orientation.y
        self._imu_q_last[:, 3]=orientation.z

        omega=msg.angular_velocity
        self._imu_omega_last[:, 0]=omega.x
        self._imu_omega_last[:, 1]=omega.y
        self._imu_omega_last[:, 2]=omega.z

        lin_acc=omega=msg.angular_velocity
        self._imu_linacc_last[:, 0]=lin_acc.x
        self._imu_linacc_last[:, 1]=lin_acc.y
        self._imu_linacc_last[:, 2]=lin_acc.z

    @override
    def set_monitored_joints(self, jointsToObserve: List[Tuple[str, str]]):
        self._xbot_joints_to_monitor = jointsToObserve # keep empty the normal jointsToObserve and use this instead
        super().set_monitored_joints(jointsToObserve)
    
    def fallback_striffness(self):
        return self._fallback_cmd_stiffness
    
    def fallback_damping(self):
        return self._fallback_cmd_damping

    def startup(self):
        super().startup()
        cfg = build_xbot_cfg(is_floating_base=self._is_floating_base)

        import time
        wait_for_sec=1.5 # [s]
        timeout_sec=60.0
        t0 = time.monotonic()
        self._robot_interface=None
        while True:
            try:
                self._robot_interface = xbot.RobotInterface(cfg)
                if self._robot_interface is not None:
                    break
            except RuntimeError:
                ggLog.error(f"{__class__}: Failed to initialize robot interface (is xbot-core running?)! Will try again in {wait_for_sec} s...")
                time.sleep(wait_for_sec)
                if time.monotonic()-t0>timeout_sec:
                    break
        
        if self._robot_interface is None:
            raise TimeoutError(f"Timed out while trying to construct robot interface")
        
        ggLog.info(get_system_recap_string(self._robot_interface))

        self._ros_control_running=False
        statistics_topicname="/xbotcore/statistics"
        self._xbot_statistics_subscriber = rospy.Subscriber(statistics_topicname, Statistics2, 
            self._xbot_statistics_callback, queue_size=1)
        self._xbot_task_stats=None
        while self._xbot_task_stats is None: # wait for first xbot stat
            # msg to arrive to retrive task order
            time.sleep(1.0)
        self._xbot_task_info_map={}
        for i in range(len(self._xbot_task_stats)):
            task_info=self._xbot_task_stats[i]
            self._xbot_task_info_map[task_info.name]=i
        ggLog.info(f"Subscribed to {statistics_topicname}")

        set_filters(True, profile_name="safe")
        self._setup_joint_control(control_mask=255)
        self._switch_control(self._enable_filters)

        enabled_joint_names = self._robot_interface.getEnabledJointNames() # this is different from robot.model().getEnabledJointNames()
        self._joints_num = len(enabled_joint_names)
        self._xbotjname_to_jid = {jname : enabled_joint_names.index(jname) for jname in enabled_joint_names}
        self._jid_to_xbotjname = {jid : jname for jname, jid in self._xbotjname_to_jid.items()}
        
        joint_dev_topicname="/xbotcore/joint_device_info"
        self._jdi_subscriber = rospy.Subscriber(joint_dev_topicname, JointDeviceInfo, 
            self._joint_device_info_callback, queue_size=1)
        ggLog.info(f"Subscribed to {joint_dev_topicname}")
        
        # imu_topic_name = "/xbotcore/imu/"+self._imu_link
        # self._imu_frame="none"
        # self._imu_q_last=np.zeros((1, 4))
        # self._imu_q_last[:, 0]=1.0
        # self._imu_omega_last=np.zeros((1, 3))
        # self._imu_linacc_last=np.zeros((1, 3))
        # self._imu_subscriber = rospy.Subscriber(imu_topic_name, Imu, self._imu_device_callback, queue_size=1)
        
        self._xbot_imu = self._robot_interface.getImu()[self._imu_link] # we assume there's only one imu
        self._R_imu_link=self._robot_interface.model().getPose(self._base_link, self._imu_link).matrix()[:3,:3] # we assume a static tranform

        # base link state (defaults to imu_link), updated when imu callback is called
        self._base_q_last=np.zeros((1, 4))
        self._base_q_last[:, 0]=1.0
        self._base_omega_last=np.zeros((1, 3))
        self._base_linacc_last=np.zeros((1, 3))

        # preallocating cmds
        self._prefs =np.zeros(shape=(self._joints_num,), dtype=np.float64)
        self._vrefs =np.zeros(shape=(self._joints_num,), dtype=np.float64)
        self._erefs =np.zeros(shape=(self._joints_num,), dtype=np.float64)
        self._pgains =np.zeros(shape=(self._joints_num,), dtype=np.float64)
        self._vgains =np.zeros(shape=(self._joints_num,), dtype=np.float64)

    def set_filters(self, set_enabled : bool, profile_name = "safe"):
        set_filters(set_enabled=set_enabled,profile_name=profile_name)
    
    # def get_imu_data(self):
    #     return (self._imu_frame, self._imu_q_last, self._imu_omega_last, self._imu_linacc_last)
    
    def get_base_link_state(self):
        return (self._base_link, self._base_q_last, self._base_omega_last, self._base_linacc_last)
    
    def read_imu_data(self):
        
        # update base link state (if not provided == imu_link)
        R_world_imu=self._xbot_imu.getOrientation()
        R_world_link=R_world_imu@self._R_imu_link # orientation of base link wrt world frame
        omega_imu_loc=self._xbot_imu.getAngularVelocity() # IMU local !!
        
        self._base_q_last[:, :]=mat2quat(R_world_link) # IMPORTANT: returns quaterion in w,x,y,z order    
        self._base_omega_last[:, :]= self._R_imu_link.T @ omega_imu_loc # rotate from IMU to base link frame
        self._base_linacc_last[:, :]=self._R_imu_link.T @ self._xbot_imu.getLinearAcceleration() # we would need to account
        # for linear velocity and angular acc to do it properly (accurate only if imu==base_link)
        
    def _is_xbot_task_running(self,task_name: str):
        task_id=self._xbot_task_info_map[task_name]
        return self._xbot_task_stats[task_id].state=="Running"

    def trigger_homing(self):
        homing_switch_srv_name = "/xbotcore/homing/switch"
        homing_state_srv_name = "/xbotcore/homing/state"
        timeout_s= float("+inf")
        wait_for_ros_service(homing_switch_srv_name, timeout=1.0)
        homing_ros_switch = rospy.ServiceProxy(homing_switch_srv_name, SetBool)
        ggLog.info(f"Created service proxy for {homing_switch_srv_name}")

        wait_for_ros_service(homing_state_srv_name, timeout=1.0)
        homing_ros_state = rospy.ServiceProxy(homing_state_srv_name, PluginStatus)
        ggLog.info(f"Created service proxy for {homing_state_srv_name}")

        t0 = time.monotonic()
        homing_running=True
        status = None
        self._switch_control(switch_on=False) # deactivate ros control to
        # avoid race conditions on the joints
        while homing_running:
            if time.monotonic()-t0>timeout_s:
                raise TimeoutError(f"Timed out waiting for homing to be completed switch. Status = '{status}'")
            try:
                # ggLog.info(f"Calling {state_srv_name}")
                resp = homing_ros_state()
            except rospy.ServiceException as e:
                ggLog.warn(f"homing_ros_state call failed: {e}")
                raise e
            status = resp.status
            homing_running = resp.status == "Running"
            # ggLog.info(f"ros_control state: {resp}")
            if not homing_running:
                try:
                    resp = homing_ros_switch(True) # DOES NOT WORK IF THE SIMULATION IS PAUSED. A sadly, services have no timeouts (https://github.com/ros/ros_comm/pull/2144)
                except rospy.ServiceException as e:
                    ggLog.info(f"homing_ros_switch call failed: {e}")
                    raise e
                # ggLog.info(f"ros_ctrl_switch service responded {resp}")        
            time.sleep(0.5)
        ggLog.info(f"homing performed with response: {resp}")
        self._switch_control(switch_on=True) # we can reactivate ros control
        time.sleep(2.0)

    def is_ros_control_running(self):
        return self._is_xbot_task_running("ros_control")

    def get_robot_interface(self):
        return self._robot_interface
    
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
        # jvel = self._robot_interface.getJointVelocity()
        jvel = self._robot_interface.getMotorVelocity() # cleaner signal on motor side (more resolution)
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
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        if not self._listenersStarted:
            raise RuntimeError("called getLinksState without having called startController. The proper way to initialize the controller is to first build the controller, then call set_monitored_links, and then call startController")

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
                           pos_velocity_xyz=th.as_tensor(ltwist[:3]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
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
            raise NotImplementedError()
        
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
        # self.apply_joint_impedances_with_ramp(self._commanded_joint_impedances_by_name) # ramp to avoid dangerous torque discontinuities

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

        curr_pos = self._robot_interface.getJointPosition()
        used_fallback = False
        for jid in range(self._joints_num):
            cmd = commanded_joint_impedances_by_jid.get(jid,None)
            if cmd is None and self._allow_fallback:
                used_fallback = True
                ggLog.warn(f"Missing command for joint {self._jid_to_xbotjname[jid]} ({jid}), using fallback.")
                # keeps current position
                cmd = (curr_pos[jid], 0, 0, self._fallback_cmd_stiffness, self._fallback_cmd_damping)
            self._prefs[jid], self._vrefs[jid], self._erefs[jid], self._pgains[jid], self._vgains[jid] = cmd
        if used_fallback:
            ggLog.warn(f"Used fallback because only had commands for joints_ids:\n {list(commanded_joint_impedances_by_jid.keys())}")
            ggLog.warn(f"Which correspond to joint names:\n {[jn for jn,ji in joint_impedances_pvesd_dict.items()]}")

        self._robot_interface.setStiffness(self._pgains)
        self._robot_interface.setDamping(self._vgains)
        self._robot_interface.setPositionReference(self._prefs)
        self._robot_interface.setVelocityReference(self._vrefs)
        self._robot_interface.setEffortReference(self._erefs)
        self._robot_interface.move()
        # ggLog.info(f"Sent robot_interface command")

    @override
    def apply_joint_impedances_with_ramp(self, joint_impedances_pvesd : Dict[Tuple[str,str],Tuple[float,float,float,float,float] | th.Tensor],
                                impedance_ramp_time = None):
        """
        Linearly ramp stiffness (p gains) and damping (v gains) from current
        values to targets over the configured ramp time.

        Per-instance attributes used (with defaults if missing):
          - self.impedance_ramp_time      : desired ramp time in seconds
          - self._impedance_ramp_max_time  : fallback ramp time in seconds
          - self._impedance_ramp_sleep     : sleep between iterations (s), default 0.005
        """
        # quick return for empty input
        if len(joint_impedances_pvesd) == 0:
            return

        if impedance_ramp_time is None:
            impedance_ramp_time=self.impedance_ramp_time

        # convert tensor or mapping to dict keyed by (model_name,jname)
        if isinstance(joint_impedances_pvesd, th.Tensor):
            joint_impedances_pvesd_dict = dict(zip(self._jimpedance_controlled_joints, joint_impedances_pvesd))
        elif isinstance(joint_impedances_pvesd, Mapping):
            joint_impedances_pvesd_dict = joint_impedances_pvesd
        else:
            raise TypeError("joint_impedances_pvesd must be a Mapping or a torch.Tensor")

        # map commanded values to joint ids
        commanded_joint_impedances_by_jid = {}
        for full_jname, jcmd in joint_impedances_pvesd_dict.items():
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(
                    f"Commanded joint impedance for model different from the controlled one "
                    f"(asked '{model_name, jname}', but have '{self._model_name}')"
                )
            jid = self._xbotjname_to_jid[jname]
            commanded_joint_impedances_by_jid[jid] = jcmd

        # prepare current refs and target arrays
        curr_pos = self._robot_interface.getJointPosition()
        curr_stiffness = list(self._robot_interface.getStiffness())
        curr_damping = list(self._robot_interface.getDamping())

        # prepare target arrays (default to current to avoid NaNs)
        target_stiffness = [float(curr_stiffness[j]) for j in range(self._joints_num)]
        target_damping   = [float(curr_damping[j])   for j in range(self._joints_num)]

        used_fallback = False
        for jid in range(self._joints_num):
            cmd = commanded_joint_impedances_by_jid.get(jid, None)
            if cmd is None and self._allow_fallback:
                used_fallback = True
                ggLog.warn(f"Missing command for joint {self._jid_to_xbotjname[jid]} ({jid}), using fallback.")
                cmd = (curr_pos[jid], 0, 0, self._fallback_cmd_stiffness, self._fallback_cmd_damping)
            elif cmd is None:
                raise RuntimeError(f"No impedance command provided for joint {jid} and fallback is not allowed.")

            # cmd expected: (pos_ref, vel_ref, effort_ref, stiffness, damping)
            pref, vref, eref, targ_s, targ_d = cmd

            # set immediate refs (we continue to re-send these each loop)
            self._prefs[jid] = pref
            self._vrefs[jid] = vref
            self._erefs[jid] = eref

            # record numeric targets
            target_stiffness[jid] = float(targ_s)
            target_damping[jid]   = float(targ_d)

        if used_fallback:
            ggLog.warn(f"Used fallback because only had commands for joints_ids:\n {list(commanded_joint_impedances_by_jid.keys())}")
            ggLog.warn(f"Which correspond to joint names:\n {[jn for jn,ji in joint_impedances_pvesd_dict.items()]}")

        # If ramp time is zero, set targets immediately and return
        if self.impedance_ramp_time <= 0.0:
            self.impedance_ramp_time=-self.impedance_ramp_time

        # Linear ramp: compute from initial to target over ramp_time
        start_time = time.perf_counter()
        initial_p = [float(x) for x in curr_stiffness]
        initial_v = [float(x) for x in curr_damping]

        ggLog.info(f"Starting linear impedance ramp for {self.impedance_ramp_time:.3f}s (sleep {self._impedance_ramp_tinysleep:.4f}s)...")

        try:
            while True:
                now = time.perf_counter()
                elapsed = now - start_time
                frac = min(1.0, max(0.0, elapsed / self.impedance_ramp_time))

                # compute interpolated gains
                interp_p = [0.0] * self._joints_num
                interp_v = [0.0] * self._joints_num
                for j in range(self._joints_num):
                    interp_p[j] = initial_p[j] + frac * (target_stiffness[j] - initial_p[j])
                    interp_v[j] = initial_v[j] + frac * (target_damping[j] - initial_v[j])

                    # update stored so other code sees intermediate values
                    self._pgains[j] = interp_p[j]
                    self._vgains[j] = interp_v[j]

                # send to robot
                self._robot_interface.setStiffness(interp_p)
                self._robot_interface.setDamping(interp_v)
                self._robot_interface.setPositionReference(self._prefs)
                self._robot_interface.setVelocityReference(self._vrefs)
                self._robot_interface.setEffortReference(self._erefs)
                self._robot_interface.move()

                # finish condition
                if frac >= 1.0:
                    ggLog.info(f"Linear impedance ramp finished after {elapsed:.3f}s.")
                    break

                # small sleep to yield CPU
                time.sleep(self._impedance_ramp_tinysleep)

        except KeyboardInterrupt:
            ggLog.warn("Impedance linear ramp interrupted by KeyboardInterrupt; leaving current interpolated values in effect.")

        # final enforce exact target values (sets exact targets if ramp completed;
        self.apply_joint_impedances(joint_impedances_pvesd)

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

    def apply_cmds_now(self):
        self._apply_controls()

    @override
    def run(self, duration_sec: float):
        self._apply_controls()
        super().run(duration_sec)

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

    def _switch_control(self, switch_on : bool, timeout_s : float = float("+inf")) -> bool:
        # for i in range(20):
        #     time.sleep(1)
        #     print(i)
        # ggLog.info(f"switch_xbotros_control({switch_on})")
        switch_srv_name = "/xbotcore/ros_control/switch"
        state_srv_name = "/xbotcore/ros_control/state"
        wait_for_ros_service(switch_srv_name, timeout=1.0)
        ros_ctrl_switch = rospy.ServiceProxy(switch_srv_name, SetBool)
        ggLog.info(f"Created service proxy for {switch_srv_name}")
        wait_for_ros_service(state_srv_name, timeout=1.0)
        ros_ctrl_state = rospy.ServiceProxy(state_srv_name, PluginStatus)
        ggLog.info(f"Created service proxy for {state_srv_name}")

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
        time.sleep(2.0)
        ggLog.info(f"switched ros_control to state: {resp}")
        return switched_on

    def _setup_joint_control(self, control_mask : int, timeout_s = 300.0) -> int:
        control_mask_srv_name = "/xbotcore/joint_master/set_control_mask"
        wait_for_ros_service(control_mask_srv_name, timeout=timeout_s)
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
    def resetWorld(self):
        super().resetWorld()
        self.clear_commands()
        self._last_jdi_time = float("-inf")
        self._last_joint_device_info = None
        if is_simulated():
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
                                    max_time_s : float = 60) -> None:
        self.clear_commands()
        if velocity_scaling is None:
            velocity_scaling = 1.0
        if acceleration_scaling is None:
            acceleration_scaling = 1.0
        js = self.getJointsState(list(jointPositions.keys()))
        max_traj_duration = 0
        joint_trajs = {}
        for jn,p_ref in jointPositions.items():
            # just to compute the duration
            traj_tpva = build_1D_vramp_trajectory(  t0 = self.getEnvTimeFromStartup(),
                                                    p0 = js[jn].position.item(),
                                                    v0 = js[jn].rate.item(),
                                                    pf = p_ref,
                                                    ctrl_freq_hz = 1000.0,
                                                    max_vel=self._jpos_cmd_max_vel.get(jn, self._jpos_cmd_max_vel_default)*velocity_scaling,
                                                    max_acc=self._jpos_cmd_max_acc.get(jn, self._jpos_cmd_max_acc_default)*acceleration_scaling)
            joint_trajs[jn] = traj_tpva
            traj_duration = traj_tpva[-1][0]
            max_traj_duration = max(0,traj_duration)
        timeout_env = max_traj_duration*2
        timeout_wall = timeout_env*20
        if max_traj_duration > max_time_s:
            raise RuntimeError(f"Computed trajectory is excessively long, would last {max_traj_duration}s, max_time is set to {max_time_s}s. \n"
                               f"Initial joint state was: {[(jn,ji.position.item(),ji.rate.item()) for jn,ji in js.items()]}\n"
                               f"Target joint position was: {jointPositions}\n"
                               f"Durations {[(jn,traj_tpva[-1][0]) for jn,traj_tpva in joint_trajs.items()]}\n"
                               f"Raise it if it is actually ok.")
        # print(f"joint_trajs max_v = {max([max(t[2]) for t in joint_trajs.values() ])}")
        self._setJointTrajectoryCommand(jointTrajectories_tpva = joint_trajs)

        # self.setJointsPositionCommand(jointPositions=jointPositions)
        t0_env = self.getEnvTimeFromStartup()
        t0_wall = time.monotonic()
        js = self.getJointsState(list(jointPositions.keys()))
        errors = [ji.position.item() - jointPositions[jn] for jn,ji in js.items()]
        reached_position = all([abs(e) < joint_position_tolerance for e in errors])
        elapsed_env_time = 0.0
        elapsed_wall_time = 0.0
        while not reached_position:
            self.run(self._stepLength_sec)
            js = self.getJointsState(list(jointPositions.keys()))
            errors = [ji.position.item() - jointPositions[jn] for jn,ji in js.items()]
            reached_position = all([abs(e) < joint_position_tolerance for e in errors])
            elapsed_env_time = self.getEnvTimeFromStartup() - t0_env
            elapsed_wall_time = time.monotonic() - t0_wall
            if elapsed_env_time > timeout_env:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (env timeout {elapsed_env_time}>{timeout_env})\n"
                                    f"    target = {jointPositions}\n"
                                    f"    joint state = {[ji.position.item() for jn,ji in js.items()]}\n"
                                    f"    errors = {errors}\n"
                                    f"    tolerance = {joint_position_tolerance}")
            if elapsed_wall_time > timeout_wall:
                self.clear_commands()
                raise MoveFailError(f"Timed out waiting for sync joint move (wall timeout {elapsed_wall_time}>{timeout_wall}) errors = {errors} tolerance = {joint_position_tolerance}")

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        raise NotImplementedError()
    
    def get_joints_state_step_stats(self):
        raise NotImplementedError()
    
    def quat2rotation(self, qwijk: np.ndarray):

        q_w, q_i, q_j, q_k = qwijk[0, 0], qwijk[0, 1], qwijk[0, 2], qwijk[0, 3]

        R=np.zeros((3,3))
        R[0, 0] = 1 - 2 * (q_j ** 2 + q_k ** 2)
        R[0, 1] = 2 * (q_i * q_j - q_k * q_w)
        R[0, 2] = 2 * (q_i * q_k + q_j * q_w)
        
        R[1, 0] = 2 * (q_i * q_j + q_k * q_w)
        R[1, 1] = 1 - 2 * (q_i ** 2 + q_k ** 2)
        R[1, 2] = 2 * (q_j * q_k - q_i * q_w)
        
        R[2, 0] = 2 * (q_i * q_k - q_j * q_w)
        R[2, 1] = 2 * (q_j * q_k + q_i * q_w)
        R[2, 2] = 1 - 2 * (q_i ** 2 + q_j ** 2)

        return R


