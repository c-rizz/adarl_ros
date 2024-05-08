#!/usr/bin/env python3
import os
import time
from threading import Lock
from typing import Dict, List, Tuple, Union

import lr_gym.utils.beep
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils
import rospkg
import rospy
import sensor_msgs.msg
from lr_gym_ros.adapters.RosAdapter import RosAdapter
from lr_gym.utils.utils import JointState, LinkState, RequestFailError
import numpy as np

from xbot_interface import config_options as opt
from xbot_interface import xbot_interface as xbot
from urdf_parser_py.urdf import URDF

from std_srvs.srv import SetBool
from xbot_msgs.srv import PluginStatus, SetControlMask
from xbot_msgs.msg import JointDeviceInfo
import torch as th
from lr_gym.adapters.BaseJointImpedanceAdapter import BaseJointImpedanceAdapter
from typing_extensions import override
from cartesian_interface.affine3 import Affine3 # needed by xbot_interface as it doesn't import it correctly
import traceback

def build_xbot_cfg(is_floating_base):
    """
    A function to construct the xbotinterface config object from ros
    """
    t0 = time.monotonic()
    while True:
        urdf = rospy.get_param('/xbotcore/robot_description', default=None) # type: ignore
        if urdf is not None:
            break
        if time.monotonic() - t0 > 30:
            raise TimeoutError()
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


def setup_control():
    control_mask_srv_name = "/xbotcore/joint_master/set_control_mask"
    enable_filter_srv_name = "/xbotcore/enable_joint_filter"
    rospy.wait_for_service(control_mask_srv_name)
    control_mask_srv = rospy.ServiceProxy(control_mask_srv_name, SetControlMask)
    rospy.wait_for_service(enable_filter_srv_name)
    enable_filter_srv = rospy.ServiceProxy(enable_filter_srv_name, SetBool)
    filters_active = False
    joint_active = False
    while not filters_active or not joint_active:
        topic_name = "/xbotcore/joint_device_info"
        jdi = rospy.wait_for_message(topic_name, JointDeviceInfo, timeout = 10)
        if not isinstance(jdi, JointDeviceInfo):
            raise RuntimeError(f"Unexpeced type received from {topic_name}, should be JointDeviceInfo but it's {type(jdi)}")
        filters_active = jdi.filter_active
        joint_active = jdi.mask!=0
        if not filters_active:
            ggLog.info(f"Enabling filters...")
            resp = enable_filter_srv(True)
            if not resp.success:
                raise RuntimeError(f"Failed to enable filters: {resp}")
            print(f"Enabled joint filter")
        if filters_active and not joint_active:
            ggLog.info(f"Enabling control...")
            resp = control_mask_srv(ctrl_mask = 255)
            if not resp.success:
                raise RuntimeError(f"Failed to set control mask: {resp}")
            print(f"Enabled joint control")


def start_control():
    switch_srv_name = "/xbotcore/ros_control/switch"
    state_srv_name = "/xbotcore/ros_control/state"
    rospy.wait_for_service(switch_srv_name)
    ros_ctrl_switch = rospy.ServiceProxy(switch_srv_name, SetBool)
    rospy.wait_for_service(state_srv_name)
    ros_ctrl_state = rospy.ServiceProxy(state_srv_name, PluginStatus)


    running = False
    while not running:
        try:
            resp = ros_ctrl_state()
        except rospy.ServiceException as e:
            print(f"ros_ctrl_state call failed: {e}")
            exit()
        running = resp.status == "Running"
        if not running:
            try:
                resp = ros_ctrl_switch(True)
            except rospy.ServiceException as e:
                print(f"ros_ctrl_switch call failed: {e}")
                exit()
            print(f"ros_ctrl_switch service responded {resp}")        
        time.sleep(0.5)
    print(f"ros_control state: {resp}")


def is_simulated():
    # Is here some better way to do this?
    # Can I ask xbot?
    topic_names = [t[0] for t in rospy.get_published_topics()]
    if "/gazebo/link_states" in topic_names:
        return True
    else:
        return False






class RosXbotAdapter(RosAdapter, BaseJointImpedanceAdapter):

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
                        allow_fallback : bool = True):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__(stepLength_sec, forced_ros_master_uri, maxObsDelay, blocking_observation)
        self._is_floating_base = is_floating_base
        self._model_name = model_name
        self._reference_frame = reference_frame
        self._torch_device = torch_device
        self._commanded_joint_impedances_by_name : Dict[Tuple[str,str], Tuple[float,float,float,float,float]]= {}
        self._joint_cmd_fallback_by_jid = {}
        self._fallback_cmd_stiffness = fallback_cmd_stiffness
        self._fallback_cmd_damping = fallback_cmd_damping
        self._allow_fallback = allow_fallback

    def setJointsToObserve(self, jointsToObserve: List[Tuple[str]]):
        self._xbot_joints_to_monitor = jointsToObserve # keep empty the normal jointsToObserve and use this instead

    def startController(self):
        super().startController()

        cfg = build_xbot_cfg(is_floating_base=self._is_floating_base)
        self._robot_interface = xbot.RobotInterface(cfg)
        ggLog.info(get_system_recap_string(self._robot_interface))
        setup_control()
        self._robot_enabled_joint_names = self._robot_interface.getEnabledJointNames() # this is different from robot.model().getEnabledJointNames()
        self._joints_num = len(self._robot_enabled_joint_names)
        self._joint_name_to_id = {jname : self._robot_enabled_joint_names.index(jname) for jname in self._robot_enabled_joint_names}
        self._joint_id_to_name = {jid : jname for jname, jid in self._joint_name_to_id.items()}
        if is_simulated():
            start_control()

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call setJointsToObserve, and then call startController")
        

        self._robot_interface.sense(update_model=False)

        #TODO: get the delay time somehow
        # obsDelay = float("+inf")
        # while obsDelay > self._maxObsAge:
        #     self.freerun(0.001)
        #     self._robot_interface.sense(update_model=False)
        #     obsDelay = self._robot_interface.getTime() - self._robot_interface.getTimestampRx()            
        # self._jointStateMsgAgeAvg.addValue(obsDelay)

        jpos = self._robot_interface.getJointPosition()
        jvel = self._robot_interface.getJointVelocity()
        jeff = self._robot_interface.getJointEffort()


        ret = {}
        for full_joint_name in requestedJoints:
            model, jname = full_joint_name
            if model != self._model_name:
                raise RuntimeError(f"Requested joint for model different from the monitored one (asked '{model, jname}', but have '{self._model_name}')")
            jid = self._joint_name_to_id[jname]
            ret[full_joint_name] = JointState(position=th.as_tensor(jpos[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                                                rate=th.as_tensor(jvel[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32),
                                                effort=th.as_tensor(jeff[jid]).to(device=self._torch_device, non_blocking=True, dtype=th.float32))
        if self._torch_device.type == "cuda":
            # sync non_blocking cuda transfers
            th.cuda.synchronize(self._torch_device)
        return ret


    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        if not self._listenersStarted:
            raise RuntimeError("called getLinksState without having called startController. The proper way to initialize the controller is to first build the controller, then call setLinksToObserve, and then call startController")

        #print("self._linksToObserve = "+str(self._linksToObserve))
        for l in requestedLinks:
            if l not in self._linksToObserve:
                raise RuntimeError(f"Requested link '{l}' that was not requested in setLinksToObserve (only observing {self._linksToObserve})")

        self._robot_interface.sense(update_model=True)
        #TODO: get the delay time somehow
        # obsDelay = float("+inf")
        # while obsDelay > self._maxObsAge:
        #     self.freerun(0.001)
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
    def setJointsImpedanceCommand(self, joint_impedances_pvesd : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]) -> None:
        # ggLog.info(f"Setting impedances: {joint_impedances_pvesd}")
        for full_jname, jcmd in joint_impedances_pvesd:
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controlled one (asked '{model_name, jname}', but have '{self._model_name}')")
            self._commanded_joint_impedances_by_name[full_jname] = jcmd

    def _apply_commanded_joint_impedances(self):
        self.apply_joint_impedances(list(self._commanded_joint_impedances_by_name.items()))

    @override
    def apply_joint_impedances(self, joint_impedances_pvesd : List[Tuple[Tuple[str,str],Tuple[float,float,float,float,float]]]):

        commanded_joint_impedances_by_jid = {}
        for full_jname, jcmd in joint_impedances_pvesd:
            model_name, jname = full_jname
            if model_name != self._model_name:
                raise RuntimeError(f"Commanded joint impedance for model different from the controleld one (asked '{model_name, jname}', but have '{self._model_name}')")
            jid = self._joint_name_to_id[jname]
            commanded_joint_impedances_by_jid[jid] = jcmd
        
        prefs, vrefs, erefs, pgains, vgains = (np.zeros(shape=(self._joints_num,), dtype=np.float64) 
                                               for _ in range(5))

        curr_pos = self._robot_interface.getJointPosition()
        used_fallback = False
        for jid in range(self._joints_num):
            cmd = commanded_joint_impedances_by_jid.get(jid,None)
            if cmd is None and self._allow_fallback:
                used_fallback = True
                ggLog.warn(f"Missing command for joint {self._joint_id_to_name[jid]} ({jid}), using fallback.")
                # keeps current position
                cmd = (curr_pos[jid], 0, 0, self._fallback_cmd_stiffness, self._fallback_cmd_damping)
            prefs[jid], vrefs[jid], erefs[jid], pgains[jid], vgains[jid] = cmd
        if used_fallback:
            ggLog.warn(f"Used fallback because only had commands for joints_ids:\n {list(commanded_joint_impedances_by_jid.keys())}")
            ggLog.warn(f"I.e. joint names:\n {[ji[0] for ji in joint_impedances_pvesd]}")

        self._robot_interface.setStiffness(pgains)
        self._robot_interface.setDamping(vgains)
        self._robot_interface.setPositionReference(prefs)
        self._robot_interface.setVelocityReference(vrefs)
        self._robot_interface.setEffortReference(erefs)
        self._robot_interface.move()
        # ggLog.info(f"Sent robot_interface command")

    def clear_commands(self):
        self._commanded_joint_impedances_by_name = {}

    def step(self) -> float:
        # traceback.print_stack()
        # ggLog.info(f"applying impedances")
        self._apply_commanded_joint_impedances()
        # ggLog.info(f"stepping")
        step_duration = super().step()
        # ggLog.info(f"clearing")
        self.clear_commands()
        return step_duration