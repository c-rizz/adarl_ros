"""This file implements the EffortRosControlAdapter class."""
#!/usr/bin/env python3
from typing import List, Tuple, Dict, Union

from adarl_ros.adapters.RosAdapter import RosAdapter
from adarl_ros.rosControlUtils import ControllerManagementHelper
from adarl_ros.rosControlUtils import TrajectoryControllerHelper

from adarl.adapters.BaseJointEffortAdapter import BaseJointEffortAdapter

import rospy
import std_msgs.msg

class EffortRosControlAdapter(RosAdapter, BaseJointEffortAdapter):
    """This class allows to control the execution of a ROS-based environment.

    Controls robot joints using ros_control's effort controllers and trajectory controllers.
    Effort controllers are to used for performing the actions of the RL agent. rajectory
    controllers are only used to reset robot positions.


    This is meant to be used only in simulation, some movements of the robots may
    not be safe in the real world (both for people and robots health)

    """

    def __init__(self,
                 effortControllersInfos : Dict[str,Tuple[str,str,Tuple[str]]],
                 trajectoryControllersInfos : Dict[str,Tuple[str,str,Tuple[str]]],
                 initialJointPositions : List[Tuple[str,str,float]],
                 stepLength_sec : float = 0.001,
                 forced_ros_master_uri : Union[str,None] = None):
        """Initialize the environment controller.

        Parameters
        ----------
        effortControllersInfos :  Dict[str,Tuple[str,str,Tuple[str]]]
            Dictionary specifing the effort controllers to be used. Each entry is identified by the controller name and contains a Tuple of the
            format [topic_name, model_name, joint_names] where joint_names is a Tuple of strings, indicating the joints controlled by the controller,
            in the order of the controller definition

        trajectoryControllersInfos :  Dict[str,Tuple[str,str,Tuple[str]]]
            Dictionary specifing the trajectory controllers to be used. Each entry is identified by the controller name and contains a Tuple of the
            format [topic_name, model_name, joint_names] where joint_names is a Tuple of strings, indicating the joints controlled by the controller,
            in the order of the controller definition
            These controllers are used to move the joints to their start position (e.g. when the environments gets resetted)

        initialJointPositions :  List[Tuple[str,str,float]]
            List of joint positions to enforce at every reset. Each element in the list
            is in the format [model_name, joint_name, position]. All the joint controlled
            by effort and trajectory controllers should also appear here.

        Raises
        -------
        ROSException
            If it fails to find required services or topics

        """
        rospy.loginfo("setting stepLength_sec to "+str(stepLength_sec))
        super().__init__(stepLength_sec = stepLength_sec, forced_ros_master_uri = forced_ros_master_uri)

        self._effortControllersInfos = effortControllersInfos
        self._trajectoryControllersInfos = trajectoryControllersInfos
        self._initialJointPositions = initialJointPositions

        self._initialTrajectoryControllersSetup = {}
        
        usedInitialpositions = [] #just for a safety check

        print(str(type(self._trajectoryControllersInfos)))
        for controllerName in self._trajectoryControllersInfos:
            controllerInfo = self._trajectoryControllersInfos[controllerName]
            jointNames = []
            positions = []
            for jointName in controllerInfo[2]:
                position = None
                for initialPosition in self._initialJointPositions:
                    if initialPosition[1] == jointName:
                        position = initialPosition[2]
                        usedInitialpositions.append(initialPosition)
                if position is None:
                    raise RuntimeError("No initial position was specified for joint '"+jointName+"' of controller "+controllerName)
                positions.append(position)
                jointNames.append(jointName)
            self._initialTrajectoryControllersSetup[controllerName] = (jointNames,positions)

        for initialPosition in self._initialJointPositions:
            if initialPosition not in usedInitialpositions:
                raise RuntimeError("Initial position "+str(initialPosition)+" is not used by any controller")

    def startup(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the set_monitored_cameras, set_monitored_joints, and set_monitored_links methods

        Returns
        -------
        type
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """
        super().startup()

        self._controllerManagementHelper = ControllerManagementHelper()
        self._trajectoryControllerHelpers = {}

        for controllerName in self._trajectoryControllersInfos.keys():
            self._trajectoryControllerHelpers[controllerName] = TrajectoryControllerHelper(controllerName, self._controllerManagementHelper)

        self._controllerManagementHelper.waitForControllersLoad(self._effortControllersInfos.keys())
        self._controllerManagementHelper.waitForControllersLoad(self._trajectoryControllersInfos.keys())
        rospy.loginfo("EffortRosControlAdapter: controllers are loaded.")

        self._effortControllerPubs = {}
        for controllerName in self._effortControllersInfos.keys():
            controllerInfo = self._effortControllersInfos[controllerName]
            self._effortControllerPubs[controllerName] = rospy.Publisher(controllerInfo[0]+"/command", std_msgs.msg.Float64MultiArray, queue_size=1)


    def setJointsEffortCommand(self, jointTorques : List[Tuple[str,str,float]]) -> None:
        for controllerName in self._effortControllersInfos.keys():
            controllerInfo = self._effortControllersInfos[controllerName]
            command = []
            for jointName in controllerInfo[2]:
                val = None
                for requestedTorque in jointTorques:
                    if requestedTorque[1] == jointName:
                        val = requestedTorque[2]
                if val is None:
                    rospy.logwarn("No torque was specified for joint "+jointName+" will use zero torque")
                    val = 0
                command.append(val)
            commandMsg = std_msgs.msg.Float64MultiArray()
            commandMsg.data = command
            #rospy.loginfo("Commanding effort "+str(command)+" to controller "+controllerName)
            self._effortControllerPubs[controllerName].publish(commandMsg)

    def resetWorld(self):
        rospy.logdebug("EffortRosControlAdapter: switching to trajectory controllers")
        trajectoryControllers = list(self._trajectoryControllersInfos.keys())
        effortControllers = list(self._effortControllersInfos.keys())
        self._controllerManagementHelper.switchControllers(trajectoryControllers, effortControllers)
        self._controllerManagementHelper.waitForControllersStart(trajectoryControllers)
        rospy.logdebug("EffortRosControlAdapter: trajectory controllers started")


        for tch in self._trajectoryControllerHelpers.values():
            setup = self._initialTrajectoryControllersSetup[tch.getControllerName()]
            tch.moveToJointPosition(setup[0],setup[1], 1)


        rospy.logdebug("EffortRosControlAdapter: switching to effort controllers")
        self._controllerManagementHelper.switchControllers(effortControllers, trajectoryControllers)
        self._controllerManagementHelper.waitForControllersStart(effortControllers)
        rospy.logdebug("EffortRosControlAdapter: effort controllers started")

        self._startup_env_time = rospy.get_time()
