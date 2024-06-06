"""
This file implements the MoveitGazeboAdapter class.
"""


from typing import List, Any, Tuple, Dict, Optional

from adarl_ros.adapters.RosAdapter import RequestFailError
from adarl_ros.adapters.MoveitRosAdapter import MoveitRosAdapter
from adarl_ros.adapters.GazeboAdapter import GazeboAdapter
from adarl.adapters.BaseSimulationAdapter import BaseSimulationAdapter

import rospy
import sensor_msgs
import sensor_msgs.msg

from adarl.utils.utils import JointState, LinkState, Pose
import numpy as np
from overrides import override

class MoveitGazeboAdapter(MoveitRosAdapter, BaseSimulationAdapter):
    """
    """

    def __init__(self,
                 jointsOrder : List[Tuple[str,str]],
                 endEffectorLink : Tuple[str,str],
                 referenceFrame : str,
                 initialJointPose : Optional[Dict[Tuple[str,str],float]],
                 gripperActionTopic : Optional[str] = None,
                 gripperInitialWidth : float = -1,
                 default_velocity_scaling = 0.1,
                 default_acceleration_scaling = 0.1,
                 maxObsDelay = float("+inf"),
                 blocking_observation = False):
        """Initialize the environment controller.

        """
        self._gazeboAdapter = GazeboAdapter() #Could do with multiple inheritance but this is more readable
        super().__init__(   jointsOrder = jointsOrder,
                            endEffectorLink = endEffectorLink,
                            referenceFrame = referenceFrame,
                            initialJointPose= initialJointPose,
                            gripperActionTopic = gripperActionTopic,
                            gripperInitialWidth = gripperInitialWidth,
                            default_velocity_scaling = default_velocity_scaling,
                            default_acceleration_scaling = default_acceleration_scaling,
                            maxObsDelay = maxObsDelay,
                            blocking_observation = blocking_observation)



    @override
    def startup(self):
        """Start the ROS listeners for receiving images, link states and joint states.
        The topics to listen to must be specified using the set_monitored_cameras, set_monitored_joints, and set_monitored_links methods
        """

        super().startup()
        self._gazeboAdapter._makeRosConnections()


    # @override
    def spawn_model(self,   model_file : Tuple[str,str],
                            model_name : str,
                            pose : Pose,
                            model_kwargs : Dict[Any,Any] = {},
                            model_format = None) -> str:
        return self._gazeboAdapter.spawn_model(model_file = model_file,
                                            pose=pose,
                                            model_name=model_name,
                                            model_kwargs=model_kwargs,
                                            model_format=model_format)
    @override
    def delete_model(self, model_name : str):
        """Delete a model from the environment"""
        self._gazeboAdapter.delete_model(model_name = model_name)
    
    @override
    def setupLight(self, *args, **kwargs):
        self._gazeboAdapter.setupLight(*args, **kwargs)

    @override
    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        wasPaused = self._gazeboAdapter.isPaused()
        if wasPaused:
            self._gazeboAdapter.unpauseSimulation()
        self._gazeboAdapter.setJointsStateDirect(jointStates = jointStates)
        if wasPaused:
            self._gazeboAdapter.pauseSimulation()
    
    @override
    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        wasPaused = self._gazeboAdapter.isPaused()
        if wasPaused:
            self._gazeboAdapter.unpauseSimulation()
        self._gazeboAdapter.setLinksStateDirect(linksStates = linksStates)
        if wasPaused:
            self._gazeboAdapter.pauseSimulation()

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
    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        try:
            js = self._gazeboAdapter.getJointsState(requestedJoints=requestedJoints)
        except RequestFailError as e:
            missing_jonts = [jr for jr in requestedJoints if jr not in e.partialResult]
            js = super().getJointsState(requestedJoints=missing_jonts)
            js.update(e.partialResult)
        return js

    @override
    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        try:
            ls =  self._gazeboAdapter.getLinksState(requestedLinks=requestedLinks) # WARNING! These may not be the same frames as the urdf ones!
        except RequestFailError as e:
            missing_links = [rl for rl in requestedLinks if rl not in e.partialResult]
            ls = super().getLinksState(requestedLinks=missing_links)
            ls.update(e.partialResult)
        return ls

    @override
    def step(self) -> float:
        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not step.")
        self._gazeboAdapter.unpauseSimulation()
        r = super().step()
        self._gazeboAdapter.pauseSimulation()
        return r

    @override
    def resetWorld(self):
        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not reset.")
        self._gazeboAdapter.unpauseSimulation()
        r = super().resetWorld()
        self._gazeboAdapter.pauseSimulation()
        return r

    def moveGripperSync(self, width : float, max_effort : float):
        wasPaused = self._gazeboAdapter.isPaused()
        if wasPaused:
            self._gazeboAdapter.unpauseSimulation()
        super().moveGripperSync(width, max_effort)
        if wasPaused:
            self._gazeboAdapter.pauseSimulation()

    @override
    def moveToEePoseSync(self,  poses : Dict[Tuple[str,str],List[float]] = None,
                                do_cartesian = False, velocity_scaling : Optional[float] = None,
                                acceleration_scaling : Optional[float] = None, ee_link : Optional[Tuple[str,str]] = None,
                                reference_frame : Optional[str] = None):
        wasPaused = self._gazeboAdapter.isPaused()
        if wasPaused:
            self._gazeboAdapter.unpauseSimulation()
        super().moveToEePoseSync(poses = poses, do_cartesian = do_cartesian, velocity_scaling = velocity_scaling, acceleration_scaling = acceleration_scaling,
                                ee_link = ee_link, reference_frame = reference_frame)
        if wasPaused:
            self._gazeboAdapter.pauseSimulation()

    @override
    def moveToJointPoseSync(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None) -> None:
        wasPaused = self._gazeboAdapter.isPaused()
        if wasPaused:
            self._gazeboAdapter.unpauseSimulation()
        super().moveToJointPoseSync(jointPositions = jointPositions, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)
        if wasPaused:
            self._gazeboAdapter.pauseSimulation()

    @override
    def freerun(self, duration_sec : float):
        self._gazeboAdapter.freerun(duration_sec)

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