"""
This file implements the MoveitGazeboController class.
"""


from typing import List, Any, Tuple, Dict, Optional

from lr_gym_ros.envControllers.RosEnvController import RequestFailError
from lr_gym_ros.envControllers.MoveitRosController import MoveitRosController
from lr_gym_ros.envControllers.GazeboController import GazeboController
from lr_gym.envControllers.SimulatedEnvController import SimulatedEnvController

import rospy
import sensor_msgs
import sensor_msgs.msg

from lr_gym.utils.utils import JointState, LinkState, Pose


class MoveitGazeboController(MoveitRosController, SimulatedEnvController):
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
        self._gazeboController = GazeboController() #Could do with multiple inheritance but this is more readable
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




    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods



        """

        super().startController()
        self._gazeboController._makeRosConnections()


    
    def spawn_model(self, model_definition : Tuple[str,str], model_name : str, pose : Pose, model_kwargs : Dict[Any,Any], format = None):
        self._gazeboController.spawn_model(model_definition = model_definition,
                                            pose=pose,
                                            model_name=model_name,
                                            model_kwargs=model_kwargs,
                                            format=format)

    def delete_model(self, model_name : str):
        """Delete a model from the environment"""
        self._gazeboController.delete_model(model_name = model_name)

    def setupLight(self, *args, **kwargs):
        self._gazeboController.setupLight(*args, **kwargs)

    def setJointsStateDirect(self, jointStates : Dict[Tuple[str,str],JointState]):
        """Set the state for a set of joints

        Parameters
        ----------
        jointStates : Dict[Tuple[str,str],JointState]
            Keys are in the format (model_name, joint_name), the value is the joint state to enforce
        """
        wasPaused = self._gazeboController.isPaused()
        if wasPaused:
            self._gazeboController.unpauseSimulation()
        self._gazeboController.setJointsStateDirect(jointStates = jointStates)
        if wasPaused:
            self._gazeboController.pauseSimulation()
    

    def setLinksStateDirect(self, linksStates : Dict[Tuple[str,str],LinkState]):
        """Set the state for a set of links

        Parameters
        ----------
        linksStates : Dict[Tuple[str,str],LinkState]
            Keys are in the format (model_name, link_name), the value is the link state to enforce
        """
        wasPaused = self._gazeboController.isPaused()
        if wasPaused:
            self._gazeboController.unpauseSimulation()
        self._gazeboController.setLinksStateDirect(linksStates = linksStates)
        if wasPaused:
            self._gazeboController.pauseSimulation()

    
    def getRenderings(self, requestedCameras : List[str]) -> List[sensor_msgs.msg.Image]: #TODO: change this to use cv2 images (i.e. ndarrays)
        try:
            r = super().getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from ros")
        except:
            r = self._gazeboController.getRenderings(requestedCameras=requestedCameras)
            # ggLog.info("got image from gazebo plugin")
        return r

    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        try:
            js = self._gazeboController.getJointsState(requestedJoints=requestedJoints)
        except RequestFailError as e:
            missing_jonts = [jr for jr in requestedJoints if jr not in e.partialResult]
            js = super().getJointsState(requestedJoints=missing_jonts)
            js.update(e.partialResult)
        return js

    def getLinksState(self, requestedLinks : List[Tuple[str,str]]) -> Dict[Tuple[str,str],LinkState]:
        try:
            ls =  self._gazeboController.getLinksState(requestedLinks=requestedLinks) # WARNING! These may not be the same frames as the urdf unes!
            # ggLog.info("Got link state from ros")
        except RequestFailError as e:
            # This allows to get the pose of links that are not tracked by ros e.g. manipulated objects
            missing_links = [rl for rl in requestedLinks if rl not in e.partialResult]
            ls = super().getLinksState(requestedLinks=missing_links)
            # ggLog.info(f"Got link state for {ls.keys()} from gazebo plugin and link state for {e.partialResult.keys()} from ros")
            ls.update(e.partialResult)
        
        # ggLog.info(f"Link state is {ls}")
        return ls
            
    def step(self):
        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not step.")
        self._gazeboController.unpauseSimulation()
        r = super().step()
        self._gazeboController.pauseSimulation()
        return r

    def resetWorld(self):
        if rospy.is_shutdown():
            raise RuntimeError("ROS has been shut down. Will not reset.")
        self._gazeboController.unpauseSimulation()
        r = super().resetWorld()
        self._gazeboController.pauseSimulation()
        return r

    
    def moveGripperSync(self, width : float, max_effort : float):
        wasPaused = self._gazeboController.isPaused()
        if wasPaused:
            self._gazeboController.unpauseSimulation()
        super().moveGripperSync(width, max_effort)
        if wasPaused:
            self._gazeboController.pauseSimulation()

    def moveToEePoseSync(self,  pose : List[float], do_cartesian = False, velocity_scaling : Optional[float] = None,
                                acceleration_scaling : Optional[float] = None, ee_link : Optional[str] = None,
                                reference_frame : Optional[str] = None):
        wasPaused = self._gazeboController.isPaused()
        if wasPaused:
            self._gazeboController.unpauseSimulation()
        super().moveToEePoseSync(pose = pose, do_cartesian = do_cartesian, velocity_scaling = velocity_scaling, acceleration_scaling = acceleration_scaling,
                                ee_link = ee_link, reference_frame = reference_frame)
        if wasPaused:
            self._gazeboController.pauseSimulation()
    
    def moveToJointPoseSync(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : Optional[float] = None,
                                    acceleration_scaling : Optional[float] = None) -> None:
        wasPaused = self._gazeboController.isPaused()
        if wasPaused:
            self._gazeboController.unpauseSimulation()
        super().moveToJointPoseSync(jointPositions = jointPositions, velocity_scaling=velocity_scaling, acceleration_scaling=acceleration_scaling)
        if wasPaused:
            self._gazeboController.pauseSimulation()

    
    def freerun(self, duration_sec : float):
        self._gazeboController.freerun(duration_sec)


    def setJointsToObserve(self, jointsToObserve : List[Tuple[str,str]]):
       super().setJointsToObserve(jointsToObserve=jointsToObserve)
       self._gazeboController.setJointsToObserve(jointsToObserve=jointsToObserve)

    def setLinksToObserve(self, linksToObserve : List[Tuple[str,str]]):
       super().setLinksToObserve(linksToObserve=linksToObserve)
       self._gazeboController.setLinksToObserve(linksToObserve=linksToObserve)

    def setCamerasToObserve(self, camerasToRender : List[str] = []):
       super().setCamerasToObserve(camerasToRender=camerasToRender)
       self._gazeboController.setCamerasToObserve(camerasToRender=camerasToRender)