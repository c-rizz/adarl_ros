#!/usr/bin/env python3
import os
import time
from threading import Lock
from typing import Dict, List, Tuple, Union

import lr_gym.utils.beep
import lr_gym.utils.dbg.ggLog as ggLog
import lr_gym.utils.utils
import lr_gym_ros_utils.ros_launch_utils
import rospkg
import rospy
import sensor_msgs.msg
from lr_gym.adapters.BaseAdapter import BaseAdapter
from lr_gym.utils.utils import JointState, LinkState, RequestFailError
from lr_gym_ros_utils.msg import LinkStates
import numpy as np


class RosAdapter(BaseAdapter):
    """This class allows to control the execution of a ROS-based environment.

    This is meant to be able to control both simulated and real environments, by using ROS.

    """

    def __init__(   self, stepLength_sec : float = 0.001, forced_ros_master_uri : Union[str, None] = None, maxObsDelay = float("+inf"), blocking_observation = False):
        """Initialize the Simulator controller.

        Raises
        -------
        ROSException
            If it fails to find the gazebo services

        """
        super().__init__()
        self._stepLength_sec = stepLength_sec

        self._forced_ros_master_uri = forced_ros_master_uri
        self._listenersStarted = False

        self._lastImagesReceived = {}
        self._lastJointStatesReceived = None
        # self._lastLinkStatesReceived = None
        self._linkStates = {}

        self._jointStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._linkStatesMutex = Lock() #To synchronize _jointStateCallback with getJointsState

        self._jointStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._linkStateMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._cameraMsgAgeAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)

        self._cameraMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._linkMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)
        self._jointMsgWaitAvg = lr_gym.utils.utils.AverageKeeper(bufferSize = 100)

        self._maxObsAge = maxObsDelay
        self._blocking_observation = blocking_observation
        self._mmRosLauncher : lr_gym_ros_utils.ros_launch_utils.MultiMasterRosLauncher = None



    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods

        Returns
        -------
        type
            Description of returned object.

        Raises
        -------
        ExceptionName
            Why the exception is raised.

        """

        if self._forced_ros_master_uri is not None:
            os.environ['ROS_MASTER_URI'] = self._forced_ros_master_uri

        # init_node uses use_sim_time to determine which time to use, but I can't
        # find a reliable way for it to be set before init_node is being called
        # So we wait for it to be set to either true or false
        useSimTime = None
        while useSimTime is None:
            try:
                useSimTime = rospy.get_param("/use_sim_time")
            except KeyError:
                ggLog.warn("Could not get /use_sim_time. Will retry")
                time.sleep(1)
            except ConnectionRefusedError:
                ggLog.error("No connection to ROS parameter server. Will retry")
                time.sleep(1)

        rospy.init_node('ros_env_controller', anonymous=True)
        lr_gym.utils.utils.setupSigintHandler()

        self._simTimeStart = rospy.get_time() #Will be overwritten by resetWorld
        self._lastStepEnd = self.getEnvTimeFromStartup() #Will be overwritten by resetWorld

        self._imageSubscribers = []
        for cam_topic in self._camerasToObserve:
            self._lastImagesReceived[cam_topic] = None
            self._imageSubscribers.append(rospy.Subscriber(cam_topic, sensor_msgs.msg.Image, self._imagesCallback, callback_args=(self,cam_topic)))
            ggLog.info(f"Subscribed to {cam_topic}")

        if len(self._jointsToObserve)>0:
            topic = "joint_states"
            self._jointStateSubscriber = rospy.Subscriber(topic, sensor_msgs.msg.JointState, self._jointStateCallback, queue_size=1)
            ggLog.info(f"Subscribed to {topic}")

        if len(self._linksToObserve)>0:
            topic = "link_states"
            self._linkStatesSubscriber = rospy.Subscriber(topic, LinkStates, self._linkStatesCallback, queue_size=1)
            ggLog.info(f"Subscribed to {topic}")




        self._listenersStarted = True



    def getJointsState(self, requestedJoints : List[Tuple[str,str]]) -> Dict[Tuple[str,str],JointState]:
        if not self._listenersStarted:
            raise RuntimeError("called getJointsState without having called startController. The proper way to initialize the controller is to first build the controller, then call setJointsToObserve, and then call startController")
        

        gottenJoints = {}
        missingJoints = requestedJoints
        for j in requestedJoints:
            if j not in self._jointsToObserve:
                raise RuntimeError("Requested joint that was not requested in setJointsToObserve")



            # ggLog.info("RosAdapter.getJointsState() called")

            call_time = rospy.get_time()

            lastErrTime = call_time
            while True:
                with self._jointStatesMutex:
                    jointStatesMsg = self._lastJointStatesReceived
                    if jointStatesMsg is not None:
                        msgAge = call_time - jointStatesMsg.header.stamp.to_sec()
                        # print(f"msgAge = {msgAge}")
                        if msgAge < self._maxObsAge or self._maxObsAge == float("+inf"):
                            for j in requestedJoints:
                                modelName = j[0]
                                jointName = j[1]                    
                                try:
                                    jointIndex = jointStatesMsg.name.index(jointName)
                                except ValueError:
                                    jointIndex = None
                                if jointIndex is not None:
                                    gottenJoints[j] = JointState([jointStatesMsg.position[jointIndex]], [jointStatesMsg.velocity[jointIndex]], [jointStatesMsg.effort[jointIndex]])
                    else:
                        msgAge = float("+inf")
                missingJoints = []
                for j in requestedJoints:
                    if j not in gottenJoints:
                        missingJoints.append(j)
                if len(missingJoints) == 0 or not self._blocking_observation:
                    break
                self.freerun(0.01)

                if rospy.get_time() - lastErrTime > 10:
                    ggLog.warn(f"Waiting for joints since {rospy.get_time()-call_time}s. Still missing: {missingJoints}")
                    lr_gym.utils.beep.beep()                
                    lastErrTime = rospy.get_time()


            self._jointStateMsgAgeAvg.addValue(msgAge)
            waitTime = rospy.get_time() - call_time
            self._jointMsgWaitAvg.addValue(waitTime)



        if len(missingJoints)>0:
            err = f"Failed to get state for joints {missingJoints}, requested {requestedJoints} "
            #rospy.logerr(err)
            raise RequestFailError(message=err, partialResult=gottenJoints)

        return gottenJoints
