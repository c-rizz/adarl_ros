#!/usr/bin/env python3

"""This file implements the PandaMoveitRosAdapter class."""

from lr_gym_ros.adapters.MoveitRosAdapter import MoveitRosAdapter

import rospy
import franka_msgs.msg
import time
from threading import Lock
import lr_gym.utils.dbg.ggLog as ggLog
from typing import Dict, List, Tuple, Optional
import lr_gym_ros_utils
import lr_gym_ros
from lr_gym.utils.utils import MoveFailError
import lr_gym.utils.beep
from actionlib_msgs.msg import GoalStatus
from overrides import override

class PandaMoveitRosAdapter(MoveitRosAdapter):
    """This class allows to control the execution of a ROS-based environment.

    Allows to control the robot via cartesian end-effector control. Inverse kinematics and
    planning are performed via Moveit!.

    TODO: Now that we have python3 avoid using move_helper
    """

    

    def startController(self):
        """Start the ROS listeners for receiving images, link states and joint states.

        The topics to listen to must be specified using the setCamerasToObserve, setJointsToObserve, and setLinksToObserve methods



        """
        self._action_fail_count = 0
        self._default_max_tries = 10
        super().startController()
        self._frankaStateMutex = Lock() #To synchronize _jointStateCallback with getJointsState
        self._errorRecoveryActionClient = self._connectRosAction('/franka_control/error_recovery', franka_msgs.msg.ErrorRecoveryAction)
        self._frankaStateSubscriber = rospy.Subscriber('/franka_state_controller/franka_states', franka_msgs.msg.FrankaState, self._frankaStateCallback, queue_size=1)

    def _frankaStateCallback(self, msg):
        self._frankaStateMutex.acquire()
        self._lastFrankaStateReceived = msg
        self._frankaStateMutex.release()

    def _getFrankaState(self):
        self._frankaStateMutex.acquire()
        ret = self._lastFrankaStateReceived
        self._frankaStateMutex.release()
        return ret

    def _getNewFrankaState(self):
        t0 = rospy.get_rostime()
        timeout = 10.0
        while True:
            ret = self._getFrankaState()
            td = (ret.header.stamp - t0).to_sec()
            if td > 0:
                break
            if (rospy.get_rostime() - t0).to_sec() > timeout:
                raise RuntimeError(f"Failed to get new franka state in {timeout} seconds.")
            time.sleep(0.01)
        return ret

    def _checkArmErrorAndRecover(self, max_tries = 10, blocking = True):
        fs = self._getNewFrankaState()
        tries = 0
        there_was_a_fail = False
        failing_states = []
        while fs.robot_mode != franka_msgs.msg.FrankaState.ROBOT_MODE_MOVE:
            there_was_a_fail = True
            failing_states.append(fs)
            # ggLog.warn(f"Panda arm is in robot_mode {fs.robot_mode}.\n Franka state = {fs}\n Trying to recover (retries = {tries}).")
            goal = franka_msgs.msg.ErrorRecoveryGoal()
            goal_state = self._errorRecoveryActionClient.send_goal_and_wait(goal, rospy.Duration.from_sec(10.0), rospy.Duration.from_sec(10.0))
            if goal_state != GoalStatus.SUCCEEDED:
                ggLog.warn(f"Failed to execute Panda arm recovery action. State = {goal_state}")
            else:
                pass
                # ggLog.info("Panda arm recovery action completed.")
            fs = self._getNewFrankaState()
            rospy.sleep(0.5)
            tries += 1
            if tries > max_tries:
                if blocking:
                    lr_gym.utils.beep.boop()
                    ggLog.error("Panda arm failed to recover automatically. Please try to put it back into a working pose manually and press Enter.")
                    try:
                        input("Press Enter when done.")
                    except EOFError as e:
                        ggLog.error("Error getting input: "+str(e))
                    tries=0
                else:
                    break
        return there_was_a_fail, failing_states

    def set_max_tries(self, max_tries):
        self._default_max_tries = max_tries

    def get_max_tries(self):
        return self._default_max_tries

    @override
    def resetWorld(self):
        self._checkArmErrorAndRecover()
        try:
            return super().resetWorld()
        except MoveFailError as e:
            ggLog.warn(f"resetWorld failed. Will try to recover. exception = {e}")
        self._step_start_time = self.getEnvSimTimeFromStart()

    @override
    def step(self) -> float:
        there_was_a_fail, failing_states = self._checkArmErrorAndRecover()
        if there_was_a_fail:
            self._action_fail_count += 1
            ggLog.info(f"There was a fail during action, fail rate = {self._action_fail_count/self._alltime_step_count}, state = {failing_states}")
        try:
            return super().step()
        except MoveFailError as e:
            ggLog.warn(f"Step failed. Will try to recover. exception = {e}")
        t = self.getEnvSimTimeFromStart()
        step_duration = t - self._step_start_time
        self._step_start_time = t
        return step_duration

    def _runRecoveringBlocking(self, function, functionName : str, max_tries = None, blocking = True, quiet = False):
        if max_tries is None:
            max_tries = self._default_max_tries
        self._checkArmErrorAndRecover()
        done = False
        tries = 0
        while not done:
            try:
                function()
                done = True
            except MoveFailError as e:
                ggLog.warn(f"{functionName} failed. exception = {e}\n"+
                            f"Trying to recover (retries = {tries}).")
                if not quiet or tries < max_tries*0.2: # don't beep the first two times, it's annoying
                    lr_gym.utils.beep.beep()
                rospy.sleep(1.0)
                self._checkArmErrorAndRecover()
                tries += 1
                if tries > max_tries:
                    if blocking:
                        lr_gym.utils.beep.boop()
                        ggLog.error("Panda arm failed to recover automatically. Please try to put it back into a working pose manually and press Enter.")
                        try:
                            input("Press Enter when done.")
                        except EOFError as e:
                            ggLog.error("Error getting input: "+str(e))
                        tries=0
                        a = ""
                        while a!='y' and a!='n':
                            a = input("Retry last movement? If not, the movement will be skipped (Y/n)")
                        if a=="n":
                            ggLog.error("Skipping movement")
                            break
                        else:
                            ggLog.error("retrying movement")
                    else:
                        raise MoveFailError(f"Failed to move at {functionName} and blocking is False")

    @override
    def moveToJointPoseSync(self, jointPositions : Dict[Tuple[str,str],float], velocity_scaling : Optional[float] = None, acceleration_scaling : Optional[float] = None, blocking = True) -> None:
        def function():
            super(PandaMoveitRosAdapter,self).moveToJointPoseSync(jointPositions, velocity_scaling, acceleration_scaling)
        self._runRecoveringBlocking(function, "moveToJointPoseSync", blocking = blocking)

    @override
    def moveToEePoseSync(self,  poses : Dict[Tuple[str,str],List[float]] = None,
                                do_cartesian = False, velocity_scaling : Optional[float] = None, acceleration_scaling : Optional[float] = None,
                                ee_link : Optional[Tuple[str,str]] = None, reference_frame : Optional[str] = None, blocking = True):
        def function():
            super(PandaMoveitRosAdapter,self).moveToEePoseSync(poses, do_cartesian, velocity_scaling, acceleration_scaling, ee_link, reference_frame)
        self._runRecoveringBlocking(function, "moveToEePoseSync", blocking = blocking)
                         
    def moveGripperSync(self, width : float, max_effort : float):
        def function():
            super(PandaMoveitRosAdapter,self).moveGripperSync(width, max_effort)
        self._runRecoveringBlocking(function, "moveGripperSync")


