#!/bin/bash

rosservice call /xbotcore/joint_master/set_control_mask "ctrl_mask: 255"
rosservice call /xbotcore/ros_control/switch "data: true"