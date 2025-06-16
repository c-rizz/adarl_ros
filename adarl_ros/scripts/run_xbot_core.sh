#!/bin/bash

sleep 5 # Dirty fix
echo "Launching xbot2-core. ROS_MASTER_URI='${ROS_MASTER_URI}'"
xbot2-core $@