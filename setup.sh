#!/bin/bash

# NOTE: this setup script will be executed right before the launcher file inside the container,
#       use it to configure your environment.

. ${CPK_CODE_DIR}/devel/setup.bash

# export ROS_HOSTNAME=$(hostname).local
# export ROS_MASTER_URI=http://base1.local:11311

export ROS_HOSTNAME=192.168.131.245
export ROS_MASTER_URI=http://192.168.131.11:11311
source ~/.bashrc
