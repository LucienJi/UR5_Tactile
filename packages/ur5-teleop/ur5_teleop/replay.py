#! /usr/bin/env python3


from threading import Thread

import numpy as np
import rospy
import cv2
from geometry_msgs.msg import Twist, PoseStamped, WrenchStamped
from quest2ros.msg import OVR2ROSHapticFeedback, OVR2ROSInputs
from scipy.spatial.transform import Rotation as R
from ur5_twist_control.helper import ori2numpy, point2numpy
from ur5_twist_control.twist_control import TwistControl
from collections import deque
import torch 
import pickle

def command_line_listener():
    while True:
        command = input("Enter command: ")
        if command == 'q':
            rospy.signal_shutdown("User requested shutdown")
            exit()

if __name__ == "__main__":

    file_path = "/code/src/ur5-diffusha/data/debug/2024-12-13/20-35-44.pt"
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    rospy.init_node('ur5_replay')
    twist_controller = TwistControl(init_arm=True)
    input_thread = Thread(target=command_line_listener)
    input_thread.start()


    
    rate = rospy.Rate(30)
    max_step = len(data)
    ct = 0
    while not rospy.is_shutdown():
        d = data[ct]
        cmd_trans_vel = d['cmd_trans_vel']
        cmd_rot_vel = d['cmd_rot_vel']
        cmd_grasp_pos = d['cmd_grasp_pos']
        # cmd_trans_vel = np.zeros(3)
        # cmd_rot_vel = np.zeros(3)
        cmd_grasp_pos = 0

        cmd_grasp_pos = np.clip(cmd_grasp_pos, 0, 1)
        
        twist_controller.move_vel(cmd_trans_vel, cmd_rot_vel)
        twist_controller.gripper.command.rPRA = int(255 * cmd_grasp_pos)
        twist_controller.gripper.publish_command(twist_controller.gripper.command)
        ct += 1
        rate.sleep()
        if ct >= max_step:
            break
    
    twist_controller.move_vel(np.zeros(3), np.zeros(3))
    rospy.spin()

    