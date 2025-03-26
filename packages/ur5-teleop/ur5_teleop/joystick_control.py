#! /usr/bin/env python3
"""
This provides an interface to move the arm with a joystick (TwistController).

- The arm initializes to a specific pose
- A user tries to reach one of the four goals on the table
- Action space is x y z velocity and no orientation
- Bell rings when the x, y location of the tip get inside of the goal
- Buzzer rings when the x, y location of the tip goes out of the goal
- Once the tip reaches the table, play some sound and initialize the arm --> next episode
  - Make sure to record the reached x, y location
"""

from threading import Thread

import numpy as np
import rospy
import cv2
from geometry_msgs.msg import Twist, PoseStamped, WrenchStamped
from sensor_msgs.msg import Joy 
from quest2ros.msg import OVR2ROSHapticFeedback, OVR2ROSInputs
from scipy.spatial.transform import Rotation as R
from ur5_twist_control.helper import ori2numpy, point2numpy
from ur5_twist_control.twist_control import TwistControl
from collections import deque

class JoyStickControl:
    def __init__(self) -> None:

        """
        Each time we call the joy_callback, we change the velocity of the arm  
        if we release the button, msg will also be updated and the velocity will be zero 


        """
        self.timestep = 0
        self.linvel_history = deque(maxlen=10)
        self.angvel_history = deque(maxlen=10)

        self.cur_lin_vel = np.zeros(3)
        self.cur_ang_vel = np.zeros(3) 
        self.gripper = 0.0

        self.lin_vel_coeff = 0.1
        self.ang_vel_coeff = 0.4

        rospy.init_node('ur5_joy_control')
        self._sub_joy = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)
        self.twist_controller = TwistControl(init_arm=True)
        self.input_thread = Thread(target=self.command_line_listener)
        self.input_thread.start()
    
    def joy_callback(self,msg):

        """ Discrete Version"""
        # tmp_lin_vel = np.zeros(3)
        # ## x-y axis velocity 
        # tmp_lin_vel[0] = msg.axes[0] ## Y-axis
        # tmp_lin_vel[1] = -1.0 * msg.axes[1] ## X-axis 

        # ## z-axis velocity 
        # ## msg.buttons[0,1,2,3] = up, right, down, left
        # tmp_lin_vel[2] = msg.buttons[0] - msg.buttons[2] 

        # ## rotation velocity 

        # tmp_ang_vel = np.zeros(3)
        # tmp_ang_vel[2] = msg.buttons[3] - msg.buttons[1]
        # tmp_ang_vel[1] = msg.buttons[5] - msg.buttons[4]
            
        """ Continuous Version"""
        tmp_lin_vel = np.zeros(3)
        # tmp_lin_vel[0] = msg.axes[0] ## Y-axis
        y_val = msg.axes[0]
        drift = 0.12
        if y_val < drift:
            y_val = y_val - drift
            y_val = max(y_val,-1)
            y_val = 0 if np.abs(y_val) < 0.04 else y_val
            
        tmp_lin_vel[0] = y_val
        tmp_lin_vel[1] = -1.0 * msg.axes[1] ## X-axis 
        tmp_lin_vel[2] = msg.axes[3] ## Z-axis

        tmp_ang_vel = np.zeros(3)
        tmp_ang_vel[2] = msg.axes[2]    # z-axis rotation
        tmp_ang_vel[1] = msg.buttons[5] - msg.buttons[4] # y-axis rotation

            

        ## gripper state 
        self.gripper = msg.buttons[7] * 1.0

        ##
        self.cur_lin_vel = tmp_lin_vel * self.lin_vel_coeff
        self.cur_ang_vel = tmp_ang_vel * self.ang_vel_coeff
    
    def update_velocity(self):
        ## check current joystick state and update linvel_history and angvel_history
        self.linvel_history.append(self.cur_lin_vel)
        self.angvel_history.append(self.cur_ang_vel)

    def command_line_listener(self):
        while True:
            command = input("Enter command: ")
            if command == 'q':
                rospy.signal_shutdown("User requested shutdown")
                exit()

    def publish(self,ros_rate = 70):
        rate = rospy.Rate(ros_rate)
        while not rospy.is_shutdown():
            self.update_velocity()
            lin_vel = np.mean(self.linvel_history,axis=0)
            ang_vel = np.mean(self.angvel_history,axis=0)
            self.twist_controller.move_vel(lin_vel, ang_vel)
            self.twist_controller.gripper.command.rPRA = int(255 * self.gripper)
            self.twist_controller.gripper.publish_command(self.twist_controller.gripper.command)

            rate.sleep()
        self.twist_controller.move_vel(np.zeros(3),np.zeros(3))
    
    def if_initialized(self):
        return (len(self.linvel_history) == self.linvel_history.maxlen) and self.joint_states is not None

if __name__ == "__main__":
    ur5_joy_control = JoyStickControl()
    ur5_joy_control.publish(
        ros_rate=30
    )
    rospy.spin()