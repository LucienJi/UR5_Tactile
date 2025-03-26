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
from quest2ros.msg import OVR2ROSHapticFeedback, OVR2ROSInputs
from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
from scipy.spatial.transform import Rotation as R
from ur5_twist_control.helper import ori2numpy, point2numpy
from ur5_twist_control.twist_control import TwistControl
from collections import deque
from std_msgs.msg import Float32MultiArray


# INIT_POS = np.array([0.296,-0.521,0.101])
INIT_POS = np.array([-0.0139, -0.4602,0.2653])
# INIT_QUAT = np.array([-0.389, 0.625,-0.116,0.667])
INIT_QUAT = np.array([0.9434, -0.3310,-0.012, 0.015])

def call_gohomepose_srv():
    from ur5_twist_control.srv import GoHomePose
    print('calling go home pose service!!')
    rospy.wait_for_service('/go_homepose')
    try:
        go_homepose = rospy.ServiceProxy('/go_homepose', GoHomePose)
        result = go_homepose()
        print('result', result)
        return result.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return False
    
class UR5QuestControl:
    def __init__(self, ros_rate=60, handedness="right", with_feedback = True, num_mags=5, num_samples=5) -> None:
        # Initialize the ROS node
        rospy.init_node('ur5_quest_control', anonymous=True)
        
        # Override defaults with ROS parameters if available
        self.ros_rate = rospy.get_param('~ros_rate', ros_rate)
        self.handedness = rospy.get_param('~handedness', handedness)
        
        # Create a rate object with the retrieved rate
        self.rate = rospy.Rate(self.ros_rate)
        self.with_feedback = with_feedback
        # Log the parameters being used
        rospy.loginfo(f"UR5QuestControl initialized with rate: {self.ros_rate} Hz, handedness: {self.handedness}")
        
        self.move_eef = False
        self.move_eef_slow = False
        self.rotate_z = 0.
        self.thumb_stick_horizontal = 0.0
        self.thumb_stick_vertical = 0.0 
        self.gripper = 0.0
        self.perv_gripper = 0.0

        self.lin_history = deque(maxlen=21)
        self.ang_history = deque(maxlen=50)

        self.twist_controller = TwistControl(init_arm=True)
        self.timestep = 0
        self._sub_vel = rospy.Subscriber(f'/q2r_{self.handedness}_hand_twist', Twist, self.vel_callback, queue_size=1)
        # self._sub_pose = rospy.Subscriber(f'/q2r_{self.handedness}_hand_pose', PoseStamped, self.pose_callback, queue_size=1)
        self._sub_input = rospy.Subscriber(f'/q2r_{self.handedness}_hand_inputs', OVR2ROSInputs, self.input_callback, queue_size=1)
        # self._sub_force = rospy.Subscriber(f'/wrench', WrenchStamped, self.force_callback, queue_size=1)
        if self.with_feedback:
            
            self.num_mags = num_mags
            self.num_samples = num_samples
            self.first_contact = False
            self.collecting_baseline = False
            self.tactile_queue = deque(maxlen=10)
            self.baseline_ct = 0
            self.tactile_baseline = None
            self.haptic_threshold = 50.0  # Threshold for minimum force to trigger haptic feedback
            self.haptic_scale = 2000.0     # Scale factor for haptic amplitude
            self._sub_tactile = rospy.Subscriber(f'/anyskin_data', Float32MultiArray, self.tactile_callback, queue_size=1)
        self._pub_haptic = rospy.Publisher(f'/q2r_{self.handedness}_hand_haptic_feedback', OVR2ROSHapticFeedback, queue_size=1)
        self._prev_timestamp = rospy.get_time()

        self.input_thread = Thread(target=self.command_line_listener)
        self.input_thread.start()
        
    def mv_initial_pose(self, initial_p, initial_quat,exec_time=2,change_control_method = True):
        ori = Quaternion(x=initial_quat[0], y=initial_quat[1], z=initial_quat[2], w=initial_quat[3])
        pos = Point(initial_p[0], initial_p[1], initial_p[2])
        self.twist_controller._move_to(pos, ori, exec_time=exec_time, spawn_twistvel_controller=change_control_method)
        print("####### Move Complete #########")


    def command_line_listener(self):
        while True:
            command = input("Enter command for Quest Control: ")
            if command == 'q':
                rospy.signal_shutdown("User requested shutdown")
                exit()
            elif command == 'g':
                self.mv_initial_pose(INIT_POS, INIT_QUAT, exec_time=4, change_control_method=True)

    def tactile_callback(self, msg):
        """
        Processes tactile data to provide haptic feedback.
        - On first contact: begins collecting baseline data
        - Once baseline is established: calculates delta from baseline for haptic feedback
        """
        # Parse the tactile data from the Float32MultiArray message
        tactile_data = np.array(msg.data).reshape(self.num_samples, self.num_mags, 3)  # shape ( N_sample, N_mag, 3)
        tactile_data = np.mean(tactile_data, axis=0) # shape ( N_mag, 3)
        # If first contact detected, start collecting baseline
        if self.first_contact and not self.collecting_baseline:
            rospy.loginfo("First contact detected! Collecting tactile baseline...")
            self.collecting_baseline = True
            self.tactile_queue.clear()  # Clear any previous data
            
        # If we're collecting baseline, add to queue
        if self.collecting_baseline:
            self.baseline_ct += 1
            self.tactile_queue.append(tactile_data)
            
            # When queue is full, calculate baseline
            if self.baseline_ct == 20 * self.tactile_queue.maxlen:
                # Compute mean across all collected readings
                tactile_readings = np.stack(list(self.tactile_queue), axis=0)
                self.tactile_baseline = np.mean(tactile_readings, axis=0) # shape ( N_mag, 3)
                self.collecting_baseline = False
                self.baseline_ct = 0
                rospy.loginfo(f"Tactile baseline established with {len(self.tactile_queue)} samples")
            return  # Don't generate haptic feedback while collecting baseline
        
        # If baseline is established, calculate delta for haptic feedback
        if self.tactile_baseline is not None and self.gripper > 0:
            # Calculate the difference between current reading and baseline
            tactile_delta = tactile_data - self.tactile_baseline
            
            # Calculate the norm of each vector in the delta
            delta_norms = np.linalg.norm(tactile_delta, axis=1)
            
            # Use the maximum norm as the signal strength
            max_delta = np.max(delta_norms)
            
            # Only provide feedback if the delta exceeds threshold
            if max_delta > self.haptic_threshold:
                # Scale the amplitude between 0.0 and 1.0
                amplitude = min(max_delta / self.haptic_scale, 1.0)
                
                # Create and publish the haptic feedback
                vibrate = OVR2ROSHapticFeedback(frequency=5, amplitude=amplitude)
                self._pub_haptic.publish(vibrate)
                
                # Log feedback at a reasonable rate
                if self.timestep % 30 == 0:  # Log approximately twice per second at 60Hz
                    rospy.loginfo(f"Haptic feedback: delta={max_delta:.2f}, amplitude={amplitude:.2f}")

    def vel_callback(self, msg):

        lin_vel = np.array([msg.linear.y, -msg.linear.x, msg.linear.z])
        ang_vel = np.array([msg.angular.y, -msg.angular.x, msg.angular.z])
        
        self.timestep +=1
        
        ang_vel = np.clip(ang_vel, -0.5, 0.5) # angular vel has lower accuracy

        if self.move_eef or self.move_eef_slow:
            self.lin_history.append(lin_vel)
            self.ang_history.append(ang_vel)
        else:
            self.lin_history.append(np.array([0, 0, 0]))
            self.ang_history.append(np.array([0, 0, 0]))

    def input_callback(self, msg):
        """
        button_upper: 
        button_lower: 
        thumb_stick_horizontal: 
        thumb_stick_vertical: 
        press_index: 
        press_middle: 
        """
        if msg.press_index > 0:
            self.gripper = 1.0 ### set to 0.5 to avoid damaging the robot 
            
        else:
            self.gripper = 0
            self.first_contact = False
            self.tactile_baseline = None
            self.collecting_baseline = False
            self.tactile_queue.clear()
        self.first_contact = (~self.first_contact) &( self.gripper > 0) & (self.perv_gripper == 0)
        self.perv_gripper = self.gripper
        self.move_eef = msg.button_lower
        self.move_eef_slow = msg.button_upper
        self.thumb_stick_horizontal = msg.thumb_stick_horizontal    
        self.thumb_stick_vertical = msg.thumb_stick_vertical

        self.rotate_z = msg.thumb_stick_horizontal

    def publish(self):
        
        while not rospy.is_shutdown():
            if len(self.lin_history) == self.lin_history.maxlen:
                if self.move_eef_slow:
                    self.twist_controller.move_vel(np.mean(self.lin_history, axis=0) * 0.2, np.mean(self.ang_history, axis=0) * 0.2)
                elif self.move_eef:
                    self.twist_controller.move_vel(np.mean(self.lin_history, axis=0) * 1.0, np.mean(self.ang_history, axis=0) * 1.0)
                else:
                    self.twist_controller.move_vel(np.zeros(3), np.zeros(3))

                self.twist_controller.gripper.command.rPRA = int(255 * self.gripper)
                self.twist_controller.gripper.publish_command(self.twist_controller.gripper.command)

            self.rate.sleep()

        self.twist_controller.move_vel(np.zeros(3), np.zeros(3))


if __name__ == '__main__':
    ur5_quest_controller = UR5QuestControl(
        ros_rate=60, handedness="right", with_feedback=True
    )
    print('UR5 Quest Control is instantiated')
    ur5_quest_controller.mv_initial_pose(INIT_POS, INIT_QUAT, exec_time=4, change_control_method=True)
    ur5_quest_controller.publish()
    rospy.spin()
