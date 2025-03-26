#!/usr/bin/env python

import rospy
from dynamic_reconfigure.client import Client
from std_msgs.msg import Int32  # Or use a custom message type if needed
from threading import Thread
from std_msgs.msg import Header, String
import numpy as np
import rospy
import cv2
from geometry_msgs.msg import Twist, PoseStamped, WrenchStamped
from quest2ros.msg import OVR2ROSHapticFeedback, OVR2ROSInputs
from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped, TransformStamped
from scipy.spatial.transform import Rotation as R
from ur5_comp_control.compliance_control import ComplianceControl
from ur5_comp_control.helper import INIT_POS, INIT_ORI, TARGET_FRAME, SOURCE_FRAME
from collections import deque
from std_msgs.msg import Float32MultiArray
import tf
from ur5_comp_control.helper import CARTESIAN_COMPLIANCE_CONTROLLER, CARTESIAN_FORCE_CONTROLLER, CARTESIAN_MOTION_CONTROLLER


class CompQ3_FeedbackManager:
    """
    Manages different types of feedback (tactile, haptic, force) and provides
    haptic vibration feedback to the controller.
    """
    def __init__(self, handedness="right") -> None:
        self.handedness = handedness
        
        
        self.first_contact = False
        self.collecting_baseline = False
        self.tactile_queue = deque(maxlen=10)
        self.baseline_ct = 0
        self.tactile_baseline = None
        self.haptic_threshold = 50.0  # Threshold for minimum force to trigger haptic feedback
        self.haptic_scale = 2000.0     # Scale factor for haptic amplitude
        self.timestep = 0
        
        # Publishers and subscribers
        self._pub_haptic = rospy.Publisher(f'/q2r_{self.handedness}_hand_haptic_feedback', 
                                          OVR2ROSHapticFeedback, queue_size=1)
        
    
    def send_haptic_feedback(self, amplitude, frequency=5):
        """
        Sends haptic feedback to the controller
        """
        vibrate = OVR2ROSHapticFeedback(frequency=frequency, amplitude=amplitude)
        self._pub_haptic.publish(vibrate)
        
    def provide_frame_difference_feedback(self, target_pose, current_pose):
        """
        Provides haptic feedback based on the difference between target and current frames
        """
        # Calculate position difference
        pos_diff = np.linalg.norm(target_pose.position - current_pose.position)
        ori_diff = np.linalg.norm(target_pose.orientation - current_pose.orientation)
        
        # Only provide feedback if difference exceeds threshold
        if pos_diff > 0.02 or ori_diff > 0.02:  # 2cm threshold
            # Scale amplitude between 0.0 and 1.0
            diff = pos_diff + ori_diff
            amplitude = min(diff / 0.1, 1.0)  # Max amplitude at 10cm difference
            
            # Send haptic feedback
            self.send_haptic_feedback(amplitude)
            
            # Log at reasonable rate
            if self.timestep % 30 == 0:
                rospy.loginfo(f"Frame difference feedback: diff={pos_diff:.3f}m, amplitude={amplitude:.2f}")


class PositionController:
    """
    Controller that publishes target positions to /target_frame
    """
    def __init__(self, ros_rate=60, alpha=1.0, allow_kinematic_control=True) -> None:
        self.alpha = alpha  # Scale coefficient for velocity
        self.dt = 1.0 / ros_rate
        self.allow_kinematic_control = allow_kinematic_control
        
        # Initialize TF listener and broadcaster
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Target frame (4x4 transformation matrix)
        self.target_position, self.target_orientation = None, None
        
        # Buffer for velocity smoothing
        self.lin_vel_history = deque(maxlen=3)
        self.ang_vel_history = deque(maxlen=3)
        
        # Publisher for target frame
        self._pub_target_frame = rospy.Publisher('/target_frame', PoseStamped, queue_size=1)
        self.init_target_frame()
    
    def init_target_frame(self):
        """
        Initialize the target frame
        """
        current_pose = self.get_current_pose()
        if current_pose is not None:
            position = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
            orientation = np.array([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
            self.target_position, self.target_orientation = position, orientation
        
    def get_current_pose(self):
        """
        Get the current pose of the end effector
        """
        try:
            self.tf_listener.waitForTransform(TARGET_FRAME, SOURCE_FRAME, rospy.Time(0), rospy.Duration(1.0))
            trans, rot = self.tf_listener.lookupTransform(TARGET_FRAME, SOURCE_FRAME, rospy.Time(0))
            
            trans, rot = self.tf_listener.lookupTransform(
            TARGET_FRAME, SOURCE_FRAME, rospy.Time(0)
            )
            
            position = Point(x=trans[0], y=trans[1], z=trans[2])
            orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        
            return Pose(position=position, orientation=orientation)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get current pose: {e}")
            return None
    
    def update_target_frame(self, lin_vel, ang_vel, active):
        """
        Update the target frame based on velocity inputs
        """
        # If not active, set target frame to current pose
        if not active:
            self.lin_vel_history.clear()
            self.ang_vel_history.clear()
            if self.allow_kinematic_control:
                current_pose = self.get_current_pose()
                if current_pose is not None:
                    position, orientation = current_pose.position, current_pose.orientation
                # to numpy
                position = np.array([position.x, position.y, position.z])
                orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
                self.target_position, self.target_orientation = position, orientation
            return
        
        # Add velocities to history for smoothing
        self.lin_vel_history.append(lin_vel)
        self.ang_vel_history.append(ang_vel)
        
        # Compute smoothed velocities
        if len(self.lin_vel_history) > 0 and self.target_position is not None and self.target_orientation is not None:
            smoothed_lin_vel = np.mean(self.lin_vel_history, axis=0)
            smoothed_ang_vel = np.mean(self.ang_vel_history, axis=0)
            
            # Scale velocities by alpha
            smoothed_lin_vel = smoothed_lin_vel * self.alpha
            # smoothed_ang_vel = smoothed_ang_vel * self.alpha
            
            # Create incremental transformation
            delta_rot = R.from_rotvec(smoothed_ang_vel * self.dt)
            delta_trans = smoothed_lin_vel * self.dt
            
            # Target orientation is a quaternion, convert to rotation object
            target_rotation = R.from_quat(self.target_orientation)
            # Apply rotation delta
            new_rotation = delta_rot * target_rotation
            # Update target orientation with new quaternion
            self.target_orientation = new_rotation.as_quat()
            # Update target position
            self.target_position = self.target_position + delta_trans
            
    
    def publish_target_frame(self):
        """
        Publish the target frame
        """
        # Extract translation and rotation from target frame
        if self.target_position is None or self.target_orientation is None:
            return
        trans = self.target_position
        rot = self.target_orientation
        # Create PoseStamped message
        pose_stamped = PoseStamped(
            header=Header(stamp=rospy.Time.now(), frame_id='ur_arm_base_link'),
            pose=Pose(position= Point(x=trans[0], y=trans[1], z=trans[2]), orientation=Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3]))
        )
        self._pub_target_frame.publish(pose_stamped)    
        
    def get_frame_difference(self):
        """
        Get the difference between the target and current frames
        """
        target_frame = Pose(position=self.target_position, orientation=self.target_orientation) 
        current_frame = self.get_current_pose()
        return target_frame, current_frame
        
    


class StiffnessManager:
    def __init__(self):
        # Client to communicate with dynamic_reconfigure server
        self.client = Client("my_cartesian_compliance_controller/stiffness")
        
        # Define predefined stiffness configurations
        self.stiffness_presets = {
            1: {  # Soft compliance
                "trans_x": 100.0, 
                "trans_y": 100.0, 
                "trans_z": 100.0,
                "rot_x": 1.0, 
                "rot_y": 1.0, 
                "rot_z": 1.0
            },
            2: {  # Medium compliance
                "trans_x": 200.0, 
                "trans_y": 200.0, 
                "trans_z": 200.0,
                "rot_x": 2.0, 
                "rot_y": 2.0, 
                "rot_z": 2.0
            },
            3: {  # Stiff compliance
                "trans_x": 500.0, 
                "trans_y": 500.0, 
                "trans_z": 500.0,
                "rot_x": 5.0, 
                "rot_y": 5.0, 
                "rot_z": 5.0
            }
        }
        
        self.current_preset = 0
        
        rospy.loginfo("Stiffness Manager is ready!")
    
    def set_stiffness(self, preset_id):
        """
        Set the stiffness configuration
        """
        self.client.update_configuration(self.stiffness_presets[preset_id])
        self.current_preset = preset_id
        rospy.loginfo(f"Switched to stiffness preset {preset_id}")
    


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
    
class CompQ3_Pos:
    def __init__(self, 
                 allow_kinematic_control=True,
                 with_cmd_line=True,
                 use_compliance=True,
                 ros_rate=60, 
                 handedness="right", 
                 with_feedback=False,
                 alpha=1.0) -> None:
        # Initialize the ROS node
        rospy.init_node('comp_q3_pos', anonymous=True)
        
        # Override defaults with ROS parameters if available
        self.ros_rate = rospy.get_param('~ros_rate', ros_rate)
        self.handedness = rospy.get_param('~handedness', handedness)
        self.alpha = rospy.get_param('~alpha', alpha)
        self.allow_kinematic_control = rospy.get_param('~allow_kinematic_control', allow_kinematic_control)
        
        # Create a rate object with the retrieved rate
        self.rate = rospy.Rate(self.ros_rate)
        self.with_feedback = with_feedback
        
        # Log the parameters being used
        rospy.loginfo(f"CompQ3_Pos initialized with rate: {self.ros_rate} Hz, handedness: {self.handedness}, alpha: {self.alpha}")
        
        # Controller state
        self.move_eef = False
        self.move_eef_slow = False
        self.rotate_z = 0.
        self.gripper = 0.0
        self.prev_gripper = 0.0
        self.thumb_stick_horizontal = 0.0
        self.thumb_stick_vertical = 0.0
        self.timestep = 0
        
        # Initialize position controller
        self.position_controller = PositionController(ros_rate=self.ros_rate, alpha=self.alpha, allow_kinematic_control=self.allow_kinematic_control)
        
        # Initialize feedback manager if feedback is enabled
        if self.with_feedback:
            self.feedback_manager = CompQ3_FeedbackManager(handedness=self.handedness)
        else:
            self.feedback_manager = None
        
        # For moving to initial pose only
        # We still need this temporarily just for initial pose setting
        self.compliance_control = ComplianceControl(init_arm=False)
        self.stiffness_client = None 
        # self.set_stiffness_client()
        # Subscribers
        self._sub_vel = rospy.Subscriber(f'/q2r_{self.handedness}_hand_twist', Twist, 
                                        self.vel_callback, queue_size=1)
        self._sub_input = rospy.Subscriber(f'/q2r_{self.handedness}_hand_inputs', OVR2ROSInputs, 
                                          self.input_callback, queue_size=1)
        
        # Start command line listener thread
        if with_cmd_line:
            self.input_thread = Thread(target=self.command_line_listener)
            self.input_thread.start()
        
        if use_compliance:
            self.compliance_control.switch_controller(CARTESIAN_COMPLIANCE_CONTROLLER)
        else:
            self.compliance_control.switch_controller(CARTESIAN_MOTION_CONTROLLER)
        
    def mv_initial_pose(self, initial_p, initial_quat, exec_time=2, change_control_method=True):
        """Move to initial pose using the twist controller (only for initialization)"""
        if type(initial_p) is not Point:
            pos = Point(x=initial_p[0], y=initial_p[1], z=initial_p[2])
        if type(initial_quat) is not Quaternion:
            ori = Quaternion(x=initial_quat[0], y=initial_quat[1], z=initial_quat[2], w=initial_quat[3])
        pose = Pose(position=pos, orientation=ori)
        self.compliance_control.move_to_pose(pose)
        
        # Set the initial position controller target frame
        current_frame = self.position_controller.get_current_pose()
        if current_frame is not None:
            self.position_controller.target_frame = current_frame
        
        print("####### Move Complete #########")
        
    def command_line_listener(self):
        """Listen for command line inputs"""
        while True:
            command = input("Enter command for Quest Control: ")
            if command == 'q':
                rospy.signal_shutdown("User requested shutdown")
                exit()
            elif command == 'g':
                self.mv_initial_pose(INIT_POS, INIT_ORI, exec_time=4, change_control_method=True)

    def vel_callback(self, msg):
        """Handle velocity messages from Quest controller"""
        # Convert velocities to robot frame
        lin_vel = np.array([-msg.linear.y, msg.linear.x, msg.linear.z])
        ang_vel = np.array([-msg.angular.y, msg.angular.x, msg.angular.z])
        
        self.timestep += 1
        
        # Clip angular velocity for stability
        ang_vel = np.clip(ang_vel, -0.8, 0.8)
        
        # Scale velocity if slow mode is active
        if self.move_eef_slow:
            lin_vel = lin_vel * 0.5
            ang_vel = ang_vel * 0.5
        
        # Update the target frame in the position controller
        self.position_controller.update_target_frame(lin_vel, ang_vel, self.move_eef or self.move_eef_slow)
        
    def input_callback(self, msg):
        """
        Handle input messages from Quest controller
        button_upper: slow movement mode
        button_lower: movement mode
        thumb_stick_horizontal: rotation around z
        thumb_stick_vertical: not used currently
        press_index: gripper control
        press_middle: not used currently
        """
        # Gripper control
        if msg.press_index > 0:
            self.gripper = 1.0  # set to 0.5 to avoid damaging the robot
        else:
            self.gripper = 0
            if self.feedback_manager:
                self.feedback_manager.reset_baseline()
        
        # First contact detection (fixed typo in variable name)
        if self.feedback_manager:
            first_contact = (not self.feedback_manager.first_contact) and (self.gripper > 0) and (self.prev_gripper == 0)
            self.feedback_manager.set_first_contact(first_contact)
            self.feedback_manager.update_gripper_state(self.gripper)
        
        self.prev_gripper = self.gripper
        
        # Movement control
        self.move_eef = msg.button_lower
        self.move_eef_slow = msg.button_upper
        self.thumb_stick_horizontal = msg.thumb_stick_horizontal
        self.thumb_stick_vertical = msg.thumb_stick_vertical
        self.rotate_z = msg.thumb_stick_horizontal
        
        # For gripper hardware control
        if hasattr(self, 'compliance_control') and hasattr(self.compliance_control, 'gripper'):
            self.compliance_control.gripper.command.rPRA = int(255 * self.gripper)
            self.compliance_control.gripper.publish_command(self.compliance_control.gripper.command)

    def publish(self):
        """Main control loop"""
        while not rospy.is_shutdown():
            # Publish the target frame
            self.position_controller.publish_target_frame()
            
            # If feedback is enabled, calculate frame difference and provide haptic feedback
            if self.with_feedback and self.feedback_manager:
                target_pose, current_pose = self.position_controller.get_frame_difference()
                if target_pose is not None and current_pose is not None:
                    self.feedback_manager.provide_frame_difference_feedback(target_pose, current_pose)
            
            # Sleep to maintain loop rate
            self.rate.sleep()

    def setup_stiffness_client(self):
        self.stiffness_client = StiffnessManager()
        

if __name__ == '__main__':
    controller = CompQ3_Pos(
        allow_kinematic_control=False,
        with_cmd_line=False, use_compliance=True,
        ros_rate=60, handedness="right", with_feedback=False, alpha=1.0
    )
    print('Position Controller is instantiated')
    rospy.sleep(2)
    # controller.mv_initial_pose(INIT_POS, INIT_
    # QUAT, exec_time=4, change_control_method=True)
    controller.publish()
    rospy.spin()
    
    # # test stiffness client
    # rospy.init_node('stiffness_test')
    # stiffness_client = StiffnessManager()
    # stiffness_client.set_stiffness(1)
    # rospy.spin()