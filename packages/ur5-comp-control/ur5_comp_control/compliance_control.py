#!/usr/bin/env python3

import signal
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from robotiq_s_interface import Gripper
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ur5_comp_control.srv import MoveToPose, MoveToPoseResponse

from ur5_comp_control.control_client import (
    ComplianceClient, 
    ControllerManagerClient, 
    MotionClient,
    ForceClient,
    WrenchClient
)
from ur5_comp_control.helper import (
    SOURCE_FRAME, 
    TARGET_FRAME,
    INIT_POS,
    INIT_ORI,
    CARTESIAN_MOTION_CONTROLLER,
    CARTESIAN_COMPLIANCE_CONTROLLER,
    CARTESIAN_FORCE_CONTROLLER
)

# Service definitions
from ur5_comp_control.srv import (
    GoHomePose, 
    GoHomePoseResponse,
    SwitchController,
    SwitchControllerResponse,
    GripperControl,
    GripperControlResponse
)


class ComplianceControl:
    """
    Implements Forward Dynamic Compliance Control for the UR5 robot
    with gripper control capabilities.
    """
    def __init__(self, init_arm=True) -> None:
        # ROS parameters
        self.input_topic = rospy.get_param("~joy_topic", '/external_control')
        self.gripper_mode = rospy.get_param("~gripper_mode", "pinch")
        self.gripper_force = rospy.get_param("~gripper_force", 20)
        
        # TF listener for coordinate transformations
        self._tf_listener = tf.TransformListener()
        
        self.gripper = None 
        # Initialize gripper
        print('Initializing Gripper...')
        self.gripper = Gripper(gripper_force=self.gripper_force)
        self._initialize_gripper()
        print('Initializing Gripper...done')
        
        # Controller clients
        print('Instantiating ControllerManagerClient...')
        self.controller_manager_client = ControllerManagerClient()
        
        print('Instantiating controllers...')
        self._motion_client = MotionClient(self.controller_manager_client)
        self._force_client = ForceClient(self.controller_manager_client)
        self._compliance_client = ComplianceClient(self.controller_manager_client)
        # self._wrench_client = WrenchClient()
        
        # Subscribe to joint states
        self.joint_state = None
        self._sub_jpos = rospy.Subscriber('/joint_states', JointState, self._jnt_callback, queue_size=1)
        
        # Active controller state
        self.active_controller = None
        
        # Initialize arm if requested
        if init_arm:
            self._init_arm(spawn_controller=True)
        
        # Set up SIGINT handler for clean shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        
        # Set up services
        print('Setting up services...')
        self._gohomepose_srv = rospy.Service('go_homepose', GoHomePose, self._handle_gohomepose)
        self._switchcontroller_srv = rospy.Service('switch_controller', SwitchController, self._handle_switch_controller)
        self._move_to_pose_srv = rospy.Service('move_to_pose_srv', MoveToPose, self._handle_move_to_pose_srv)
        
        print('ComplianceControl initialized')
        
    # def _move_to_pose_srv(self, req):
    
    def _handle_gohomepose(self, req):
        """Handler for the go_homepose service"""
        print('Resetting the arm...')
        success = self._init_arm(spawn_controller=True)
        return GoHomePoseResponse(success=success)
    
    def _handle_move_to_pose_srv(self, req):
        """Handler for the move_to_pose_srv service"""
        success = self.move_to_pose(req.target_pose)
        c_pose = self._get_curr_pose()
        pose_diff = np.linalg.norm(np.array([c_pose.position.x, c_pose.position.y, c_pose.position.z]) - np.array([req.target_pose.position.x, req.target_pose.position.y, req.target_pose.position.z]))

        target_quat = np.array([req.target_pose.orientation.x, req.target_pose.orientation.y, req.target_pose.orientation.z, req.target_pose.orientation.w])
        current_quat = np.array([c_pose.orientation.x, c_pose.orientation.y, c_pose.orientation.z, c_pose.orientation.w])

        orientation_threshold = 0.1 # Adjust this threshold as needed (in radians for angle difference)

        while not rospy.is_shutdown():
            c_pose = self._get_curr_pose()
            pose_diff = np.linalg.norm(np.array([c_pose.position.x, c_pose.position.y, c_pose.position.z]) - np.array([req.target_pose.position.x, req.target_pose.position.y, req.target_pose.position.z]))
            current_quat = np.array([c_pose.orientation.x, c_pose.orientation.y, c_pose.orientation.z, c_pose.orientation.w])

            # Calculate orientation difference using quaternion dot product or angle between them
            r_current = R.from_quat(current_quat)
            r_target = R.from_quat(target_quat)
            r_diff = r_current.inv() * r_target  # Get the relative rotation
            angle_diff = np.linalg.norm(r_diff.as_rotvec()) # Magnitude of the rotation vector is the angle

            if pose_diff < 0.1 and angle_diff < orientation_threshold:
                break
            rospy.sleep(0.1)

        c_pose = self._get_curr_pose() # Get the final pose after the loop
        if success:
            return MoveToPoseResponse(success=True, message=f'Moved to pose: {c_pose}')
        else:
            return MoveToPoseResponse(success=False, message=f'Failed to move to pose. Current pose: {c_pose}')

    
    def _handle_switch_controller(self, req):
        """Handler for switching between controllers"""
        
        if req.controller_name not in [CARTESIAN_MOTION_CONTROLLER, CARTESIAN_COMPLIANCE_CONTROLLER, CARTESIAN_FORCE_CONTROLLER]:
            msg = f"Unknown controller: {req.controller_name}"
            return SwitchControllerResponse(success=False, message=msg)
        
        success = False
        if req.controller_name == CARTESIAN_MOTION_CONTROLLER:
            success = self._motion_client.switch_controller()
            self.active_controller = CARTESIAN_MOTION_CONTROLLER
        elif req.controller_name == CARTESIAN_COMPLIANCE_CONTROLLER:
            success = self._compliance_client.switch_controller()
            self.active_controller = CARTESIAN_COMPLIANCE_CONTROLLER
        elif req.controller_name == CARTESIAN_FORCE_CONTROLLER:
            success = self._force_client.switch_controller()
            self.active_controller = CARTESIAN_FORCE_CONTROLLER
        
        if success:
            msg = f"Successfully switched to {req.controller_name}"
        else:
            msg = f"Failed to switch to {req.controller_name}"
        
        return SwitchControllerResponse(success=success, message=msg)

    
    def _jnt_callback(self, msg):
        """Callback for joint state messages"""
        self.joint_state = {'pos': msg.position, 'vel': msg.velocity, 'effort': msg.effort}
    
   
    def _get_curr_pose(self) -> Pose:
        """Get the current pose of the end effector"""
        timeout = 1.0
        self._tf_listener.waitForTransform(
            TARGET_FRAME, SOURCE_FRAME, rospy.Time(), rospy.Duration(timeout)
        )
        
        trans, rot = self._tf_listener.lookupTransform(
            TARGET_FRAME, SOURCE_FRAME, rospy.Time(0)
        )
        
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        
        return Pose(position=position, orientation=orientation)
    
    def _get_curr_twist(self):
        """Get the current twist (linear and angular velocity) of the end effector"""
        tooltip_frame = SOURCE_FRAME
        base_frame = TARGET_FRAME
        
        twist = self._tf_listener.lookupTwistFull(
            tracking_frame=tooltip_frame,
            observation_frame=base_frame,
            reference_frame=tooltip_frame,
            ref_point=(0, 0, 0),
            reference_point_frame=tooltip_frame,
            time=rospy.Time(0),
            averaging_interval=rospy.Duration(nsecs=int(50 * 1e6))  # 50 ms
        )
        
        return twist
    
    def _init_arm(self, spawn_controller=True) -> bool:
        """Initialize the arm by moving to the initial position"""
        # Use Cartesian motion controller
        try:
            self._motion_client.switch_controller()
            self.active_controller = CARTESIAN_MOTION_CONTROLLER
            # Send the arm to the initial pose
            time.sleep(2.0)
            init_pose = Pose(position=INIT_POS, orientation=INIT_ORI)
            success = self._motion_client.move_to_pose(init_pose, exec_time=3.0)
            # Wait for the arm to reach the position
            time.sleep(3.0)
            print("Arm initialized to home position")
            
            if spawn_controller:
                # Switch to the compliance controller if requested
                self._compliance_client.switch_controller()
                self.active_controller = CARTESIAN_COMPLIANCE_CONTROLLER
            
            return success
        except Exception as e:
            print(f"Error initializing arm: {e}")
            return False
            
    
    def _initialize_gripper(self):
        """Initialize the gripper with the specified mode"""
        print(f"Initializing gripper in {self.gripper_mode} mode")
        self.gripper.activate()
        self.gripper.set_mode(1)
        # self.gripper.set_force(self.gripper_force)
        self.gripper.open()
        self.gripper.close()
        
    
    def move_to_pose(self, pose: Pose):
        """Move to a target pose using motion control"""
        if not self.active_controller == CARTESIAN_MOTION_CONTROLLER:
            self._motion_client.switch_controller()
            self.active_controller = CARTESIAN_MOTION_CONTROLLER
        return self._motion_client.move_to_pose(pose)
    
    def apply_force(self, force: np.ndarray, torque: np.ndarray):
        """Apply a specific force and torque using force control"""
        # Switch to force controller if not already active
        if not self.active_controller == CARTESIAN_FORCE_CONTROLLER:
            self._force_client.switch_controller()
            self.active_controller = CARTESIAN_FORCE_CONTROLLER
        return self._force_client.apply_force(force, torque)
    
    def move_compliant(self, target_pose: Pose):
        """
        Move to a target pose with compliance control.
        Uses previously set compliance parameters.
        """
        # Switch to compliance controller if not already active
        if not self.active_controller == CARTESIAN_COMPLIANCE_CONTROLLER:
            self._compliance_client.switch_controller()
            self.active_controller = CARTESIAN_COMPLIANCE_CONTROLLER
        return self._compliance_client.move_to_pose(target_pose)
    def switch_controller(self, controller_name: str):
        self.controller_manager_client.switch_controller(controller_name)
    def _shutdown(self, *args):
        """Clean shutdown when SIGINT is received"""
        print("Shutting down...")
        
        # Stop any motion
        try:
            # Switch to motion controller
            self._motion_client.switch_controller()
            
            # Close gripper gently
            self.gripper.close()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        rospy.signal_shutdown("ComplianceControl node was interrupted") 

if __name__ == "__main__":
    rospy.init_node('compliance_control')
    rate = rospy.Rate(30)
    compliance_control = ComplianceControl(init_arm=False)

    compliance_control.switch_controller(CARTESIAN_COMPLIANCE_CONTROLLER)
    rospy.spin()
        
    
    # Test the compliance control