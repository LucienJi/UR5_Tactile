#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import List
import tf
import torch
import actionlib
import geometry_msgs.msg as geometry_msgs
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import rospy
from cartesian_control_msgs.msg import (
    CartesianTrajectoryPoint, 
)
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import (
    ListControllers, 
    ListControllersRequest, 
    LoadController, 
    LoadControllerRequest, 
    SwitchController, 
    SwitchControllerRequest
)
from geometry_msgs.msg import Twist, WrenchStamped, Pose, PoseStamped, Wrench
from std_msgs.msg import Header
from ur5_comp_control.helper import INIT_POS, INIT_ORI, TARGET_FRAME, SOURCE_FRAME
# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

# Joint names for the UR5
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Controllers available on the UR5 system

CARTESIAN_MOTION_CONTROLLERS = [
    "my_cartesian_motion_controller",
]

CARTESIAN_COMPLIANCE_CONTROLLERS = [
    "my_cartesian_compliance_controller",
]

CARTESIAN_FORCE_CONTROLLERS = [
    "my_cartesian_force_controller"
]


# Controllers that would conflict with each other
CONFLICTING_CONTROLLERS = [
    "my_cartesian_compliance_controller",
    "my_cartesian_force_controller",
    "my_cartesian_motion_controller"
]


def ROS_INFO(msg):
    """Utility function to print ROS info messages"""
    rospy.loginfo(msg)


def ROS_WARN(msg):
    """Utility function to print ROS warning messages"""
    rospy.logwarn(msg)


class ControllerManagerClient:
    """Client for interacting with the controller manager"""

    def __init__(self):
        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "/controller_manager/switch_controller", SwitchController
        )
        self.load_srv = rospy.ServiceProxy("/controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
            self.load_srv.wait_for_service(timeout.to_sec())
            self.list_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller manager: {}".format(err))
            sys.exit(-1)

        self.active_controllers = self.list_controllers()
        ROS_INFO("ControllerManagerClient initialized")
        ROS_INFO(f"Active controllers: {self.active_controllers}")

    def list_controllers(self):
        """Get a list of active controllers"""
        req = ListControllersRequest()
        response = self.list_srv.call(req)
        active_controllers = []
        for controller in response.controller:
            if controller.state == "running":
                active_controllers.append(controller.name)
        return active_controllers

    def switch_controller(self, target_controller):
        """Switch to the specified controller"""
        ROS_INFO(f"Switching to controller: {target_controller}")
        
        other_controllers = CARTESIAN_MOTION_CONTROLLERS + CARTESIAN_FORCE_CONTROLLERS + CARTESIAN_COMPLIANCE_CONTROLLERS
        other_controllers.remove(target_controller)
        
        # Check if the controller is loaded
        req = ListControllersRequest()
        response = self.list_srv.call(req)
        controller_loaded = False
        for controller in response.controller:
            if controller.name == target_controller:
                controller_loaded = True
                ROS_INFO(f"Controller {target_controller} is loaded")
                if controller.state == "running":
                    ROS_INFO(f"Controller {target_controller} is already running")
                    return True
            else:
                ROS_INFO(f"Controller {controller.name} is not loaded")
        # Load the controller if needed
        if not controller_loaded:
            ROS_INFO(f"Loading controller {target_controller}")
            load_req = LoadControllerRequest()
            load_req.name = target_controller
            self.load_srv.call(load_req)
        
        # Find controllers to stop (conflicting controllers)
        controllers_to_stop = []
        active_controllers = self.list_controllers()
        ROS_INFO(f"Active controllers: {active_controllers}")
        
        for controller in active_controllers:
            if controller in CONFLICTING_CONTROLLERS and controller != target_controller:
                controllers_to_stop.append(controller)
        
        # Switch controllers
        switch_req = SwitchControllerRequest()
        switch_req.start_controllers = [target_controller]
        switch_req.stop_controllers = controllers_to_stop
        switch_req.strictness = SwitchControllerRequest.BEST_EFFORT
        switch_req.start_asap = True
        switch_req.timeout = 5.0
        
        result = self.switch_srv.call(switch_req)
        
        if result.ok:
            self.active_controllers = self.list_controllers()
            ROS_INFO(f"Successfully switched to {target_controller}")
            return True
        else:
            ROS_WARN(f"Failed to switch to {target_controller}")
            return False


class MotionClient:
    """Client for sending Cartesian motion commands"""
    _controller = 'my_cartesian_motion_controller'
    
    def __init__(self, controller_manager: ControllerManagerClient):
        self.controller_manager = controller_manager
    
        
        # Also create a publisher for direct pose commands
        self._pub_pose = rospy.Publisher(
            f"/target_frame", 
            PoseStamped, 
            queue_size=1
        )
        
        ROS_INFO("MotionClient initialized")
    
    def move_to_pose(self, pose: Pose, frame_id="ur_arm_base_link", exec_time=2.0):
        """Move to the specified pose directly"""
        # Create and publish the pose stamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose
        self._pub_pose.publish(pose_stamped)
        return True

    
    def switch_controller(self, target_controller=_controller):
        """Switch to the motion controller"""
        return self.controller_manager.switch_controller(target_controller)


class ForceClient:
    """Client for sending force control commands"""
    _controller = 'my_cartesian_force_controller'
    
    def __init__(self, controller_manager: ControllerManagerClient):
        self.controller_manager = controller_manager
        
        # Publisher for force control commands
        self._pub_force = rospy.Publisher(
            f"/target_wrench",
            WrenchStamped,
            queue_size=1
        )
        
        # Parameter settings
        self.max_force = 20.0  # Newtons
        self.max_torque = 5.0  # Nm
        
        ROS_INFO("ForceClient initialized")
    
    def apply_force(self, force: np.ndarray, torque: np.ndarray, frame_id="ur_arm_base_link"):
        """Apply the specified force and torque"""
        
        # Limit forces and torques for safety
        force_norm = np.linalg.norm(force)
        if force_norm > self.max_force:
            force = force * (self.max_force / force_norm)
        
        torque_norm = np.linalg.norm(torque)
        if torque_norm > self.max_torque:
            torque = torque * (self.max_torque / torque_norm)
        
        # Create and publish the wrench message
        wrench_msg = WrenchStamped()
        wrench_msg.header.frame_id = frame_id
        wrench_msg.header.stamp = rospy.Time.now()
        
        wrench_msg.wrench.force.x = force[0]
        wrench_msg.wrench.force.y = force[1]
        wrench_msg.wrench.force.z = force[2]
        
        wrench_msg.wrench.torque.x = torque[0]
        wrench_msg.wrench.torque.y = torque[1]
        wrench_msg.wrench.torque.z = torque[2]
        
        self._pub_force.publish(wrench_msg)
        return True
    
    def switch_controller(self, target_controller=_controller):
        """Switch to the force controller"""
        return self.controller_manager.switch_controller(target_controller)


class ComplianceClient:
    """Client for controlling the arm with compliance control"""
    _controller = 'my_cartesian_compliance_controller'
    
    def __init__(self, controller_manager: ControllerManagerClient):
        self.controller_manager = controller_manager
        
        # Publisher for compliance controller commands
        self._pub_compliance_pose = rospy.Publisher(
            f"/target_frame", 
            PoseStamped, 
            queue_size=1
        )
        
        # Publisher for compliance parameters
        self._pub_cart_stiffness = rospy.Publisher(
            f"/cartesian_stiffness", 
            geometry_msgs.Vector3, 
            queue_size=1
        )
        
        self._pub_rot_stiffness = rospy.Publisher(
            f"/rotational_stiffness", 
            geometry_msgs.Vector3, 
            queue_size=1
        )
        
        ROS_INFO("ComplianceClient initialized")
        
        
    def move_to_pose(self, pose, frame_id="ur_arm_base_link"):
        """Move to the specified pose using compliance control"""
        # Create the pose stamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = pose
        
        # Publish the pose
        self._pub_compliance_pose.publish(pose_stamped)
        return True
        
    def switch_controller(self, target_controller=_controller):
        """Switch to the compliance controller"""
        return self.controller_manager.switch_controller(target_controller) 



import torch
from collections import deque
def load_net(path,input_dim):
    model = torch.jit.load(path)
    model.eval()
    return model
 
class WrenchClient:
    """Client for sending wrench commands"""
    _controller = 'my_cartesian_force_controller'
    
    def __init__(self, path='para_net.pt', input_dim=4, force_threshold=5.0, torque_threshold=0.5):
        # self.wrench_net = load_net(path,input_dim)
        self._sub_wrench = rospy.Subscriber(
            f"/robotiq_force_torque_wrench",
            WrenchStamped,
            self.wrench_callback
        )
        self._pub_cur_pose = rospy.Publisher(
            f"/current_pose",
            PoseStamped,
            queue_size=1
        )
        
        self.tf_listener = tf.TransformListener()
        
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.force_deque = deque(maxlen=10)
        self.torque_deque = deque(maxlen=10)
    
    def wrench_callback(self, wrench_msg: WrenchStamped):
        """Callback for the wrench message"""
        target_frame = TARGET_FRAME
        source_frame = SOURCE_FRAME
        try:
            self.tf_listener.waitForTransform(
                target_frame, source_frame, rospy.Time(), rospy.Duration(1.0)
            )
            
            trans, rot = self.tf_listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            
            position = Point(x=trans[0], y=trans[1], z=trans[2])
            orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
            cur_pose = Pose(position=position, orientation=orientation)
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = target_frame
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose = cur_pose
            self._pub_cur_pose.publish(pose_stamped)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
            ROS_WARN(f"Error getting transform: {e}")
            
if __name__ == "__main__":
    rospy.init_node("ur5_comp_control")
    wrench_client = WrenchClient()
    rospy.spin()
    