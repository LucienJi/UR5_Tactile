#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from scipy.spatial.transform import Rotation as R
import moveit_commander
import rospy

# Common frames
SOURCE_FRAME = '/UR_1_s_grip_point'
TARGET_FRAME = '/ur_arm_base_link'

# Sane initial configuration for UR5
INIT_JOINT_POS = [1.3123908042907715, -1.709191624318258, 1.5454894304275513, -1.1726744810687464, -1.5739596525775355, -0.7679112593280237]
# Initial pose (these are in ur_arm_tool0_controller frame, not hand_finger_tip_link frame)

INIT_POS = Point(x=0.22266099167526043, y=0.43185254983352417, z=0.4369834249045895)
INIT_ORI = Quaternion(x=0.6634025009378515, y=0.041104734026518375, z=0.1581501543418476, w=-0.7302027466886599)


# Default compliance parameters
DEFAULT_CARTESIAN_STIFFNESS = [800.0, 800.0, 800.0]  # N/m in x, y, z
DEFAULT_CARTESIAN_DAMPING = [20.0, 20.0, 20.0]       # Ns/m in x, y, z
DEFAULT_ROT_STIFFNESS = [50.0, 50.0, 50.0]           # Nm/rad in roll, pitch, yaw
DEFAULT_ROT_DAMPING = [5.0, 5.0, 5.0]                # Nms/rad in roll, pitch, yaw
DEFAULT_NULL_STIFFNESS = [10.0] * 6                  # Nm/rad for each joint
DEFAULT_NULL_DAMPING = [0.5] * 6                     # Nms/rad for each joint

# Controller names
CARTESIAN_MOTION_CONTROLLER = "my_cartesian_motion_controller"
CARTESIAN_FORCE_CONTROLLER = "my_cartesian_force_controller"
CARTESIAN_COMPLIANCE_CONTROLLER = "my_cartesian_compliance_controller"

def get_arm():
    """
    Returns the move_group for the robot arm.
    """
    robot = moveit_commander.RobotCommander()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    return move_group

def get_planning_scene():
    """
    Returns the planning scene interface.
    """
    planning_scene = moveit_commander.PlanningSceneInterface()
    return planning_scene

def point2numpy(point: Point) -> np.ndarray:
    """
    Convert ROS Point to numpy array
    """
    return np.array([point.x, point.y, point.z])

def numpy2point(np_pos: np.ndarray) -> Point:
    """
    Convert numpy array to ROS Point
    """
    return Point(x=np_pos[0], y=np_pos[1], z=np_pos[2])

def ori2numpy(quat: Quaternion) -> np.ndarray:
    """
    Convert ROS Quaternion to numpy array
    """
    return np.array([quat.x, quat.y, quat.z, quat.w])

def numpy2quat(np_quat: np.ndarray) -> Quaternion:
    """
    Convert numpy array to ROS Quaternion
    """
    return Quaternion(x=np_quat[0], y=np_quat[1], z=np_quat[2], w=np_quat[3])

def pose_from_translation_quaternion(translation, quaternion):
    """
    Create a pose from translation and quaternion
    """
    pose = Pose()
    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    return pose

def pose_to_transformation_matrix(pose):
    """
    Convert a ROS pose to a 4x4 transformation matrix
    """
    pos = point2numpy(pose.position)
    quat = ori2numpy(pose.orientation)
    rotation = R.from_quat(quat).as_matrix()
    
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pos
    
    return matrix

def transformation_matrix_to_pose(matrix):
    """
    Convert a 4x4 transformation matrix to a ROS pose
    """
    pos = matrix[:3, 3]
    rotation = matrix[:3, :3]
    quat = R.from_matrix(rotation).as_quat()
    
    pose = Pose()
    pose.position = numpy2point(pos)
    pose.orientation = numpy2quat(quat)
    
    return pose 