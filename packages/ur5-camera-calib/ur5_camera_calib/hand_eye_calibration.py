#!/usr/bin/env python3

"""
Hand-eye calibration methods for the fixed camera.
"""

import numpy as np
import cv2
import math


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        numpy.ndarray: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    q = np.zeros(4)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    
    return q


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (w, x, y, z) to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q
    
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*z*w
    R[0, 2] = 2*x*z + 2*y*w
    
    R[1, 0] = 2*x*y + 2*z*w
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*x*w
    
    R[2, 0] = 2*x*z - 2*y*w
    R[2, 1] = 2*y*z + 2*x*w
    R[2, 2] = 1 - 2*x*x - 2*y*y
    
    return R


def create_pose_matrix(rotation, translation):
    """
    Create a 4x4 pose matrix from rotation matrix and translation vector.
    
    Args:
        rotation (numpy.ndarray): 3x3 rotation matrix
        translation (numpy.ndarray): 3x1 translation vector
        
    Returns:
        numpy.ndarray: 4x4 pose matrix
    """
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation.ravel()
    return pose


def inverse_pose(pose):
    """
    Compute the inverse of a pose matrix.
    
    Args:
        pose (numpy.ndarray): 4x4 pose matrix
        
    Returns:
        numpy.ndarray: 4x4 inverse pose matrix
    """
    inv_pose = np.eye(4)
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    inv_pose[:3, :3] = R.T
    inv_pose[:3, 3] = -R.T @ t
    
    return inv_pose


def hand_eye_calibration(tag_poses, robot_poses, method="tsai"):
    """
    Perform hand-eye calibration.
    
    Args:
        tag_poses (list): List of AprilTag poses in camera frame (4x4 matrices)
        robot_poses (list): List of robot poses in base frame (4x4 matrices)
        method (str): Calibration method ('tsai' or 'park')
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix from camera to robot base
    """
    # Convert poses to rotation vectors and translations
    tag_rvecs = []
    tag_tvecs = []
    robot_rvecs = []
    robot_tvecs = []
    
    for i in range(len(tag_poses)):
        # Extract rotations and translations
        R_tag = tag_poses[i][:3, :3]
        t_tag = tag_poses[i][:3, 3]
        R_robot = robot_poses[i][:3, :3]
        t_robot = robot_poses[i][:3, 3]
        
        # Convert to rotation vectors
        rvec_tag, _ = cv2.Rodrigues(R_tag)
        rvec_robot, _ = cv2.Rodrigues(R_robot)
        
        tag_rvecs.append(rvec_tag)
        tag_tvecs.append(t_tag.reshape(3, 1))
        robot_rvecs.append(rvec_robot)
        robot_tvecs.append(t_robot.reshape(3, 1))
    
    # Use OpenCV's calibrateHandEye function
    if method.lower() == 'tsai':
        R_cam_base, t_cam_base = cv2.calibrateHandEye(
            robot_rvecs, robot_tvecs,
            tag_rvecs, tag_tvecs,
            cv2.CALIB_HAND_EYE_TSAI
        )
    else:  # park
        R_cam_base, t_cam_base = cv2.calibrateHandEye(
            robot_rvecs, robot_tvecs,
            tag_rvecs, tag_tvecs,
            cv2.CALIB_HAND_EYE_PARK
        )
    
    # Create transformation matrix
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3] = R_cam_base
    T_cam_base[:3, 3] = t_cam_base.ravel()
    
    return T_cam_base


def compute_tag_pose_in_base(tag_pose_camera, camera_to_base):
    """
    Compute the pose of the AprilTag in the robot base frame.
    
    Args:
        tag_pose_camera (numpy.ndarray): 4x4 pose matrix of tag in camera frame
        camera_to_base (numpy.ndarray): 4x4 transformation from camera to base
        
    Returns:
        numpy.ndarray: 4x4 pose matrix of tag in base frame
    """
    return camera_to_base @ tag_pose_camera


def compute_camera_pose_in_base(tag_pose_camera, tag_pose_base):
    """
    Compute the pose of a camera in the robot base frame.
    
    Args:
        tag_pose_camera (numpy.ndarray): 4x4 pose matrix of tag in camera frame
        tag_pose_base (numpy.ndarray): 4x4 pose matrix of tag in base frame
        
    Returns:
        numpy.ndarray: 4x4 transformation from camera to base
    """
    # Tag in camera frame is the inverse of camera in tag frame
    camera_in_tag = inverse_pose(tag_pose_camera)
    
    # Camera in base frame: tag_pose_base * camera_in_tag
    camera_in_base = tag_pose_base @ camera_in_tag
    
    return camera_in_base 