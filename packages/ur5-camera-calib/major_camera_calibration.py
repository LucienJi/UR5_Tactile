#!/usr/bin/env python3

"""
Script for calibrating the fixed 'major' camera with respect to the robot base.
Uses hand-eye calibration with robot poses and AprilTag detections.
"""

import argparse
import cv2
import numpy as np
import os
import time
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ur5_camera_calib.camera_utils import RealSenseCamera, get_available_cameras
from ur5_camera_calib.apriltag_detector import AprilTagDetector
from ur5_camera_calib.hand_eye_calibration import (
    hand_eye_calibration, 
    create_pose_matrix,
    rotation_matrix_to_quaternion
)
from ur5_camera_calib.config import (
    HAND_EYE_PARAMS, 
    APRILTAG_PARAMS, 
    save_calibration, 
    load_calibration, 
    ensure_dir_exists, 
    INTRINSIC_FILE
)


class HandEyeCalibrationCollector:
    """Collects robot and tag poses for hand-eye calibration."""
    
    def __init__(self, camera, tag_detector):
        """
        Initialize the collector.
        
        Args:
            camera: RealSenseCamera instance
            tag_detector: AprilTagDetector instance
        """
        self.camera = camera
        self.tag_detector = tag_detector
        
        # Load camera intrinsics
        intrinsics_data = load_calibration(INTRINSIC_FILE)
        if intrinsics_data is None:
            # If no calibration file, get from camera directly
            self.camera_matrix = self.camera.get_intrinsics_matrix()
            self.dist_coeffs = self.camera.get_distortion_coeffs()
        else:
            self.camera_matrix = intrinsics_data.get("camera_matrix")
            self.dist_coeffs = intrinsics_data.get("distortion_coefficients")
        
        # Initialize ROS node
        rospy.init_node('hand_eye_calibration', anonymous=True)
        
        # Subscribe to robot pose topic
        self.robot_pose = None
        self.robot_pose_sub = rospy.Subscriber(
            '/ur5/tool_pose', 
            Pose, 
            self.robot_pose_callback
        )
        
        # For publishing visualization
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/hand_eye_calibration/image', Image, queue_size=10)
        
        # Data storage
        self.tag_poses = []  # Tag poses in camera frame
        self.robot_poses = []  # Robot poses in base frame
        
        # Minimum rotation between poses (in degrees)
        self.min_rotation = HAND_EYE_PARAMS["min_rotation"]
        self.last_rotation = None
    
    def robot_pose_callback(self, msg):
        """Callback for robot pose messages."""
        self.robot_pose = msg
    
    def get_robot_pose_matrix(self):
        """
        Get the current robot pose as a 4x4 homogeneous transformation matrix.
        
        Returns:
            numpy.ndarray: 4x4 pose matrix
        """
        if self.robot_pose is None:
            rospy.logwarn("No robot pose received yet.")
            return None
        
        # Extract position and orientation
        pos = self.robot_pose.position
        quat = self.robot_pose.orientation
        
        # Convert quaternion to rotation matrix
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        
        # Create rotation matrix from quaternion
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
        
        # Create translation vector
        t = np.array([pos.x, pos.y, pos.z])
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def check_sufficient_rotation(self, new_pose):
        """
        Check if the new pose has sufficient rotation from the last pose.
        
        Args:
            new_pose (numpy.ndarray): 4x4 pose matrix
            
        Returns:
            bool: True if rotation is sufficient
        """
        if self.last_rotation is None or len(self.robot_poses) == 0:
            return True
        
        # Extract rotation matrices
        R_new = new_pose[:3, :3]
        R_last = self.last_rotation
        
        # Compute the relative rotation matrix
        R_rel = R_new @ R_last.T
        
        # Convert to axis-angle representation
        angle = np.arccos((np.trace(R_rel) - 1) / 2)
        angle_deg = np.degrees(angle)
        
        return angle_deg >= self.min_rotation
    
    def collect_poses(self, num_poses=HAND_EYE_PARAMS["num_poses"]):
        """
        Collect robot and tag poses for hand-eye calibration.
        
        Args:
            num_poses (int): Number of poses to collect
            
        Returns:
            tuple: (tag_poses, robot_poses)
        """
        # Clear previous data
        self.tag_poses = []
        self.robot_poses = []
        
        print(f"Collecting {num_poses} poses for hand-eye calibration.")
        print("Position the robot arm such that the AprilTag is visible to the camera.")
        print("Press 'c' to capture a pose, 'q' to finish early, or 'r' to reset collection.")
        
        self.camera.start()
        
        try:
            while len(self.tag_poses) < num_poses:
                # Get current frame
                color_img, _ = self.camera.get_frames()
                
                # Detect AprilTags
                detections = self.tag_detector.detect(color_img)
                
                # Draw detections
                vis_img = self.tag_detector.draw_detections(
                    color_img, 
                    detections, 
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                
                # Add information to the image
                cv2.putText(
                    vis_img, 
                    f"Poses: {len(self.tag_poses)}/{num_poses}",
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Show instructions
                cv2.putText(
                    vis_img, 
                    "c: capture, q: finish, r: reset",
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 
                    (255, 0, 0), 
                    2
                )
                
                # Publish visualization
                msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
                self.image_pub.publish(msg)
                
                # Show image
                cv2.imshow("Hand-Eye Calibration", vis_img)
                key = cv2.waitKey(1) & 0xFF
                
                # Capture on 'c' key press
                if key == ord('c'):
                    # Check if we have a valid tag detection
                    if not detections:
                        print("No AprilTag detected. Try again.")
                        continue
                    
                    # Use the first detected tag
                    detection = detections[0]
                    
                    # Get the current robot pose
                    robot_pose = self.get_robot_pose_matrix()
                    if robot_pose is None:
                        print("Robot pose not available. Try again.")
                        continue
                    
                    # Check for sufficient rotation
                    if not self.check_sufficient_rotation(robot_pose):
                        print(f"Not enough rotation from last pose (needs {self.min_rotation} degrees). Try a different pose.")
                        continue
                    
                    # Get the tag pose in camera frame
                    rot_mat, tvec = self.tag_detector.estimate_pose(
                        detection, 
                        self.camera_matrix, 
                        self.dist_coeffs
                    )
                    tag_pose = create_pose_matrix(rot_mat, tvec)
                    
                    # Add to dataset
                    self.tag_poses.append(tag_pose)
                    self.robot_poses.append(robot_pose)
                    self.last_rotation = robot_pose[:3, :3]
                    
                    print(f"Pose {len(self.tag_poses)}/{num_poses} captured.")
                
                # Reset on 'r' key press
                elif key == ord('r'):
                    self.tag_poses = []
                    self.robot_poses = []
                    self.last_rotation = None
                    print("Pose collection reset.")
                
                # Quit on 'q' key press
                elif key == ord('q'):
                    break
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
        
        return self.tag_poses, self.robot_poses


def save_major_camera_calibration(camera_serial, T_cam_base, output_dir):
    """
    Save major camera calibration data to a file.
    
    Args:
        camera_serial (str): Camera serial number
        T_cam_base (numpy.ndarray): Transformation from camera to base
        output_dir (str): Output directory
    """
    ensure_dir_exists(output_dir)
    
    # Extract rotation and translation
    R = T_cam_base[:3, :3]
    t = T_cam_base[:3, 3]
    
    # Convert rotation to quaternion
    q = rotation_matrix_to_quaternion(R)
    
    # Prepare data
    data = {
        "camera_serial": camera_serial,
        "is_major_camera": True,
        "transformation_matrix": T_cam_base,
        "rotation_matrix": R,
        "translation_vector": t,
        "quaternion": q,
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create filename
    filename = f"extrinsics_major_{camera_serial}.yaml"
    file_path = os.path.join(output_dir, filename)
    
    # Save to file
    save_calibration(data, file_path)
    print(f"Major camera calibration saved to: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Major Camera Calibration")
    parser.add_argument("--serial", help="Camera serial number")
    parser.add_argument("--num-poses", type=int, default=HAND_EYE_PARAMS["num_poses"], 
                        help="Number of poses to collect")
    parser.add_argument("--tag-size", type=float, default=APRILTAG_PARAMS["size"], 
                        help="AprilTag size in meters")
    parser.add_argument("--tag-id", type=int, default=0, 
                        help="AprilTag ID to use for calibration")
    parser.add_argument("--output-dir", default="~/.ur5_camera_calib/extrinsics", 
                        help="Output directory")
    args = parser.parse_args()
    
    # Get available cameras
    available_cameras = get_available_cameras()
    
    if not available_cameras:
        print("No RealSense cameras found.")
        return
    
    # Select camera
    if args.serial:
        camera_serial = args.serial
        if camera_serial not in available_cameras:
            print(f"Camera with serial {camera_serial} not found.")
            print(f"Available cameras: {available_cameras}")
            return
    else:
        print("Available cameras:")
        for i, serial in enumerate(available_cameras):
            print(f"  {i+1}. {serial}")
        
        try:
            selection = int(input("Select camera (number): ")) - 1
            camera_serial = available_cameras[selection]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
    
    print(f"Using camera: {camera_serial}")
    
    # Initialize the camera
    camera = RealSenseCamera(serial_number=camera_serial)
    
    # Initialize the tag detector
    tag_detector = AprilTagDetector(tag_size=args.tag_size)
    
    # Initialize the calibration collector
    collector = HandEyeCalibrationCollector(camera, tag_detector)
    
    # Collect poses
    tag_poses, robot_poses = collector.collect_poses(num_poses=args.num_poses)
    
    if len(tag_poses) < 5:
        print("Not enough poses collected for accurate calibration. Need at least 5.")
        return
    
    # Perform hand-eye calibration
    print("Performing hand-eye calibration...")
    T_cam_base = hand_eye_calibration(tag_poses, robot_poses)
    
    print("Calibration result:")
    print("Transformation from camera to robot base:")
    print(T_cam_base)
    
    # Save calibration results
    output_dir = os.path.expanduser(args.output_dir)
    save_major_camera_calibration(camera_serial, T_cam_base, output_dir)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 