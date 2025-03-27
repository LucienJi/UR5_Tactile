#!/usr/bin/env python3

"""
AprilTag detection and pose estimation for camera calibration.
"""

import cv2
import numpy as np
import apriltag
from .config import APRILTAG_PARAMS


class AprilTagDetector:
    """Class for detecting AprilTags and estimating their 3D poses."""
    
    def __init__(self, tag_family=APRILTAG_PARAMS["family"], tag_size=APRILTAG_PARAMS["tag_size"]):
        """
        Initialize the AprilTag detector.
        
        Args:
            tag_family (str): AprilTag family to detect.
            tag_size (float): Size of the tag in meters.
        """
        self.tag_family = tag_family
        self.tag_size = tag_size
        
        # Initialize detector
        self.detector = apriltag.Detector(families=tag_family)
    
    def detect(self, image):
        """
        Detect AprilTags in an image.
        
        Args:
            image (numpy.ndarray): Input image (grayscale or color).
            
        Returns:
            list: List of detection results from the detector.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect AprilTags
        results = self.detector.detect(gray)
        return results
    
    def estimate_pose(self, detection, camera_matrix, dist_coeffs=None):
        """
        Estimate the 3D pose of a detected AprilTag.
        
        Args:
            detection: AprilTag detection result.
            camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix.
            dist_coeffs (numpy.ndarray, optional): Distortion coefficients.
            
        Returns:
            tuple: (rotation_matrix, translation_vector)
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)
        
        # Define 3D points of the tag in object coordinate frame
        half_size = self.tag_size / 2
        object_points = np.array([
            [-half_size,  half_size, 0.0],  # Top-left
            [ half_size,  half_size, 0.0],  # Top-right
            [ half_size, -half_size, 0.0],  # Bottom-right
            [-half_size, -half_size, 0.0]   # Bottom-left
        ])
        
        # Get 2D points from detection corners
        image_points = np.array(detection.corners)
        
        # Estimate pose
        ret, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs
        )
        
        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        return rot_mat, tvec
    
    def draw_detections(self, image, detections, camera_matrix=None, dist_coeffs=None):
        """
        Draw detected AprilTags on the image.
        
        Args:
            image (numpy.ndarray): Input image.
            detections: List of AprilTag detections.
            camera_matrix (numpy.ndarray, optional): Camera matrix for pose drawing.
            dist_coeffs (numpy.ndarray, optional): Distortion coefficients.
            
        Returns:
            numpy.ndarray: Image with drawn detections.
        """
        output = image.copy()
        
        for detection in detections:
            # Draw bounding box
            for i in range(4):
                j = (i + 1) % 4
                pt1 = tuple(detection.corners[i].astype(int))
                pt2 = tuple(detection.corners[j].astype(int))
                cv2.line(output, pt1, pt2, (0, 255, 0), 2)
            
            # Draw ID
            center = tuple(detection.center.astype(int))
            cv2.putText(
                output, 
                str(detection.tag_id), 
                center, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                2
            )
            
            # Draw pose axis if camera matrix is provided
            if camera_matrix is not None:
                rot_mat, tvec = self.estimate_pose(detection, camera_matrix, dist_coeffs)
                axis_length = self.tag_size / 2
                
                # Define the 3D points for the axes
                axis_pts = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],  # X-axis (red)
                    [0, axis_length, 0],  # Y-axis (green)
                    [0, 0, axis_length]   # Z-axis (blue)
                ])
                
                # Project 3D points to 2D
                rvec, _ = cv2.Rodrigues(rot_mat)
                imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, camera_matrix, dist_coeffs if dist_coeffs is not None else np.zeros(5))
                
                # Draw the axes
                origin = tuple(imgpts[0].ravel().astype(int))
                cv2.line(output, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 2)  # X-axis (red)
                cv2.line(output, origin, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(output, origin, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 2)  # Z-axis (blue)
        
        return output 