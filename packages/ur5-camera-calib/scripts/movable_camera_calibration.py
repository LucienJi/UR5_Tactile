#!/usr/bin/env python3

"""
Script for calibrating movable cameras with respect to the robot base using a common AprilTag.
Requires a pre-calibrated 'major' camera.
"""

import argparse
import cv2
import numpy as np
import os
import time
import yaml

from ur5_camera_calib.camera_utils import RealSenseCamera, get_available_cameras
from ur5_camera_calib.apriltag_detector import AprilTagDetector
from ur5_camera_calib.hand_eye_calibration import (
    create_pose_matrix,
    compute_tag_pose_in_base,
    compute_camera_pose_in_base,
    rotation_matrix_to_quaternion
)
from ur5_camera_calib.config import (
    APRILTAG_PARAMS,
    save_calibration,
    load_calibration,
    ensure_dir_exists,
    CALIBRATION_DIR,
    EXTRINSIC_FILE
)


def get_major_camera_transformation():
    """
    Find the major camera's transformation to base frame.
    
    Returns:
        tuple: (camera_serial, transformation_matrix) or (None, None) if not found
    """
    # Check if the extrinsics file exists
    if not os.path.exists(EXTRINSIC_FILE):
        return None, None
    
    # Load the file
    data = load_calibration(EXTRINSIC_FILE)
    if data is None:
        return None, None
    
    # Look for major camera
    if data.get("is_major_camera"):
        return data.get("camera_serial"), data.get("transformation_matrix")
    
    # Alternative: Check the calibration directory for major camera files
    extrinsics_dir = os.path.join(CALIBRATION_DIR, "extrinsics")
    if os.path.exists(extrinsics_dir):
        for filename in os.listdir(extrinsics_dir):
            if filename.startswith("extrinsics_major_") and filename.endswith(".yaml"):
                file_path = os.path.join(extrinsics_dir, filename)
                data = load_calibration(file_path)
                if data and data.get("is_major_camera"):
                    return data.get("camera_serial"), data.get("transformation_matrix")
    
    return None, None


def calibrate_movable_cameras(major_camera, other_cameras, detector):
    """
    Calibrate movable cameras with respect to the robot base.
    
    Args:
        major_camera: Tuple of (RealSenseCamera, transformation_to_base)
        other_cameras: List of RealSenseCamera instances
        detector: AprilTagDetector instance
        
    Returns:
        dict: Dictionary mapping camera serials to their transformations
    """
    major_cam, T_major_to_base = major_camera
    
    # Start all cameras
    major_cam.start()
    for cam in other_cameras:
        cam.start()
    
    camera_transformations = {}
    
    try:
        # Instruct the user to place a calibration tag
        print("Place the AprilTag in view of ALL cameras.")
        print("Press 'c' to capture and calibrate, 'q' to quit.")
        
        while True:
            # Get frames from major camera
            major_color, _ = major_cam.get_frames()
            
            # Detect AprilTag in major camera
            major_detections = detector.detect(major_color)
            
            # Visualize
            major_vis = detector.draw_detections(
                major_color, 
                major_detections, 
                major_cam.get_intrinsics_matrix(), 
                major_cam.get_distortion_coeffs()
            )
            
            cv2.putText(
                major_vis, 
                "Major Camera",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("Major Camera", major_vis)
            
            # Get frames and detect AprilTags for all other cameras
            other_vis_images = []
            other_detections_list = []
            
            for i, cam in enumerate(other_cameras):
                other_color, _ = cam.get_frames()
                other_detections = detector.detect(other_color)
                other_detections_list.append(other_detections)
                
                other_vis = detector.draw_detections(
                    other_color, 
                    other_detections, 
                    cam.get_intrinsics_matrix(), 
                    cam.get_distortion_coeffs()
                )
                
                cv2.putText(
                    other_vis, 
                    f"Camera {i+1}: {cam.serial_number}",
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
                other_vis_images.append(other_vis)
                cv2.imshow(f"Camera {i+1}", other_vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture and calibrate on 'c' key press
            if key == ord('c'):
                # Check if we have valid tag detections in all cameras
                if not major_detections:
                    print("No AprilTag detected in major camera. Try again.")
                    continue
                
                valid_detections = True
                for i, detections in enumerate(other_detections_list):
                    if not detections:
                        print(f"No AprilTag detected in camera {i+1}. Try again.")
                        valid_detections = False
                        break
                
                if not valid_detections:
                    continue
                
                # Use the first detected tag for each camera
                major_detection = major_detections[0]
                
                # Get the tag pose in major camera frame
                major_rot, major_tvec = detector.estimate_pose(
                    major_detection, 
                    major_cam.get_intrinsics_matrix(), 
                    major_cam.get_distortion_coeffs()
                )
                tag_pose_major = create_pose_matrix(major_rot, major_tvec)
                
                # Compute the tag pose in base frame
                tag_pose_base = compute_tag_pose_in_base(tag_pose_major, T_major_to_base)
                
                # For each other camera, compute its pose in base frame
                for i, cam in enumerate(other_cameras):
                    detection = other_detections_list[i][0]  # First detection
                    
                    # Get the tag pose in this camera's frame
                    cam_rot, cam_tvec = detector.estimate_pose(
                        detection, 
                        cam.get_intrinsics_matrix(),
                        cam.get_distortion_coeffs()
                    )
                    tag_pose_cam = create_pose_matrix(cam_rot, cam_tvec)
                    
                    # Compute the camera pose in base frame
                    cam_pose_base = compute_camera_pose_in_base(tag_pose_cam, tag_pose_base)
                    
                    # Save the transformation
                    camera_transformations[cam.serial_number] = cam_pose_base
                    
                    print(f"Camera {i+1} ({cam.serial_number}) calibrated.")
                    print("Transformation to base frame:")
                    print(cam_pose_base)
                
                break  # Exit the calibration loop
            
            # Quit on 'q' key press
            elif key == ord('q'):
                break
    
    finally:
        # Stop all cameras
        major_cam.stop()
        for cam in other_cameras:
            cam.stop()
        
        cv2.destroyAllWindows()
    
    return camera_transformations


def save_movable_camera_calibration(camera_serials, transformations, output_dir):
    """
    Save movable camera calibration data to files.
    
    Args:
        camera_serials (list): List of camera serial numbers
        transformations (dict): Dictionary mapping camera serials to transformations
        output_dir (str): Output directory
    """
    ensure_dir_exists(output_dir)
    
    # Save each camera's calibration
    for serial in camera_serials:
        if serial not in transformations:
            continue
        
        T = transformations[serial]
        
        # Extract rotation and translation
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Convert rotation to quaternion
        q = rotation_matrix_to_quaternion(R)
        
        # Prepare data
        data = {
            "camera_serial": serial,
            "is_major_camera": False,
            "transformation_matrix": T,
            "rotation_matrix": R,
            "translation_vector": t,
            "quaternion": q,
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create filename
        filename = f"extrinsics_movable_{serial}.yaml"
        file_path = os.path.join(output_dir, filename)
        
        # Save to file
        save_calibration(data, file_path)
        print(f"Movable camera calibration saved to: {file_path}")
    
    # Also save a combined file with all cameras
    combined_data = {
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {}
    }
    
    for serial in camera_serials:
        if serial not in transformations:
            continue
        
        T = transformations[serial]
        combined_data["cameras"][serial] = {
            "transformation_matrix": T,
            "is_major_camera": False
        }
    
    # Add major camera info if available
    major_serial, T_major = get_major_camera_transformation()
    if major_serial is not None and T_major is not None:
        combined_data["cameras"][major_serial] = {
            "transformation_matrix": T_major,
            "is_major_camera": True
        }
    
    # Save combined file
    combined_file = os.path.join(output_dir, "all_cameras_extrinsics.yaml")
    save_calibration(combined_data, combined_file)
    print(f"Combined camera calibration saved to: {combined_file}")


def main():
    parser = argparse.ArgumentParser(description="Movable Camera Calibration")
    parser.add_argument("--major-serial", help="Major camera serial number (if not using saved)")
    parser.add_argument("--tag-size", type=float, default=APRILTAG_PARAMS["size"], 
                        help="AprilTag size in meters")
    parser.add_argument("--output-dir", default="~/.ur5_camera_calib/extrinsics", 
                        help="Output directory")
    args = parser.parse_args()
    
    # Get available cameras
    available_cameras = get_available_cameras()
    
    if not available_cameras:
        print("No RealSense cameras found.")
        return
    
    # Get major camera transformation
    major_serial, T_major_to_base = get_major_camera_transformation()
    
    # If not found in saved calibration and not provided, ask the user
    if (major_serial is None or T_major_to_base is None) and args.major_serial is None:
        print("Major camera not found in calibration files.")
        print("Available cameras:")
        for i, serial in enumerate(available_cameras):
            print(f"  {i+1}. {serial}")
        
        try:
            selection = int(input("Select major camera (number): ")) - 1
            major_serial = available_cameras[selection]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
        
        print(f"Using camera {major_serial} as major camera.")
        print("This camera must be already calibrated with respect to the robot base.")
        print("If not, please run major_camera_calibration.py first.")
        
        # Check if we have calibration for this camera
        extrinsics_dir = os.path.join(CALIBRATION_DIR, "extrinsics")
        if os.path.exists(extrinsics_dir):
            major_file = os.path.join(extrinsics_dir, f"extrinsics_major_{major_serial}.yaml")
            if os.path.exists(major_file):
                data = load_calibration(major_file)
                if data:
                    T_major_to_base = data.get("transformation_matrix")
    elif args.major_serial:
        major_serial = args.major_serial
        # Look for calibration
        extrinsics_dir = os.path.join(CALIBRATION_DIR, "extrinsics")
        if os.path.exists(extrinsics_dir):
            major_file = os.path.join(extrinsics_dir, f"extrinsics_major_{major_serial}.yaml")
            if os.path.exists(major_file):
                data = load_calibration(major_file)
                if data:
                    T_major_to_base = data.get("transformation_matrix")
    
    if T_major_to_base is None:
        print("Major camera transformation not found.")
        print("Please calibrate the major camera first.")
        return
    
    # Initialize the major camera
    major_camera = RealSenseCamera(serial_number=major_serial)
    
    # Select the other cameras
    other_serials = [s for s in available_cameras if s != major_serial]
    
    if not other_serials:
        print("No other cameras available.")
        return
    
    print("Available cameras for movable calibration:")
    for i, serial in enumerate(other_serials):
        print(f"  {i+1}. {serial}")
    
    # Allow user to select which cameras to calibrate
    selected_indices = input("Enter camera numbers to calibrate (comma-separated, or 'all'): ")
    
    if selected_indices.lower() == 'all':
        selected_cameras = other_serials
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selected_indices.split(',')]
            selected_cameras = [other_serials[i] for i in indices if 0 <= i < len(other_serials)]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
    
    if not selected_cameras:
        print("No cameras selected.")
        return
    
    print(f"Calibrating cameras: {', '.join(selected_cameras)}")
    
    # Initialize the selected cameras
    other_camera_objects = [RealSenseCamera(serial_number=s) for s in selected_cameras]
    
    # Initialize the tag detector
    tag_detector = AprilTagDetector(tag_size=args.tag_size)
    
    # Perform calibration
    transformations = calibrate_movable_cameras(
        (major_camera, T_major_to_base),
        other_camera_objects,
        tag_detector
    )
    
    if transformations:
        # Save calibration results
        output_dir = os.path.expanduser(args.output_dir)
        save_movable_camera_calibration(selected_cameras, transformations, output_dir)


if __name__ == "__main__":
    main() 