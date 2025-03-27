#!/usr/bin/env python3

"""
Script for camera intrinsic calibration using checkerboard.
"""

import argparse
import cv2
import numpy as np
import time
import os
import yaml
import pyrealsense2 as rs

from ur5_camera_calib.camera_utils import RealSenseCamera, get_available_cameras
from ur5_camera_calib.config import CHECKERBOARD_PARAMS, save_calibration, ensure_dir_exists


def calibrate_camera(camera, num_images=20, delay=2):
    """
    Perform intrinsic camera calibration using a checkerboard pattern.
    
    Args:
        camera: RealSenseCamera instance
        num_images (int): Number of images to capture for calibration
        delay (int): Delay between captures in seconds
        
    Returns:
        tuple: (camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    # Checkerboard parameters
    board_rows = CHECKERBOARD_PARAMS["rows"]
    board_cols = CHECKERBOARD_PARAMS["cols"]
    square_size = CHECKERBOARD_PARAMS["square_size"]
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Start the camera
    camera.start()
    
    captured_images = 0
    
    print(f"Will capture {num_images} images for calibration.")
    print("Position the checkerboard in different orientations within the camera's view.")
    print("Press 'c' to capture an image, 'q' to quit, or wait for automatic capture.")
    
    last_capture_time = time.time() - delay  # Allow immediate first capture
    
    try:
        while captured_images < num_images:
            # Get frame
            color_img, _ = camera.get_frames()
            
            # Draw info
            display_img = color_img.copy()
            cv2.putText(
                display_img, 
                f"Captured: {captured_images}/{num_images}",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF
            
            current_time = time.time()
            
            # Capture on 'c' key press or auto capture after delay
            if key == ord('c') or (current_time - last_capture_time > delay and delay > 0):
                # Convert to grayscale
                gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(
                    gray, 
                    (board_cols, board_rows), 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Draw and display the corners
                    cv2.drawChessboardCorners(display_img, (board_cols, board_rows), corners2, ret)
                    
                    # Add to dataset
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    # Update counters
                    captured_images += 1
                    last_capture_time = current_time
                    
                    # Display the captured image
                    cv2.putText(
                        display_img, 
                        f"Captured: {captured_images}/{num_images}",
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    cv2.putText(
                        display_img, 
                        "CHECKERBOARD FOUND",
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        (0, 0, 255), 
                        2
                    )
                    cv2.imshow("Calibration", display_img)
                    cv2.waitKey(500)  # Show the capture for a moment
                else:
                    print("Checkerboard not found. Try again.")
            
            # Quit on 'q' key press
            if key == ord('q'):
                break
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    
    if captured_images == 0:
        print("No images captured. Calibration failed.")
        return None, None, None, None
    
    print(f"Calibrating with {len(objpoints)} images...")
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (camera.width, camera.height), None, None
    )
    
    if ret:
        print("Calibration successful!")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"Average reprojection error: {mean_error / len(objpoints)}")
        
        return camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        print("Calibration failed.")
        return None, None, None, None


def save_intrinsics(camera_serial, camera_matrix, dist_coeffs, output_dir):
    """
    Save intrinsic calibration data to a file.
    
    Args:
        camera_serial (str): Camera serial number
        camera_matrix (numpy.ndarray): Camera matrix
        dist_coeffs (numpy.ndarray): Distortion coefficients
        output_dir (str): Output directory
    """
    ensure_dir_exists(output_dir)
    
    # Prepare data
    data = {
        "camera_serial": camera_serial,
        "camera_matrix": camera_matrix,
        "distortion_coefficients": dist_coeffs,
        "resolution": {"width": 640, "height": 480},
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create filename
    filename = f"intrinsics_{camera_serial}.yaml"
    file_path = os.path.join(output_dir, filename)
    
    # Save to file
    save_calibration(data, file_path)
    print(f"Intrinsic calibration saved to: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Camera Intrinsic Calibration")
    parser.add_argument("--serial", help="Camera serial number")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to capture")
    parser.add_argument("--delay", type=int, default=2, help="Delay between captures (0 for manual)")
    parser.add_argument("--output-dir", default="~/.ur5_camera_calib/intrinsics", help="Output directory")
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
    camera = RealSenseCamera(
        serial_number=camera_serial,
        width=args.width,
        height=args.height,
    )
    
    # Perform calibration
    camera_matrix, dist_coeffs, _, _ = calibrate_camera(
        camera, 
        num_images=args.num_images,
        delay=args.delay
    )
    
    if camera_matrix is not None:
        # Save calibration results
        output_dir = os.path.expanduser(args.output_dir)
        save_intrinsics(camera_serial, camera_matrix, dist_coeffs, output_dir)


if __name__ == "__main__":
    main() 