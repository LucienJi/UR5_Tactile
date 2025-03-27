#!/usr/bin/env python3

"""
Camera manager for handling multiple RealSense cameras.
Provides utilities for identification and consistent naming.
"""

import os
import cv2
import numpy as np
import yaml
import rospy
import threading
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs

from .camera_utils import RealSenseCamera, get_available_cameras
from .config import ensure_dir_exists, CALIBRATION_DIR, load_calibration


class CameraIdentifier:
    """
    Tool for identifying which camera is which by displaying a visual indicator.
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.cameras = {}
        self.display_thread = None
        self.running = False
    
    def add_camera(self, serial, camera):
        """Add a camera to the identifier."""
        self.cameras[serial] = camera
    
    def start(self):
        """Start the identification display thread."""
        if self.display_thread is not None and self.display_thread.is_alive():
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
    
    def stop(self):
        """Stop the identification display thread."""
        self.running = False
        if self.display_thread is not None:
            self.display_thread.join(timeout=1.0)
    
    def _display_loop(self):
        """Main display loop for camera identification."""
        window_name = "Camera Identifier"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        serials = list(self.cameras.keys())
        current_index = 0
        
        while self.running:
            # Get current camera
            current_serial = serials[current_index]
            camera = self.cameras[current_serial]
            
            # Get frame
            color_img, _ = camera.get_frames()
            
            # Draw identification info
            display_img = color_img.copy()
            
            # Draw camera info with colored background
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (display_img.shape[1], 60), (0, 100, 200), -1)
            
            # Draw serial number
            cv2.putText(
                overlay,
                f"Camera: {current_serial}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Add index info
            cv2.putText(
                overlay,
                f"Index: {current_index+1}/{len(serials)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Blend overlay and original image
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
            
            # Show image
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(100) & 0xFF
            
            # Handle key presses
            if key == ord('n'):  # Next camera
                current_index = (current_index + 1) % len(serials)
            elif key == ord('p'):  # Previous camera
                current_index = (current_index - 1) % len(serials)
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyWindow(window_name)


class CameraManager:
    """
    Manager for multiple RealSense cameras.
    Handles identification, starting, stopping, and publishing to ROS topics.
    """
    def __init__(self, namespace="camera"):
        """
        Initialize the camera manager.
        
        Args:
            namespace (str): Base namespace for camera topics
        """
        self.namespace = namespace
        self.cameras = {}
        self.publishers = {}
        self.bridge = CvBridge()
        self.camera_info = {}
        self.extrinsics = {}
        self.load_calibration_data()
    
    def load_calibration_data(self):
        """Load camera calibration data if available."""
        # Load intrinsics
        intrinsics_dir = os.path.join(CALIBRATION_DIR, "intrinsics")
        if os.path.exists(intrinsics_dir):
            for filename in os.listdir(intrinsics_dir):
                if filename.startswith("intrinsics_") and filename.endswith(".yaml"):
                    serial = filename.replace("intrinsics_", "").replace(".yaml", "")
                    file_path = os.path.join(intrinsics_dir, filename)
                    data = load_calibration(file_path)
                    if data:
                        self.camera_info[serial] = data
        
        # Load extrinsics
        extrinsics_dir = os.path.join(CALIBRATION_DIR, "extrinsics")
        if os.path.exists(extrinsics_dir):
            # Try loading the combined file first
            combined_file = os.path.join(extrinsics_dir, "all_cameras_extrinsics.yaml")
            if os.path.exists(combined_file):
                data = load_calibration(combined_file)
                if data and "cameras" in data:
                    self.extrinsics = data["cameras"]
            else:
                # Load individual files
                for filename in os.listdir(extrinsics_dir):
                    if (filename.startswith("extrinsics_major_") or 
                        filename.startswith("extrinsics_movable_")) and filename.endswith(".yaml"):
                        serial = filename.split("_")[-1].replace(".yaml", "")
                        file_path = os.path.join(extrinsics_dir, filename)
                        data = load_calibration(file_path)
                        if data:
                            self.extrinsics[serial] = {
                                "transformation_matrix": data.get("transformation_matrix"),
                                "is_major_camera": data.get("is_major_camera", False)
                            }
    
    def discover_cameras(self):
        """Discover available RealSense cameras."""
        return get_available_cameras()
    
    def initialize_cameras(self, serials=None, width=640, height=480, fps=30):
        """
        Initialize cameras by serial numbers.
        
        Args:
            serials (list): List of camera serials to initialize. If None, discovers all cameras.
            width (int): Image width
            height (int): Image height
            fps (int): Frames per second
        
        Returns:
            list: List of initialized camera serials
        """
        if serials is None:
            serials = self.discover_cameras()
        
        initialized = []
        for serial in serials:
            try:
                camera = RealSenseCamera(
                    serial_number=serial,
                    width=width,
                    height=height,
                    fps=fps
                )
                self.cameras[serial] = camera
                initialized.append(serial)
                rospy.loginfo(f"Initialized camera {serial}")
            except Exception as e:
                rospy.logerr(f"Failed to initialize camera {serial}: {e}")
        
        return initialized
    
    def start_cameras(self, serials=None):
        """
        Start cameras and initialize ROS publishers.
        
        Args:
            serials (list): List of camera serials to start. If None, starts all initialized cameras.
        
        Returns:
            list: List of started camera serials
        """
        if serials is None:
            serials = list(self.cameras.keys())
        
        started = []
        for serial in serials:
            if serial not in self.cameras:
                rospy.logwarn(f"Camera {serial} not initialized")
                continue
            
            try:
                # Start the camera
                self.cameras[serial].start()
                
                # Create publishers
                camera_ns = f"{self.namespace}/{serial}"
                self.publishers[serial] = {
                    "color": rospy.Publisher(f"{camera_ns}/color/image_raw", Image, queue_size=10),
                    "depth": rospy.Publisher(f"{camera_ns}/depth/image_raw", Image, queue_size=10),
                    "camera_info": rospy.Publisher(f"{camera_ns}/color/camera_info", CameraInfo, queue_size=10)
                }
                
                # Create camera info message
                if serial in self.camera_info:
                    camera_info_msg = self._create_camera_info_msg(serial)
                    self.publishers[serial]["camera_info_msg"] = camera_info_msg
                
                started.append(serial)
                rospy.loginfo(f"Started camera {serial}")
            except Exception as e:
                rospy.logerr(f"Failed to start camera {serial}: {e}")
        
        return started
    
    def stop_cameras(self, serials=None):
        """
        Stop cameras.
        
        Args:
            serials (list): List of camera serials to stop. If None, stops all cameras.
        """
        if serials is None:
            serials = list(self.cameras.keys())
        
        for serial in serials:
            if serial in self.cameras:
                try:
                    self.cameras[serial].stop()
                    rospy.loginfo(f"Stopped camera {serial}")
                except Exception as e:
                    rospy.logerr(f"Error stopping camera {serial}: {e}")
    
    def publish_frames(self, serials=None):
        """
        Publish frames from cameras to ROS topics.
        
        Args:
            serials (list): List of camera serials to publish. If None, publishes all started cameras.
        """
        if serials is None:
            serials = list(self.publishers.keys())
        
        for serial in serials:
            if serial not in self.cameras or serial not in self.publishers:
                continue
            
            try:
                # Get frames
                color_img, depth_img = self.cameras[serial].get_frames()
                
                # Convert to ROS messages
                color_msg = self.bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
                depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="passthrough")
                
                # Set headers
                stamp = rospy.Time.now()
                color_msg.header.stamp = stamp
                depth_msg.header.stamp = stamp
                color_msg.header.frame_id = f"camera_{serial}_color_optical_frame"
                depth_msg.header.frame_id = f"camera_{serial}_depth_optical_frame"
                
                # Publish
                self.publishers[serial]["color"].publish(color_msg)
                self.publishers[serial]["depth"].publish(depth_msg)
                
                # Publish camera info if available
                if "camera_info_msg" in self.publishers[serial]:
                    camera_info_msg = self.publishers[serial]["camera_info_msg"]
                    camera_info_msg.header.stamp = stamp
                    self.publishers[serial]["camera_info"].publish(camera_info_msg)
            
            except Exception as e:
                rospy.logerr(f"Error publishing frames for camera {serial}: {e}")
    
    def run_identification(self, serials=None):
        """
        Run camera identification tool.
        
        Args:
            serials (list): List of camera serials to identify. If None, identifies all initialized cameras.
        """
        if serials is None:
            serials = list(self.cameras.keys())
        
        # Start cameras if needed
        for serial in serials:
            if serial in self.cameras:
                if not hasattr(self.cameras[serial], 'intrinsics') or self.cameras[serial].intrinsics is None:
                    try:
                        self.cameras[serial].start()
                    except Exception as e:
                        rospy.logerr(f"Error starting camera {serial} for identification: {e}")
                        serials.remove(serial)
        
        if not serials:
            rospy.logerr("No cameras available for identification")
            return
        
        # Create identifier and add cameras
        identifier = CameraIdentifier()
        for serial in serials:
            identifier.add_camera(serial, self.cameras[serial])
        
        # Run identification
        rospy.loginfo("Starting camera identification. Press 'n' for next camera, 'p' for previous, 'q' to quit.")
        identifier.start()
        
        # Wait for user to finish
        try:
            while not rospy.is_shutdown() and identifier.running:
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            identifier.stop()
    
    def _create_camera_info_msg(self, serial):
        """Create a CameraInfo message from calibration data."""
        if serial not in self.camera_info:
            return None
        
        data = self.camera_info[serial]
        
        msg = CameraInfo()
        msg.header.frame_id = f"camera_{serial}_color_optical_frame"
        
        # Set resolution
        if "resolution" in data:
            msg.width = data["resolution"].get("width", 640)
            msg.height = data["resolution"].get("height", 480)
        else:
            msg.width = 640
            msg.height = 480
        
        # Set camera matrix (K)
        if "camera_matrix" in data:
            K = data["camera_matrix"]
            msg.K = [K[0, 0], 0, K[0, 2], 0, K[1, 1], K[1, 2], 0, 0, 1]
        
        # Set distortion coefficients
        if "distortion_coefficients" in data:
            D = data["distortion_coefficients"]
            if D.size == 5:  # Most common case
                msg.D = D.tolist()
                msg.distortion_model = "plumb_bob"
            else:
                msg.D = D.tolist()
                msg.distortion_model = "rational_polynomial"
        
        # Set remaining matrices with identity if not available
        msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # Rectification matrix (identity)
        
        # Set projection matrix (P) - use K with a 0 in the last column
        if "camera_matrix" in data:
            K = data["camera_matrix"]
            msg.P = [K[0, 0], 0, K[0, 2], 0, 0, K[1, 1], K[1, 2], 0, 0, 0, 1, 0]
        
        return msg


def save_camera_layout(cameras, layout_name="camera_layout"):
    """
    Save camera layout information to a file.
    
    Args:
        cameras (dict): Dictionary mapping serial numbers to user-assigned names
        layout_name (str): Name for the layout file
    """
    layout_dir = os.path.join(CALIBRATION_DIR, "layouts")
    ensure_dir_exists(layout_dir)
    
    layout_file = os.path.join(layout_dir, f"{layout_name}.yaml")
    
    with open(layout_file, 'w') as f:
        yaml.dump(cameras, f)
    
    print(f"Camera layout saved to: {layout_file}")


def load_camera_layout(layout_name="camera_layout"):
    """
    Load camera layout information from a file.
    
    Args:
        layout_name (str): Name of the layout file
        
    Returns:
        dict: Dictionary mapping serial numbers to user-assigned names
    """
    layout_file = os.path.join(CALIBRATION_DIR, "layouts", f"{layout_name}.yaml")
    
    if not os.path.exists(layout_file):
        return {}
    
    with open(layout_file, 'r') as f:
        cameras = yaml.safe_load(f)
    
    return cameras 