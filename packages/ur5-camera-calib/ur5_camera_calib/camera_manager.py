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
import multiprocessing
import time
import signal
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


def camera_publisher_process(camera_config, namespace="camera", fps=30):
    """
    Process function for publishing camera frames.
    
    Args:
        camera_config (dict): Configuration for the camera process
        namespace (str): Base namespace for camera topics
        fps (int): Frames per second for publishing
    """
    # Extract configuration
    serial = camera_config['serial']
    width = camera_config.get('width', 640)
    height = camera_config.get('height', 480)
    camera_fps = camera_config.get('fps', 30)
    camera_info_data = camera_config.get('camera_info', None)
    
    # Set process name for easier identification
    process_name = f"camera_pub_{serial}"
    try:
        # In Python 3.x we can set the process name
        multiprocessing.current_process().name = process_name
    except:
        pass
    
    # Initialize ROS node within the process
    unique_node_name = f"camera_publisher_{serial}_{os.getpid()}"
    rospy.init_node(unique_node_name, anonymous=True)
    
    # Initialize camera
    try:
        camera = RealSenseCamera(
            serial_number=serial,
            width=width,
            height=height,
            fps=camera_fps
        )
        
        # Start camera
        camera.start()
        rospy.loginfo(f"Started camera {serial} in dedicated process")
        
        # Set up publishers
        bridge = CvBridge()
        camera_ns = f"{namespace}/{serial}"
        color_pub = rospy.Publisher(f"{camera_ns}/color/image_raw", Image, queue_size=10)
        depth_pub = rospy.Publisher(f"{camera_ns}/depth/image_raw", Image, queue_size=10)
        info_pub = rospy.Publisher(f"{camera_ns}/color/camera_info", CameraInfo, queue_size=10)
        
        # Create camera info message if available
        camera_info_msg = None
        if camera_info_data:
            camera_info_msg = _create_camera_info_msg(serial, camera_info_data)
        
        # Publishing loop
        rate = rospy.Rate(fps)
        
        # Handle process termination gracefully
        def handle_sigterm(signum, frame):
            rospy.loginfo(f"Camera {serial} process received terminate signal")
            camera.stop()
            rospy.signal_shutdown("Process terminated")
        
        signal.signal(signal.SIGTERM, handle_sigterm)
        
        last_diagnostics_time = time.time()
        frame_count = 0
        
        while not rospy.is_shutdown():
            try:
                # Get frames
                color_img, depth_img = camera.get_frames()
                
                # Create timestamp (all messages use same timestamp for synchronization)
                stamp = rospy.Time.now()
                
                # Convert and publish color image
                color_msg = bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
                color_msg.header.stamp = stamp
                color_msg.header.frame_id = f"camera_{serial}_color_optical_frame"
                color_pub.publish(color_msg)
                
                # Convert and publish depth image
                depth_msg = bridge.cv2_to_imgmsg(depth_img, encoding="passthrough")
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = f"camera_{serial}_depth_optical_frame"
                depth_pub.publish(depth_msg)
                
                # Publish camera info if available
                if camera_info_msg:
                    camera_info_msg.header.stamp = stamp
                    info_pub.publish(camera_info_msg)
                
                frame_count += 1
                current_time = time.time()
                
                # Log diagnostics every 10 seconds
                if current_time - last_diagnostics_time > 10.0:
                    actual_fps = frame_count / (current_time - last_diagnostics_time)
                    rospy.loginfo(f"Camera {serial} publishing at {actual_fps:.2f} FPS")
                    last_diagnostics_time = current_time
                    frame_count = 0
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in camera {serial} process: {e}")
                # Slow down in case of persistent errors
                rospy.sleep(1.0)
    
    except Exception as e:
        rospy.logerr(f"Failed to initialize camera {serial} in process: {e}")
    
    finally:
        if 'camera' in locals():
            try:
                camera.stop()
                rospy.loginfo(f"Stopped camera {serial} in process")
            except:
                pass


def _create_camera_info_msg(serial, camera_info_data):
    """Create a CameraInfo message from calibration data."""
    msg = CameraInfo()
    msg.header.frame_id = f"camera_{serial}_color_optical_frame"
    
    # Set resolution
    if "resolution" in camera_info_data:
        msg.width = camera_info_data["resolution"].get("width", 640)
        msg.height = camera_info_data["resolution"].get("height", 480)
    else:
        msg.width = 640
        msg.height = 480
    
    # Set camera matrix (K)
    if "camera_matrix" in camera_info_data:
        K = camera_info_data["camera_matrix"]
        msg.K = [K[0, 0], 0, K[0, 2], 0, K[1, 1], K[1, 2], 0, 0, 1]
    
    # Set distortion coefficients
    if "distortion_coefficients" in camera_info_data:
        D = camera_info_data["distortion_coefficients"]
        if D.size == 5:  # Most common case
            msg.D = D.tolist()
            msg.distortion_model = "plumb_bob"
        else:
            msg.D = D.tolist()
            msg.distortion_model = "rational_polynomial"
    
    # Set remaining matrices with identity if not available
    msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # Rectification matrix (identity)
    
    # Set projection matrix (P) - use K with a 0 in the last column
    if "camera_matrix" in camera_info_data:
        K = camera_info_data["camera_matrix"]
        msg.P = [K[0, 0], 0, K[0, 2], 0, 0, K[1, 1], K[1, 2], 0, 0, 0, 1, 0]
    
    return msg


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
        self.processes = {}
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
    
    def start_cameras(self, serials=None, use_multiprocessing=True, publishing_fps=30):
        """
        Start cameras and initialize ROS publishers.
        
        Args:
            serials (list): List of camera serials to start. If None, starts all initialized cameras.
            use_multiprocessing (bool): Whether to use multiprocessing for publishing frames
            publishing_fps (int): Frames per second for publishing
        
        Returns:
            list: List of started camera serials
        """
        if serials is None:
            serials = list(self.cameras.keys())
        
        started = []
        
        # If not using multiprocessing, use the traditional approach
        if not use_multiprocessing:
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
        
        # Using multiprocessing approach
        for serial in serials:
            if serial in self.cameras:
                try:
                    # Stop the camera in the main process if it was started
                    if hasattr(self.cameras[serial], 'intrinsics') and self.cameras[serial].intrinsics is not None:
                        self.cameras[serial].stop()
                    
                    # Create configuration for the camera process
                    camera_config = {
                        'serial': serial,
                        'width': self.cameras[serial].width,
                        'height': self.cameras[serial].height,
                        'fps': self.cameras[serial].fps
                    }
                    
                    # Add camera info if available
                    if serial in self.camera_info:
                        camera_config['camera_info'] = self.camera_info[serial]
                    
                    # Start a dedicated process for this camera
                    process = multiprocessing.Process(
                        target=camera_publisher_process,
                        args=(camera_config, self.namespace, publishing_fps),
                        name=f"camera_pub_{serial}"
                    )
                    process.daemon = True
                    process.start()
                    
                    # Store the process
                    self.processes[serial] = process
                    
                    started.append(serial)
                    rospy.loginfo(f"Started camera {serial} in a dedicated process (PID: {process.pid})")
                
                except Exception as e:
                    rospy.logerr(f"Failed to start camera {serial} in a dedicated process: {e}")
            else:
                rospy.logwarn(f"Camera {serial} not initialized")
        
        return started
    
    def stop_cameras(self, serials=None):
        """
        Stop cameras.
        
        Args:
            serials (list): List of camera serials to stop. If None, stops all cameras.
        """
        if serials is None:
            serials = list(set(list(self.cameras.keys()) + list(self.processes.keys())))
        
        for serial in serials:
            # Stop processes if running in multiprocessing mode
            if serial in self.processes and self.processes[serial].is_alive():
                try:
                    rospy.loginfo(f"Terminating camera process for {serial}")
                    self.processes[serial].terminate()
                    self.processes[serial].join(timeout=3.0)
                    if self.processes[serial].is_alive():
                        rospy.logwarn(f"Camera process for {serial} did not terminate gracefully, forcing kill")
                        # On Unix, we can send SIGKILL
                        try:
                            os.kill(self.processes[serial].pid, signal.SIGKILL)
                        except:
                            pass
                    del self.processes[serial]
                except Exception as e:
                    rospy.logerr(f"Error stopping camera process for {serial}: {e}")
            
            # Stop camera in main process if it exists and is started
            if serial in self.cameras:
                try:
                    if hasattr(self.cameras[serial], 'intrinsics') and self.cameras[serial].intrinsics is not None:
                        self.cameras[serial].stop()
                        rospy.loginfo(f"Stopped camera {serial} in main process")
                except Exception as e:
                    rospy.logerr(f"Error stopping camera {serial} in main process: {e}")
    
    def publish_frames(self, serials=None):
        """
        Publish frames from cameras to ROS topics.
        Only used when not using multiprocessing.
        
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
    
    def are_processes_alive(self):
        """
        Check if camera processes are alive.
        
        Returns:
            bool: True if any camera process is alive
        """
        return any(p.is_alive() for p in self.processes.values())
    
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
        
        return _create_camera_info_msg(serial, self.camera_info[serial])


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