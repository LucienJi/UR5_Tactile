#!/usr/bin/env python3

"""
Utility functions for camera operations.
"""

import cv2
import numpy as np
import pyrealsense2 as rs


def get_available_cameras():
    """
    Get a list of available RealSense cameras.
    
    Returns:
        list: List of camera serial numbers
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    camera_serials = []
    for dev in devices:
        camera_serials.append(dev.get_info(rs.camera_info.serial_number))
    
    return camera_serials


class RealSenseCamera:
    """Class for interfacing with RealSense cameras."""
    
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        """
        Initialize a RealSense camera.
        
        Args:
            serial_number (str, optional): Camera serial number. If None, uses first available camera.
            width (int): Image width.
            height (int): Image height.
            fps (int): Frames per second.
        """
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial_number:
            self.config.enable_device(serial_number)
        
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.align = rs.align(rs.stream.color)
        
        # Camera intrinsics - will be populated after start
        self.intrinsics = None
    
    def start(self):
        """Start the camera."""
        profile = self.pipeline.start(self.config)
        
        # Get the intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        return self.intrinsics
    
    def stop(self):
        """Stop the camera."""
        self.pipeline.stop()
    
    def get_frames(self):
        """
        Get the next frames from the camera.
        
        Returns:
            tuple: (color_image, depth_image)
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def get_intrinsics_matrix(self):
        """
        Get camera intrinsics as a 3x3 matrix.
        
        Returns:
            numpy.ndarray: 3x3 camera intrinsics matrix
        """
        if self.intrinsics is None:
            raise ValueError("Camera is not started or no intrinsics available")
        
        K = np.zeros((3, 3))
        K[0, 0] = self.intrinsics.fx  # fx
        K[1, 1] = self.intrinsics.fy  # fy
        K[0, 2] = self.intrinsics.ppx  # cx
        K[1, 2] = self.intrinsics.ppy  # cy
        K[2, 2] = 1.0
        
        return K
    
    def get_distortion_coeffs(self):
        """
        Get camera distortion coefficients.
        
        Returns:
            numpy.ndarray: Distortion coefficients
        """
        if self.intrinsics is None:
            raise ValueError("Camera is not started or no intrinsics available")
        
        # RealSense uses Brown-Conrady distortion model (same as OpenCV)
        return np.array(self.intrinsics.coeffs) 