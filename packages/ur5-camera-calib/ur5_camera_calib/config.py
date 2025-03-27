#!/usr/bin/env python3

"""
Configuration file for the camera calibration package.
"""

import os
import yaml
import numpy as np

# Default paths for saving calibration data
CALIBRATION_DIR = os.path.expanduser("~/.ur5_camera_calib")
INTRINSIC_FILE = os.path.join(CALIBRATION_DIR, "intrinsics.yaml")
EXTRINSIC_FILE = os.path.join(CALIBRATION_DIR, "extrinsics.yaml")

# Camera parameters
DEFAULT_CAMERA_PARAMS = {
    "width": 640,
    "height": 480,
    "fps": 30,
}

# Checkerboard parameters for intrinsic calibration
CHECKERBOARD_PARAMS = {
    "rows": 6,  # Number of inner corners in rows
    "cols": 9,  # Number of inner corners in columns
    "square_size": 0.025,  # Square size in meters
}

# AprilTag parameters
APRILTAG_PARAMS = {
    "family": "tag36h11",
    "size": 0.065,  # Tag size in meters
    "border": 1,
}

# Hand-eye calibration parameters
HAND_EYE_PARAMS = {
    "num_poses": 15,  # Number of poses to collect for calibration
    "min_rotation": 5.0,  # Minimum rotation between poses (degrees)
}

def ensure_dir_exists(dir_path):
    """Ensure that directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_calibration(data, file_path):
    """Save calibration data to YAML file."""
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(file_path))
    
    # Convert numpy arrays to lists for YAML serialization
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            processed_data[key] = value.tolist()
        else:
            processed_data[key] = value
    
    with open(file_path, 'w') as f:
        yaml.dump(processed_data, f)

def load_calibration(file_path):
    """Load calibration data from YAML file."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convert lists back to numpy arrays
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            processed_data[key] = np.array(value)
        else:
            processed_data[key] = value
    
    return processed_data 