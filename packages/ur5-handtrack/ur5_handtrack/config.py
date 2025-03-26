#!/usr/bin/env python
import numpy as np

# Camera Parameters
CAMERA_CONFIGS = {
    'camera1': {
        'camera_matrix': np.array([[615.7328491210938, 0.0, 326.9174499511719],
                                 [0.0, 616.8341674804688, 225.59703063964844],
                                 [0.0, 0.0, 1.0]]),
        'distortion_coeffs': np.array([0.08189064264297485, -0.19392056763172150, 
                                     -0.00043430625415593, -0.00251007852703333, 0.0]),
        'camera_topic': '/outer_cam/color/image_raw',
        'camera_frame': 'camera1_frame'
    },
    # 'camera2': {
    #     'camera_matrix': np.array([[614.9328491210938, 0.0, 327.9174499511719],
    #                              [0.0, 615.8341674804688, 224.59703063964844],
    #                              [0.0, 0.0, 1.0]]),
    #     'distortion_coeffs': np.array([0.07189064264297485, -0.18392056763172150, 
    #                                  -0.00042430625415593, -0.00250007852703333, 0.0]),
    #     'camera_topic': '/camera2/image_raw',
    #     'camera_frame': 'camera2_frame'
    # }
}

# AprilTag Configuration
APRILTAG_CONFIGS = {
    'tag_family': 'tagStandard41h12',
    'tag_ids': [0, 1, 2],  # IDs of the three tags on the object
    'tag_size': 0.02  # Size of the tags in meters
}

# Tag to Object Transformations (4x4 transformation matrices)
TAG_TO_OBJECT = {
    0: np.array([[1, 0, 0, 0.05],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]),
    1: np.array([[1, 0, 0, -0.05],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]),
    2: np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0.05],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
}

# Kalman Filter Parameters
KALMAN_PARAMS = {
    'process_noise_pos': 0.1,
    'process_noise_vel': 0.1,
    'measurement_noise_pos': 0.01,
    'measurement_noise_rot': 0.01,
    'initial_pos_std': 0.1,
    'initial_vel_std': 0.1
}
