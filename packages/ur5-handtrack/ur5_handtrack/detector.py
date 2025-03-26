#!/usr/bin/env python
import cv2
import numpy as np
from pupil_apriltags import Detector
import tf.transformations as tf

class TagDetector:
    def __init__(self, camera_config, tag_config):
        self.camera_matrix = camera_config['camera_matrix']
        self.dist_coeffs = camera_config['distortion_coeffs']
        self.camera_frame = camera_config['camera_frame']
        
        self.tag_size = tag_config['tag_size']
        self.tag_to_object = tag_config['tag_to_object']
        
        self.detector = Detector(
            families=tag_config['tag_family'],
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
    def detect_tags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=True,
                                  camera_params=[self.camera_matrix[0,0], self.camera_matrix[1,1],
                                               self.camera_matrix[0,2], self.camera_matrix[1,2]],
                                  tag_size=self.tag_size)
        
        object_poses = []
        for tag in tags:
            if tag.tag_id in self.tag_to_object:
                # Get tag pose
                R = tag.pose_R
                t = tag.pose_t
                
                # Convert to transformation matrix
                T_camera_tag = np.eye(4)
                T_camera_tag[:3, :3] = R
                T_camera_tag[:3, 3] = t.flatten()
                
                # Transform to object frame
                T_tag_object = self.tag_to_object[tag.tag_id]
                T_camera_object = T_camera_tag @ T_tag_object
                
                # Extract position and quaternion
                pos = T_camera_object[:3, 3]
                rot_matrix = T_camera_object[:3, :3]
                quat = tf.quaternion_from_matrix(T_camera_object)
                
                object_poses.append((pos, quat))
        
        return object_poses
