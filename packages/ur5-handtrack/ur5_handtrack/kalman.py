#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ObjectStateEstimator:
    def __init__(self, params):
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.state = np.zeros(10)
        self.state[6] = 1.0  # Initial quaternion w component
        
        # Initialize covariance matrix
        self.P = np.eye(10)
        self.P[:3, :3] *= params['initial_pos_std']**2
        self.P[3:6, 3:6] *= params['initial_vel_std']**2
        
        # Process noise
        self.Q = np.eye(10)
        self.Q[:3, :3] *= params['process_noise_pos']
        self.Q[3:6, 3:6] *= params['process_noise_vel']
        
        # Measurement noise
        self.R = np.eye(7)  # position (3) + quaternion (4)
        self.R[:3, :3] *= params['measurement_noise_pos']
        self.R[3:, 3:] *= params['measurement_noise_rot']
        
        self.dt = 0.1  # Time step
        
    def predict(self):
        # Predict state
        F = np.eye(10)
        F[:3, 3:6] = np.eye(3) * self.dt
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        # Normalize quaternion
        q_norm = np.linalg.norm(self.state[6:])
        self.state[6:] /= q_norm
        
    def update(self, measurement):
        """
        Update state with measurement (position and quaternion)
        measurement: np.array([x, y, z, qw, qx, qy, qz])
        """
        H = np.zeros((7, 10))
        H[:3, :3] = np.eye(3)  # Position measurement
        H[3:, 6:] = np.eye(4)  # Quaternion measurement
        
        y = measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.P = (np.eye(10) - K @ H) @ self.P
        
        # Normalize quaternion
        q_norm = np.linalg.norm(self.state[6:])
        self.state[6:] /= q_norm
        
    def get_state(self):
        """Returns position and rotation quaternion"""
        return self.state[:3], self.state[6:]
