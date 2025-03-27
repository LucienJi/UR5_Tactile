#!/usr/bin/env python3
"""A very simple script to record trajectory and dump it into a file"""

import os
import time
# from robotiq_s_model_articulated_msgs.msg import SModelRobotOutput, SModelRobotInput
from copy import deepcopy
from pathlib import Path
from threading import Thread
from typing import Callable
import sys
import termios
import tty
import select

import cv2
import message_filters
import numpy as np
import h5py
import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, WrenchStamped
from robotiq_s_interface.gripper import SModelRobotInput, SModelRobotOutput
from sensor_msgs.msg import Image
from ur5_twist_control.helper import ori2numpy, point2numpy, vec32numpy
import pickle
from datetime import datetime
from std_msgs.msg import Float32MultiArray
from ur5_sensor.msg import StampedFloat32MultiArray

bridge = CvBridge()

class KeyboardReader:
    def __init__(self):
        self.is_running = True
        self.last_key = None
        self.thread = Thread(target=self._read_keys)
        self.thread.daemon = True
        self.thread.start()

    def _read_keys(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.is_running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    self.last_key = sys.stdin.read(1)
                time.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_last_key(self):
        key = self.last_key
        self.last_key = None
        return key

    def stop(self):
        self.is_running = False
        self.thread.join()

class UR5TrajRecorder:
    def __init__(self, freq=30, 
                 record_outer_cam=True, 
                 record_wrist_cam=False, 
                 record_tactile=False, 
                 record_wrench=True, 
                 save_path='/code/src/ur5-tactile/data',
                 num_mags = 5, num_samples = 5):
        rospy.init_node('ur5_traj_recorder')

        # Override defaults with ROS parameters if available
        self.freq = rospy.get_param('~freq', freq)
        self.record_outer_cam = rospy.get_param('~record_outer_cam', record_outer_cam)
        self.record_wrist_cam = rospy.get_param('~record_wrist_cam', record_wrist_cam)
        self.record_tactile = rospy.get_param('~record_tactile', record_tactile)
        self.record_wrench = rospy.get_param('~record_wrench', record_wrench)
        self.save_path = rospy.get_param('~save_path', save_path)

        self.rate = rospy.Rate(self.freq)

        # Create h5 file for recording
        current_time = datetime.now()
        self.date_str = current_time.strftime("%Y-%m-%d")
        self.timestamp_str = current_time.strftime("%H-%M-%S")
        self.save_dir = os.path.join(self.save_path, self.date_str)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.h5_path = os.path.join(self.save_dir, f"{self.timestamp_str}.h5")
        self.demo_count = 0
        
        # Initialize h5 file
        with h5py.File(self.h5_path, 'w') as f:
            # No need to create 'demos' group, demos will be stored at root level
            f.attrs['created_at'] = self.timestamp_str
        
        # Existing topics
        twistvel_topic = '/twist_controller/command'
        gripper_topic = '/UR_1/SModelRobotOutput'
        joints_topic = '/joint_states'
        
        # Camera and tactile topics
        self.outer_camera_topic = '/outer_cam/color/image_raw'
        self.wrist_camera_topic = '/wrist_cam/color/image_raw'
        self.tactile_topic = '/anyskin_data'
        self.wrench_topic = '/wrench'  # Force/torque sensor topic

        self._tf_listener = tf.TransformListener()
        
        # Existing subscribers
        self._sub_twistvel = rospy.Subscriber(
            twistvel_topic, Twist, self._twistvel_callback, queue_size=1
        )
        self._sub_gripper = rospy.Subscriber(
            gripper_topic, SModelRobotOutput, self._gripper_callback, queue_size=1
        )
        self._sub_joint_angles = rospy.Subscriber(
            joints_topic, JointState, self._joints_callback, queue_size=1
        )

        # Initialize state variables
        self.states = []
        self.joint_states = None
        self.trans_vel = np.zeros(3)
        self.rot_vel = np.zeros(3)
        self.gripper_pra_requested = 0.0
        self.current_outer_image = None
        self.current_wrist_image = None
        self.current_tactile = None
        self.current_wrench = None
        
        self.step = 0
        self.record_flag = False
        
        # Set up camera and tactile subscribers based on configuration
        self._setup_sensor_subscribers()
        self.num_mags = num_mags
        self.num_samples = num_samples
        
        rospy.loginfo(f"Recording configuration: outer_cam={self.record_outer_cam}, wrist_cam={self.record_wrist_cam}, tactile={self.record_tactile}, wrench={self.record_wrench}")

    def _setup_sensor_subscribers(self):
        """Set up simple independent subscribers for camera and tactile data"""
        # Set up individual subscribers for each data source
        if self.record_outer_cam:
            self._sub_outer_camera = rospy.Subscriber(
                self.outer_camera_topic, Image, self._outer_camera_callback, queue_size=1
            )
            rospy.loginfo(f"Subscribed to outer camera: {self.outer_camera_topic}")
        
        if self.record_wrist_cam:
            self._sub_wrist_camera = rospy.Subscriber(
                self.wrist_camera_topic, Image, self._wrist_camera_callback, queue_size=1
            )
            rospy.loginfo(f"Subscribed to wrist camera: {self.wrist_camera_topic}")
        
        if self.record_tactile:
    
            self._sub_tactile = rospy.Subscriber(
                self.tactile_topic, Float32MultiArray, self._tactile_callback, queue_size=1
            )
            rospy.loginfo(f"Subscribed to tactile: {self.tactile_topic}")
        
        if self.record_wrench:
            self._sub_wrench = rospy.Subscriber(
                self.wrench_topic, WrenchStamped, self._wrench_callback, queue_size=1
            )
            rospy.loginfo(f"Subscribed to wrench: {self.wrench_topic}")

    def _wrench_callback(self, msg: WrenchStamped):
        """Callback for force/torque sensor data"""
        self.current_wrench = msg.wrench

    def _outer_camera_callback(self, image_msg):
        """Callback for outer camera data"""
        try:
            current_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.current_outer_image = cv2.resize(current_image, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            rospy.logerr(f"Error in outer camera callback: {str(e)}")

    def _wrist_camera_callback(self, image_msg):
        """Callback for wrist camera data"""
        try:
            current_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.current_wrist_image = cv2.resize(current_image, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            rospy.logerr(f"Error in wrist camera callback: {str(e)}")

    def _tactile_callback(self, tactile_msg):
        """Callback for tactile data"""
        try:
            self.current_tactile = np.array(tactile_msg.data).reshape(self.num_samples, self.num_mags, 3)
        except Exception as e:
            rospy.logerr(f"Error in tactile callback: {str(e)}")

    def clear_states(self):
        self.states = []

    def get_curr_pose(self):
        from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
        source_frame = 'ur_arm_tool0_controller' #'/hand_finger_tip_link'
        target_frame = '/ur_arm_base'
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        return Pose(position=position, orientation=orientation)

    def _gripper_callback(self, msg: SModelRobotOutput):
        # print('gripper callback!!')
        requested = msg.rPRA / 255.  # (0: open, 1: close)
        self.gripper_pra_requested = requested
        # self.gripper_pra_current = self.twist_controller.gripper.status_raw.gPRA / 255.

    def _joints_callback(self, msg: JointState):
        # print('joint callback!!')
        self.joint_states = msg.position

    def _twistvel_callback(self, msg: Twist):
        # print('twistvel callback!!')
        self.trans_vel = vec32numpy(msg.linear)
        self.rot_vel = vec32numpy(msg.angular)

    def record(self):
        # Get gripper pose
        curr_pose = self.get_curr_pose()
        gripper_pos = point2numpy(curr_pose.position)
        gripper_quat = ori2numpy(curr_pose.orientation)
        
        state = {
            'step': self.step,
            'gripper_pos': gripper_pos,
            'gripper_quat': gripper_quat,
            'cmd_trans_vel': self.trans_vel,
            'cmd_rot_vel': self.rot_vel,
            'cmd_grasp_pos': self.gripper_pra_requested,
            'joint_states': self.joint_states,
        }

        # Add camera and tactile data if available
        if self.record_outer_cam and self.current_outer_image is not None:
            state['outer_image'] = self.current_outer_image.copy()
        
        if self.record_wrist_cam and self.current_wrist_image is not None:
            state['wrist_image'] = self.current_wrist_image.copy()
            
        if self.record_tactile and self.current_tactile is not None:
            state['tactile'] = self.current_tactile.copy()
            
        # Add force/torque data if available
        if self.record_wrench and self.current_wrench is not None:
            # Store force and torque as numpy arrays
            force = np.array([
                self.current_wrench.force.x,
                self.current_wrench.force.y,
                self.current_wrench.force.z
            ])
            torque = np.array([
                self.current_wrench.torque.x,
                self.current_wrench.torque.y,
                self.current_wrench.torque.z
            ])
            state['force'] = force
            state['torque'] = torque

        if self.record_flag:
            self.states.append(state)
            self.step += 1
    
    def start_record(self):
        self.clear_states()
        self.step = 0
        self.record_flag = True 

    def save_states(self, compress=False):
        """Save current states to the h5 file as a new demo"""
        self.record_flag = False
        
        if not self.states:
            rospy.logwarn("No states to save!")
            return
        
        with h5py.File(self.h5_path, 'a') as f:
            # Create a new demo group directly at the root level
            demo_group = f.create_group(f'demo_{self.demo_count}')
            demo_group.attrs['timestamp'] = datetime.now().strftime("%H-%M-%S")
            demo_group.attrs['num_steps'] = len(self.states)
            
            # Create groups for different data types
            action_group = demo_group.create_group('action')
            robot_state_group = demo_group.create_group('robot_state')
            obs_group = demo_group.create_group('obs')
            
            # Prepare data arrays
            steps = len(self.states)
            
            # Action data
            cmd_trans_vel = np.zeros((steps, 3))
            cmd_rot_vel = np.zeros((steps, 3))
            cmd_grasp_pos = np.zeros(steps)
            
            # Robot state data
            joint_states_array = np.zeros((steps, 6))  # Assuming 6 joints
            gripper_pos_array = np.zeros((steps, 3))
            gripper_quat_array = np.zeros((steps, 4))
            
            # Observation data - check which data types are available
            has_outer_images = self.record_outer_cam and 'outer_image' in self.states[0]
            has_wrist_images = self.record_wrist_cam and 'wrist_image' in self.states[0]
            has_tactile = self.record_tactile and 'tactile' in self.states[0]
            has_force_torque = self.record_wrench and 'force' in self.states[0] and 'torque' in self.states[0]
            
            if has_outer_images:
                # Get image dimensions from first state
                img_shape = self.states[0]['outer_image'].shape
                outer_images_array = np.zeros((steps, *img_shape), dtype=np.uint8)
            
            if has_wrist_images:
                # Get image dimensions from first state
                img_shape = self.states[0]['wrist_image'].shape
                wrist_images_array = np.zeros((steps, *img_shape), dtype=np.uint8)
            
            if has_tactile:
                # Get tactile dimensions from first state
                tactile_shape = self.states[0]['tactile'].shape
                tactile_array = np.zeros((steps, *tactile_shape))
                
            if has_force_torque:
                # Force and torque arrays
                force_array = np.zeros((steps, 3))
                torque_array = np.zeros((steps, 3))
            
            # Fill data arrays
            for i, state in enumerate(self.states):
                # Action data
                cmd_trans_vel[i] = state['cmd_trans_vel']
                cmd_rot_vel[i] = state['cmd_rot_vel']
                cmd_grasp_pos[i] = state['cmd_grasp_pos']
                
                # Robot state data
                if state['joint_states'] is not None:
                    joint_states_array[i] = state['joint_states'][:6]  # Assuming 6 joints
                gripper_pos_array[i] = state['gripper_pos']
                gripper_quat_array[i] = state['gripper_quat']
                
                # Observation data
                if has_outer_images:
                    outer_images_array[i] = state['outer_image']
                
                if has_wrist_images:
                    wrist_images_array[i] = state['wrist_image']
                
                if has_tactile:
                    tactile_array[i] = state['tactile']
                    
                if has_force_torque:
                    force_array[i] = state['force']
                    torque_array[i] = state['torque']
            
            # Store action data
            action_group.create_dataset('cmd_trans_vel', data=cmd_trans_vel)
            action_group.create_dataset('cmd_rot_vel', data=cmd_rot_vel)
            action_group.create_dataset('cmd_grasp_pos', data=cmd_grasp_pos)
            
            # Store robot state data
            robot_state_group.create_dataset('joint_states', data=joint_states_array)
            robot_state_group.create_dataset('gripper_pos', data=gripper_pos_array)
            robot_state_group.create_dataset('gripper_quat', data=gripper_quat_array)
            
            # Store observation data
            if has_outer_images:
                obs_group.create_dataset('outer_image', data=outer_images_array, 
                                         compression="gzip" if compress else None, 
                                         compression_opts=4 if compress else None)
            
            if has_wrist_images:
                obs_group.create_dataset('wrist_image', data=wrist_images_array, 
                                         compression="gzip" if compress else None, 
                                         compression_opts=4 if compress else None)
            
            if has_tactile:
                obs_group.create_dataset('tactile', data=tactile_array)
                
            if has_force_torque:
                obs_group.create_dataset('force', data=force_array)
                obs_group.create_dataset('torque', data=torque_array)
        
        print(f'Demo {self.demo_count} saved to {self.h5_path}')
        print(f'Number of states recorded: {len(self.states)}')
        
        self.demo_count += 1
        self.clear_states()
        self.step = 0

    def stop(self, compress=True):
        """Stop recording and optionally compress the h5 file"""
        if compress:
            print(f"Compressing file {self.h5_path}...")
            # Create a temporary compressed file
            temp_path = self.h5_path + ".compressed"
            
            with h5py.File(self.h5_path, 'r') as src, h5py.File(temp_path, 'w') as dst:
                # Copy all attributes
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                
                # Copy all groups and datasets with compression
                def copy_with_compression(name, obj):
                    if isinstance(obj, h5py.Group):
                        # Create group in destination
                        group = dst.create_group(name)
                        # Copy attributes
                        for key, value in obj.attrs.items():
                            group.attrs[key] = value
                    elif isinstance(obj, h5py.Dataset):
                        # Create dataset with compression
                        kwargs = {
                            'data': obj[()],
                            'compression': 'gzip',
                            'compression_opts': 4
                        }
                        dst.create_dataset(name, **kwargs)
                        # Copy attributes
                        for key, value in obj.attrs.items():
                            dst[name].attrs[key] = value
                
                # Visit all objects in the source file
                src.visititems(copy_with_compression)
            
            # Replace original file with compressed one
            os.replace(temp_path, self.h5_path)
            print(f"File compressed successfully.")
        
        rospy.signal_shutdown("User requested shutdown")
        import sys
        sys.exit(0)

class CmdlineListener:
    def __init__(self, 
                 start_fn: Callable, 
                 save_fn: Callable,
                 stop_fn: Callable,
                 gohome_fn: Callable
                 ):
        self.key_reader = KeyboardReader()
        self.input_thread = Thread(target=self._listener)
        self.input_thread.daemon = True
        self.input_thread.start()
        self.start_fn = start_fn
        self.save_fn = save_fn
        self.stop_fn = stop_fn
        self.gohome_fn = gohome_fn
        self.is_running = True
        
        # Print instructions
        self.print_instructions()
        
    def print_instructions(self):
        print("\n===== UR5 Trajectory Recorder Instructions =====")
        print("Press 's' to START recording")
        print("Press 'e' to END recording and save the demonstration")
        print("Press 'q' to quit the program")
        print("Press 'g' to go to home position")
        print("Press 'h' to show these instructions again")
        print("=========================================\n")

    def _listener(self):
        while self.is_running:
            key = self.key_reader.get_last_key()
            if key == 's':
                print("========== Start Recording ==============")
                self.start_fn()
            elif key == 'e':
                print("========== Save Recording ==============")
                self.save_fn()
            elif key == 'q':
                print("========== Stop Recording ==============")
                print("Compress the file? (y/n)")
                # Wait for compression response
                compression_response = None
                while compression_response not in ['y', 'n']:
                    compression_response = self.key_reader.get_last_key()
                    time.sleep(0.1)
                    
                is_compressed = True if compression_response == 'y' else False
                print(f"Compressing: {is_compressed}")
                self.stop_fn(compress=is_compressed)
                self.is_running = False
                self.key_reader.stop()
            elif key == 'g':
                print("========== Go Home ==============")
                self.gohome_fn()
            elif key == 'h':
                self.print_instructions()
                
            time.sleep(0.1)  # Short sleep to prevent CPU hogging

def call_gohomepose_srv():
    from ur5_twist_control.srv import GoHomePose
    print('calling go home pose service!!')
    rospy.wait_for_service('/go_homepose')
    try:
        go_homepose = rospy.ServiceProxy('/go_homepose', GoHomePose)
        result = go_homepose()
        print('result', result)
        return result.success
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return False


if __name__ == '__main__':
    fpath = '/code/src/ur5-tactile/data'

    # call_gohomepose_srv()

    traj_recorder = UR5TrajRecorder(
        freq=30,
        record_outer_cam=True,
        record_wrist_cam=True,
        record_tactile=True,
        save_path=fpath
    )
    print('Started trajectory recorder!')
    
    listener = CmdlineListener(
        start_fn=traj_recorder.start_record,
        save_fn=lambda: traj_recorder.save_states(compress=False),
        stop_fn=traj_recorder.stop,
        gohome_fn=call_gohomepose_srv
    )

    try:
        while not rospy.is_shutdown():
            traj_recorder.record()
            traj_recorder.rate.sleep()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure proper cleanup
        if hasattr(listener, 'key_reader'):
            listener.key_reader.stop()
        print("Trajectory recorder stopped.")

