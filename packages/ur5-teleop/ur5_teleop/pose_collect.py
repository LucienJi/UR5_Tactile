#! /usr/bin/env python3
from ur5_comp_control.compliance_control import ComplianceControl
from ur5_comp_control.srv import MoveToPose, MoveToPoseRequest
from ur5_comp_control.helper import TARGET_FRAME, SOURCE_FRAME
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from sensor_msgs.msg import JointState
import numpy as np
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import h5py
import threading
import sys
import termios
import tty
import select
import time


class KeyboardReader:
    def __init__(self):
        self.is_running = True
        self.last_key = None
        self.thread = threading.Thread(target=self._read_keys)
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


class PoseCollector:
    def __init__(self, save_path):
        rospy.init_node('pose_collector_v2', anonymous=True)
        self.wrench_sub = rospy.Subscriber('/robotiq_force_torque_wrench', WrenchStamped, self.wrench_callback)
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        
        self.controller = ComplianceControl(init_arm=False)
        self.controller._motion_client.switch_controller()

        self._tf_listener = tf.TransformListener()
        self.save_path = save_path
        
        self.wrench = None 
        self.joint_states = None
        
        self.data_points_saved = 0
        self.demo_count = 0
        
        # Recording state variables
        self.is_recording = False
        self.recorded_wrenches = []
        self.recorded_transformations = []
        self.recorded_joint_states = []
        
        try:
            rospy.wait_for_service('move_to_pose_srv', timeout=5)
        except rospy.ServiceException as e:
            rospy.logerr("Service 'move_to_pose_srv' not available: %s" % e)
            exit(1)
        self.move_to_pose_client = rospy.ServiceProxy('move_to_pose_srv', MoveToPose)
        
        self.key_reader = KeyboardReader()
        
        # Print instructions
        self.print_instructions()
    
    def print_instructions(self):
        print("\n===== Pose Collector v2 Instructions =====")
        print("Press 's' to START recording a demonstration")
        print("Press 'e' to END recording and save the demonstration")
        print("Press 'q' to quit the program")
        print("Press 'i' to show these instructions again")
        print("Press 'c' to show current pose data")
        print("========================================\n")
    
    def wrench_callback(self, msg):
        self.wrench = msg.wrench
    
    def joint_states_callback(self, msg):
        self.joint_states = msg
    
    def get_curr_pose(self):
        timeout = 1.0
        
        target_frame = TARGET_FRAME
        source_frame = SOURCE_FRAME
        try:
            self._tf_listener.waitForTransform(
                target_frame, source_frame, rospy.Time(), rospy.Duration(timeout)
            )
            
            trans, rot = self._tf_listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            
            position = Point(x=trans[0], y=trans[1], z=trans[2])
            orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
            
            return Pose(position=position, orientation=orientation)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {e}")
            return None

    def save_data_h5py(self, wrench_data, transformation_data):
        with h5py.File(self.save_path, 'a') as hf:
            if 'wrenches' not in hf:
                hf.create_dataset('wrenches', data=wrench_data.reshape(1, -1), maxshape=(None, 6))
                hf.create_dataset('transformations', data=transformation_data.reshape(1, -1), maxshape=(None, 7))
            else:
                hf['wrenches'].resize((hf['wrenches'].shape[0] + 1), axis=0)
                hf['wrenches'][-1] = wrench_data
                hf['transformations'].resize((hf['transformations'].shape[0] + 1), axis=0)
                hf['transformations'][-1] = transformation_data
        
        self.data_points_saved += 1
        print(f"Data point {self.data_points_saved} saved to: {self.save_path}")

    def save_demo_h5py(self, wrenches, transformations, joint_states):
        """Save a complete demonstration to the H5 file"""
        if not wrenches or not transformations or not joint_states:
            rospy.logwarn("No data to save. Demo not recorded.")
            return False
            
        # Convert lists to numpy arrays
        wrenches_array = np.array(wrenches)
        transformations_array = np.array(transformations)
        
        # Process joint states to extract positions, velocities, and efforts
        joint_positions = np.array([js.position for js in joint_states])
        joint_velocities = np.array([js.velocity for js in joint_states])
        joint_efforts = np.array([js.effort for js in joint_states])
        
        # Save to H5 file
        with h5py.File(self.save_path, 'a') as hf:
            # Create demos group if it doesn't exist
            if 'demos' not in hf:
                demos_group = hf.create_group('demos')
            else:
                demos_group = hf['demos']
                
            # Create a new demo group with a unique index
            demo_id = f'demo_{self.demo_count}'
            demo_group = demos_group.create_group(demo_id)
            
            # Save data in the demo group
            demo_group.create_dataset('wrenches', data=wrenches_array)
            demo_group.create_dataset('transformations', data=transformations_array)
            demo_group.create_dataset('joint_positions', data=joint_positions)
            demo_group.create_dataset('joint_velocities', data=joint_velocities)
            demo_group.create_dataset('joint_efforts', data=joint_efforts)
            
            # Store timestamp as attribute
            demo_group.attrs['timestamp'] = time.time()
            demo_group.attrs['num_frames'] = len(wrenches)
        
        self.demo_count += 1
        print(f"Demo {self.demo_count} with {len(wrenches)} frames saved to: {self.save_path}")
        return True

    def capture_current_data(self):
        """Capture current data point for recording"""
        current_transformation = self.get_curr_pose()
        current_wrench_msg = self.wrench
        current_joint_states = self.joint_states

        if current_transformation is not None and current_wrench_msg is not None and current_joint_states is not None:
            # Transform data
            current_transformation_np = np.array([
                current_transformation.position.x,
                current_transformation.position.y,
                current_transformation.position.z,
                current_transformation.orientation.x,
                current_transformation.orientation.y,
                current_transformation.orientation.z,
                current_transformation.orientation.w
            ])
            
            current_wrench_np = np.array([
                current_wrench_msg.force.x,
                current_wrench_msg.force.y,
                current_wrench_msg.force.z,
                current_wrench_msg.torque.x,
                current_wrench_msg.torque.y,
                current_wrench_msg.torque.z
            ])
            
            # Store data in recording lists
            self.recorded_wrenches.append(current_wrench_np)
            self.recorded_transformations.append(current_transformation_np)
            self.recorded_joint_states.append(current_joint_states)
            
            frames_recorded = len(self.recorded_wrenches)
            if frames_recorded % 10 == 0:  # Print status every 10 frames
                print(f"Recording... {frames_recorded} frames captured")
                
            return True
        else:
            if current_transformation is None:
                rospy.logwarn("Could not get current transformation. Frame skipped.")
            if current_wrench_msg is None:
                rospy.logwarn("Could not get current wrench. Frame skipped.")
            if current_joint_states is None:
                rospy.logwarn("Could not get current joint states. Frame skipped.")
            return False

    def save_current_data(self):
        current_transformation = self.get_curr_pose()
        current_wrench_msg = self.wrench

        if current_transformation is not None and current_wrench_msg is not None:
            current_transformation_np = np.array([
                current_transformation.position.x,
                current_transformation.position.y,
                current_transformation.position.z,
                current_transformation.orientation.x,
                current_transformation.orientation.y,
                current_transformation.orientation.z,
                current_transformation.orientation.w
            ])
            
            current_wrench_np = np.array([
                current_wrench_msg.force.x,
                current_wrench_msg.force.y,
                current_wrench_msg.force.z,
                current_wrench_msg.torque.x,
                current_wrench_msg.torque.y,
                current_wrench_msg.torque.z
            ])
            
            self.save_data_h5py(current_wrench_np, current_transformation_np)
            return True
        else:
            rospy.logwarn("Could not get current transformation or wrench. Data not saved.")
            return False
    
    def display_current_data(self):
        current_transformation = self.get_curr_pose()
        current_wrench_msg = self.wrench
        
        if current_transformation is not None and current_wrench_msg is not None:
            print("\n===== Current Data =====")
            print(f"Position: x={current_transformation.position.x:.4f}, y={current_transformation.position.y:.4f}, z={current_transformation.position.z:.4f}")
            print(f"Orientation: x={current_transformation.orientation.x:.4f}, y={current_transformation.orientation.y:.4f}, z={current_transformation.orientation.z:.4f}, w={current_transformation.orientation.w:.4f}")
            print(f"Wrench - Force: x={current_wrench_msg.force.x:.4f}, y={current_wrench_msg.force.y:.4f}, z={current_wrench_msg.force.z:.4f}")
            print(f"Wrench - Torque: x={current_wrench_msg.torque.x:.4f}, y={current_wrench_msg.torque.y:.4f}, z={current_wrench_msg.torque.z:.4f}")
            
            if self.joint_states is not None:
                print(f"Joint Positions: {[f'{pos:.4f}' for pos in self.joint_states.position]}")
            
            print("======================\n")
        else:
            rospy.logwarn("Could not get current transformation or wrench.")

    def run(self):
        print("Starting interactive pose collection. Move the robot to desired poses manually.")
        print("Press 's' to start recording a demonstration.")
        
        rate = rospy.Rate(5)  # 10Hz
        
        try:
            while not rospy.is_shutdown():
                key = self.key_reader.get_last_key()
                
                if key == 's' and not self.is_recording:
                    print("Starting to record demonstration...")
                    self.is_recording = True
                    # Clear previous recordings
                    self.recorded_wrenches = []
                    self.recorded_transformations = []
                    self.recorded_joint_states = []
                
                elif key == 'e' and self.is_recording:
                    print("Ending demonstration recording...")
                    self.is_recording = False
                    # Save the recorded demonstration
                    if self.save_demo_h5py(self.recorded_wrenches, self.recorded_transformations, self.recorded_joint_states):
                        print(f"Demonstration recorded with {len(self.recorded_wrenches)} frames.")
                        self.recorded_wrenches = []
                        self.recorded_transformations = []
                        self.recorded_joint_states = []
                    else:
                        print("Failed to save demonstration.")
                
                elif key == 'q':
                    if self.is_recording:
                        print("Ending current recording before quitting...")
                        self.is_recording = False
                        # Save any recorded data
                        self.save_demo_h5py(self.recorded_wrenches, self.recorded_transformations, self.recorded_joint_states)
                    print("Quitting pose collector...")
                    break
                    
                elif key == 'i':
                    self.print_instructions()
                    
                elif key == 'c':
                    self.display_current_data()
                
                # If we're recording, capture current data
                if self.is_recording:
                    self.capture_current_data()
                    
                rate.sleep()
                
        except KeyboardInterrupt:
            print("Program interrupted by user")
            if self.is_recording:
                print("Saving final recording due to interruption...")
                self.is_recording = False
                self.save_demo_h5py(self.recorded_wrenches, self.recorded_transformations, self.recorded_joint_states)
        finally:
            self.key_reader.stop()
            print(f"Pose collection finished. Total demos saved: {self.demo_count}")


if __name__ == '__main__':
    try:
        save_path = "/code/src/ur5-tactile/data/collected_data_interactive_v2.h5"
        collector = PoseCollector(save_path)
        collector.run()
    except rospy.ROSInterruptException:
        pass 