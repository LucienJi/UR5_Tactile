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


class PoseCollectorV2:
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
        print("Press 's' to save current data point")
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
            print("======================\n")
        else:
            rospy.logwarn("Could not get current transformation or wrench.")

    def run(self):
        print("Starting interactive pose collection. Move the robot to desired poses manually.")
        print("Data will be saved when you press 's'.")
        
        rate = rospy.Rate(10)  # 10Hz
        
        try:
            while not rospy.is_shutdown():
                key = self.key_reader.get_last_key()
                
                if key == 's':
                    print("Saving current data point...")
                    self.save_current_data()
                    
                elif key == 'q':
                    print("Quitting pose collector...")
                    break
                    
                elif key == 'i':
                    self.print_instructions()
                    
                elif key == 'c':
                    self.display_current_data()
                    
                rate.sleep()
                
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.key_reader.stop()
            print(f"Pose collection finished. Total data points saved: {self.data_points_saved}")


if __name__ == '__main__':
    try:
        save_path = "/code/src/ur5-tactile/data/collected_data_interactive.h5"
        collector = PoseCollectorV2(save_path)
        collector.run()
    except rospy.ROSInterruptException:
        pass 