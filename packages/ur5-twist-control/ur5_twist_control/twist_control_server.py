#! /usr/bin/env python3

import signal
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, Pose, Quaternion, QuaternionStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ur5_twist_control.control_client import ControllerManagerClient, TrajectoryClient, TwistVelocityClient
from ur5_twist_control.helper import get_arm, get_planning_scene, numpy2quat, ori2numpy, point2numpy
from typing import Union

# Services
from ur5_twist_control.srv import GoHomePose, GoHomePoseResponse
from ur5_twist_control.srv import GoCartesianPose, GoCartesianPoseResponse

eps = 1e-3
source_frame = '/hand_finger_tip_link'
target_frame = '/ur_arm_base'

# Sane initial configuration
init_joint_pos = [1.3123908042907715, -1.709191624318258, 1.5454894304275513, -1.1726744810687464, -1.5739596525775355, -0.7679112593280237]
# NOTE: BUG: these init pos and ori are of `ur_arm_tool0_controller` frame! NOT `hand_finger_tip_link` frame
# Initial pose
init_pos = Point(x=0.1, y=-0.4, z=0.4)
# self._init_ori = Quaternion(x=0.928, y=-0.371, z=0, w=0)  # OLD
init_ori = Quaternion(x=0.364, y=-0.931, z=0, w=0)

def quat2np(quat: Quaternion):
    return np.array([quat.x, quat.y, quat.z, quat.w])

def point2np(point: Point):
    return np.array([point.x, point.y, point.z])

def np2point(np_pos: np.ndarray):
    from geometry_msgs.msg import Point
    return Point(x=np_pos[0], y=np_pos[1], z=np_pos[2])


def np2quat(np_quat: np.ndarray):
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=np_quat[0], y=np_quat[1], z=np_quat[2], w=np_quat[3])


class TwistControlServer:
    def __init__(self, init_arm=True) -> None:
        self._tf_listener = tf.TransformListener()

        print('Instantiating ControllerManagerClient...')
        controller_manager_client = ControllerManagerClient()

        print('Instantiating move clients...')
        self._traj_client = TrajectoryClient(controller_manager_client)

        # Just to get the joint positions
        from sensor_msgs.msg import JointState
        self.joint_state = None
        self._sub_jpos = rospy.Subscriber('/joint_states', JointState, self._jnt_callback, queue_size=1)
        self._pub_jpos = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)

        # Get current pose and move to the initial pose
        if init_arm:
            self._init_arm(use_joint_controller=True)

        # Define a service
        print('Running go_homepose service...')
        self._gohomepose_srv = rospy.Service('go_homepose', GoHomePose, self._handle_gohomepose)

        self._gocartesianpose_srv = rospy.Service('go_cartesianpose', GoCartesianPose, self._handle_go_cartesianpose)

    def _handle_gohomepose(self, req):
        print('Resetting the arm...')
        success = self._init_arm(use_joint_controller=True)
        return GoHomePoseResponse(success=success)

    def _handle_go_cartesianpose(self, req):
        print('Handle Go CartesianPose!')
        print('request', req)
        pos = req.target_pose.position
        ori = req.target_pose.orientation  # This is Quaternion!!
        with np.printoptions(precision=3):
            print(f'Moving the arm toward pos: {pos}, ori: {ori}')

        success = self._move_to(pos, ori, exec_time=req.exec_time, spawn_twistvel_controller=True, use_tool0_frame=False)
        return GoCartesianPoseResponse(success=success)

    def _jnt_callback(self, msg):
        self.joint_state = {'pos': msg.position, 'vel': msg.velocity, 'effort': msg.effort}

    def _get_curr_pose(self) -> Pose:
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        import time
        start = time.perf_counter()
        trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        elapsed = time.perf_counter() - start
        print(f'>>>>>>>>>>>>> elapsed (lookupTransform): {elapsed:.2f}')
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])

        return Pose(position=position, orientation=orientation)

    def _get_curr_twist(self):
        tooltip_frame = source_frame
        base_frame = target_frame

        twist = self._tf_listener.lookupTwistFull(
            tracking_frame=tooltip_frame,
            observation_frame=base_frame,
            reference_frame=tooltip_frame,
            ref_point=(0, 0, 0),
            reference_point_frame=tooltip_frame,
            time=rospy.Time(0),
            averaging_interval=rospy.Duration(nsecs=int(50 * 1e6))  # 50 ms
        )
        return twist

    def _init_arm(self, spawn_twistvel_controller=True, use_joint_controller=True) -> bool:
        """Move to the initial position"""
        if use_joint_controller:
            # Use scaled_pos_joint_traj_controller (i.e., arm_controller) rather than cartesian ones!
            # Load arm_controller
            self._traj_client.controller_manager.switch_controller('arm_controller')

            # Get current joint pos
            while self.joint_state is None:
                print("Waiting for joint_state to be populated...")
                time.sleep(0.1)

            assert self.joint_state['pos'] is not None
            while np.linalg.norm(np.array(self.joint_state['pos'])) < 1e-6:
                print("Waiting for none-zero joint_state...")
                time.sleep(0.1)

            joint_traj = [self.joint_state['pos'], init_joint_pos]
            print('====== Moving Home ======')
            print('point 0', joint_traj[0])
            print('point 1', joint_traj[1])
            print('====== Moving Home DONE ======')
            success = self._traj_client.send_joint_trajectory(joint_traj, time_step=2.)

            # Switch back to twist_controller!
            self._traj_client.controller_manager.switch_controller('twist_controller')
        else:
            # self._traj_client.controller_manager.switch_controller('forward_cartesian_traj_controller')
            success = self._move_to(init_pos, init_ori, spawn_twistvel_controller=spawn_twistvel_controller)
        return success

    def _move_to(self, tgt_pos: Point, tgt_quat: Union[Quaternion, None] = None, exec_time=3, spawn_twistvel_controller=True, use_tool0_frame=True):
        """
        NOTE: This method expects the pos and ori in `ur_arm_tool0_controller` frame! NOT `hand_finger_tip_link` frame
        You can set `use_tool0_frame=False` to specify the pose in `hand_finger_tip_link` frame
        """
        # TODO: Convert the pose in `ur_arm_tool0_controller` to `hand_finger_tip_link`
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, 'hand_finger_tip_link', rospy.Time(0))

        T_bh = np.eye(4)  # hand_finger_tip_link to base_link
        T_bh[:3, :3] = R.from_quat(rot).as_matrix()
        T_bh[3, :3] = trans


        print('Loading forward_cartesian_traj_controller...')
        self._traj_client.switch_controller()

        # NOTE: We need this rather than self._get_curr_pose() just because the traj_client expects the pose of ur_arm_tool0_controller.
        timeout = 1.0
        self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
        trans, rot = self._tf_listener.lookupTransform(target_frame, 'ur_arm_tool0_controller', rospy.Time(0))
        position = Point(x=trans[0], y=trans[1], z=trans[2])
        orientation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
        curr_pose = Pose(position=position, orientation=orientation)

        # target_pose = Pose(position=pos, orientation=ori)

        T_bu = np.eye(4)  # ur_arm_tool0_controller to base_link
        T_bu[:3, :3] = R.from_quat(rot).as_matrix()
        T_bu[:3, 3] = trans

        # Assume the given pos and ori is about hand_finger_tip_link
        T_tgt_bh = np.eye(4)
        T_tgt_bh[:3, :3] = R.from_quat(quat2np(tgt_quat)).as_matrix()
        T_tgt_bh[:3, 3] = point2np(tgt_pos)

        T_hu = np.linalg.inv(T_bh) @ T_bu
        T_tgt_bu = T_tgt_bh @ T_hu

        if use_tool0_frame:
            tgt_pos_u = tgt_pos
            tgt_quat_u = tgt_quat
            assert tgt_quat is not None
        else:
            tgt_pos_u = np2point(T_tgt_bu[:3, 3])
            tgt_quat_u = np2quat(R.from_matrix(T_tgt_bu[:3, :3]).as_quat())
            if tgt_quat is None:
                tgt_quat_u = self._init_ori

        target_pose = Pose(position=tgt_pos_u, orientation=tgt_quat_u)

        # Move to the init location
        print('curr pose', curr_pose)
        print(f'Moving to {target_pose}')

        traj = [curr_pose, target_pose]
        success = self._traj_client.send_cartesian_trajectory(traj, init_time=0.0, time_step=exec_time)

        if spawn_twistvel_controller:
            # Spawn TwistController
            self._traj_client.controller_manager.switch_controller('twist_controller')

        return success


if __name__ == '__main__':
    rospy.init_node('twist_control_server')
    twist_contrl = TwistControlServer()

    # keep spinning
    rospy.spin()
