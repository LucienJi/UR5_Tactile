import sys
import copy
import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander
import moveit_msgs.msg
from enum import IntEnum
from time import sleep
from ros_utils import *
from scipy.interpolate import interp1d
from ur5_interface import IntermediateLink, TerminalLink
from threading import BoundedSemaphore


class ArmMode(IntEnum):
    UNITIALIZED = -1,
    READY = 0,
    PLANNING = 1,
    MOVING = 2,
    AT_GOAL = 3,
    IN_COLLISION = 4,
    E_STOP = 5


class LinkType(IntEnum):
    INTERMEDIATE = 0,
    TERMINAL = 1


class Arm():

    default_planner = "BKPIECEkConfigDefault"
    link_type_map = {
        LinkType.INTERMEDIATE: IntermediateLink,
        LinkType.TERMINAL: TerminalLink
    }


    def __init__(
        self,
        planner=default_planner,
        links={"ee": (LinkType.INTERMEDIATE, "manipulator")}
        ):

        # initialize interface node (if needed)
        if rospy.get_node_uri() is None:
            self.node_name = 'ur5_interface_node'
            rospy.init_node(self.node_name)
            ROS_INFO( "The Arm interface was created outside a ROS node, a new node will be initialized!" )
        else:
            ROS_INFO( "The Arm interface was created within a ROS node, no need to create a new one!" )


        # store parameters
        self._planner = planner
        self._links = links

        # create a RobotCommander
        self._robot = RobotCommander()

        # create semaphore
        self._semaphore = BoundedSemaphore(value=1)

        # default link (for utility functions only)
        self._default_link = None

        # create links objects
        num_valid_links = 0
        for link in self._links.items():
            link_name, link_description = link
            link_type, link_group_name = link_description
            link_class = Arm.link_type_map[link_type]
            link_obj = link_class(self, link_group_name, self._planner)
            if not self._robot.has_group(link_group_name):
                ROS_ERR("Move group `%s` not found!", link_group_name)
            else:
                self.__dict__[link_name] = link_obj
                self._default_link = link_obj
                num_valid_links += 1
        if num_valid_links <= 0:
            ROS_ERR("No valid links were found")
            return None

        # keep track of the status of the node
        self._is_shutdown = False

        self.verbose = False


    def get_robot_state(self):
        return self._robot.get_current_state()

    def get_group_names(self):
        return self._robot.get_group_names()

    def get_planning_frame(self):
        return self._default_link._group.get_planning_frame()

    def get_joint_values(self):
        return self._default_link._group.get_current_joint_values()

    def plan_to_joint_config(self, joint_angles):
        ''' Plan to a given joint configuration '''
        # Clear old pose targets
        self._default_link._group.clear_pose_targets()
        # Set Joint configuration target
        self._default_link._group.set_joint_value_target(joint_angles)
        numTries = 0
        while numTries < 5:
            success, plan, _, _ = self._default_link._group.plan()
            numTries+=1
            if success:
                if self.verbose: print('succeeded in %d tries' % numTries)
                return True, plan
        if self.verbose: print("Planning failed")
        return False, None

    def execute_plan(self, plan):
        ''' Execute a given plan '''
        self._semaphore.acquire()
        out = self._default_link._group.execute(plan)
        sleep(0.05)
        self._semaphore.release()
        return out

    def _plan_to_pose(self, group, pose):
        ''' Plan to a given pose for the end-effector '''
        # Clear old pose targets
        group.clear_pose_targets()

        constr = group.get_path_constraints()
        if len(constr.orientation_constraints) > 0:
            pose.orientation = constr.orientation_constraints[0].orientation

        # Set target pose
        group.set_pose_target(pose)

        numTries = 0;
        while numTries < 5:
            success, plan, _, _ = group.plan()
            numTries += 1
            # check that planning succeeded
            if success:
                if self.verbose: print('succeeded in %d tries' % numTries)
                return True, plan
         # if we get here, we couldn't find a plan in max number of tries
        if self.verbose: print('Planning failed')
        return False, None

    def _plan_cartesian_path(self, group, waypoints):
        numTries = 0;
        while numTries < 5:
            success, plan, _, _ = group.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0          # jump_threshold
            )
            numTries += 1
            # check that planning succeeded
            if success:
                if self.verbose: print('succeeded in %d tries' % numTries)
                return True, plan
         # if we get here, we couldn't find a plan in max number of tries
        return False, None

    def _execute_plan(self, group, plan):
        ''' Execute a given plan '''
        self._semaphore.acquire()
        out = group.execute(plan)
        self._semaphore.release()
        return out

    def shutdown(self):
        print('Shutting down Arm...')
        self._is_shutdown = True
        print('Arm released!')
        return True


    # def plan_cartesian_parametric(
    #     self,
    #     goal_pose,
    #     num_waypoints=10,
    #     x_evol='linear',
    #     y_evol='linear',
    #     z_evol='linear',
    #     roll_evol='linear',
    #     pitch_evol='linear',
    #     yaw_evol='linear' ):
    #
    #     ''' Plan from the current pose of the EndEffector to the given `goal_pose`
    #         by interpolating position (x,y,z) and orientation (r,p,y) over time
    #         according to the given evolution descriptors.
    #
    #         @arg goal_pose  (list, np.array):
    #             List or Numpy array of 6 elements [X, Y, Z, R, P, Y].
    #
    #         @arg x_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of X from
    #             start to goal described as a monotonic sequence with values in [0,1].
    #
    #         @arg y_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of Y from
    #             start to goal described as a monotonic sequence with values in [0,1].
    #
    #         @arg z_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of Z from
    #             start to goal described as a monotonic sequence with values in [0,1].
    #
    #         @arg r_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of the Roll
    #             angle from start to goal described as a monotonic sequence with
    #             values in [0,1].
    #
    #         @arg p_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of the Pitch
    #             angle from start to goal described as a monotonic sequence with
    #             values in [0,1].
    #
    #         @arg y_evol     (string, list, np.array):
    #             It can be either a string, defining a standard evolution
    #             (supported: linear), or an interable with the evolution of the Yaw
    #             angle from start to goal described as a monotonic sequence with
    #             values in [0,1].
    #     '''
    #
    #     # TODO: use self.group.execute( plan ) to execute the plan computed above,
    #     #       self.group.go() sends the arm somewhere else
    #     # TODO: replace standard evolutions (strings) with numeric range
    #     waypoints = []
    #
    #     # start with the current pose
    #     start = self.get_ee_pose()
    #     end = goal_pose
    #
    #     # Get x,y coordinates of start and end pose
    #     x1, y1 = start.position.x, start.position.y
    #     x2, y2 = end.position.x, end.position.y
    #     z1 = start.position.z
    #     H = 0.2
    #
    #     # interpolate x,y,z
    #     x = np.linspace(x1, x2, num_waypoints)
    #     y = np.linspace(y1, y2, num_waypoints)
    #     s = np.linspace(0, 1 , num_waypoints)
    #     z = z1 + 4 * H * (-s**2 + s)
    #
    #
    #     parabola = interp1d(x, z)
    #     z = parabola(x)
    #
    #
    #     wpose = copy.deepcopy(start)
    #
    #     for i in range(0,len(s),1):
    #         wpose.position.x = x[i]
    #         wpose.position.y = y[i]
    #         wpose.position.z = z[i]
    #         waypoints.append(copy.deepcopy(wpose))
    #
    #     plan, fraction = self.group.compute_cartesian_path(
    #         waypoints,   # waypoints to follow
    #         0.01,        # eef_step
    #         0.0)         # jump_threshold
    #     return plan, fraction
