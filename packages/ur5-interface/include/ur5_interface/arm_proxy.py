import copy
import rospy
from ros_utils import *
import numpy as np
from moveit_commander import MoveGroupCommander, Constraints
from moveit_msgs.msg import OrientationConstraint

class LinkProxy():

    def __init__(self, arm):
        self._arm = arm
        self._group_name = None
        self._group = None
        self._planner = None
        self.orientation_frozen = False

    def _initialize_group(self):
        # get list of planners available to this move group
        available_planners = rospy.get_param(
            "/move_group/%s/planner_configs" % self._group_name,
            None
        )
        if available_planners is None:
            ROS_ERR("The Move group `%s` does not have planners configured.", self._group_name)
            return False
        if self._planner not in available_planners:
            ROS_WARN(
                "The planner `%s` is not available for the group `%s`, using `%s` instead.",
                self._planner,
                self._group_name,
                available_planners[0]
            )
            self._planner = available_planners[0]
        self._group = MoveGroupCommander(self._group_name)
        # set planner
        self._group.set_planner_id(self._planner)

    def _is_group_valid(self):
        if self._group is None:
            if isinstance(self, TerminalLink) or isinstance(self, IntermediateLink):
                ROS_ERR("Link object was created without a group")
                return False
            else:
                ROS_ERR("Cannot instantiate abstract classes")
                return False
        return True

    def plan_to_pose(self, pose, *args):
        raise NotImplentedError()

    def plan_cartesian_path(self, waypoints, *args):
        raise NotImplentedError()

    def execute_plan(self, plan):
        if self._is_group_valid():
            return self._arm._execute_plan(self._group, plan)
        return False

    def get_pose(self):
        return self._group.get_current_pose().pose

    def is_at(self, pose, epsilon):
        cur_pose = self.get_pose()
        cur_position = [ cur_pose.position.x, cur_pose.position.y, cur_pose.position.z ]
        goal_position = [ pose.position.x, pose.position.y, pose.position.z ]
        eucl_dist = np.linalg.norm(np.array(cur_position)-np.array(goal_position))
        return eucl_dist < epsilon

    def get_planning_frame(self):
        if self._is_group_valid():
            return self._group.get_planning_frame()
        return False

    def freeze_orientation(self):
        # get current pose
        cur_pose = self._group.get_current_pose()
        # create pose constraint
        constraints = Constraints()
        constraints.name = "orientation_constraints"
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = cur_pose.header
        orientation_constraint.header.stamp = rospy.Time.now()
        orientation_constraint.link_name = self._group.get_end_effector_link()
        orientation_constraint.orientation = cur_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.5
        orientation_constraint.absolute_y_axis_tolerance = 0.5
        orientation_constraint.absolute_z_axis_tolerance = 0.5
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)
        self._group.set_path_constraints(constraints)
        self.orientation_frozen = True

    def clear_path_constraints(self):
        self._group.set_path_constraints(None)
        self.orientation_frozen = False


# fingertip, hand center, palm
class TerminalLink(LinkProxy):
    def __init__(self, arm, group_name, planner):
        self._arm = arm
        self._group_name = group_name
        self._planner = planner
        self._initialize_group()

    def plan_to_pose(self, pose, *args):
        if self._is_group_valid():
            return self._arm._plan_to_pose(self._group, pose)
        return False, None

    def plan_cartesian_path(self, waypoints, *args):
        if self._is_group_valid():
            return self._arm._plan_cartesian_path(self._group, waypoints)


# camera, wrist
class IntermediateLink(LinkProxy):
    def __init__(self, arm, group_name, planner):
        self._arm = arm
        self._group_name = group_name
        self._planner = planner
        self._initialize_group()

    def plan_to_pose(self, pose, offset_x, offset_y, offset_z, *args):
        if self._is_group_valid():
            pose = copy.deepcopy(pose)
            pose.position.x += offset_x
            pose.position.y += offset_y
            pose.position.z += offset_z
            return self._arm._plan_to_pose(self._group, pose)
        return False, None

    def plan_cartesian_path(self, waypoints, offset_x, offset_y, offset_z, *args):
        if self._is_group_valid():
            wpoints = [copy.deepcopy(p) for p in waypoints]
            for pose in wpoints:
                pose.position.x += offset_x
                pose.position.y += offset_y
                pose.position.z += offset_z
            return self._arm._plan_cartesian_path(self._group, wpoints)
