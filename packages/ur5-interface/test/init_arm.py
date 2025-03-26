from ur5_interface import Arm

# initialize arm interface
a = Arm( move_group="finger_tip_manipulator" )

# create aliases
arm = a
ur = a
ur5 = a
