#!/usr/bin/env python3

import rospy
from ur5_comp_control.compliance_control import ComplianceControl

if __name__ == "__main__":
    rospy.init_node('compliance_control_server')
    
    # Create and initialize the compliance control
    controller = ComplianceControl(init_arm=True)
    
    # Initialize ros node parameters
    # These can be set as params in the launch file
    rate = rospy.get_param('~rate', 100.0)  # Hz
    
    # Main loop
    r = rospy.Rate(rate)
    while not rospy.is_shutdown():
        # Main update loop - can be used for continuous monitoring
        # or other periodic tasks
        r.sleep()
        
    print("Compliance control server has stopped.")
        
 