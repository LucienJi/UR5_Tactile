#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from config import CAMERA_CONFIGS, APRILTAG_CONFIGS, TAG_TO_OBJECT, KALMAN_PARAMS
from detector import TagDetector
from kalman import ObjectStateEstimator

class HandTracker:
    def __init__(self):
        rospy.init_node('hand_tracker')
        
        self.bridge = CvBridge()
        self.detectors = {}
        self.kalman = ObjectStateEstimator(KALMAN_PARAMS)
        
        # Initialize detectors for each camera
        for camera_name, config in CAMERA_CONFIGS.items():
            self.detectors[camera_name] = TagDetector(
                camera_config=config,
                tag_config={
                    'tag_family': APRILTAG_CONFIGS['tag_family'],
                    'tag_size': APRILTAG_CONFIGS['tag_size'],
                    'tag_to_object': TAG_TO_OBJECT
                }
            )
        
        # Initialize subscribers
        self.subscribers = {}
        for camera_name, config in CAMERA_CONFIGS.items():
            self.subscribers[camera_name] = rospy.Subscriber(
                config['camera_topic'],
                Image,
                self.image_callback,
                callback_args=camera_name,
                queue_size=1
            )
        
        # Initialize publisher
        self.pose_pub = rospy.Publisher('hand_state', PoseStamped, queue_size=1)
        
        self.last_update_time = rospy.Time.now()
    
    def image_callback(self, msg, camera_name):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detect tags and get object poses
            object_poses = self.detectors[camera_name].detect_tags(cv_image)
            
            if len(object_poses) > 0:
                # Average multiple detections if necessary
                avg_pos = np.mean([pose[0] for pose in object_poses], axis=0)
                avg_quat = np.mean([pose[1] for pose in object_poses], axis=0)
                avg_quat /= np.linalg.norm(avg_quat)
                
                # Update Kalman filter
                measurement = np.concatenate([avg_pos, avg_quat])
                
                current_time = rospy.Time.now()
                dt = (current_time - self.last_update_time).to_sec()
                self.kalman.dt = dt
                self.last_update_time = current_time
                
                self.kalman.predict()
                self.kalman.update(measurement)
                
                # Publish result
                self.publish_pose()
        
        except Exception as e:
            rospy.logerr(f"Error processing image from {camera_name}: {str(e)}")
    
    def publish_pose(self):
        pos, quat = self.kalman.get_state()
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_frame"
        
        pose_msg.pose.position.x = pos[0]
        pose_msg.pose.position.y = pos[1]
        pose_msg.pose.position.z = pos[2]
        
        pose_msg.pose.orientation.w = quat[0]
        pose_msg.pose.orientation.x = quat[1]
        pose_msg.pose.orientation.y = quat[2]
        pose_msg.pose.orientation.z = quat[3]
        
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        tracker = HandTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
