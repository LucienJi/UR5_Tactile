#!/usr/bin/env python3

"""
Script for identifying already running cameras.
Useful for distinguishing between cameras that are already streaming.
"""

import argparse
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ur5_camera_calib.camera_manager import load_camera_layout


class CameraIdentifier:
    """Class for identifying already running cameras."""
    
    def __init__(self, camera_serials, camera_names=None):
        """
        Initialize the camera identifier.
        
        Args:
            camera_serials (list): List of camera serial numbers
            camera_names (dict): Dictionary mapping serial numbers to names
        """
        self.camera_serials = camera_serials
        self.camera_names = camera_names or {}
        self.bridge = CvBridge()
        self.subscribers = {}
        self.images = {}
        self.current_index = 0
        self.running = True
    
    def subscribe_to_cameras(self):
        """Subscribe to all camera topics."""
        for serial in self.camera_serials:
            topic = f"/camera/{serial}/color/image_raw"
            sub = rospy.Subscriber(
                topic,
                Image,
                self.image_callback,
                callback_args=serial,
                queue_size=1,
                buff_size=2**24
            )
            self.subscribers[serial] = sub
    
    def image_callback(self, msg, serial):
        """Callback for camera image messages."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.images[serial] = (cv_image, rospy.Time.now())
        except Exception as e:
            rospy.logerr(f"Error converting image for camera {serial}: {e}")
    
    def run(self):
        """Run the camera identification loop."""
        window_name = "Camera Identifier"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        rate = rospy.Rate(30)  # 30 Hz
        
        print("\n=== Camera Identification ===")
        print("This tool will help you identify which camera is which.")
        print("Press 'n' for next camera, 'p' for previous, 'q' to quit.")
        
        # Wait for some images to arrive
        rospy.sleep(1.0)
        
        while self.running and not rospy.is_shutdown():
            if not self.images:
                rospy.loginfo_throttle(1.0, "Waiting for camera images...")
                rate.sleep()
                continue
            
            # Get current camera
            serials = list(self.camera_serials)
            if not serials:
                break
            
            current_serial = serials[self.current_index]
            
            # Check if we have an image for this camera
            if current_serial not in self.images:
                # Skip this camera
                self.current_index = (self.current_index + 1) % len(serials)
                rate.sleep()
                continue
            
            # Get the image and timestamp
            image, timestamp = self.images[current_serial]
            
            # Check if image is stale (older than 1 second)
            if (rospy.Time.now() - timestamp).to_sec() > 1.0:
                # Mark image as stale
                cv2.putText(
                    image,
                    "STALE IMAGE",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Draw identification info
            display_img = image.copy()
            
            # Create overlay
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (display_img.shape[1], 90), (0, 0, 0), -1)
            
            # Draw camera info
            name = self.camera_names.get(current_serial, "Unnamed")
            cv2.putText(
                overlay,
                f"Camera: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2
            )
            
            cv2.putText(
                overlay,
                f"Serial: {current_serial}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2
            )
            
            # Add index info
            cv2.putText(
                overlay,
                f"Camera {self.current_index+1}/{len(serials)}",
                (display_img.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 0),
                2
            )
            
            # Blend overlay and original image
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
            
            # Show image
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('n'):  # Next camera
                self.current_index = (self.current_index + 1) % len(serials)
            elif key == ord('p'):  # Previous camera
                self.current_index = (self.current_index - 1) % len(serials)
            elif key == ord('q'):  # Quit
                self.running = False
                break
            
            rate.sleep()
        
        cv2.destroyAllWindows()
    
    def stop(self):
        """Stop the identifier and clean up."""
        self.running = False
        for sub in self.subscribers.values():
            sub.unregister()


def find_running_cameras():
    """
    Find running camera topics.
    
    Returns:
        list: List of camera serials
    """
    # Get all published topics
    topics = rospy.get_published_topics()
    
    # Filter to find camera topics
    camera_topics = [t for t, _ in topics if '/camera/' in t and '/color/image_raw' in t]
    
    # Extract serial numbers
    serials = []
    for topic in camera_topics:
        parts = topic.split('/')
        if len(parts) >= 3:
            serials.append(parts[2])
    
    return serials


def main():
    parser = argparse.ArgumentParser(description="Identify already running cameras")
    parser.add_argument("--layout-name", default="camera_layout", help="Name of the camera layout to load")
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('camera_identifier', anonymous=True)
    
    # Find running cameras
    print("Looking for running cameras...")
    camera_serials = find_running_cameras()
    
    if not camera_serials:
        print("No running cameras found!")
        print("Make sure cameras are running with the 'launch_cameras.py' script.")
        return 1
    
    print(f"Found {len(camera_serials)} running cameras:")
    for i, serial in enumerate(camera_serials):
        print(f"  {i+1}. {serial}")
    
    # Load camera layout
    camera_names = load_camera_layout(args.layout_name)
    
    # Create identifier
    identifier = CameraIdentifier(camera_serials, camera_names)
    
    # Subscribe to camera topics
    print("\nSubscribing to camera topics...")
    identifier.subscribe_to_cameras()
    
    # Run identification
    try:
        identifier.run()
    except KeyboardInterrupt:
        pass
    finally:
        identifier.stop()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass 