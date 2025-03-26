#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Header
from ur5_sensor.msg import StampedFloat32MultiArray  
from anyskin import AnySkinProcess
import pygame
import os
import time
import threading

class AnySkinPublisher:
    """
    ROS Publisher for AnySkin sensor data.
    Reads data from the sensor and publishes it as Float32MultiArray messages.
    """
    
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('anyskin_publisher', anonymous=True)
        
        # Get parameters from ROS parameter server with defaults
        self.port = rospy.get_param('~port', '/dev/ttyACM0')
        self.num_mags = rospy.get_param('~num_mags', 5)
        self.baudrate = rospy.get_param('~baudrate', 115200)
        self.publish_rate = rospy.get_param('~publish_rate', 60)  # Hz
        self.samples_per_publish = rospy.get_param('~samples_per_publish', 5)
    
        
        # Initialize the sensor process
        self.sensor = AnySkinProcess(
            num_mags=self.num_mags,
            port=self.port,
            baudrate=self.baudrate,
            temp_filtered=True,
            burst_mode=True
        )
        
        # Create publisher with stamped message
        # self.pub = rospy.Publisher('anyskin_data', StampedFloat32MultiArray, queue_size=10)
        self.pub = rospy.Publisher('anyskin_data', Float32MultiArray, queue_size=10)
        
        # Start the sensor
        self.sensor.start()
        rospy.sleep(1)  # Give time for sensor to initialize
        self.sensor.start_streaming()
        
        rospy.loginfo(f"AnySkin publisher initialized on port {self.port}")
        self.baseline = np.zeros((self.num_mags, 3))
    def _get_baseline(self):
        samples = self.sensor.get_data_wo_time(
                        num_samples=self.samples_per_publish)
        if samples:
            samples = np.array(samples).reshape(self.samples_per_publish,self.num_mags, 3)
            samples = np.mean(samples, axis=0)
            self.baseline = samples

    def publish_data(self):
        """
        Main publishing loop that reads sensor data and publishes it
        """
        rate = rospy.Rate(self.publish_rate)
        
        try:
            while not rospy.is_shutdown():
                try:
                    # Get samples from the sensor
                    samples = self.sensor.get_data_wo_time(
                        num_samples=self.samples_per_publish)
                    
                    if samples:
                        # Convert samples to numpy array
                        data_array = np.array(samples).reshape(self.samples_per_publish,self.num_mags, 3)
                        data_array = data_array - self.baseline.reshape(1,self.num_mags,3)
                        msg = Float32MultiArray()
                        msg.data = data_array.flatten().tolist()

                        
                        # Create message with header
                        # msg = StampedFloat32MultiArray()
                        # msg.header = Header()
                        # msg.header.stamp = rospy.Time.now()
                        
                        # # 设置 array 字段
                        # msg.array = Float32MultiArray()
                        # msg.array.layout.dim = []  # Add empty dimension
                        # msg.array.layout.data_offset = 0
                        # msg.array.data = data_array.flatten().tolist()
                        
                        # Publish the message
                        self.pub.publish(msg)
                        
                    
                    rate.sleep()
                    
                except Exception as e:
                    rospy.logerr(f"Error in publishing loop: {str(e)}")
                    break
        finally:
            # Ensure cleanup happens
            self.shutdown()
    
    def shutdown(self):
        """
        Clean shutdown of the publisher
        """
        rospy.loginfo("Shutting down AnySkin publisher...")
        if hasattr(self, 'sensor'):
            self.sensor.pause_streaming()  # Stop the streaming first
            rospy.sleep(0.5)  # Give some time for the streaming to stop
            self.sensor.join()  # Then join the process

class CmdlineListener:
    def __init__(self, get_baseline_fn: callable):
        """Initialize a command-line listener thread.
        
        Args:
            get_baseline_fn: Function to call when baseline calibration is requested
        """
        self.get_baseline_fn = get_baseline_fn
        self.thread = threading.Thread(target=self._listener)
        self.thread.daemon = True
        self.thread.start()
        
    def _listener(self):
        """Listen for user commands from the terminal."""
        rospy.loginfo("Command-line listener started. Available commands:")
        rospy.loginfo("  'b' or 'baseline': Re-calibrate baseline")
        rospy.loginfo("  'q' or 'quit': Quit the program")
        
        while not rospy.is_shutdown():
            try:
                cmd = input().strip().lower()
                if cmd in ['b', 'baseline']:
                    rospy.loginfo("Recalibrating baseline...")
                    self.get_baseline_fn()
                    rospy.loginfo("Baseline recalibration complete")
                elif cmd in ['q', 'quit']:
                    rospy.loginfo("Quit command received, shutting down...")
                    rospy.signal_shutdown("User requested shutdown")
                else:
                    rospy.loginfo(f"Unknown command: '{cmd}'")
                    rospy.loginfo("Available commands: 'b'/'baseline', 'q'/'quit'")
            except Exception as e:
                rospy.logerr(f"Error in command-line listener: {e}")

def main():
    publisher = None
    try:
        publisher = AnySkinPublisher()
        # Start the command-line listener
        cmd_listener = CmdlineListener(publisher._get_baseline)
        publisher.publish_data()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Caught keyboard interrupt, shutting down...")
    finally:
        if publisher is not None:
            publisher.shutdown()

if __name__ == '__main__':
    main()
