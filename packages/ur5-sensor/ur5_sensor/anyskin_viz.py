#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Header
from ur5_sensor.msg import StampedFloat32MultiArray  # 导入自定义消息类型
from anyskin import AnySkinProcess
import pygame
import os
import time
import pygame

class AnySkinViz:
    """
    Visualization class for AnySkin sensor data.
    """
    def __init__(self, ros_rate,num_mags=5,num_samples=5):
        rospy.init_node('anyskin_viz', anonymous=True)
        self.ros_rate = ros_rate
        self.num_mags = num_mags
        self.num_samples = num_samples
        self.data = np.zeros((self.num_samples, self.num_mags, 3))
        # self.sub = rospy.Subscriber('anyskin_data', StampedFloat32MultiArray, self.callback)
        self.sub = rospy.Subscriber('anyskin_data', Float32MultiArray, self.callback)
        self.rate = rospy.Rate(self.ros_rate)
        # Initialize the ROS node
        bg_image_path = "/code/src/ur5-tactile/packages/ur5-sensor/ur5_sensor/anyskin/visualizations/images/viz_bg.png"
        # bg_image = plt.imread("anyskin.png")
        bg_image = pygame.image.load(bg_image_path)
        image_width, image_height = bg_image.get_size()
        aspect_ratio = image_height / image_width
        desired_width = 400
        desired_height = int(desired_width * aspect_ratio)
        self.scaling = 7.0
        self.viz_mode = '3axis'
        self.chip_locations = np.array(
            [
                [204, 222],  # center
                [130, 222],  # left
                [279, 222],  # right
                [204, 157],  # up
                [204, 290],  # down
            ]
        )
        self.chip_xy_rotations = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0])

        # Resize the background image to the new dimensions
        bg_image = pygame.transform.scale(bg_image, (desired_width, desired_height))
        # Create the pygame display window
        self.window = pygame.display.set_mode((desired_width, desired_height), pygame.SRCALPHA)
        self.background_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)
        self.background_color = (234, 237, 232, 255)
        self.background_surface.fill(self.background_color)
        self.background_surface.blit(bg_image, (0, 0))
        pygame.display.set_caption("Sensor Data Visualization")
        self.baseline = np.zeros((self.num_mags, 3))
        self._get_baseline()
        
    def callback(self, msg):
        data = np.array(msg.data).reshape(self.num_samples,self.num_mags, 3)
        self.data = data
        
    def _get_baseline(self):
        data = self.data
        baseline_data = np.mean(data, axis=0)
        self.baseline = baseline_data
    
    def visualize_data(self, data):
        data = data.reshape(-1, 3)
        data_mag = np.linalg.norm(data, axis=1)
        # print(angles)
        # Draw the chip locations
        for magid, chip_location in enumerate(self.chip_locations):
            if self.viz_mode == "magnitude":
                pygame.draw.circle(
                    self.window, (255, 83, 72), chip_location, data_mag[magid] / self.scaling
                )
            elif self.viz_mode == "3axis":
                if data[magid, -1] < 0:
                    width = 2
                else:
                    width = 0
                pygame.draw.circle(
                    self.window,
                    (255, 0, 0),
                    chip_location,
                    np.abs(data[magid, -1]) / self.scaling,
                    width,
                )
                arrow_start = chip_location
                rotation_mat = np.array(
                    [
                        [
                            np.cos(self.chip_xy_rotations[magid]),
                            -np.sin(self.chip_xy_rotations[magid]),
                        ],
                        [
                            np.sin(self.chip_xy_rotations[magid]),
                            np.cos(self.chip_xy_rotations[magid]),
                        ],
                    ]
                )
                data_xy = np.dot(rotation_mat, data[magid, :2])
                arrow_end = (
                    chip_location[0] + data_xy[0] / self.scaling,
                    chip_location[1] + data_xy[1] / self.scaling,
                )
                pygame.draw.line(self.window, (0, 255, 0), arrow_start, arrow_end, 2)

    
    def publish(self):
        running = True
        clock = pygame.time.Clock()
        FPS  = self.ros_rate
        while not rospy.is_shutdown() and running:
            self.window.blit(self.background_surface, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    print(f"Mouse clicked at ({x}, {y})")
            # Check if user pressed b
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        self._get_baseline()
            self.visualize_data(self.data[-1,:,:] - self.baseline)
            pygame.display.update()
            clock.tick(FPS)
            self.rate.sleep()
        pygame.quit()

if __name__ == '__main__':
    ros_rate = 60
    anyskin_viz = AnySkinViz(ros_rate)
    ## sleep for a while to get the sensor data
    time.sleep(5)
    anyskin_viz._get_baseline()
    anyskin_viz.publish()
    rospy.spin()
            
