roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.131.245 tcp_port:=10000
the tcp ip and tcp port should be the same as the Q2R in the quest headset


1. maximum freq for tactile is  around 360hz, we can downsample them.
2. Quest3: roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.8.159  tcp_port:=22222
3. rosrun quest2ros ros2quest.py
4. husky_node also publish "---
header: 
  seq: 6082871
  stamp: 
    secs: 1740950879
    nsecs: 170990454
  frame_id: ''
name: 
  - front_left_wheel
  - front_right_wheel
  - rear_left_wheel
  - rear_right_wheel
position: [0.0, 0.0, 0.0, 0.0]
velocity: [0.0, 0.0, 0.0, 0.0]
effort: [0.0, 0.0, 0.0, 0.0]
---
"

pos: -0.0139, -0.4602,0.2653
quat: 0.9434, -0.3310,-0.012, 0.015