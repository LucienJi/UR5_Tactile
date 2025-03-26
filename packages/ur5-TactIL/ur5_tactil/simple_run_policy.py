#!/usr/bin/env python3
"""
Simplified neural network-based control policy for UR5 robot.
This script loads a trained policy and uses it to control the robot in real-time,
with a simple sequential execution flow instead of multithreading.
"""

import os
import time
import numpy as np
import rospy
import tf
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist, Point, Pose, Quaternion
from std_msgs.msg import Float32MultiArray
from ur5_twist_control.helper import ori2numpy, point2numpy, vec32numpy
from ur5_twist_control.twist_control import TwistControl
import message_filters
from collections import deque
import torch
import threading
import sys

# Bridge for converting ROS Image messages to OpenCV images
bridge = CvBridge()

# Default initial position and orientation
INIT_POS = np.array([-0.0139, -0.4602, 0.2653])
INIT_QUAT = np.array([0.9434, -0.3310, -0.012, 0.015])

class LatencyTracker:
    """Tracks and reports latency statistics for different stages of the pipeline."""
    def __init__(self, enabled=True, log_interval=10.0, log_file=None):
        self.enabled = enabled
        self.log_interval = log_interval  # Log interval in seconds
        self.stage_times = {}
        self.stage_latencies = {}
        self.stage_stats = {}
        self.last_log_time = time.time()
        
        # Create log file if specified
        self.log_file = log_file
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(self.log_file, 'w') as f:
                f.write("timestamp,stage,latency\n")
    
    def start_stage(self, stage_name):
        """Start timing a stage of the pipeline."""
        if not self.enabled:
            return
        
        self.stage_times[stage_name] = time.time()
    
    def end_stage(self, stage_name):
        """End timing a stage and record its latency."""
        if not self.enabled or stage_name not in self.stage_times:
            return
        
        latency = time.time() - self.stage_times[stage_name]
        
        if stage_name not in self.stage_latencies:
            self.stage_latencies[stage_name] = []
        
        self.stage_latencies[stage_name].append(latency)
        
        # Log to file if enabled
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()},{stage_name},{latency}\n")
        
        # Check if it's time to log stats
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.compute_and_log_stats()
            self.last_log_time = current_time
    
    def compute_and_log_stats(self):
        """Compute and log statistics for all stages."""
        if not self.enabled:
            return
        
        for stage, latencies in self.stage_latencies.items():
            if not latencies:
                continue
                
            # Compute statistics
            latency_array = np.array(latencies)
            stats = {
                'mean': float(np.mean(latency_array)),
                'median': float(np.median(latency_array)),
                'min': float(np.min(latency_array)),
                'max': float(np.max(latency_array)),
                'std': float(np.std(latency_array)),
                'p95': float(np.percentile(latency_array, 95)),
                'p99': float(np.percentile(latency_array, 99)),
                'count': len(latencies)
            }
            
            self.stage_stats[stage] = stats
            
            # Clear latencies to avoid memory growth
            self.stage_latencies[stage] = []
            
            # Log the statistics
            rospy.loginfo(f"Latency stats for {stage}: "
                         f"mean={stats['mean']:.4f}s, "
                         f"median={stats['median']:.4f}s, "
                         f"min={stats['min']:.4f}s, "
                         f"max={stats['max']:.4f}s, "
                         f"p95={stats['p95']:.4f}s, "
                         f"p99={stats['p99']:.4f}s, "
                         f"count={stats['count']}")
    
    def get_stats(self, stage_name=None):
        """Get statistics for a specific stage or all stages."""
        if not self.enabled:
            return {}
            
        if stage_name:
            return self.stage_stats.get(stage_name, {})
        else:
            return self.stage_stats

class SimplePolicyRunner:
    def __init__(self, policy=None, data_transformer=None, ros_rate=30, policy_freq=10):
        """
        Initialize the simplified policy runner.
        
        Args:
            policy: The neural network policy to use for control
            data_transformer: Transformer for normalizing inputs and denormalizing outputs
            ros_rate: Rate for the main ROS loop (Hz)
            policy_freq: Frequency at which to run the policy (Hz)
        """
        rospy.init_node('ur5_simple_policy_runner')
        
        # Override defaults with ROS parameters if available
        self.ros_rate = rospy.get_param('~ros_rate', ros_rate)
        self.policy_freq = rospy.get_param('~policy_freq', policy_freq)
        self.use_gpu = rospy.get_param('~use_gpu', True)
        self.debug_mode = rospy.get_param('~debug_mode', False)
        
        # Set device for computation
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Create rate objects
        self.rate = rospy.Rate(self.ros_rate)
        self.policy_rate = rospy.Rate(self.policy_freq)
        
        
        
        # Initialize TF listener for getting robot pose
        self._tf_listener = tf.TransformListener()
        
        # Initialize twist controller
        self.twist_controller = TwistControl(init_arm=True)
        ## send zero velocity to the robot, zero grasp
        self.twist_controller.move_vel(np.zeros(3), np.zeros(3))
        self.twist_controller.gripper.command.rPRA = int(255 * 0.0)
        self.twist_controller.gripper.publish_command(self.twist_controller.gripper.command)
        
        # Initialize state variables
        self.joint_states = None
        self.current_image = None
        self.current_tactile = None
        self.gripper_pos = None
        self.gripper_quat = None
        
        # For smoothing control outputs
        self.lin_vel_history = deque(maxlen=3)
        self.ang_vel_history = deque(maxlen=3)
        
        # Load policy if provided
        self.policy = policy
        self.data_transformer = data_transformer
        
        # Prepare the policy for execution
        self.prepare_policy()
        
        # Set up subscribers for the same topics used in record.py
        self._sub_joint_angles = rospy.Subscriber(
            '/joint_states', JointState, self._joints_callback, queue_size=1
        )
        
        # Set up synchronized subscribers for camera and tactile data
        self._sub_camera = message_filters.Subscriber('/outer_cam/color/image_raw', Image)
        self._sub_tactile = message_filters.Subscriber('/anyskin_data', Float32MultiArray)
        
        # Synchronize camera and tactile data
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self._sub_camera, self._sub_tactile],
            queue_size=10,
            slop=0.1,  # 100ms time difference tolerance
            allow_headerless=True  # Allow messages without headers
        )
        self.ts.registerCallback(self._sensor_callback)
        
        # Publisher for twist commands
        self._pub_twist = rospy.Publisher('/twist_controller/command', Twist, queue_size=1)
        
        # Flag to control execution
        self.running = False
        
        # Start command-line listener thread (just for user input, not for processing)
        self.cmd_listener = CmdlineListener(
            start_fn=self.start_policy,
            stop_fn=self.stop_policy,
            gohome_fn=self.go_home,
            debug_fn=self._toggle_debug_mode
        )
        
        # Initialize CUDA if using GPU
        if self.use_gpu and torch.cuda.is_available():
            self._setup_gpu()
        
        rospy.loginfo("SimplePolicyRunner initialized")
        
        if self.device.type == "cuda":
            rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            rospy.loginfo("Using CPU for computation")

    def _toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.debug_mode = not self.debug_mode
        rospy.loginfo(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        

    def _setup_gpu(self):
        """Set up GPU for inference"""
        try:
            # Set CUDA options for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Pre-allocate tensors on GPU for reuse
            self.gpu_tensors = {
                'image': torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=self.device),
                'tactile': torch.zeros(1, 5, 5, 3, dtype=torch.float32, device=self.device),
                'proprioception': torch.zeros(1, 13, dtype=torch.float32, device=self.device)
            }
            
            # Warm up GPU
            dummy_tensor = torch.zeros(1, 3, 224, 224, device=self.device)
            for _ in range(10):
                _ = dummy_tensor + dummy_tensor
            
            rospy.loginfo("GPU initialized successfully")
        except Exception as e:
            rospy.logerr(f"Error initializing GPU: {e}")
            self.device = torch.device("cpu")
            self.use_gpu = False

    def mv_initial_pose(self, initial_p, initial_quat, exec_time=2, change_control_method=True):
        """Move the robot to an initial pose."""
        ori = Quaternion(x=initial_quat[0], y=initial_quat[1], z=initial_quat[2], w=initial_quat[3])
        pos = Point(initial_p[0], initial_p[1], initial_p[2])
        self.twist_controller._move_to(pos, ori, exec_time=exec_time, spawn_twistvel_controller=change_control_method)
        rospy.loginfo("Move to initial pose complete")
    
    def prepare_policy(self):
        """Prepare the policy for inference."""
        try:
            if self.policy is None:
                rospy.logwarn("No policy provided")
                return
                
            rospy.loginfo("Preparing policy for execution")
            
            # Set policy to evaluation mode if it's a torch model
            if hasattr(self.policy, 'eval'):
                self.policy.eval()
                
            # Use torch inference mode for better performance
            if hasattr(torch, 'inference_mode'):
                self.inference_context = torch.inference_mode()
            else:
                self.inference_context = torch.no_grad()
            
            # Try to optimize with TorchScript if possible
            try:
                example_input = {
                    'observations': {
                        'image': torch.zeros(1, 3, 224, 224, device=self.device),
                        'tactile': torch.zeros(1, 5, 5, 3, device=self.device),
                        'proprioception': {
                            'joint_states': torch.zeros(1, 6, device=self.device),
                            'gripper_pos': torch.zeros(1, 3, device=self.device),
                            'gripper_quat': torch.zeros(1, 4, device=self.device)
                        }
                    }
                }
                self.policy = torch.jit.trace(self.policy, example_input)
                rospy.loginfo("Policy optimized with TorchScript")
            except Exception as e:
                rospy.logwarn(f"Could not optimize policy with TorchScript: {e}")
                
            rospy.loginfo("Policy prepared successfully")
        except Exception as e:
            rospy.logerr(f"Failed to prepare policy: {str(e)}")

    def _joints_callback(self, msg: JointState):
        """Callback for joint states."""
        self.joint_states = msg.position

    def _sensor_callback(self, image_msg, tactile_msg):
        """Callback for synchronized camera and tactile data."""
        try:
            # Process image data
            current_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            current_image = cv2.resize(current_image, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Process tactile data
            current_tactile = np.array(tactile_msg.data)
            
            # Store the processed data
            self.current_image = current_image
            self.current_tactile = current_tactile
        except Exception as e:
            rospy.logerr(f"Error in sensor callback: {str(e)}")

    def get_curr_pose(self):
        """Get the current pose of the robot end-effector."""
        source_frame = 'ur_arm_tool0_controller'
        target_frame = '/ur_arm_base'
        timeout = 1.0
        try:
            self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
            trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            
            self.gripper_pos = np.array(trans)
            self.gripper_quat = np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {str(e)}")

    def collect_data(self):
        """Collect all necessary data for policy execution."""
        # Start timing data collection
        
        # Get current robot pose
        self.get_curr_pose()
        
        # Check if we have all the necessary data
        if (self.current_image is None or 
            self.current_tactile is None or 
            self.joint_states is None or
            self.gripper_pos is None or
            self.gripper_quat is None):
                rospy.logwarn_throttle(1.0, "Missing data for policy inference")
                return None
        
        # Create input data dictionary
        input_data = {
            'image': self.current_image.copy(),
            'tactile': self.current_tactile.copy(),
            'proprioception': np.concatenate([
                self.gripper_pos, 
                self.gripper_quat,
                self.joint_states
            ], axis=-1),
            'timestamp': rospy.Time.now()
        }
        
        # End timing data collection

        
        return input_data

    def prepare_data_for_policy(self, input_data):
        """Prepare collected data for policy inference."""
        # Start timing data preparation
        s_time = rospy.Time.now()
        try:
            # Image: HWC -> CHW format and normalize
            img_np = input_data['image'].astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
            image_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
            
            # Tactile data
            tactile_np = input_data['tactile'].astype(np.float32).reshape(5, 5, 3).transpose(1, 0, 2)
            tactile_tensor = torch.from_numpy(tactile_np).unsqueeze(0).to(self.device)
            
            # Proprioception data
            proprioceptions_np = input_data['proprioception'].astype(np.float32)
            proprioceptions_tensor = torch.from_numpy(proprioceptions_np).unsqueeze(0).to(self.device)
            
            # Create policy input dictionary
            policy_input = {
                'observations': {
                    'image': [image_tensor],
                    'tactile': tactile_tensor,
                    'proprioception': proprioceptions_tensor,
                },
                'timestamp': input_data['timestamp'],
                'prep_complete_time': rospy.Time.now()
            }
            
            # End timing data preparation
            e_time = rospy.Time.now()
            rospy.loginfo(f"Data preparation time: {e_time - s_time}")
            return policy_input
        except Exception as e:
            rospy.logerr(f"Error in data preparation: {str(e)}")
            return None

    def execute_policy(self, policy_input):
        """Execute the policy on the prepared input data."""
        # Start timing inference
        s_time = rospy.Time.now()
        try:
            if self.policy is None:
                rospy.logwarn_throttle(5.0, "No policy available for execution")
                policy_output = {
                    'lin_vel': np.zeros(3),
                    'ang_vel': np.zeros(3),
                    'gripper_cmd': 0.0
                }
            else:
                # Normalize inputs if data transformer is available
                
                if self.data_transformer:
                    policy_input['observations'] = self.data_transformer.normalize_observation(policy_input['observations'])
                
                # Execute policy inference with performance optimizations
                with self.inference_context:
                    policy_output_tensor = self.policy(policy_input['observations'])
                    
                    # Denormalize outputs if data transformer is available
                    if self.data_transformer:
                        policy_output_tensor = self.data_transformer.denormalize_action(policy_output_tensor)
                    
                    # Convert tensor output to numpy
                    try:
                        # Assuming policy_output_tensor has shape [1, N_chunks, N_actions]
                        policy_output_np = policy_output_tensor.cpu().numpy()[0][0]
                        
                        policy_output = {
                            'lin_vel': policy_output_np[1:4],  # Linear velocity
                            'ang_vel': policy_output_np[4:7],  # Angular velocity
                            'gripper_cmd': policy_output_np[0]  # Gripper command
                        }
                    except Exception as e:
                        rospy.logerr(f"Error processing policy output: {e}")
                        policy_output = {
                            'lin_vel': np.zeros(3),
                            'ang_vel': np.zeros(3),
                            'gripper_cmd': 0.0
                        }
            
            # Add timestamps for latency tracking
            policy_output['input_timestamp'] = policy_input['timestamp']
            policy_output['prep_complete_time'] = policy_input['prep_complete_time']
            policy_output['output_timestamp'] = rospy.Time.now()
            e_time = rospy.Time.now()
            
            ## human time
            rospy.loginfo(f"Inference time: {(e_time - s_time).to_sec()} seconds")
            
            # End timing inference
            
            return policy_output
        except Exception as e:
            rospy.logerr(f"Error in policy execution: {str(e)}")
            
            # Return zero commands in case of failure
            return {
                'lin_vel': np.zeros(3),
                'ang_vel': np.zeros(3),
                'gripper_cmd': 0.0,
                'input_timestamp': policy_input['timestamp'] if 'timestamp' in policy_input else rospy.Time.now(),
                'prep_complete_time': policy_input.get('prep_complete_time', rospy.Time.now()),
                'output_timestamp': rospy.Time.now()
            }

    def execute_control(self, policy_output):
        """Execute control based on policy output."""
        # Start timing control execution
        
        try:
            # Extract control commands
            lin_vel = policy_output.get('lin_vel', np.zeros(3))
            ang_vel = policy_output.get('ang_vel', np.zeros(3))
            gripper_cmd = policy_output.get('gripper_cmd', 0.0)
            
            # Apply safety limiting for gripper command
            print(f"gripper_cmd: {gripper_cmd}")
            gripper_cmd = 0.0
            
            # Calculate and log latency if in debug mode
            if self.debug_mode and 'input_timestamp' in policy_output:
                total_latency = (policy_output['output_timestamp'] - policy_output['input_timestamp']).to_sec()
                prep_latency = (policy_output['prep_complete_time'] - policy_output['input_timestamp']).to_sec()
                inference_latency = (policy_output['output_timestamp'] - policy_output['prep_complete_time']).to_sec()
                
                if total_latency > 0.1:  # Log only if latency is high
                    rospy.logwarn_throttle(1.0, f"High total latency: {total_latency:.3f}s "
                                              f"(prep: {prep_latency:.3f}s, "
                                              f"inference: {inference_latency:.3f}s)")
            
            # Apply smoothing to velocity commands
            self.lin_vel_history.append(lin_vel)
            self.ang_vel_history.append(ang_vel)
            
            smoothed_lin_vel = np.mean(self.lin_vel_history, axis=0) if len(self.lin_vel_history) > 0 else lin_vel
            smoothed_ang_vel = np.mean(self.ang_vel_history, axis=0) if len(self.ang_vel_history) > 0 else ang_vel
            
            # Apply velocity commands to the robot
            # self.twist_controller.move_vel(smoothed_lin_vel, smoothed_ang_vel)
            self.twist_controller.move_vel(np.zeros(3), np.zeros(3))
            
            # Apply gripper commands if available
            if gripper_cmd is not None:
                self.twist_controller.gripper.command.rPRA = int(255 * gripper_cmd)
                self.twist_controller.gripper.publish_command(self.twist_controller.gripper.command)
            
            # End timing control execution
            
            # Log velocities periodically
            rospy.loginfo_throttle(1.0, f"Linear velocity: {smoothed_lin_vel}, Angular velocity: {smoothed_ang_vel}")
            
        except Exception as e:
            rospy.logerr(f"Error in control execution: {str(e)}")
            
            # Stop robot on error
            self.twist_controller.move_vel(np.zeros(3), np.zeros(3))

    def policy_loop(self):
        """Main policy execution loop."""
        while self.running and not rospy.is_shutdown():
            try:
                # 1. Collect data
                input_data = self.collect_data()
                if input_data is None:
                    self.policy_rate.sleep()
                    continue
                
                # 2. Prepare data for policy
                policy_input = self.prepare_data_for_policy(input_data)
                if policy_input is None:
                    self.policy_rate.sleep()
                    continue
                
                # 3. Execute policy
                policy_output = self.execute_policy(policy_input)
                
                # 4. Execute control
                self.execute_control(policy_output)
                
            except Exception as e:
                rospy.logerr(f"Error in policy loop: {str(e)}")
            
            # Sleep to maintain desired frequency
            self.policy_rate.sleep()

    def start_policy(self):
        """Start running the policy execution loop."""
        if not self.running:
            if self.policy is None and not isinstance(self.policy, DummyPolicy):
                rospy.logwarn("No policy loaded, cannot start")
                return
            
            self.running = True
            rospy.loginfo("Policy execution started")
        else:
            rospy.loginfo("Policy is already running")

    def stop_policy(self):
        """Stop running the policy."""
        if self.running:
            self.running = False
            
            # Stop the robot
            self.twist_controller.move_vel(np.zeros(3), np.zeros(3))

            
            rospy.loginfo("Policy execution stopped")
        else:
            rospy.loginfo("Policy is not running")

    def go_home(self):
        """Move the robot to the home position."""
        rospy.loginfo("Moving to home position...")
        
        # Stop policy execution if running
        was_running = self.running
        if was_running:
            self.stop_policy()
        
        # Move to the home position
        try:
            self.mv_initial_pose(INIT_POS, INIT_QUAT, exec_time=2, change_control_method=True)
        except Exception as e:
            rospy.logerr(f"Failed to move to home position: {str(e)}")
        
        # Restart policy if it was running before
        if was_running:
            self.start_policy()

    def run(self):
        """Main execution loop."""
        rospy.loginfo("SimplePolicyRunner is running.")
        rospy.loginfo("Available commands:")
        rospy.loginfo("  's': Start policy execution")
        rospy.loginfo("  'e': Stop policy execution")
        rospy.loginfo("  'g': Go to home position")
        rospy.loginfo("  'd': Toggle debug mode (latency tracking)")
        rospy.loginfo("  'q': Quit the program")
        
        while not rospy.is_shutdown():
            if self.running:
                self.policy_loop()
            else:
                # If not running, just sleep to save CPU
                self.rate.sleep()


class CmdlineListener:
    """Handles command-line input from the user."""
    def __init__(self, start_fn, stop_fn, gohome_fn, debug_fn):
        """
        Initialize a command-line listener thread.
        
        Args:
            start_fn: Function to call when starting policy
            stop_fn: Function to call when stopping policy
            gohome_fn: Function to call when going to home position
            debug_fn: Function to toggle debug mode
        """
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.gohome_fn = gohome_fn
        self.debug_fn = debug_fn
        
        self.thread = threading.Thread(target=self._listener)
        self.thread.daemon = True
        self.thread.start()
        
    def _listener(self):
        """Listen for user commands from the terminal."""
        rospy.loginfo("Command-line listener started. Available commands:")
        rospy.loginfo("  's': Start policy execution")
        rospy.loginfo("  'e': Stop policy execution")
        rospy.loginfo("  'g': Go to home position")
        rospy.loginfo("  'd': Toggle debug mode (latency tracking)")
        rospy.loginfo("  'q': Quit the program")
        
        while not rospy.is_shutdown():
            try:
                cmd = input().strip().lower()
                if cmd == 's':
                    rospy.loginfo("Starting policy execution...")
                    self.start_fn()
                elif cmd == 'e':
                    rospy.loginfo("Stopping policy execution...")
                    self.stop_fn()
                elif cmd == 'g':
                    rospy.loginfo("Going to home position...")
                    self.gohome_fn()
                elif cmd == 'd':
                    rospy.loginfo("Toggling debug mode...")
                    self.debug_fn()
                elif cmd == 'q':
                    rospy.loginfo("Quit command received, shutting down...")
                    rospy.signal_shutdown("User requested shutdown")
                    break
                else:
                    rospy.loginfo(f"Unknown command: '{cmd}'")
                    rospy.loginfo("Available commands: 's', 'e', 'g', 'd', 'q'")
            except Exception as e:
                rospy.logerr(f"Error in command-line listener: {e}")




if __name__ == '__main__':
    # Try to import required modules for TactIL
    try:
        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(script_dir)
        
        # Add the current directory to the path so Python can find the TactIL module
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Now we can import from TactIL
        from TactIL.tester import Tester
        from TactIL.utils.train_utils import load_config
        
        # Build the config path relative to the script location
        config_path = os.path.join(script_dir, "TactIL/config/config.yaml")
        
        # Load config and create tester
        config = load_config(config_path)
        tester = Tester(config)
        
        # Create and run the policy runner with the trained model
        policy_runner = SimplePolicyRunner(
            policy=tester.model,
            data_transformer=tester.data_transformer,
            ros_rate=30,
            policy_freq=10,
            # latency_log_file=os.path.join(script_dir, "latency_log.txt")
        )
        
        try:
            policy_runner.run()
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            rospy.loginfo("Caught keyboard interrupt, shutting down...")
        finally:
            # Ensure the robot stops
            if hasattr(policy_runner, 'twist_controller'):
                policy_runner.twist_controller.move_vel(np.zeros(3), np.zeros(3))
    
    except ImportError as e:
        # Fall back to dummy policy if imports fail
        rospy.logerr(f"Failed to import TactIL modules: {e}")
        rospy.logerr("Falling back to dummy policy for testing")
        main() 