#!/usr/bin/env python3
"""
Neural network-based control policy for UR5 robot.
This script loads a trained policy and uses it to control the robot in real-time.
"""

import os
import time
import threading
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
from queue import Queue, Empty, Full
from collections import deque
import json
from datetime import datetime
import sys
import torch
# Bridge for converting ROS Image messages to OpenCV images
bridge = CvBridge()
# INIT_POS = np.array([0.296,-0.521,0.101])
INIT_POS = np.array([-0.0139, -0.4602,0.2653])
# INIT_QUAT = np.array([-0.389, 0.625,-0.116,0.667])
INIT_QUAT = np.array([0.9434, -0.3310,-0.012, 0.015])

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

class PolicyRunner:
    def __init__(self, policy=None, data_transformer=None, ros_rate=30, policy_freq=10, queue_size=10):
        """
        Initialize the policy runner.
        
        Args:
            policy: The neural network policy to use for control
            ros_rate: Rate for the main ROS loop (Hz)
            policy_freq: Frequency at which to run the policy (Hz)
            queue_size: Size of the data queue for producer-consumer pattern
        """
        rospy.init_node('ur5_policy_runner')
        
        # Override defaults with ROS parameters if available
        self.ros_rate = rospy.get_param('~ros_rate', ros_rate)
        self.policy_freq = rospy.get_param('~policy_freq', policy_freq)
        self.queue_size = rospy.get_param('~queue_size', queue_size)
        self.use_gpu = rospy.get_param('~use_gpu', True)
        self.debug_mode = rospy.get_param('~debug_mode', False)
        self.latency_log_file = rospy.get_param('~latency_log_file', None)
        self.device = torch.device("cuda")
        # Create rate objects
        self.rate = rospy.Rate(self.ros_rate)
        
        # Initialize latency tracker
        self.latency_tracker = LatencyTracker(
            enabled=self.debug_mode,
            log_interval=5.0,  # Log every 5 seconds in debug mode
            log_file=self.latency_log_file
        )
        
        # Initialize TF listener for getting robot pose
        self._tf_listener = tf.TransformListener()
        
        # Initialize twist controller
        self.twist_controller = TwistControl(init_arm=True)
        
        # Initialize state variables
        self.joint_states = None
        self.current_image = None
        self.current_tactile = None
        self.gripper_pos = None
        self.gripper_quat = None
        
        # Create queues for multi-stage pipeline
        self.raw_data_queue = Queue(maxsize=self.queue_size * 2)        # Raw sensor data
        self.gpu_data_queue = Queue(maxsize=self.queue_size)         # GPU-ready tensors
        self.result_queue = Queue(maxsize=self.queue_size)          # Model outputs
    
        
        # Create locks for thread-safe access to shared resources
        self.state_lock = threading.Lock()
        
        # For smoothing control outputs
        self.lin_vel_history = deque(maxlen=3)
        self.ang_vel_history = deque(maxlen=3)
        # Initialize GPU if available and requested
        
        # Load policy if provided
        self.policy = policy
        self.data_transformer = data_transformer
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
        
        # Flags to control execution
        self.running = False
        self.threads = []
        
        # Initialize CUDA streams for overlapping operations (if using GPU)
        self.streams = []
        
        # Start command-line listener thread
        self.cmd_listener = CmdlineListener(
            start_fn=self.start_policy,
            stop_fn=self.stop_policy,
            gohome_fn=self.go_home,
            debug_fn=self._toggle_debug_mode
        )
        if self.use_gpu:
            self._setup_gpu()
        
        
        rospy.loginfo("PolicyRunner initialized")

    def _toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.debug_mode = not self.debug_mode
        self.latency_tracker.enabled = self.debug_mode
        rospy.loginfo(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        
        if self.debug_mode:
            self.latency_tracker.compute_and_log_stats()

    def _setup_gpu(self):
        """Set up GPU for inference if available"""
        try:
            # Try to import torch and set up GPU
            import torch
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                rospy.loginfo(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                # Set CUDA options for better performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Create CUDA streams for overlapping operations
                for i in range(3):  # Create 3 streams for different operations
                    self.streams.append(torch.cuda.Stream())
                
                # Allocate pinned memory for faster CPU-GPU transfers
                self.pinned_memory_pool = {
                    'image': torch.zeros(1, 3, 224, 224, dtype=torch.float32).pin_memory(),
                    'tactile': torch.zeros(1, 5,5,3, dtype=torch.float32).pin_memory(),
                    'proprioception': torch.zeros(1, 13, dtype=torch.float32).pin_memory(),
                }
                
                # Pre-allocate tensors on GPU for reuse
                self.gpu_tensors = {
                    'image': torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=self.device),
                    'tactile': torch.zeros(1, 5,5,3, dtype=torch.float32, device=self.device),
                    'proprioception':torch.zeros(1,13, dtype = torch.float32, device = self.device)
                }
                
                # Warm up GPU
                dummy_tensor = torch.zeros(1, 3, 224, 224, device=self.device)
                for _ in range(10):
                    _ = dummy_tensor + dummy_tensor
                
                rospy.loginfo("GPU initialized successfully with pinned memory and CUDA streams")
            else:
                self.device = torch.device("cpu")
                rospy.logwarn("GPU not available, using CPU instead")
        except ImportError:
            rospy.logwarn("PyTorch not found, GPU acceleration disabled")
            self.device = None
            self.use_gpu = False

    def mv_initial_pose(self, initial_p, initial_quat,exec_time=2,change_control_method = True):
        ori = Quaternion(x=initial_quat[0], y=initial_quat[1], z=initial_quat[2], w=initial_quat[3])
        pos = Point(initial_p[0], initial_p[1], initial_p[2])
        self.twist_controller._move_to(pos, ori, exec_time=exec_time, spawn_twistvel_controller=change_control_method)
        print("####### Move Complete #########")
    
    def prepare_policy(self):
        """
        Load a policy from a file.
        
        Args:
            policy_path: Path to the policy file
        """
        try:
            # This is a placeholder - replace with actual policy loading code
            rospy.loginfo(f"Prepare policy")
            
            # Example for PyTorch model loading
            if self.use_gpu:
                import torch
                self.policy.eval()  # Set to evaluation mode
                
                # Apply optimizations for inference
                if hasattr(torch, 'inference_mode'):
                    self.inference_context = torch.inference_mode()
                else:
                    self.inference_context = torch.no_grad()
                
                # Try to optimize with TorchScript if possible
                try:
                    example_input = {
                        'observations': {
                            'image': torch.zeros(1, 3, 224, 224, device=self.device),
                            'tactile': torch.zeros(1, 5,5,3, device=self.device),
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
                    rospy.logwarn(f"Could not optimize with TorchScript: {e}")
            else:
                # Load for CPU
                # self.policy = torch.load(policy_path, map_location='cpu')
                pass
                
            rospy.loginfo("Policy loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load policy: {str(e)}")
            self.policy = None

    def _joints_callback(self, msg: JointState):
        """Callback for joint states"""
        # with self.state_lock:
        self.joint_states = msg.position

    def _sensor_callback(self, image_msg, tactile_msg):
        """Callback for synchronized camera and tactile data"""
        try:
            current_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            current_image = cv2.resize(current_image, (224, 224), interpolation=cv2.INTER_AREA)
            current_tactile = np.array(tactile_msg.data)
            
            # with self.state_lock:
            self.current_image = current_image
            self.current_tactile = current_tactile
        except Exception as e:
            rospy.logerr(f"Error in sensor callback: {str(e)}")

    def get_curr_pose(self):
        """Get the current pose of the robot end-effector"""
        source_frame = 'ur_arm_tool0_controller'
        target_frame = '/ur_arm_base'
        timeout = 1.0
        try:
            self._tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(timeout))
            trans, rot = self._tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            
            # with self.state_lock:
            self.gripper_pos = np.array(trans)
            self.gripper_quat = np.array(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF Error: {str(e)}")

    def _data_collector(self):
        """Thread function to collect data and put it in the queue"""
        rate = rospy.Rate(self.policy_freq * 2)  # Collect data at twice the policy frequency
        
        while self.running and not rospy.is_shutdown():
            # Start timing data collection
            self.latency_tracker.start_stage('data_collection')
            
            # Get current robot state
            self.get_curr_pose()
            
            # Skip if we don't have all the necessary data
            # with self.state_lock:
            if (self.current_image is None or 
                self.current_tactile is None or 
                self.joint_states is None or
                self.gripper_pos is None or
                self.gripper_quat is None):
                    rospy.logwarn_throttle(1.0, "Missing data for policy inference")
                    rate.sleep()
                    continue
                
                # Create a copy of the current state
            policy_input = {
                    'image': self.current_image.copy(),
                    'tactile': self.current_tactile.copy(),
                    'proprioception': np.concatenate([ self.gripper_pos, self.gripper_quat,self.joint_states],axis = -1),
                    'timestamp': rospy.Time.now()
                }
            
            # End timing data collection
            self.latency_tracker.end_stage('data_collection')
            
            # Put the data in the queue, but don't block if queue is full (discard old data)
            try:
                self.raw_data_queue.put(policy_input, block=False)
            except Full:
                # Queue is full, log it and continue
                rospy.logwarn_throttle(2.0, "Raw data queue is full, dropping oldest item")
                try:
                    # Try to remove the oldest item and then add the new one
                    _ = self.raw_data_queue.get_nowait()
                    self.raw_data_queue.task_done()
                    self.raw_data_queue.put(policy_input, block=False)
                except:
                    # If any other error occurs, just continue
                    pass
                
            rate.sleep()

    def _data_preparation(self):
        """Thread function to prepare data for GPU inference"""
        import torch  # Import here to ensure it's only used if GPU is enabled
        
        while self.running and not rospy.is_shutdown():
            try:
                # Get data from the raw queue with timeout
                raw_data = self.raw_data_queue.get(timeout=1.0)
                
                # Start timing data preparation
                
                self.latency_tracker.start_stage('data_preparation')
                
                # Process data for GPU
                with torch.cuda.stream(self.streams[0]):
                    # Convert numpy arrays to tensors using pinned memory
                    # Image: HWC -> CHW format and normalize
                    img_np = raw_data['image'].astype(np.float32) / 255.0  # Normalize to [0,1]
                    img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
                    self.pinned_memory_pool['image'].copy_(torch.from_numpy(img_np))
                    image_tensor = self.pinned_memory_pool['image'].to(self.device, non_blocking=True)
                    
                    # Tactile data
                    tactile_np = raw_data['tactile'].astype(np.float32).reshape(5,5,3).transpose(1,0,2)
                    self.pinned_memory_pool['tactile'].copy_(torch.from_numpy(tactile_np))
                    tactile_tensor = self.pinned_memory_pool['tactile'].to(self.device, non_blocking=True)
                    
                    # proprioception
                    proprioceptions_np = raw_data['proprioception'].astype(np.float32)
                    self.pinned_memory_pool['proprioception'].copy_(torch.from_numpy(proprioceptions_np))
                    proprioceptions_tensor = self.pinned_memory_pool['proprioception'].to(self.device, non_blocking=True)
                    
                    # Create GPU-ready input dictionary
                    
                    gpu_ready_input = {
                        'observations': {
                            'image': [image_tensor],
                            'tactile': tactile_tensor,
                            'proprioception': proprioceptions_tensor,
                        },
                        'timestamp': raw_data['timestamp'],
                        'prep_complete_time': rospy.Time.now()
                    }
                    
                    
                
                # Ensure all CUDA operations are complete
                torch.cuda.synchronize()
                
                # End timing data preparation
                self.latency_tracker.end_stage('data_preparation')
                
                # Put the GPU-ready data in the queue
                try:
                    self.gpu_data_queue.put(gpu_ready_input, block=False)
                except Full:
                    # Queue is full, log it and continue
                    rospy.logwarn_throttle(2.0, "GPU data queue is full, dropping oldest item")
                    try:
                        # Try to remove the oldest item and then add the new one
                        _ = self.gpu_data_queue.get_nowait()
                        self.gpu_data_queue.task_done()
                        self.gpu_data_queue.put(gpu_ready_input, block=False)
                    except:
                        # If any other error occurs, just continue
                        pass
                
                rospy.loginfo(f"Length of raw data queue: {self.raw_data_queue.qsize()}")
                # Mark raw data task as done
                self.raw_data_queue.task_done()
                
            except Empty:
                # Handle queue timeout
                pass
            except Exception as e:
                rospy.logerr(f"Error in data preparation: {str(e)}")

    def _policy_executor(self):
        """Thread function to run policy inference on GPU-ready data"""
        import torch  # Import here to ensure it's only used if GPU is enabled
        
        while self.running and not rospy.is_shutdown():
            try:
                # Get GPU-ready data from the queue with timeout
                gpu_input = self.gpu_data_queue.get(timeout=1.0)
                
                # Start timing inference
                self.latency_tracker.start_stage('inference')
                
                # Run inference with GPU optimization
                if self.policy:
                    try:
                        with self.inference_context, torch.cuda.stream(self.streams[1]):
                            
                            gpu_input['observations'] = self.data_transformer.normalize_observation(gpu_input['observations'])
                        
                                
                            policy_output_tensor = self.policy(gpu_input['observations'])
                            
                            policy_output_tensor = self.data_transformer.denormalize_action(policy_output_tensor)
                            # should be in shape (1, N_chunks, N_actions)
                            # Ensure all operations are complete
                            torch.cuda.synchronize()
                            
                            policy_output = policy_output_tensor.cpu().numpy()[0][0]
                            # Convert back to numpy (CPU)
                            policy_output = {
                                'lin_vel': policy_output[1:4],
                                'ang_vel': policy_output[4:7],
                                'gripper_cmd': policy_output[0]
                            }
                            raise Exception("Stop here for debug")
                    except Exception as e:
                        rospy.logerr(f"Policy inference error: {str(e)}")
                        policy_output = {
                            'lin_vel': np.zeros(3),
                            'ang_vel': np.zeros(3),
                            'gripper_cmd': 0.0
                        }
                else:
                    policy_output = {
                        'lin_vel': np.zeros(3),
                        'ang_vel': np.zeros(3),
                        'gripper_cmd': 0.0
                    }
                
                # End timing inference
                self.latency_tracker.end_stage('inference')
                
                # Add timestamps for latency tracking
                policy_output['input_timestamp'] = gpu_input['timestamp']
                policy_output['prep_complete_time'] = gpu_input['prep_complete_time']
                policy_output['output_timestamp'] = rospy.Time.now()
                
                # Put the result in the queue
                try:
                    self.result_queue.put(policy_output, block=False)
                except Full:
                    # Queue is full, log it and continue
                    rospy.logwarn_throttle(2.0, "Result queue is full, dropping oldest item")
                    try:
                        # Try to remove the oldest item and then add the new one
                        _ = self.result_queue.get_nowait()
                        self.result_queue.task_done()
                        self.result_queue.put(policy_output, block=False)
                    except:
                        # If any other error occurs, just continue
                        pass
                
                # Mark GPU data task as done
                self.gpu_data_queue.task_done()
                
            except Exception as e:
                rospy.logerr(f"Error in policy executor: {str(e)}")

    def _control_executor(self):
        """Thread function to apply control commands from policy output"""
        rate = rospy.Rate(self.policy_freq)
        
        while self.running and not rospy.is_shutdown():
            try:
                # Start timing control execution
                self.latency_tracker.start_stage('control_execution')
                
                # Log queue size at this point
                queue_size = self.result_queue.qsize()
                
                try:
                    # Get result from the queue with timeout
                    policy_output = self.result_queue.get(timeout=0.1)
                except Empty:
                    # Handle empty queue case gracefully
                    rospy.loginfo_throttle(5.0, "Waiting for policy output... (Queue empty)")
                    self.latency_tracker.end_stage('control_execution')
                    rate.sleep()
                    continue
                
                # Extract control commands
                lin_vel = policy_output.get('lin_vel', np.zeros(3))
                ang_vel = policy_output.get('ang_vel', np.zeros(3))
                gripper_cmd = policy_output.get('gripper_cmd', 0.0)
                gripper_cmd = (1 if gripper_cmd > 0 else 0) * 0.5 ### for security 
                
                # Calculate latency for each stage
                if self.debug_mode:
                    total_latency = (policy_output['output_timestamp'] - policy_output['input_timestamp']).to_sec()
                    prep_latency = (policy_output['prep_complete_time'] - policy_output['input_timestamp']).to_sec()
                    inference_latency = (policy_output['output_timestamp'] - policy_output['prep_complete_time']).to_sec()
                    
                    if total_latency > 0.1:  # Log if total latency is high
                        rospy.logwarn_throttle(1.0, f"High total latency: {total_latency:.3f}s "
                                                    f"(prep: {prep_latency:.3f}s, "
                                                    f"inference: {inference_latency:.3f}s)")
                
                # Apply smoothing to velocity commands
                
                self.lin_vel_history.append(lin_vel)
                self.ang_vel_history.append(ang_vel)
                
                smoothed_lin_vel = np.mean(self.lin_vel_history, axis=0) if len(self.lin_vel_history) > 0 else lin_vel
                smoothed_ang_vel = np.mean(self.ang_vel_history, axis=0) if len(self.ang_vel_history) > 0 else ang_vel
                
                # Apply velocity commands
                self.twist_controller.move_vel(smoothed_lin_vel, smoothed_ang_vel)
                
                # Apply gripper commands if available
                if gripper_cmd is not None:
                    self.twist_controller.gripper.command.rPRA = int(255 * gripper_cmd)
                    self.twist_controller.gripper.publish_command(self.twist_controller.gripper.command)
                
                # End timing control execution
                self.latency_tracker.end_stage('control_execution')
                
                rospy.loginfo_throttle(1.0, f"Queue size before/after: {queue_size}/{self.result_queue.qsize()}")
                # Mark task as done
                self.result_queue.task_done()
            
            except Exception as e:
                rospy.logerr(f"Error in control executor: {str(e)}")
            
            rate.sleep()

    def start_policy(self):
        """Start running the policy with multi-threading pipeline"""
        if not self.running:
            if self.policy is None and not isinstance(self.policy, DummyPolicy):
                rospy.logwarn("No policy loaded, cannot start")
                return
            
            self.running = True
            
            # Clear any existing data in queues
            for queue in [self.raw_data_queue, self.gpu_data_queue, self.result_queue]:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except:
                        break
            
            # Start threads for different stages of the pipeline
            self.threads = []
            
            # Data collector thread
            data_thread = threading.Thread(target=self._data_collector, name="DataCollector")
            data_thread.daemon = True
            data_thread.start()
            self.threads.append(data_thread)
            
            # Data preparation thread (only if using GPU)
            ## do not start this thread if the size of data queue is less than 10
            if self.use_gpu and self.raw_data_queue.qsize() > 10:
                prep_thread = threading.Thread(target=self._data_preparation, name="DataPreparation")
                prep_thread.daemon = True
                prep_thread.start()
                self.threads.append(prep_thread)
            
            # Policy executor thread
            ## we start this thread only when the queue is not empty 
            while not self.raw_data_queue.empty() and not self.gpu_data_queue.empty():
                time.sleep(0.01)
            rospy.loginfo("Starting policy executor thread")
            policy_thread = threading.Thread(target=self._policy_executor, name="PolicyExecutor")
            policy_thread.daemon = True
            policy_thread.start()
            self.threads.append(policy_thread)
            
            ## we start this thread only when the queue is not empty 
            while not self.result_queue.empty():
                time.sleep(0.01)
            rospy.loginfo("Starting control executor thread")
            # Control executor thread
            control_thread = threading.Thread(target=self._control_executor, name="ControlExecutor")
            control_thread.daemon = True
            control_thread.start()
            self.threads.append(control_thread)
            
            rospy.loginfo(f"Policy execution started with {len(self.threads)}-stage pipeline")
        else:
            rospy.loginfo("Policy is already running")

    def stop_policy(self):
        """Stop running the policy"""
        if self.running:
            self.running = False
            
            # Wait for threads to finish
            for thread in self.threads:
                thread.join(timeout=1.0)
            
            # Stop the robot
            self.twist_controller.move_vel(np.zeros(3), np.zeros(3))
            
            # Clear the threads list
            self.threads = []
            
            # Log final latency statistics
            if self.debug_mode:
                self.latency_tracker.compute_and_log_stats()
            
            rospy.loginfo("Policy execution stopped")
        else:
            rospy.loginfo("Policy is not running")

    def go_home(self):
        """Move the robot to the home position"""
        rospy.loginfo("Moving to home position...")
        
        # Stop policy execution if running
        was_running = self.running
        if was_running:
            self.stop_policy()
        
        # Call the home position service
        try:
            self.mv_initial_pose(INIT_POS, INIT_QUAT, exec_time=2, change_control_method=True)
        except Exception as e:
            rospy.logerr(f"Failed to move to home position: {str(e)}")
        
        # Restart policy if it was running before
        if was_running:
            self.start_policy()

    def run(self):
        """Main loop"""
        rospy.loginfo("PolicyRunner is running.")
        rospy.loginfo("Available commands:")
        rospy.loginfo("  's': Start policy execution")
        rospy.loginfo("  'e': Stop policy execution")
        rospy.loginfo("  'g': Go to home position")
        rospy.loginfo("  'd': Toggle debug mode (latency tracking)")
        rospy.loginfo("  'q': Quit the program")
        
        while not rospy.is_shutdown():
            # Main loop just keeps the node alive
            # Actual policy execution happens in separate threads
            self.rate.sleep()


class CmdlineListener:
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


class DummyPolicy:
    """
    A dummy policy for testing purposes.
    In a real implementation, this would be replaced with a trained neural network.
    """
    def __init__(self):
        self.counter = 0
    
    def forward(self, input_dict):
        """
        Run inference on the policy.
        
        Args:
            input_dict: Dictionary containing 'image', 'tactile', and 'proprioception' keys
            
        Returns:
            Dictionary containing 'lin_vel', 'ang_vel', and optionally 'gripper_cmd'
        """
        # Just a simple oscillating motion for testing
        self.counter += 1
        amplitude = 0.05
        frequency = 0.1
        
        # Simple sinusoidal motion in x direction
        lin_vel = np.array([amplitude * np.sin(frequency * self.counter), 0, 0])
        ang_vel = np.zeros(3)
        
        return {
            'lin_vel': lin_vel,
            'ang_vel': ang_vel,
            'gripper_cmd': 0.0  # Keep gripper open
        }


def main():
    # Create a dummy policy for testing
    dummy_policy = DummyPolicy()
    
    # Create the policy runner with the dummy policy
    policy_runner = PolicyRunner(
        policy=dummy_policy,
        ros_rate=30,
        policy_freq=10,
        queue_size=5
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


if __name__ == '__main__':
    # main()
    import os
    import sys
    
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    
    # Add the current directory to the path so Python can find the TactIL module
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Now we can import from TactIL regardless of how the script is run
    from TactIL.tester import Tester
    from TactIL.utils.train_utils import load_config
    
    # Build the config path relative to the script location
    config_path = os.path.join(script_dir, "TactIL/config/config.yaml")
    
    config = load_config(config_path)
    tester = Tester(config)
    
    policy_runner = PolicyRunner(
        policy=tester.model,
        data_transformer=tester.data_transformer,
        ros_rate=30,
        policy_freq=10,
        queue_size=10
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
    
