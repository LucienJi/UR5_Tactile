# UR5 Camera Calibration

This package provides tools for calibrating multiple cameras for use with a UR5 robot system. It supports:

1. **Intrinsic calibration** for all cameras using a checkerboard pattern
2. **Extrinsic calibration** for a fixed "major" camera using hand-eye calibration
3. **Online calibration** for movable cameras using AprilTags
4. **Camera management tools** for identifying and managing multiple cameras

## Requirements

- RealSense cameras (D400 series recommended)
- AprilTags for extrinsic calibration
- Checkerboard pattern for intrinsic calibration
- ROS environment with the UR5 robot

## Installation

1. Make sure you have all the required Python dependencies:
   ```bash
   pip install numpy opencv-python pyyaml apriltag pyrealsense2
   ```

2. Build the ROS package:
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

## Usage

The package provides the following scripts:

### Camera Calibration

#### 1. Intrinsic Calibration

Calibrate the internal parameters of a camera using a checkerboard pattern:

```bash
rosrun ur5_camera_calib intrinsic_calibration.py [--serial SERIAL] [--num-images NUM_IMAGES] [--delay DELAY]
```

Options:
- `--serial`: Camera serial number (if not specified, will prompt for selection)
- `--num-images`: Number of images to capture for calibration (default: 20)
- `--delay`: Delay between automatic captures in seconds, 0 for manual capture (default: 2)

#### 2. Major Camera Calibration

Calibrate the fixed "major" camera relative to the robot base frame:

```bash
rosrun ur5_camera_calib major_camera_calibration.py [--serial SERIAL] [--num-poses NUM_POSES] [--tag-size TAG_SIZE]
```

Options:
- `--serial`: Camera serial number (if not specified, will prompt for selection)
- `--num-poses`: Number of robot poses to collect (default: 15)
- `--tag-size`: AprilTag size in meters (default: 0.065)

#### 3. Movable Camera Calibration

Calibrate movable cameras relative to the robot base frame using a common AprilTag:

```bash
rosrun ur5_camera_calib movable_camera_calibration.py [--major-serial MAJOR_SERIAL] [--tag-size TAG_SIZE]
```

Options:
- `--major-serial`: Serial number of the major (already calibrated) camera
- `--tag-size`: AprilTag size in meters (default: 0.065)

### Camera Management

#### 4. Launching Multiple Cameras

Launch and identify multiple RealSense cameras:

```bash
rosrun ur5_camera_calib launch_cameras.py [--identify] [--save-layout] [--layout-name NAME]
```

Options:
- `--identify`: Run interactive camera identification and naming
- `--save-layout`: Save the camera layout (mapping between serials and names)
- `--layout-name`: Name for the layout file (default: "camera_layout")
- `--width`, `--height`, `--fps`: Camera resolution and frame rate

This script:
- Discovers all connected RealSense cameras
- Initializes and starts them with consistent topic naming
- Optionally helps you identify each camera and give it a meaningful name
- Publishes all camera streams to ROS topics using the serial number for topic naming

#### 5. Identifying Cameras

Identify which running camera is which:

```bash
rosrun ur5_camera_calib identify_cameras.py [--layout-name NAME]
```

Options:
- `--layout-name`: Name of the layout file to load (default: "camera_layout")

This script helps identify already running cameras by:
- Finding all active camera topics
- Displaying each camera's feed with its serial and user-assigned name
- Allowing you to cycle through cameras to match them with physical positions

## Calibration Process

### One-time Calibration Process

1. **Intrinsic Calibration**: Use the `intrinsic_calibration.py` script to calibrate the internal parameters of all cameras.

2. **Major Camera Calibration**: Mount the major camera in a fixed position relative to the robot base. Use the `major_camera_calibration.py` script to calibrate its extrinsic parameters. This involves moving the robot to different positions while holding an AprilTag.

### Multi-time (Online) Calibration

3. **Movable Camera Calibration**: When the movable cameras are repositioned, use the `movable_camera_calibration.py` script to recalibrate them relative to the robot base. This requires placing an AprilTag visible to all cameras.

## Camera Identification and Management

To avoid confusion when working with multiple cameras, use the following workflow:

1. **Initial Setup**:
   - Run `launch_cameras.py --identify --save-layout` to launch all cameras
   - Look at each camera feed and assign it a meaningful name (e.g., "front_camera", "wrist_camera")
   - The script will save this mapping between serial numbers and names

2. **Regular Use**:
   - Run `launch_cameras.py` to start all cameras with consistent topic naming
   - If needed, run `identify_cameras.py` to refresh your memory about which camera is which

3. **Accessing Camera Streams**:
   - Each camera publishes to topics using its serial number: `/camera/SERIAL/color/image_raw`
   - Use the saved layout file to map between camera names and topic paths in your code

## Output

Calibration results are saved in:
- `~/.ur5_camera_calib/intrinsics/` for intrinsic parameters
- `~/.ur5_camera_calib/extrinsics/` for extrinsic parameters
- `~/.ur5_camera_calib/layouts/` for camera naming layouts

For each camera, separate YAML files are generated, as well as a combined file containing all camera parameters.

## License

MIT License 