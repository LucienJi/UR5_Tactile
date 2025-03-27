#!/usr/bin/env python3

"""
Script for launching multiple RealSense cameras with clear identification.
Allows for associating camera serial numbers with meaningful names.
"""

import argparse
import sys
import time
import yaml
import rospy
import threading
from .ur5_camera_calib.camera_manager import CameraManager, save_camera_layout, load_camera_layout


def identify_and_name_cameras(manager):
    """
    Interactive tool to identify cameras and assign names to them.
    
    Args:
        manager (CameraManager): Camera manager instance
        
    Returns:
        dict: Dictionary mapping serial numbers to user-assigned names
    """
    print("\n=== Camera Identification and Naming ===")
    print("This tool will help you identify each camera and assign a name to it.")
    print("We'll show each camera's view one by one.")
    
    serials = list(manager.cameras.keys())
    if not serials:
        print("No cameras found!")
        return {}
    
    # Dictionary to store camera names
    camera_names = {}
    
    # Load existing layout if available
    existing_layout = load_camera_layout()
    if existing_layout:
        print("\nExisting camera layout found:")
        for serial, name in existing_layout.items():
            if serial in serials:
                print(f"  {serial}: {name}")
        
        use_existing = input("\nDo you want to use this layout? (y/n): ").lower() == 'y'
        if use_existing:
            # Only include serials that are currently connected
            return {s: existing_layout[s] for s in serials if s in existing_layout}
    
    # Start identification
    print("\nStarting camera identification...")
    print("A window will open showing each camera's view.")
    print("For each camera, enter a descriptive name (or press Enter to skip).")
    
    # Run identification in a separate thread
    identification_active = True
    
    def id_thread_function():
        manager.run_identification(serials)
        nonlocal identification_active
        identification_active = False
    
    id_thread = threading.Thread(target=id_thread_function)
    id_thread.daemon = True
    id_thread.start()
    
    try:
        for i, serial in enumerate(serials):
            if not identification_active:
                break
            
            print(f"\nCamera {i+1}/{len(serials)}")
            print(f"Serial: {serial}")
            
            # If this camera was in the existing layout, show the previous name
            if existing_layout and serial in existing_layout:
                prev_name = existing_layout[serial]
                print(f"Previous name: {prev_name}")
                name = input(f"Enter a name for this camera (or press Enter to keep '{prev_name}'): ")
                if not name:
                    name = prev_name
            else:
                name = input("Enter a name for this camera (or press Enter to skip): ")
            
            if name:
                camera_names[serial] = name
            
            # Wait for user to press 'n' to continue to next camera
            print("Press 'n' in the camera window to move to the next camera...")
            input("Press Enter here when ready for the next camera...")
    
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure identification is stopped
        identification_active = False
        id_thread.join(timeout=2.0)
    
    return camera_names


def main():
    parser = argparse.ArgumentParser(description="Launch and identify multiple RealSense cameras")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--fps", type=int, default=30, help="Camera frames per second")
    parser.add_argument("--identify", action="store_true", help="Run camera identification")
    parser.add_argument("--save-layout", action="store_true", help="Save camera layout")
    parser.add_argument("--layout-name", default="camera_layout", help="Name for the camera layout")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing for camera publishing")
    parser.add_argument("--publishing-fps", type=int, default=30, help="Frame publishing rate")
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('camera_launcher', anonymous=True)
    
    # Create camera manager
    manager = CameraManager(namespace="camera")
    
    # Discover and initialize cameras
    print("Discovering cameras...")
    serials = manager.discover_cameras()
    
    if not serials:
        print("No RealSense cameras found!")
        return 1
    
    print(f"Found {len(serials)} cameras:")
    for i, serial in enumerate(serials):
        print(f"  {i+1}. {serial}")
    
    # Initialize cameras
    print("\nInitializing cameras...")
    initialized = manager.initialize_cameras(
        serials=serials,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    if not initialized:
        print("Failed to initialize any cameras!")
        return 1
    
    print(f"Initialized {len(initialized)} cameras")
    
    # For identification, we'll need to use the cameras without multiprocessing
    if args.identify:
        # Run camera identification and naming
        camera_names = identify_and_name_cameras(manager)
        
        # Save layout if requested
        if args.save_layout and camera_names:
            save_camera_layout(camera_names, args.layout_name)
            print(f"Camera layout saved")
    else:
        # Just load existing layout
        camera_names = load_camera_layout(args.layout_name)
    
    # Stop existing camera instances before starting in multiprocessing mode
    if not args.no_multiprocessing:
        manager.stop_cameras()
    
    # Start cameras
    print("\nStarting cameras...")
    use_multiprocessing = not args.no_multiprocessing
    started = manager.start_cameras(
        serials=initialized,
        use_multiprocessing=use_multiprocessing,
        publishing_fps=args.publishing_fps
    )
    
    if not started:
        print("Failed to start any cameras!")
        return 1
    
    print(f"Started {len(started)} cameras" + 
          (" using multiprocessing" if use_multiprocessing else " in single-process mode"))
    
    # Print camera information
    print("\n=== Camera Information ===")
    for serial in started:
        name = camera_names.get(serial, "Unnamed")
        print(f"Camera: {name} (Serial: {serial})")
        print(f"  Topics:")
        print(f"    Color: /camera/{serial}/color/image_raw")
        print(f"    Depth: /camera/{serial}/depth/image_raw")
        print(f"    Info:  /camera/{serial}/color/camera_info")
        print("")
    
    # If we're using multiprocessing, we don't need to publish frames in the main process
    if not use_multiprocessing:
        # Start publishing thread
        publish_thread = threading.Thread(target=lambda: publish_camera_frames(manager, args.publishing_fps))
        publish_thread.daemon = True
        publish_thread.start()
    
    print("\nCameras are running. Press Ctrl+C to stop...")
    
    try:
        # Keep the main thread alive
        while not rospy.is_shutdown():
            # If using multiprocessing, check if processes are still alive
            if use_multiprocessing and not manager.are_processes_alive():
                print("All camera processes have stopped.")
                break
            rospy.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop all cameras
        print("\nStopping cameras...")
        manager.stop_cameras()
        print("All cameras stopped")
    
    return 0


def publish_camera_frames(manager, rate=30):
    """
    Continuously publish camera frames at the specified rate.
    
    Args:
        manager (CameraManager): Camera manager instance
        rate (int): Publishing rate in Hz
    """
    rate_obj = rospy.Rate(rate)
    while not rospy.is_shutdown():
        manager.publish_frames()
        rate_obj.sleep()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass 