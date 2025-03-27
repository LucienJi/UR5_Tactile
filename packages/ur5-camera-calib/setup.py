from setuptools import setup, find_packages

setup(
    name="ur5_camera_calib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pyyaml",
        "apriltag",
        "pyrealsense2",
    ],
) 