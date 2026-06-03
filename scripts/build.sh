#!/bin/bash
# Build the Calibration ROS2 workspace.
# This compiles the copied Camera_Vision and ros2_rm_robot packages.

set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
set -u

echo "[INFO] Building Calibration workspace..."
cd "$(dirname "$0")/.."

# Clean previous builds if requested
if [[ "${1:-}" == "--clean" ]]; then
    echo "[INFO] Cleaning previous build artifacts..."
    rm -rf build install log
fi

export LD_LIBRARY_PATH="$PWD/src/ros2_packages/ros2_rm_robot/rm_driver/lib:${LD_LIBRARY_PATH:-}"

colcon build \
    --packages-select \
        camera_vision \
        rm_driver \
        rm_description \
        rm_control \
        rm_bringup \
        rm_ros_interfaces \
        rm_65_config \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "[OK] Build complete."
echo "     Source the workspace before running: source install/setup.bash"
