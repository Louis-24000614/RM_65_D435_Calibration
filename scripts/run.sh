#!/bin/bash
# Calibration runner — sources ROS2 workspace and runs a script with PYTHONPATH set.
#
# Usage:
#   ./run.sh 01_test_charuco.py --ros-topic /vision/raw
#   ./run.sh 02_collect_handeye.py --config config/calibration.yaml
#   ./run.sh 03_solve_handeye.py --config config/calibration.yaml
#   ./run.sh 04_verify_handeye.py --config config/calibration.yaml

set -euo pipefail

SCRIPT_NAME="${1:-}"
if [[ -z "$SCRIPT_NAME" ]]; then
    echo "Usage: ./run.sh <script_name> [args...]"
    echo "Examples:"
    echo "  ./run.sh 01_test_charuco.py --ros-topic /vision/raw"
    echo "  ./run.sh 02_collect_handeye.py --config config/calibration.yaml"
    echo "  ./run.sh 03_solve_handeye.py"
    echo "  ./run.sh 04_verify_handeye.py"
    exit 1
fi

# Move to workspace root (parent of scripts/)
cd "$(dirname "$0")/.."

# Source ROS2 environment and THIS workspace's install space
if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
fi

if [[ -f "$(pwd)/install/setup.bash" ]]; then
    source "$(pwd)/install/setup.bash"
fi

# Ensure rm_driver lib is on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$(pwd)/src/ros2_packages/ros2_rm_robot/rm_driver/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$(pwd)/src/calibration_py:${PYTHONPATH:-}"

shift
python3 "apps/$SCRIPT_NAME" "$@"
