#!/bin/bash
# Start hardware needed for hand-eye calibration.
# This uses the ROS2 packages built INSIDE this Calibration workspace.
#
# Usage:
#   ./scripts/start_hw.sh           # start in tmux (recommended)
#   ./scripts/start_hw.sh --bg      # start as background processes (no tmux)

set -euo pipefail

SESSION="calib_hw"
LOG_DIR="$PWD/data/logs"

set +u
source /opt/ros/humble/setup.bash
set -u
if [[ -f "$PWD/install/setup.bash" ]]; then
    set +u
    source "$PWD/install/setup.bash"
    set -u
fi
export LD_LIBRARY_PATH="$PWD/src/ros2_rm_robot/rm_driver/lib:${LD_LIBRARY_PATH:-}"

start_bg() {
    local name="$1"
    local cmd="$2"
    mkdir -p "$LOG_DIR"
    if [[ -f "$LOG_DIR/$name.pid" ]]; then
        local old_pid
        old_pid="$(cat "$LOG_DIR/$name.pid" 2>/dev/null || true)"
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[INFO] $name already running: pid=$old_pid"
            return 0
        fi
    fi
    echo "[INFO] Starting $name in background. Log: $LOG_DIR/$name.log"
    nohup bash -lc "$cmd" >"$LOG_DIR/$name.log" 2>&1 &
    echo "$!" > "$LOG_DIR/$name.pid"
}

stop_bg() {
    echo "[INFO] Stopping background processes..."
    for pidfile in "$LOG_DIR"/*.pid; do
        [[ -f "$pidfile" ]] || continue
        pid="$(cat "$pidfile" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "[INFO] Killed $(basename "$pidfile" .pid) (pid=$pid)"
        fi
        rm -f "$pidfile"
    done
}

# Handle --stop
if [[ "${1:-}" == "--stop" ]]; then
    stop_bg
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        echo "[INFO] Killed tmux session: $SESSION"
    fi
    exit 0
fi

# If tmux is not installed, fall back to background mode automatically
if ! command -v tmux >/dev/null 2>&1; then
    echo "[WARN] tmux not found; falling back to background mode."
    start_bg bringup "ros2 launch rm_bringup rm_65_6f_bringup.launch.py"
    start_bg camera  "ros2 launch camera_vision dep_cam.launch.py"
    echo "[OK] Hardware started in background."
    echo "     Logs: $LOG_DIR"
    echo "     Stop: $0 --stop"
    exit 0
fi

# Background mode (no tmux)
if [[ "${1:-}" == "--bg" ]]; then
    start_bg bringup "ros2 launch rm_bringup rm_65_6f_bringup.launch.py"
    start_bg camera  "ros2 launch camera_vision dep_cam.launch.py"
    echo "[OK] Hardware started in background."
    echo "     Logs: $LOG_DIR"
    echo "     Stop: $0 --stop"
    exit 0
fi

# Tmux mode
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Hardware tmux session already exists: $SESSION"
    echo "     Attach: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n bringup
tmux send-keys -t "$SESSION:bringup" "
export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}
ros2 launch rm_bringup rm_65_6f_bringup.launch.py
" C-m

tmux new-window -t "$SESSION" -n camera
tmux send-keys -t "$SESSION:camera" "
export ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0}
ros2 launch camera_vision dep_cam.launch.py
" C-m

echo "[OK] Hardware started in tmux session: $SESSION"
echo "     Attach: tmux attach -t $SESSION"
echo "     Stop:   tmux kill-session -t $SESSION"
echo "     Or:     $0 --stop"
