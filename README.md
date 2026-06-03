# Hand-Eye Calibration Workspace

Fully independent calibration workspace for ChArUco eye-in-hand calibration.
Contains its own copy of robot and camera ROS2 packages — no dependency on the original project.

## Structure

```
.
├── scripts/
│   ├── build.sh                   # Build this ROS2 workspace
│   ├── run.sh                     # Launcher that sources THIS workspace's ROS2 env
│   └── start_hw.sh                # Start robot + camera (uses THIS workspace's install)
├── apps/
│   ├── 01_test_charuco.py         # Test board detection
│   ├── 02_collect_handeye.py      # Collect samples (needs ROS2 + robot + camera)
│   ├── 03_solve_handeye.py        # Solve T_link6_camera (pure Python, no ROS2)
│   └── 04_verify_handeye.py       # Verify accuracy in base frame (needs ROS2)
├── config/
│   └── calibration.yaml           # Board params, ROS topics, TF frames, paths
├── src/
│   ├── ros2_packages/             # ROS2 workspace source packages
│   │   ├── Camera_Vision/         # Camera driver & launch (copied)
│   │   └── ros2_rm_robot/         # Robot driver, bringup, control, description (copied)
│   └── calibration_py/            # Python calibration core libraries
│       ├── __init__.py
│       ├── charuco_board.py       # Detection & drawing
│       ├── transforms.py          # 4x4 matrix utilities
│       ├── handeye_solver.py      # AX=XB solver + evaluation
│       ├── ros_utils.py           # ROS2 node helpers
│       ├── io_utils.py            # JSON read/write
│       └── config_loader.py       # YAML config loader
├── data/
│   ├── samples/                   # Raw samples (auto-timestamped subdirs)
│   └── results/
│       └── charuco_eye_in_hand.json
└── README.md
```

## Build (First Time)

```bash
cd /home/nvidia/Calibration
./scripts/build.sh
```

If you need to rebuild from scratch:

```bash
./scripts/build.sh --clean
```

After building, `build/`, `install/`, and `log/` will be created in this directory.

## Quick Start

### 1. Start hardware

```bash
cd /home/nvidia/Calibration
./scripts/start_hw.sh              # tmux mode (recommended)
# or
./scripts/start_hw.sh --bg         # background mode
```

Stop later with:
```bash
./scripts/start_hw.sh --stop
```

### 2. Test ChArUco detection

```bash
./scripts/run.sh 01_test_charuco.py --ros-topic /vision/raw
```

### 3. Collect samples

```bash
./scripts/run.sh 02_collect_handeye.py --config config/calibration.yaml
```

Move the arm to varied poses. Press **Enter** to capture each sample.
Press **q + Enter** to finish.

Samples are saved under `data/samples/YYYYMMDD_HHMMSS/`.

### 4. Solve hand-eye matrix

```bash
./scripts/run.sh 03_solve_handeye.py --config config/calibration.yaml
```

Or point to a specific session:

```bash
./scripts/run.sh 03_solve_handeye.py --session-dir ./data/samples/20260603_120000
```

Result is written to `data/results/charuco_eye_in_hand.json`.

### 5. Verify calibration

```bash
./scripts/run.sh 04_verify_handeye.py --config config/calibration.yaml
```

This prints the ChArUco board center in `base_link` and a safe hover point.
Compare with a physical measurement or jog the arm to the hover point to check.

## Configuring the Board

Edit `config/calibration.yaml` to match your physical ChArUco board:

```yaml
charuco:
  dictionary: "DICT_5X5_100"
  squares_x: 12
  squares_y: 9
  square_mm: 30.0
  marker_mm: 22.5
```

## Notes

- `02_collect_handeye.py` and `04_verify_handeye.py` require the ROS2 environment and running hardware.
- `03_solve_handeye.py` is pure Python (OpenCV + NumPy) and can run offline.
- This workspace is **fully independent**; the original project `Capture_RealMan_OmniPicker-main` is not touched or referenced at runtime.
