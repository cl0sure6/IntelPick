# IntelPick

Webcam + YOLO colour recognition + Waveshare RoArm: sorts coloured objects into bins.
An HSV-only backend is available for initial setup before training weights exist.
ROS 2 Humble, Python, runs in WSL2 (Ubuntu 22.04).

```
camera_node → detector_node (HSV / YOLO + calibration) → sorter_node → arm_node → RoArm
```

Design, assumption check and milestones: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build (in WSL)

```bash
source /opt/ros/humble/setup.bash
cd /mnt/c/Users/adils/IntelPickIRO
PYTHONNOUSERSITE=1 colcon build --symlink-install
source install/setup.bash
colcon test --packages-select intelpick && colcon test-result --all
```

`PYTHONNOUSERSITE=1` makes the build ignore Python packages installed with `pip --user`.
Installing the YOLO requirements puts setuptools 78 into `~/.local`, which breaks ROS
Humble's build (`canonicalize_version() got an unexpected keyword argument
'strip_trailing_zero'`). The build only needs the system packages; running still uses the
`--user` ones (Ultralytics, PyTorch). Without the YOLO packages installed, plain
`colcon build` works too.

## Get USB devices into WSL (once per boot)

[usbipd-win](https://github.com/dorssel/usbipd-win) in PowerShell (open a new terminal after
installing so it is on PATH):

```powershell
usbipd list                                  # arm = 10c4:ea60 (CP2102); webcam = your camera's line
usbipd bind --force --busid <BUSID>          # admin, first time only; --force because USBPcap is installed
usbipd attach --wsl --busid <BUSID>          # every time the device is plugged in
```

While attached, the device disappears from Windows. In Ubuntu, once:
`sudo usermod -aG dialout,video $USER`, then `wsl --shutdown` and reopen. Check with
`ls /dev/video* /dev/ttyUSB*`.

## Run

```bash
# First contact with the real arm (M1): measures what the docs don't say, prints config values
ros2 run intelpick probe_arm --port /dev/ttyUSB0      # add --dry-run to rehearse without the arm

# Tune the grasp: pick an object from a known spot with the sorter's exact sequence, put it back.
# Defaults come from the config; override to experiment, then copy what works back into it.
ros2 run intelpick grasp_trial --x 0.25 --y 0.0 --tries 3
ros2 run intelpick grasp_trial --x 0.25 --grasp-height 0.01 --standoff 0.05 --open 1.3

# Arm only, no hardware: exercise the services
ros2 run intelpick arm_node --ros-args -p dry_run:=true
ros2 service call /arm/move_to intelpick_interfaces/srv/MoveTo "{x: 0.25, y: 0.0, z: 0.05}"

# Camera + detector only, tune HSV while watching the overlay
ros2 launch intelpick intelpick.launch.py sorter:=false arm:=false
ros2 launch intelpick intelpick.launch.py sorter:=false arm:=false camera:=http://<phone-ip>:4747/video
ros2 run rqt_image_view rqt_image_view /detections/image

# Calibrate (arm_node must NOT be running). Markers go on the corners of the work mat:
# the area they enclose becomes the pick zone, everything outside it is ignored.
ros2 run intelpick calibrate --model m2 --port /dev/ttyUSB0

# Full cell
ros2 launch intelpick intelpick.launch.py model:=m2 serial_port:=/dev/ttyUSB0
```

### Wi-Fi camera (phone app or IP camera)

`camera:=` (and `calibrate --camera`) take a device, an index or a stream URL:

| Source | URL (check the app's screen for the exact one) |
|---|---|
| DroidCam (Android/iOS) | `http://<phone-ip>:4747/video` |
| IP Webcam (Android) | `http://<phone-ip>:8080/video` |
| Wi-Fi security camera | `rtsp://user:pass@<ip>:554/<stream>` (pass on the command line, not in the YAML) |

The phone must be on the same Wi-Fi as the PC; WSL reaches it through Windows, no usbipd needed.
Two things that bite:

- **University / office Wi-Fi usually blocks device-to-device traffic** (client isolation; the
  KBTU network does). The page times out even with both on the same network. Turn on the
  iPhone's Personal Hotspot and join it from the laptop: the phone is then `172.20.10.1`
  (`http://172.20.10.1:4747/video`). The video stays local and uses no mobile data.
- **DroidCam serves one viewer at a time.** An open browser tab or the DroidCam PC client
  takes the slot and everything else gets "DroidCam is Busy". Close them before launching.
  (The DroidCam PC client isn't needed at all.)

Set the resolution in the app (640×480 is plenty), then calibrate. The calibration records the
image size and the detector refuses to compute positions if the stream's resolution changes.
Network streams lag; keep `settle_time` above the lag you see.

Only one program can drive the arm at a time (`arm_node`, `probe_arm`, `calibrate`,
`grasp_trial`); the driver locks the serial port and a second one is refused.

Tunables live in [src/intelpick/config/intelpick.yaml](src/intelpick/config/intelpick.yaml).

## Finish the YOLO colour workflow

Motion/grasp bring-up and the camera test have passed. The remaining physical steps are
collecting labelled images, training/validating the colour model, calibration and a sorting
trial with the actual bins. The code includes capture/annotation and dataset validation tools.

Follow [docs/YOLO.md](docs/YOLO.md) for setup, training and the full run sequence:

```bash
ros2 run intelpick capture_dataset --camera /dev/video0 --out datasets/colors --split train
ros2 run intelpick capture_dataset --camera /dev/video0 --out datasets/colors --split val
ros2 run intelpick train_yolo --data datasets/colors/data.yaml --check-only
ros2 run intelpick train_yolo --data datasets/colors/data.yaml --epochs 80 --device cpu
ros2 launch intelpick intelpick.launch.py backend:=yolo \
  yolo_model:=/absolute/path/to/weights/best.pt sorter:=false arm:=false
```

Train detection classes `red`, `green`, `blue`, `yellow`; the default YOLO label mode passes
them directly to the colour bins. A YOLO + HSV mode also supports shape-trained models.
The launch default stays `backend:=color` until you explicitly supply trained YOLO weights.
