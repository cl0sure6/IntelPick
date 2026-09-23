# IntelPick

Webcam + computer vision + Waveshare RoArm: sorts coloured / shaped objects into bins.
ROS 2 Humble, Python, runs in WSL2 (Ubuntu 22.04).

```
camera_node → detector_node (HSV / YOLO + calibration) → sorter_node → arm_node → RoArm
```

Design, assumption check and milestones: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build (in WSL)

```bash
source /opt/ros/humble/setup.bash
cd /mnt/c/Users/adils/IntelPickIRO
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select intelpick && colcon test-result --all
```

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

# Arm only, no hardware: exercise the services
ros2 run intelpick arm_node --ros-args -p dry_run:=true
ros2 service call /arm/move_to intelpick_interfaces/srv/MoveTo "{x: 0.25, y: 0.0, z: 0.05}"

# Camera + detector only, tune HSV while watching the overlay
ros2 launch intelpick intelpick.launch.py sorter:=false dry_run:=true
ros2 launch intelpick intelpick.launch.py sorter:=false dry_run:=true camera:=http://<phone-ip>:4747/video
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
Set the resolution in the app (640×480 is plenty), then calibrate. The calibration records the
image size and the detector refuses to compute positions if the stream's resolution changes.
Network streams lag; keep `settle_time` above the lag you see.

Tunables live in [src/intelpick/config/intelpick.yaml](src/intelpick/config/intelpick.yaml).

## YOLO (shape sorting, milestone M5)

```bash
pip install ultralytics "numpy<2"    # numpy<2 keeps ROS cv_bridge working
yolo train model=yolo26n.pt data=datasets/shapes/data.yaml epochs=80 imgsz=640
ros2 launch intelpick intelpick.launch.py backend:=yolo   # set yolo_model to runs/.../best.pt
```
