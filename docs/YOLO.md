# YOLO colour recognition: capture → train → sort

IntelPick supports two colour-recognition paths:

| Backend / mode | What recognises colour | Output labels |
|---|---|---|
| `backend:=yolo`, `yolo_label_mode: class` (recommended) | YOLO trained on your colour classes | `red`, `green`, `blue`, `yellow` |
| `backend:=yolo`, `yolo_label_mode: color` | YOLO finds objects; HSV recognises their colour | `red`, `green`, `blue`, `yellow` |
| `backend:=yolo`, `yolo_label_mode: color_class` | YOLO finds shapes; HSV recognises colour | `red_cube`, etc.; configure matching bin labels |
| `backend:=color` (launch default until weights exist) | HSV alone | `red`, `green`, `blue`, `yellow` |

The recommended path uses YOLO itself to learn colour recognition. Generic COCO weights
do not have these colour classes or your cubes/caps. They are a starting point for training.
Runtime requires an existing local weights file and never downloads one automatically.
Classification, segmentation and OBB models are rejected; use **object detection** weights.

## 1. Prepare WSL

Start Ubuntu from Windows Command Prompt or PowerShell:

```powershell
wsl -d Ubuntu-22.04
```

Wait for a Linux prompt such as `adils@...:~$`. A prompt starting with `C:\Users\...>`
is still Windows; running `python3` there can open the Microsoft Store alias instead of
the project's Linux Python. Run all remaining commands **inside Ubuntu/WSL**, using the
same `/usr/bin/python3` as ROS Humble:

```bash
cd /mnt/c/Users/adils/IntelPickIRO
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y python3-pip
# CPU-only PyTorch avoids the much larger CUDA dependencies on a laptop without an NVIDIA GPU.
python3 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user -r requirements-yolo.txt
# Build with the --user packages hidden: they include setuptools 78, which breaks ROS
# Humble's build tools (error: canonicalize_version() ... 'strip_trailing_zero').
PYTHONNOUSERSITE=1 colcon build --symlink-install
source install/setup.bash
python3 -c "import cv2, numpy, ultralytics; from cv_bridge import CvBridge; print(cv2.__version__, numpy.__version__, ultralytics.__version__)"
```

The optional requirements constrain NumPy below 2 for ROS `cv_bridge` and OpenCV below 4.12
to keep that NumPy constraint compatible. The HSV-only path still uses the existing ROS/apt
dependencies. For an NVIDIA GPU, install the appropriate PyTorch build using its
[official installation selector](https://pytorch.org/get-started/locally/), then install
the same requirements. Keep training and runtime versions consistent.

This laptop has an **RTX 4060** that WSL can see (`nvidia-smi` works), but the CPU-only
PyTorch above can't use it (`torch.cuda.is_available()` is False). CPU training of the small
model on ~250 images takes on the order of an hour; with a CUDA build of PyTorch (a few GB
download, chosen in the selector above) it takes minutes. Train with `--device 0` then.

## 2. Capture and label real examples

Stop the camera node so the capture tool has exclusive access. Keep the camera fixed in
its final mount and use the resolution and lighting intended for the demo.

```bash
ros2 run intelpick capture_dataset --camera /dev/video0 --out datasets/colors --split train
# Phone camera alternative:
ros2 run intelpick capture_dataset --camera http://172.20.10.1:4747/video --out datasets/colors --split train
```

- **SPACE** freezes a fresh frame. The HSV detector (ranges from the installed config, as
  tuned with `tune_colors`) suggests a coloured box for each object it finds.
- Check every box. Click one to select it, then **1 red**, **2 green**, **3 blue** or
  **4 yellow** to recolour it; **D** or right-click deletes it; drag on the image to add a box
  HSV missed (it shows a magenta `?` until you give it a colour). **TAB** cycles through the
  boxes, **H** hides the help text if it covers an object.
- **ENTER** saves the photo with the boxes shown and returns to the live view. **ESC** skips
  the photo without saving.
- **B** in the live view saves an explicitly reviewed empty-table/background frame. Use it
  only when no target objects are present. If HSV sees objects, B opens the same check
  window first; delete the false boxes and ENTER saves what is left (none = background).
  **Q** exits the live view.

The suggestions only save drawing; you are still the source of truth. Pressing ENTER without
looking would teach YOLO to copy HSV, mistakes included. Fix boxes that are loose, cut off,
on shadows, bins or distractors, and keep photos where HSV gets it wrong (after fixing them):
those are what the network learns from beyond HSV. `--no-suggest` starts every photo empty.
Start with roughly 200 varied training images, balanced across colours, positions, object
sizes, orientations, shadows and combinations. This is a starting target, not an accuracy
guarantee. Include some empty-table images and non-target distractors.

Rearrange the objects and capture a separate validation session, around 40–60 images:

```bash
ros2 run intelpick capture_dataset --camera /dev/video0 --out datasets/colors --split val
```

Do not put adjacent video frames or the same scene into both splits. Each colour needs
labelled examples in both. The tools create:

```text
datasets/colors/
  data.yaml             # absolute dataset path and class names
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Labels follow Ultralytics' [YOLO detection format](https://docs.ultralytics.com/datasets/detect/):
`class_id center_x center_y width height`, one object per line, coordinates normalized to
0–1. IDs are `0 red`, `1 green`, `2 blue`, `3 yellow`. The helper requires an empty `.txt`
file for background images so missing annotation work cannot silently become background.
External annotation exports work if they follow this layout and class order. If moving the
dataset to another machine, update the absolute `path` in `data.yaml`.

## 3. Check and train

```bash
ros2 run intelpick train_yolo --data datasets/colors/data.yaml --check-only
ros2 run intelpick train_yolo --data datasets/colors/data.yaml --epochs 80 --device cpu
# CUDA GPU alternative: --device 0
```

The checker rejects missing labels, unreadable images, invalid boxes, missing classes and
identical decoded images crossing the train/validation boundary. It cannot detect every
near-duplicate scene; separate capture sessions still matter.

Training starts from `yolo26n.pt` by default (downloaded by Ultralytics on first use).
Use `--model` to choose other compatible detection weights. It disables hue, saturation and
RGB/BGR channel augmentation: changing a red object to another colour while keeping the
`red` label would teach the wrong task. Brightness and geometric variation remain enabled.
See the [Ultralytics augmentation reference](https://docs.ultralytics.com/guides/yolo-data-augmentation/).

The tool prints the actual `weights/best.pt` path; repeated runs may use numbered directories.
Inspect training results and predictions on the held-out images. For example, using the
**actual** printed run directory:

```bash
~/.local/bin/yolo detect val model=runs/colors/train/weights/best.pt data=datasets/colors/data.yaml device=cpu plots=True
~/.local/bin/yolo detect predict model=runs/colors/train/weights/best.pt source=datasets/colors/images/val device=cpu save=True
```

Assess per-colour precision/recall and inspect confusion between colours; a single overall
score can hide a weak class. Add real examples where recognition fails and retrain.

## 4. Preview the trained model, without opening the arm

The default config has `yolo_label_mode: class`, matching this dataset and the existing bins.
Use an absolute WSL path to the printed model:

```bash
ros2 launch intelpick intelpick.launch.py backend:=yolo \
  yolo_model:=/mnt/c/Users/adils/IntelPickIRO/runs/colors/train/weights/best.pt \
  camera:=/dev/video0 sorter:=false arm:=false
ros2 run rqt_image_view rqt_image_view /detections/image
```

The overlay shows each label and confidence. Before calibration, image detections work but
table coordinates are NaN, so the sorter ignores them. In HSV modes, weak or ambiguous colour
crops are skipped rather than guessed. Tune `color_min_fraction`, `color_min_margin` and
HSV ranges under your lighting. `min_area`/`max_area` also apply to YOLO boxes.

For a shape model, copy the parameter file and set `yolo_label_mode: color` to sort its detected
objects into the four colour bins. `color_class` outputs combined labels and needs matching
`sorter.bin_labels` and `bin_positions`. `yolo_label_mode` replaces the old `yolo_color_tag`:
old `true` corresponds to `color_class`, old `false` to `class`.

## 5. Calibrate and run the cell

Camera tests and motion tests do not establish the camera-to-arm mapping. With the arm node
stopped, use the existing calibration tool at the same camera resolution:

```bash
ros2 run intelpick calibrate --camera /dev/video0 --model m2 --port /dev/ttyUSB0
```

Check RMS below 5 mm and hover alignment on real objects. Confirm the physical bins match
`sorter.bin_positions` in `src/intelpick/config/intelpick.yaml`.

First exercise the complete loop with the simulated arm:

```bash
ros2 launch intelpick intelpick.launch.py backend:=yolo \
  yolo_model:=/mnt/c/Users/adils/IntelPickIRO/runs/colors/train/weights/best.pt dry_run:=true
```

Then run with the real arm:

```bash
ros2 launch intelpick intelpick.launch.py backend:=yolo \
  yolo_model:=/mnt/c/Users/adils/IntelPickIRO/runs/colors/train/weights/best.pt \
  model:=m2 serial_port:=/dev/ttyUSB0
```

Pass the same `camera:=...` URL to preview and full launch if using a phone. Pass a custom
`params:=/absolute/path/to/intelpick.yaml` for tuned settings. Fresh detections must have
matching bin labels, sufficient confidence, reachable grasp/approach points and consecutive
stable positions. The sorter drops pre-home camera frames and expires targets when the
camera stops. `detection_max_age` must exceed normal camera + inference latency;
`settle_time` must exceed stream lag after homing.

Training images, calibration, held-out model accuracy and an unattended real-arm sorting
trial remain physical validation steps. The hardware-free suite tests the YOLO result adapter
with controlled predictions and exercises both detection paths through actual ROS topics
and simulated arm services; it does not measure neural-network accuracy.
