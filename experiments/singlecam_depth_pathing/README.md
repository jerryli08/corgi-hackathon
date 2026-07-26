# Logitech C920 Depth and Floor Path Planner

Minimal Python project for showing a live Logitech C920 webcam feed, Depth Anything V2 Small relative depth, SegFormer B0 ADE20K floor segmentation, image-space perspective floor rectangles, edge-detected side guides, and a smoothed path through the segmented floor.

## Files

- `main.py` - webcam capture, depth inference, floor segmentation, perspective floor rectangles, edge side guides, path drawing, display loop
- `serial_servo_test.py` - separate Python serial test for two continuous servos on Arduino
- `arduino_continuous_servos.ino` - Arduino sketch for D9 left drive and D10 right drive continuous servos
- `requirements.txt` - Python dependencies
- `README.md` - setup and usage

## Stages

1. Create and activate a Python virtual environment.
2. Install the dependencies.
3. Connect the Logitech C920 webcam.
4. Run the live depth and floor path planner.
5. Press `Q` or Escape to quit.

## Windows Setup

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation scripts, run this once for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python main.py
```

The first run downloads these Hugging Face models:

- `depth-anything/Depth-Anything-V2-Small-hf`
- `nvidia/segformer-b0-finetuned-ade-512-512`

After that, they load from the local cache when available.

## Camera Selection

The webcam indices can be left on auto-detect or set per computer with environment variables:

```powershell
$env:CORGI_FLOOR_CAMERA_INDEX="1"
$env:CORGI_TARGET_CAMERA_INDEX="2"
```

If these are unset, the app auto-lists DirectShow cameras and prefers names matching the USB Logitech C920 keywords. It opens one C920 as the floor/path camera, then opens a second camera for target detection while avoiding the first camera index.

You can still edit `CAMERA_INDEX` and `TARGET_CAMERA_INDEX` near the top of `main.py` for quick local testing, but environment variables are better when several teammates run the same repo.

Depth Anything V2 often behaves like inverse depth, where larger raw values are closer. The code converts that to clearance, where larger values are safer/farther:

```python
DEPTH_VALUES_ARE_INVERSE = True
```

## Model and Planner Settings

Depth Anything and SegFormer use the same floor/path camera frame. A second side-mounted C920 is optional for orange pill-bottle detection. Arduino serial control is optional and is only used when servo steering is enabled.

Key tuning constants are near the top of `main.py`:

```python
DEPTH_INPUT_WIDTH = 384
DEPTH_INPUT_HEIGHT = 384
SEGMENTATION_INPUT_WIDTH = 512
SEGMENTATION_INPUT_HEIGHT = 512
INFERENCE_EVERY_N_FRAMES = 3
USE_DEPTH_OBSTACLE_MASK = False
SUDDEN_MASK_CHANGE_FRACTION = 0.32
SUDDEN_MASK_CONFIRM_FRAMES = 2
```

Increase model input sizes for better masks and depth detail. Decrease them, or increase `INFERENCE_EVERY_N_FRAMES`, for faster live updates.

By default the overlay does not depend on camera pitch. It draws the floor grid directly from the floor segmentation and draws side perspective guides from edge detection:

```python
USE_IMAGE_SPACE_PERSPECTIVE_GRID = True
FLOOR_GRID_ROWS = 12
FLOOR_GRID_COLS = 8
FLOOR_GRID_TOP_RATIO = 0.22
FLOOR_GRID_ALPHA = 0.26
PATH_FUTURE_TARGET_RATIO = 0.72
PATH_EARLY_TURN_POWER = 0.55
PATH_FLOOR_CENTER_BLEND = 0.35
ROBOT_WIDTH_PIXELS_BOTTOM = 150
ROBOT_WIDTH_PIXELS_TOP = 58
ROBOT_PATH_BOX_ALPHA = 0.42
EDGE_CANNY_LOW = 70
EDGE_CANNY_HIGH = 150
SIDE_EDGE_MAX_LINES = 10
```

The side target camera detects an orange pill bottle using HSV color thresholding:

```python
ENABLE_TARGET_CAMERA = True
TARGET_CAMERA_SIDE = "right"
TARGET_ORANGE_HSV_LOW = (5, 80, 70)
TARGET_ORANGE_HSV_HIGH = (28, 255, 255)
TARGET_REACH_DISTANCE_CM = 24.0
TARGET_KNOWN_WIDTH_CM = 5.5
TARGET_FOCAL_LENGTH_PIXELS = 700.0
TARGET_CENTER_DEADBAND_PIXELS = 55
```

Distance is estimated from apparent bottle width:

```text
distance_cm = known_width_cm * focal_length_pixels / detected_width_pixels
```

Calibrate `TARGET_FOCAL_LENGTH_PIXELS` by placing the bottle at a known distance, measuring its detected box width, and using:

```text
focal_length_pixels = detected_width_pixels * known_distance_cm / known_width_cm
```

`USE_DEPTH_OBSTACLE_MASK` and `USE_GRID_DEPTH_OBSTACLE_TRIM` are off by default so depth does not erase the floor grid while tuning. Set them to `True` later if you want depth boundaries to subtract from the floor region.

The safe mask has a sudden-change guard. If a large part of the detected floor changes in one frame, the app holds the previous stable mask until the change repeats for `SUDDEN_MASK_CONFIRM_FRAMES`. The debug text shows `HOLD` while this filter is rejecting a likely bad frame.

The older calibrated ground-plane mode is still available in the code by setting `USE_IMAGE_SPACE_PERSPECTIVE_GRID = False`, but the default mode is better when the final camera mounting angle is unknown.

## What It Does

- Opens the webcam using OpenCV.
- Loads Depth Anything V2 Small automatically.
- Loads lightweight SegFormer B0 pretrained on ADE20K automatically.
- Uses CUDA when available, otherwise CPU.
- Shows four views: original RGB with floor perspective rectangles, side edge guides, and path; colorized depth; floor mask; and safe floor mask.
- Creates `floor_mask` from configurable traversable classes such as floor, carpet, road, pavement, earth, and ground.
- Creates `obstacle_mask` from nearby depth regions when `USE_DEPTH_OBSTACLE_MASK = True`; otherwise it trusts the floor segmentation alone.
- Creates `safe_mask = floor_mask AND NOT obstacle_mask`.
- Removes small isolated regions, fills small holes, erodes floor boundaries, dilates obstacles, and keeps only the safe connected component touching the bottom-center of the frame.
- Draws perspective floor rectangles from the segmented floor boundaries.
- Draws red robot-width footprint boxes around the planned path so you can see the corridor the robot body needs.
- Draws side guide lines using Canny edge detection and Hough line detection.
- Draws a future-biased smoothed path through the segmented floor, so the start of the path bends toward the upcoming floor center instead of always beginning straight forward.
- Uses the second side-mounted C920 to detect an orange pill bottle and override path following when the target is visible.
- Stops when the bottle is centered in the target frame and close enough for the arm to reach.
- If the bottle is centered but too far, it uses tank drive to maneuver toward the target side.
- Prints live debug counts on the camera view: floor pixels, safe pixels, top-down grid cells, and path points.
- Smooths depth, masks, and the selected path to reduce flicker and jitter.

This project only displays visual guidance.

## Notes

- A CUDA-capable GPU makes live inference much faster.
- CPU mode works, but may update slowly depending on your computer.
- Good lighting and a stable camera position improve depth stability.
- The default overlay is perspective-guided image geometry, not measured metric 3D. For true metric 3D, you would need camera calibration plus stereo, LiDAR, or another measured-depth sensor.

## Arduino Serial Servo Test

This is separate from the camera pathing code. Use it first to confirm Python can command the Arduino over USB.

### Wiring

- Left continuous servo signal wire: Arduino `D9`
- Right continuous servo signal wire: Arduino `D10`
- Vertical linear-axis winch servo signal wire: Arduino `D3`
- Servo power: use a separate 5-6V supply that can provide enough current
- Servo ground and Arduino `GND`: connect together

Do not power drive servos from the Arduino 5V pin. Continuous servos can pull enough current to brown out or damage the board.

### Arduino Sketch Setup

1. Open Arduino IDE.
2. Create a new sketch.
3. Paste the contents of `arduino_continuous_servos.ino`.
4. Select your board from `Tools > Board`.
5. Select the Arduino port from `Tools > Port`.
6. Upload the sketch.
7. Close Serial Monitor before running Python, because only one program can use the serial port at a time.

The sketch listens for text commands:

```text
L:90 R:90 A:90
```

For most continuous servos:

- `90` means stop
- below `90` spins one direction
- above `90` spins the other direction

The sketch also has a safety timeout: if it stops receiving commands, it sends both servos back to `90`.

The Arduino sketch already supports the D3 winch servo through the `A:` value. You do not need to edit the sketch for timed winch moves unless you change the pin or want the timing to live on the Arduino instead of Python.

### Python Serial Test

Install the updated Python dependencies:

```powershell
pip install -r requirements.txt
```

List available serial ports:

```powershell
python serial_servo_test.py --list
```

Run the automatic test sequence, replacing `COM3` with your Arduino port:

```powershell
python serial_servo_test.py --port COM3
```

Run interactive mode:

```powershell
python serial_servo_test.py --port COM3 --interactive
```

Move the D3 winch upward for a fixed amount of time, then stop:

```powershell
python serial_servo_test.py --port COM3 --axis-up
```

Move the D3 winch downward for a fixed amount of time, then stop:

```powershell
python serial_servo_test.py --port COM3 --axis-down
```

## Server mission mode

The merged server starts this program with `--mission "pill bottle"`. Mission mode
follows webcam 1 until webcam 2 sees orange, then stops path following and uses slow
two-wheel forward/back motion until the bottle is horizontally centered. Webcam 2 is
rotated 180 degrees by default.

Because the physical arm is not installed, payload motion is disabled by default.
The pickup, basket drop, and winch stages still appear in the FSM and server status,
but are dry runs and D3 remains stopped. Set `CORGI_SINGLECAM_PAYLOAD_ENABLED=1` only
when the arm and payload motion are ready for a physical test.

With no number, `--axis-up` and `--axis-down` default to 10 seconds. You can still pass a different duration after you measure the real lowest-to-highest travel time:

```powershell
python serial_servo_test.py --port COM3 --axis-up 6.5
```

Interactive commands:

- `f` forward
- `b` backward
- `l` left
- `r` right
- `up` vertical axis up
- `down` vertical axis down
- `stop` stop
- `q` quit
- raw values like `108 72 90`

If one wheel spins backward relative to the other, swap the direction values in `serial_servo_test.py` for that movement, or physically rotate the servo mounting. For continuous servos, you may also need to trim the servo so `90` is truly stopped.

## Camera-Based Servo Steering

`main.py` now includes an optional slow PD/PID steering controller for the two continuous servos. It uses the horizontal error between the path lookahead point and the image center:

- path target right of center: steer right
- path target left of center: steer left
- no path: send stop

It does not use encoders yet.

After the standalone serial test works, set these per computer:

```powershell
$env:CORGI_SINGLECAM_SERVO_ENABLED="1"
$env:CORGI_SINGLECAM_ARDUINO_PORT="COM5"
```

The matching constants still exist near the top of `main.py` as defaults.

Start slow with these defaults:

```python
BASE_SERVO_SPEED = 10
MAX_SERVO_SPEED = 32
MAX_STEERING_CORRECTION = 14
PIVOT_TURN_ERROR_THRESHOLD = 0.10
PIVOT_TURN_MIN_SPEED = 10
PIVOT_TURN_MAX_SPEED = 18
PID_KP = 26.0
PID_KI = 0.0
PID_KD = 6.0
```

For very small path errors, the robot drives forward. For any meaningful turn, it uses both wheels in opposite directions for a pivot turn.

Continuous servos are usually mirrored on a robot, so one wheel often needs the opposite sign to drive forward. These defaults assume left forward is above `90` and right forward is below `90`:

```python
LEFT_SERVO_FORWARD_SIGN = 1
RIGHT_SERVO_FORWARD_SIGN = -1
```

If forward drives backward, flip both signs. If it spins in place when it should go forward, flip only one sign.
