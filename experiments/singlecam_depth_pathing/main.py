import argparse
import json
import os
import sys
import time
from enum import Enum

import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
)

try:
    import serial
except ImportError:
    serial = None


def env_flag(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def env_optional_int(name, default=None):
    value = os.getenv(name)
    return default if value is None or not value.strip() else int(value)


# Camera
CAMERA_INDEX = env_optional_int("CORGI_FLOOR_CAMERA_INDEX")
TARGET_CAMERA_INDEX = env_optional_int("CORGI_TARGET_CAMERA_INDEX")
PREFERRED_CAMERA_NAME_KEYWORDS = ("c920", "logitech", "hd pro webcam", "usb")
FALLBACK_CAMERA_INDICES = (1, 2, 3, 4, 0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Models
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
SEGMENTATION_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
DEPTH_INPUT_WIDTH = 384
DEPTH_INPUT_HEIGHT = 384
SEGMENTATION_INPUT_WIDTH = 512
SEGMENTATION_INPUT_HEIGHT = 512
INFERENCE_EVERY_N_FRAMES = 3
DEPTH_VALUES_ARE_INVERSE = True

# Traversable ADE20K label matching
TRAVERSABLE_LABEL_KEYWORDS = (
    "floor",
    "carpet",
    "road",
    "sidewalk",
    "pavement",
    "earth",
    "ground",
    "field",
    "path",
    "runway",
)

# Smoothing
DEPTH_EMA_ALPHA = 0.35
FLOOR_MASK_EMA_ALPHA = 0.45
OBSTACLE_MASK_EMA_ALPHA = 0.45
PATH_EMA_ALPHA = 0.25

# Depth and mask cleanup
DEPTH_BLUR_KERNEL = 7
DEPTH_COLOR_MAP = cv2.COLORMAP_TURBO
NEAR_OBSTACLE_PERCENTILE = 28
NEAR_OBSTACLE_MARGIN = 0.04
USE_DEPTH_OBSTACLE_MASK = False
FLOOR_MASK_THRESHOLD = 0.50
OBSTACLE_MASK_THRESHOLD = 0.45
MIN_SAFE_REGION_AREA = 1200
SMALL_COMPONENT_AREA = 900
MORPH_KERNEL_SIZE = 5
FLOOR_ERODE_ITERATIONS = 2
OBSTACLE_DILATE_ITERATIONS = 3
BOTTOM_CENTER_RADIUS_PIXELS = 42
SUDDEN_MASK_CHANGE_FRACTION = 0.32
SUDDEN_MASK_CONFIRM_FRAMES = 2
STABLE_SAFE_MASK_BLEND_ALPHA = 0.35

# Pseudo-3D floor-space planner
CAMERA_HEIGHT_METERS = 0.55
CAMERA_PITCH_DEGREES = 28.0
AUTO_CALIBRATE_CAMERA_PITCH = True
AUTO_PITCH_MIN_DEGREES = 5.0
AUTO_PITCH_MAX_DEGREES = 65.0
AUTO_PITCH_STEP_DEGREES = 2.5
CAMERA_HORIZONTAL_FOV_DEGREES = 70.0
CAMERA_VERTICAL_FOV_DEGREES = 43.0
GRID_WIDTH_METERS = 3.0
GRID_FORWARD_METERS = 4.0
GRID_CELL_SIZE_METERS = 0.05
GRID_PATH_STEPS = 64
MIN_GRID_ROW_CELLS = 6
ROBOT_CLEARANCE_METERS = 0.22
FLOOR_BOUNDARY_CLEARANCE_METERS = 0.12
PERSPECTIVE_RECTANGLE_STEP_CELLS = 4
PERSPECTIVE_RECTANGLE_MIN_SAFE_CELLS = 4
PERSPECTIVE_RECTANGLE_ALPHA = 0.28
GRID_CELL_MIN_FLOOR_SAMPLES = 2
GRID_DEPTH_OBSTACLE_THRESHOLD = 0.18
USE_GRID_DEPTH_OBSTACLE_TRIM = False

# Image-space perspective overlay for unknown camera mounting angle
USE_IMAGE_SPACE_PERSPECTIVE_GRID = True
FLOOR_GRID_ROWS = 12
FLOOR_GRID_COLS = 8
FLOOR_GRID_TOP_RATIO = 0.22
FLOOR_GRID_MIN_WIDTH_PIXELS = 24
FLOOR_GRID_ALPHA = 0.26
PATH_FUTURE_TARGET_RATIO = 0.72
PATH_FUTURE_TARGET_AVERAGE_POINTS = 4
PATH_EARLY_TURN_POWER = 0.55
PATH_FLOOR_CENTER_BLEND = 0.35
ROBOT_WIDTH_PIXELS_BOTTOM = 150
ROBOT_WIDTH_PIXELS_TOP = 58
ROBOT_PATH_BOX_ALPHA = 0.42
ROBOT_PATH_BOX_STEP_POINTS = 1
EDGE_CANNY_LOW = 70
EDGE_CANNY_HIGH = 150
EDGE_HOUGH_THRESHOLD = 45
EDGE_MIN_LINE_LENGTH = 45
EDGE_MAX_LINE_GAP = 18
SIDE_EDGE_MIN_ABS_SLOPE = 0.35
SIDE_EDGE_MAX_LINES = 10

# Direction labels
TURN_DEADBAND_PIXELS = 70
STOP_TEXT = "STOP"

# Optional Arduino continuous-servo control. Leave ARDUINO_SERIAL_PORT = None
# while tuning vision. Set it to something like "COM3" when ready.
ARDUINO_SERIAL_PORT = os.getenv("CORGI_SINGLECAM_ARDUINO_PORT", "COM5") or None
ARDUINO_BAUD_RATE = 115200
ARDUINO_RESET_WAIT_SECONDS = 2.0
SERIAL_SEND_INTERVAL_SECONDS = 0.10

# Continuous servo speed mapping
SERVO_STOP = 90
BASE_SERVO_SPEED = 10
MAX_SERVO_SPEED = 32
MAX_STEERING_CORRECTION = 14
PIVOT_TURN_ERROR_THRESHOLD = 0.10
PIVOT_TURN_MIN_SPEED = 10
PIVOT_TURN_MAX_SPEED = 18
LEFT_SERVO_FORWARD_SIGN = -1
RIGHT_SERVO_FORWARD_SIGN = 1

# Path steering PID. Starts mostly proportional and slow.
ENABLE_SERVO_CONTROL = env_flag("CORGI_SINGLECAM_SERVO_ENABLED", True)
PID_KP = 26.0
PID_KI = 0.0
PID_KD = 6.0
PID_INTEGRAL_LIMIT = 0.75
PATH_LOOKAHEAD_RATIO = 0.84
PATH_ERROR_AVERAGE_LAST_POINTS = 5

# Side camera orange pill-bottle targeting
ENABLE_TARGET_CAMERA = True
TARGET_CAMERA_SIDE = "right"
TARGET_ORANGE_HSV_LOW = (5, 80, 70)
TARGET_ORANGE_HSV_HIGH = (28, 255, 255)
TARGET_MIN_AREA_PIXELS = 600
TARGET_MORPH_KERNEL_SIZE = 5
TARGET_CENTER_DEADBAND_PIXELS = 55
TARGET_REACH_DISTANCE_CM = 24.0
TARGET_KNOWN_WIDTH_CM = 5.5
TARGET_FOCAL_LENGTH_PIXELS = 700.0
TARGET_TURN_MIN_SPEED = 9
TARGET_TURN_MAX_SPEED = 18
TARGET_APPROACH_SPEED = 10

# Mission-mode sequencing. The physical arm is not integrated yet, so these are
# timed stand-ins for the future arm pick, winch lift, basket drop, and winch lower.
MISSION_BOTTLE_STABLE_FRAMES = 5
MISSION_ARM_PICK_SECONDS = 3.0
MISSION_WINCH_UP_SECONDS = 22.0
MISSION_BASKET_DROP_SECONDS = 2.0
MISSION_WINCH_DOWN_SECONDS = 22.0
MISSION_AXIS_UP_VALUE = 104
MISSION_AXIS_DOWN_VALUE = 76
MISSION_TARGET_CENTER_DEADBAND_PIXELS = 25
MISSION_TARGET_CENTER_STABLE_FRAMES = 5
MISSION_TARGET_FB_KP = 7.0
MISSION_TARGET_FB_MIN_SPEED = 5.0
MISSION_TARGET_FB_MAX_SPEED = 9.0
MISSION_TARGET_FB_SIGN = 1.0
MISSION_TARGET_ROTATE_180 = True
MISSION_ENABLE_PAYLOAD_SEQUENCE = False
MISSION_RESUME_PATH_SECONDS = 2.0
MISSION_EVENT_PREFIX = "CORGI_EVENT "


class MissionState(str, Enum):
    PATH_FOLLOWING = "PATH_FOLLOWING"
    TARGET_CENTERING = "TARGET_CENTERING"
    TARGET_LOCKED = "TARGET_LOCKED"
    ARM_PICK_PLACEHOLDER = "ARM_PICK_PLACEHOLDER"
    WINCH_UP = "WINCH_UP"
    BASKET_DROP_PLACEHOLDER = "BASKET_DROP_PLACEHOLDER"
    WINCH_DOWN = "WINCH_DOWN"
    RESUMING_PATH = "RESUMING_PATH"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


def emit_mission_event(state, item, human_text, progress, **details):
    payload = {
        "phase": state.value if isinstance(state, MissionState) else str(state),
        "item": item,
        "human_text": human_text,
        "progress": float(progress),
        **details,
    }
    print(MISSION_EVENT_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def load_depth_model(device):
    print(f"Loading {DEPTH_MODEL_NAME} on {device.upper()}...")
    try:
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load Depth Anything V2 Small. Check your internet connection "
            "for the first run and confirm that dependencies installed correctly."
        ) from exc
    return processor, model


def load_segmentation_model(device):
    print(f"Loading {SEGMENTATION_MODEL_NAME} on {device.upper()}...")
    try:
        processor = AutoImageProcessor.from_pretrained(SEGMENTATION_MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGMENTATION_MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not load SegFormer B0 ADE20K segmentation. Check your internet "
            "connection for the first run and confirm dependencies installed correctly."
        ) from exc

    traversable_ids = label_ids_from_keywords(model.config.id2label, TRAVERSABLE_LABEL_KEYWORDS)
    if not traversable_ids:
        raise RuntimeError(
            "SegFormer loaded, but no traversable ADE20K labels matched the configured keywords."
        )
    print(f"Traversable classes: {traversable_ids}")
    return processor, model, traversable_ids


def label_ids_from_keywords(id2label, keywords):
    ids = []
    for raw_id, label in id2label.items():
        normalized = str(label).lower().replace("-", " ").replace("_", " ")
        if any(keyword in normalized for keyword in keywords):
            ids.append(int(raw_id))
    return set(ids)


def open_camera(role="floor", preferred_index=None, excluded_indices=()):
    if preferred_index is None and role == "floor":
        preferred_index = CAMERA_INDEX
    if preferred_index is None and role == "target":
        preferred_index = TARGET_CAMERA_INDEX
    tried = []
    for index, name in camera_candidates(preferred_index):
        if index in excluded_indices:
            continue
        tried.append(index)
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)

        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            camera_name = f" ({name})" if name else ""
            print(f"Using {role} camera index {index}{camera_name}.")
            return capture, index

        capture.release()

    raise RuntimeError(
        f"Could not open {role} webcam. Tried camera indices: {tried}. "
        "Check that the Logitech C920 is connected and not already in use."
    )


def camera_candidates(manual_index=None):
    if manual_index is not None:
        yield manual_index, "manual camera index"

    named_devices = list_directshow_camera_devices()
    preferred = []
    fallback = []
    for index, name in named_devices:
        normalized = name.lower()
        if any(keyword in normalized for keyword in PREFERRED_CAMERA_NAME_KEYWORDS):
            preferred.append((index, name))
        else:
            fallback.append((index, name))

    yielded = set()
    for index, name in [*preferred, *fallback]:
        if index not in yielded:
            yielded.add(index)
            yield index, name

    for index in FALLBACK_CAMERA_INDICES:
        if index not in yielded:
            yielded.add(index)
            yield index, "index probe"


def list_directshow_camera_devices():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("pygrabber is not installed; falling back to camera index probing.")
        return []
    except Exception as exc:
        print(f"Could not initialize DirectShow camera enumeration: {exc}")
        return []

    try:
        graph = FilterGraph()
        return list(enumerate(graph.get_input_devices()))
    except Exception as exc:
        print(f"Could not list DirectShow camera devices: {exc}")
        return []


@torch.inference_mode()
def estimate_depth(frame_bgr, processor, model, device):
    model_input = cv2.resize(
        frame_bgr,
        (DEPTH_INPUT_WIDTH, DEPTH_INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt", do_resize=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.unsqueeze(1)
    prediction = torch.nn.functional.interpolate(
        predicted_depth,
        size=(FRAME_HEIGHT, FRAME_WIDTH),
        mode="bicubic",
        align_corners=False,
    )

    depth = prediction.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    depth = cv2.GaussianBlur(depth, (DEPTH_BLUR_KERNEL, DEPTH_BLUR_KERNEL), 0)
    return normalize_depth_as_clearance(depth)


@torch.inference_mode()
def estimate_floor_mask(frame_bgr, processor, model, device, traversable_ids):
    model_input = cv2.resize(
        frame_bgr,
        (SEGMENTATION_INPUT_WIDTH, SEGMENTATION_INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame_rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt", do_resize=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(FRAME_HEIGHT, FRAME_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    labels = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
    return np.isin(labels, list(traversable_ids)).astype(np.float32)


def normalize_depth_as_clearance(depth):
    low, high = np.percentile(depth, [2, 98])
    if high - low < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    if DEPTH_VALUES_ARE_INVERSE:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def smooth_array(previous, current, alpha):
    if previous is None:
        return current.astype(np.float32)
    return (alpha * current + (1.0 - alpha) * previous).astype(np.float32)


def remove_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == component_id] = 1
    return cleaned


def keep_bottom_center_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    h, w = mask.shape
    seed = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(seed, (w // 2, h - 1), BOTTOM_CENTER_RADIUS_PIXELS, 1, -1)
    touched_labels = labels[(seed == 1) & (labels > 0)]
    if touched_labels.size == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    component_id = int(np.bincount(touched_labels).argmax())
    if stats[component_id, cv2.CC_STAT_AREA] < MIN_SAFE_REGION_AREA:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == component_id).astype(np.uint8)


def build_masks(depth_clearance, floor_probability):
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    raw_floor = (floor_probability >= FLOOR_MASK_THRESHOLD).astype(np.uint8)
    raw_floor = remove_small_components(raw_floor, SMALL_COMPONENT_AREA)
    raw_floor = cv2.morphologyEx(raw_floor, cv2.MORPH_CLOSE, kernel, iterations=2)
    floor_mask = cv2.erode(raw_floor, kernel, iterations=FLOOR_ERODE_ITERATIONS)

    if USE_DEPTH_OBSTACLE_MASK:
        near_threshold = float(np.percentile(depth_clearance, NEAR_OBSTACLE_PERCENTILE))
        near_threshold += NEAR_OBSTACLE_MARGIN
        obstacle_mask = (depth_clearance <= near_threshold).astype(np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=OBSTACLE_DILATE_ITERATIONS)
    else:
        obstacle_mask = np.zeros_like(floor_mask, dtype=np.uint8)

    safe_mask = ((floor_mask == 1) & (obstacle_mask == 0)).astype(np.uint8)
    safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    safe_mask = remove_small_components(safe_mask, SMALL_COMPONENT_AREA)
    connected_safe = keep_bottom_center_component(safe_mask)
    return floor_mask, obstacle_mask, connected_safe


def colorize_depth(depth):
    depth_u8 = np.uint8(np.clip(depth, 0.0, 1.0) * 255)
    return cv2.applyColorMap(depth_u8, DEPTH_COLOR_MAP)


def detect_orange_bottle(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(TARGET_ORANGE_HSV_LOW, dtype=np.uint8),
        np.array(TARGET_ORANGE_HSV_HIGH, dtype=np.uint8),
    )
    kernel = np.ones((TARGET_MORPH_KERNEL_SIZE, TARGET_MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < TARGET_MIN_AREA_PIXELS:
        return None, mask

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0:
        return None, mask

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    error_pixels = center_x - frame_bgr.shape[1] / 2.0
    distance_cm = (TARGET_KNOWN_WIDTH_CM * TARGET_FOCAL_LENGTH_PIXELS) / max(1.0, float(w))
    return {
        "bbox": (x, y, w, h),
        "area": area,
        "center": (center_x, center_y),
        "error_pixels": error_pixels,
        "distance_cm": distance_cm,
        "centered": abs(error_pixels) <= TARGET_CENTER_DEADBAND_PIXELS,
        "reachable": distance_cm <= TARGET_REACH_DISTANCE_CM,
    }, mask


def draw_target_view(frame_bgr, detection, mask):
    view = frame_bgr.copy()
    center_x = view.shape[1] // 2
    cv2.line(view, (center_x, 0), (center_x, view.shape[0]), (255, 255, 255), 1, cv2.LINE_AA)
    center_y = view.shape[0] // 2
    cv2.line(view, (0, center_y), (view.shape[1], center_y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(
        view,
        (center_x - TARGET_CENTER_DEADBAND_PIXELS, 0),
        (center_x - TARGET_CENTER_DEADBAND_PIXELS, view.shape[0]),
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        view,
        (center_x + TARGET_CENTER_DEADBAND_PIXELS, 0),
        (center_x + TARGET_CENTER_DEADBAND_PIXELS, view.shape[0]),
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    label = "NO BOTTLE"
    color = (40, 40, 255)
    if detection is not None:
        x, y, w, h = detection["bbox"]
        color = (40, 220, 40) if detection["centered"] and detection["reachable"] else (0, 180, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 3, cv2.LINE_AA)
        label = f"BOTTLE {detection['distance_cm']:.0f}cm x:{detection['error_pixels']:+.0f}px"
    cv2.putText(view, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(view, "TARGET CAMERA", (18, view.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def mask_to_bgr(mask, color):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def floor_bounds_by_rows(floor_mask):
    height, width = floor_mask.shape
    top_y = int(height * FLOOR_GRID_TOP_RATIO)
    y_values = np.linspace(height - 1, top_y, FLOOR_GRID_ROWS + 1).astype(np.int32)
    bounds = []
    last_left = width // 2
    last_right = width // 2

    for y in y_values:
        band_half = max(3, int(height / (FLOOR_GRID_ROWS * 2.5)))
        y0 = max(0, y - band_half)
        y1 = min(height, y + band_half + 1)
        xs = np.where(np.any(floor_mask[y0:y1, :] > 0, axis=0))[0]
        if xs.size >= FLOOR_GRID_MIN_WIDTH_PIXELS:
            left = int(xs[0])
            right = int(xs[-1])
            if bounds:
                left = int(round(0.7 * left + 0.3 * last_left))
                right = int(round(0.7 * right + 0.3 * last_right))
            bounds.append((int(y), left, right))
            last_left, last_right = left, right
        elif bounds:
            break

    if len(bounds) < 3:
        return []
    return bounds


def image_space_path_from_floor(floor_mask):
    bounds = floor_bounds_by_rows(floor_mask)
    if not bounds:
        return None

    centers = np.array([(left + right) / 2.0 for _, left, right in bounds], dtype=np.float32)
    target_index = int(round((len(centers) - 1) * PATH_FUTURE_TARGET_RATIO))
    target_index = int(np.clip(target_index, 0, len(centers) - 1))
    target_end = min(len(centers), target_index + PATH_FUTURE_TARGET_AVERAGE_POINTS)
    future_target_x = float(np.mean(centers[target_index:target_end]))

    points = []
    start_x = FRAME_WIDTH / 2.0
    for index, (y, left, right) in enumerate(bounds):
        row_center_x = (left + right) / 2.0
        progress = index / max(1, len(bounds) - 1)
        early_turn = progress ** PATH_EARLY_TURN_POWER
        future_curve_x = start_x * (1.0 - early_turn) + future_target_x * early_turn
        center_x = (
            PATH_FLOOR_CENTER_BLEND * row_center_x
            + (1.0 - PATH_FLOOR_CENTER_BLEND) * future_curve_x
        )
        center_x = int(round(np.clip(center_x, 0, FRAME_WIDTH - 1)))
        points.append([center_x, y])
    return np.array(points, dtype=np.int32)


def draw_floor_perspective_grid(image, floor_mask):
    bounds = floor_bounds_by_rows(floor_mask)
    if len(bounds) < 2:
        return

    overlay = image.copy()
    outlines = []
    for row_index in range(len(bounds) - 1):
        y0, left0, right0 = bounds[row_index]
        y1, left1, right1 = bounds[row_index + 1]
        for col_index in range(FLOOR_GRID_COLS):
            t0 = col_index / FLOOR_GRID_COLS
            t1 = (col_index + 1) / FLOOR_GRID_COLS
            x00 = int(round(left0 + (right0 - left0) * t0))
            x01 = int(round(left0 + (right0 - left0) * t1))
            x10 = int(round(left1 + (right1 - left1) * t0))
            x11 = int(round(left1 + (right1 - left1) * t1))
            poly = np.array([[x00, y0], [x01, y0], [x11, y1], [x10, y1]], dtype=np.int32)

            cx = int(round(np.mean(poly[:, 0])))
            cy = int(round(np.mean(poly[:, 1])))
            if not (0 <= cx < FRAME_WIDTH and 0 <= cy < FRAME_HEIGHT):
                continue
            if floor_mask[cy, cx] == 0:
                continue

            cv2.fillConvexPoly(overlay, poly, (35, 170, 80), cv2.LINE_AA)
            outlines.append(poly.reshape((-1, 1, 2)))

    cv2.addWeighted(overlay, FLOOR_GRID_ALPHA, image, 1.0 - FLOOR_GRID_ALPHA, 0, image)
    for poly in outlines:
        cv2.polylines(image, [poly], True, (215, 255, 215), 1, cv2.LINE_AA)


def draw_side_edge_guides(image, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=EDGE_HOUGH_THRESHOLD,
        minLineLength=EDGE_MIN_LINE_LENGTH,
        maxLineGap=EDGE_MAX_LINE_GAP,
    )
    if lines is None:
        return

    height, width = frame.shape[:2]
    candidates = []
    lines = np.asarray(lines).reshape(-1, 4)
    for line in lines:
        x1, y1, x2, y2 = [int(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            slope = 999.0
        else:
            slope = dy / dx
        if abs(slope) < SIDE_EDGE_MIN_ABS_SLOPE:
            continue
        length = float(np.hypot(dx, dy))
        bottomness = max(y1, y2) / height
        side_bias = abs(((x1 + x2) / 2.0) - width / 2.0) / (width / 2.0)
        candidates.append((length * (0.6 + bottomness + side_bias), x1, y1, x2, y2))

    candidates.sort(reverse=True)
    for _, x1, y1, x2, y2 in candidates[:SIDE_EDGE_MAX_LINES]:
        cv2.line(image, (x1, y1), (x2, y2), (80, 210, 255), 2, cv2.LINE_AA)


def camera_intrinsics():
    fx = FRAME_WIDTH / (2.0 * np.tan(np.deg2rad(CAMERA_HORIZONTAL_FOV_DEGREES) / 2.0))
    fy = FRAME_HEIGHT / (2.0 * np.tan(np.deg2rad(CAMERA_VERTICAL_FOV_DEGREES) / 2.0))
    cx = FRAME_WIDTH / 2.0
    cy = FRAME_HEIGHT / 2.0
    return fx, fy, cx, cy


def pixel_to_ground(u, v, pitch_degrees):
    fx, fy, cx, cy = camera_intrinsics()
    x = (u - cx) / fx
    y = (v - cy) / fy
    ray_camera = np.array([x, y, 1.0], dtype=np.float32)

    pitch = np.deg2rad(pitch_degrees)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    ray_down = cos_p * ray_camera[1] + sin_p * ray_camera[2]
    ray_forward = -sin_p * ray_camera[1] + cos_p * ray_camera[2]

    if ray_down <= 1e-5:
        return None

    scale = CAMERA_HEIGHT_METERS / ray_down
    ground_x = float(ray_camera[0] * scale)
    ground_z = float(ray_forward * scale)
    if ground_z <= 0.0:
        return None
    return ground_x, ground_z


def ground_to_pixel(ground_x, ground_z, pitch_degrees):
    fx, fy, cx, cy = camera_intrinsics()
    pitch = np.deg2rad(pitch_degrees)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)

    down = CAMERA_HEIGHT_METERS
    camera_x = ground_x
    camera_y = cos_p * down - sin_p * ground_z
    camera_z = sin_p * down + cos_p * ground_z
    if camera_z <= 1e-5:
        return None

    u = fx * (camera_x / camera_z) + cx
    v = fy * (camera_y / camera_z) + cy
    if not (0 <= u < FRAME_WIDTH and 0 <= v < FRAME_HEIGHT):
        return None
    return int(round(u)), int(round(v))


def grid_shape():
    rows = int(round(GRID_FORWARD_METERS / GRID_CELL_SIZE_METERS))
    cols = int(round(GRID_WIDTH_METERS / GRID_CELL_SIZE_METERS))
    return rows, cols


def ground_to_grid(ground_x, ground_z):
    rows, cols = grid_shape()
    col = int(round((ground_x + GRID_WIDTH_METERS / 2.0) / GRID_CELL_SIZE_METERS))
    row = int(round(ground_z / GRID_CELL_SIZE_METERS))
    if 0 <= row < rows and 0 <= col < cols:
        return row, col
    return None


def grid_to_ground(row, col):
    ground_x = col * GRID_CELL_SIZE_METERS - GRID_WIDTH_METERS / 2.0
    ground_z = row * GRID_CELL_SIZE_METERS
    return ground_x, ground_z


def sample_mask_at(mask, pixel, radius=2):
    if pixel is None:
        return 0.0
    x, y = pixel
    h, w = mask.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(np.mean(mask[y0:y1, x0:x1] > 0))


def floor_grid_from_masks(floor_mask, obstacle_mask, depth_clearance, pitch_degrees):
    rows, cols = grid_shape()
    floor_grid = np.zeros((rows, cols), dtype=np.uint8)
    obstacle_grid = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            center = ground_to_pixel(*grid_to_ground(row, col), pitch_degrees)
            if center is None:
                continue

            half = GRID_CELL_SIZE_METERS * 0.45
            ground_x, ground_z = grid_to_ground(row, col)
            sample_pixels = [
                center,
                ground_to_pixel(ground_x - half, ground_z - half, pitch_degrees),
                ground_to_pixel(ground_x + half, ground_z - half, pitch_degrees),
                ground_to_pixel(ground_x + half, ground_z + half, pitch_degrees),
                ground_to_pixel(ground_x - half, ground_z + half, pitch_degrees),
            ]
            floor_samples = sum(
                sample_mask_at(floor_mask, pixel, radius=3) >= 0.5 for pixel in sample_pixels
            )
            if floor_samples >= GRID_CELL_MIN_FLOOR_SAMPLES:
                floor_grid[row, col] = 1

            if sample_mask_at(obstacle_mask, center, radius=4) > 0.2:
                obstacle_grid[row, col] = 1
            elif USE_GRID_DEPTH_OBSTACLE_TRIM and depth_clearance[center[1], center[0]] < GRID_DEPTH_OBSTACLE_THRESHOLD:
                obstacle_grid[row, col] = 1

    kernel = np.ones((3, 3), np.uint8)
    floor_grid = cv2.morphologyEx(floor_grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    floor_grid = remove_small_components(floor_grid, max(10, SMALL_COMPONENT_AREA // 80))

    clearance_cells = max(1, int(round(ROBOT_CLEARANCE_METERS / GRID_CELL_SIZE_METERS)))
    obstacle_grid = cv2.dilate(obstacle_grid, kernel, iterations=clearance_cells)

    boundary_clearance_cells = max(1, int(round(FLOOR_BOUNDARY_CLEARANCE_METERS / GRID_CELL_SIZE_METERS)))
    floor_grid = cv2.erode(floor_grid, kernel, iterations=boundary_clearance_cells)

    safe_grid = ((floor_grid == 1) & (obstacle_grid == 0)).astype(np.uint8)
    return floor_grid, obstacle_grid, keep_bottom_center_grid_component(safe_grid)


def pitch_candidates():
    if not AUTO_CALIBRATE_CAMERA_PITCH:
        return [CAMERA_PITCH_DEGREES]
    values = np.arange(
        AUTO_PITCH_MIN_DEGREES,
        AUTO_PITCH_MAX_DEGREES + AUTO_PITCH_STEP_DEGREES * 0.5,
        AUTO_PITCH_STEP_DEGREES,
    )
    preferred = [CAMERA_PITCH_DEGREES]
    ordered = sorted(values, key=lambda value: abs(float(value) - CAMERA_PITCH_DEGREES))
    return [*preferred, *[float(value) for value in ordered if abs(float(value) - CAMERA_PITCH_DEGREES) > 1e-6]]


def choose_pitch_and_grid(floor_mask, obstacle_mask, depth_clearance, previous_pitch):
    candidates = pitch_candidates()
    if previous_pitch is not None:
        candidates = [previous_pitch, *[pitch for pitch in candidates if abs(pitch - previous_pitch) > 1e-6]]

    best = None
    for pitch in candidates:
        grid_floor, grid_obstacle, grid_safe = floor_grid_from_masks(
            floor_mask, obstacle_mask, depth_clearance, pitch
        )
        score = int(np.count_nonzero(grid_safe))
        if best is None or score > best["score"]:
            best = {
                "pitch": float(pitch),
                "score": score,
                "floor": grid_floor,
                "obstacle": grid_obstacle,
                "safe": grid_safe,
            }

    if best is None:
        empty = np.zeros(grid_shape(), dtype=np.uint8)
        return CAMERA_PITCH_DEGREES, empty, empty, empty
    return best["pitch"], best["floor"], best["obstacle"], best["safe"]


def keep_bottom_center_grid_component(grid):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(grid, dtype=np.uint8)

    rows, cols = grid.shape
    bottom_rows = max(2, int(round(0.35 / GRID_CELL_SIZE_METERS)))
    center_cols = max(2, int(round(0.35 / GRID_CELL_SIZE_METERS)))
    seed = np.zeros_like(grid, dtype=np.uint8)
    c0 = max(0, cols // 2 - center_cols)
    c1 = min(cols, cols // 2 + center_cols + 1)
    seed[:bottom_rows, c0:c1] = 1

    touched_labels = labels[(seed == 1) & (labels > 0)]
    if touched_labels.size == 0:
        return np.zeros_like(grid, dtype=np.uint8)

    component_id = int(np.bincount(touched_labels).argmax())
    return (labels == component_id).astype(np.uint8)


def path_from_floor_grid(safe_grid, pitch_degrees):
    if np.count_nonzero(safe_grid) < MIN_GRID_ROW_CELLS:
        return None, None

    rows, cols = safe_grid.shape
    row_values = np.linspace(0, rows - 1, GRID_PATH_STEPS).astype(np.int32)
    grid_points = []
    last_col = cols // 2

    for row in row_values:
        safe_cols = np.where(safe_grid[row, :] > 0)[0]
        if safe_cols.size < MIN_GRID_ROW_CELLS:
            if grid_points:
                break
            return None, None

        left = int(safe_cols[0])
        right = int(safe_cols[-1])
        center_col = int(round((left + right) / 2.0))
        center_col = int(round(0.70 * center_col + 0.30 * last_col))
        center_col = int(np.clip(center_col, 0, cols - 1))
        grid_points.append([row, center_col])
        last_col = center_col

    if len(grid_points) < max(8, GRID_PATH_STEPS // 4):
        return None, None

    image_points = []
    for row, col in grid_points:
        pixel = ground_to_pixel(*grid_to_ground(row, col), pitch_degrees)
        if pixel is not None:
            image_points.append(pixel)

    if len(image_points) < max(8, len(grid_points) // 2):
        return None, None
    return np.array(image_points, dtype=np.int32), np.array(grid_points, dtype=np.int32)


def smooth_path(previous_points, selected_points):
    if previous_points is None:
        return selected_points.astype(np.float32)
    if previous_points.shape != selected_points.shape:
        return selected_points.astype(np.float32)
    return (
        PATH_EMA_ALPHA * selected_points.astype(np.float32)
        + (1.0 - PATH_EMA_ALPHA) * previous_points
    )


def direction_from_path(path_points, frame_width):
    if path_points is None:
        return STOP_TEXT

    offset = path_horizontal_error(path_points, frame_width) * (frame_width / 2.0)
    if offset < -TURN_DEADBAND_PIXELS:
        return "LEFT"
    if offset > TURN_DEADBAND_PIXELS:
        return "RIGHT"
    return "FORWARD"


class SteeringPid:
    def __init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def update(self, error):
        now = time.monotonic()
        if self.previous_time is None:
            dt = SERIAL_SEND_INTERVAL_SECONDS
        else:
            dt = max(1e-3, now - self.previous_time)

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT))
        derivative = (error - self.previous_error) / dt

        self.previous_error = error
        self.previous_time = now

        correction = PID_KP * error + PID_KI * self.integral + PID_KD * derivative
        return float(np.clip(correction, -MAX_STEERING_CORRECTION, MAX_STEERING_CORRECTION))


class StableMaskFilter:
    def __init__(self):
        self.stable_probability = None
        self.pending_mask = None
        self.pending_count = 0

    def update(self, candidate_mask):
        candidate = candidate_mask.astype(np.float32)
        if self.stable_probability is None:
            self.stable_probability = candidate
            return candidate_mask.astype(np.uint8), False

        stable_mask = self.stable_probability >= 0.5
        changed_fraction = float(np.mean(stable_mask != (candidate > 0.5)))
        sudden_change = changed_fraction >= SUDDEN_MASK_CHANGE_FRACTION

        if sudden_change:
            if self.pending_mask is not None:
                pending_changed = float(np.mean((self.pending_mask > 0.5) != (candidate > 0.5)))
            else:
                pending_changed = 1.0

            if pending_changed < SUDDEN_MASK_CHANGE_FRACTION:
                self.pending_count += 1
            else:
                self.pending_mask = candidate
                self.pending_count = 1

            if self.pending_count < SUDDEN_MASK_CONFIRM_FRAMES:
                return stable_mask.astype(np.uint8), True

        self.pending_mask = None
        self.pending_count = 0
        self.stable_probability = (
            STABLE_SAFE_MASK_BLEND_ALPHA * candidate
            + (1.0 - STABLE_SAFE_MASK_BLEND_ALPHA) * self.stable_probability
        )
        return (self.stable_probability >= 0.5).astype(np.uint8), False


def path_horizontal_error(path_points, frame_width):
    if path_points is None or len(path_points) == 0:
        return 0.0

    index = int(round((len(path_points) - 1) * PATH_LOOKAHEAD_RATIO))
    start = max(0, index - PATH_ERROR_AVERAGE_LAST_POINTS + 1)
    target_x = float(np.mean(path_points[start : index + 1, 0]))
    error_pixels = target_x - frame_width / 2.0
    return float(np.clip(error_pixels / (frame_width / 2.0), -1.0, 1.0))


def servo_values_from_path(path_points, controller, frame_width):
    if path_points is None:
        controller.reset()
        return SERVO_STOP, SERVO_STOP, 0.0, 0.0

    error = path_horizontal_error(path_points, frame_width)
    correction = controller.update(error)

    if abs(error) >= PIVOT_TURN_ERROR_THRESHOLD:
        turn_direction = 1.0 if error > 0.0 else -1.0
        turn_amount = (abs(error) - PIVOT_TURN_ERROR_THRESHOLD) / max(1e-6, 1.0 - PIVOT_TURN_ERROR_THRESHOLD)
        turn_speed = PIVOT_TURN_MIN_SPEED + turn_amount * (PIVOT_TURN_MAX_SPEED - PIVOT_TURN_MIN_SPEED)
        left_speed = turn_direction * turn_speed
        right_speed = -turn_direction * turn_speed
    else:
        left_speed = BASE_SERVO_SPEED
        right_speed = BASE_SERVO_SPEED

    left_speed = float(np.clip(left_speed, -MAX_SERVO_SPEED, MAX_SERVO_SPEED))
    right_speed = float(np.clip(right_speed, -MAX_SERVO_SPEED, MAX_SERVO_SPEED))

    left_value = SERVO_STOP + LEFT_SERVO_FORWARD_SIGN * left_speed
    right_value = SERVO_STOP + RIGHT_SERVO_FORWARD_SIGN * right_speed
    left_value = int(round(np.clip(left_value, 0, 180)))
    right_value = int(round(np.clip(right_value, 0, 180)))
    return left_value, right_value, error, correction


def servo_values_for_tank_speeds(left_speed, right_speed):
    left_speed = float(np.clip(left_speed, -MAX_SERVO_SPEED, MAX_SERVO_SPEED))
    right_speed = float(np.clip(right_speed, -MAX_SERVO_SPEED, MAX_SERVO_SPEED))
    left_value = SERVO_STOP + LEFT_SERVO_FORWARD_SIGN * left_speed
    right_value = SERVO_STOP + RIGHT_SERVO_FORWARD_SIGN * right_speed
    return int(round(np.clip(left_value, 0, 180))), int(round(np.clip(right_value, 0, 180)))


def servo_values_from_target(detection):
    if detection is None:
        return None

    error = float(np.clip(detection["error_pixels"] / (FRAME_WIDTH / 2.0), -1.0, 1.0))
    if detection["centered"] and detection["reachable"]:
        return SERVO_STOP, SERVO_STOP, "TARGET_STOP", error, 0.0

    if detection["centered"] and not detection["reachable"]:
        turn_direction = 1.0 if TARGET_CAMERA_SIDE.lower() == "right" else -1.0
        left_value, right_value = servo_values_for_tank_speeds(
            turn_direction * TARGET_APPROACH_SPEED,
            -turn_direction * TARGET_APPROACH_SPEED,
        )
        return left_value, right_value, "TARGET_APPROACH_SIDE", error, TARGET_APPROACH_SPEED

    turn_direction = 1.0 if error > 0.0 else -1.0
    turn_amount = abs(error)
    turn_speed = TARGET_TURN_MIN_SPEED + turn_amount * (TARGET_TURN_MAX_SPEED - TARGET_TURN_MIN_SPEED)
    left_value, right_value = servo_values_for_tank_speeds(
        turn_direction * turn_speed,
        -turn_direction * turn_speed,
    )
    return left_value, right_value, "TARGET_ALIGN", error, turn_speed


def open_arduino_serial():
    if not ENABLE_SERVO_CONTROL or ARDUINO_SERIAL_PORT is None:
        print("Servo serial control disabled. Set ENABLE_SERVO_CONTROL=True and ARDUINO_SERIAL_PORT='COMx' to enable.")
        return None
    if serial is None:
        raise RuntimeError("pyserial is not installed. Run: pip install -r requirements.txt")

    try:
        connection = serial.Serial(ARDUINO_SERIAL_PORT, ARDUINO_BAUD_RATE, timeout=0.1)
    except serial.SerialException as exc:
        raise RuntimeError(f"Could not open Arduino serial port {ARDUINO_SERIAL_PORT}: {exc}") from exc

    time.sleep(ARDUINO_RESET_WAIT_SECONDS)
    connection.reset_input_buffer()
    print(f"Servo serial control enabled on {ARDUINO_SERIAL_PORT}.")
    return connection


def send_servo_values(connection, left_value, right_value, axis_value=SERVO_STOP):
    if connection is None:
        return
    message = f"L:{left_value} R:{right_value} A:{axis_value}\n"
    connection.write(message.encode("ascii"))
    connection.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live floor path planner with optional pill-bottle mission mode."
    )
    parser.add_argument(
        "--mission",
        default="",
        help="Run as a fetch mission for this item. Arm actions are simulated for now.",
    )
    parser.add_argument(
        "--enable-payload-sequence",
        action="store_true",
        default=os.getenv("CORGI_SINGLECAM_PAYLOAD_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        help="Enable physical D3 winch motion. Leave off until the arm is installed.",
    )
    parser.add_argument(
        "--target-rotate-180",
        action=argparse.BooleanOptionalAction,
        default=MISSION_TARGET_ROTATE_180,
        help="Rotate the side camera image by 180 degrees.",
    )
    return parser.parse_args()


def run_timed_servo_phase(connection, seconds, left=SERVO_STOP, right=SERVO_STOP, axis=SERVO_STOP, label=""):
    end_time = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < end_time:
        send_servo_values(connection, left, right, axis)
        remaining = end_time - time.monotonic()
        print(f"{label} {max(0.0, remaining):.1f}s remaining", end="\r")
        time.sleep(SERIAL_SEND_INTERVAL_SECONDS)
    send_servo_values(connection, SERVO_STOP, SERVO_STOP, SERVO_STOP)
    if label:
        print(f"{label} complete.{' ' * 24}")


def draw_path(image, path_points, label):
    draw_robot_path_boxes(image, path_points)
    if path_points is not None:
        pts = np.round(path_points).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.polylines(image, [pts], False, (40, 220, 40), 4, cv2.LINE_AA)

    color = (50, 220, 50) if label != STOP_TEXT else (40, 40, 255)
    cv2.putText(image, label, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)


def robot_width_at_y(y):
    progress = 1.0 - float(np.clip(y / max(1, FRAME_HEIGHT - 1), 0.0, 1.0))
    width = ROBOT_WIDTH_PIXELS_BOTTOM * (1.0 - progress) + ROBOT_WIDTH_PIXELS_TOP * progress
    return float(width)


def draw_robot_path_boxes(image, path_points):
    if path_points is None or len(path_points) < 2:
        return

    points = np.round(path_points).astype(np.float32)
    overlay = image.copy()
    outlines = []
    step = max(1, ROBOT_PATH_BOX_STEP_POINTS)

    for index in range(0, len(points) - 1, step):
        p0 = points[index]
        p1 = points[min(index + step, len(points) - 1)]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length < 1e-3:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32) / length
        half0 = robot_width_at_y(p0[1]) / 2.0
        half1 = robot_width_at_y(p1[1]) / 2.0
        poly = np.array(
            [
                p0 - normal * half0,
                p0 + normal * half0,
                p1 + normal * half1,
                p1 - normal * half1,
            ],
            dtype=np.int32,
        )
        poly[:, 0] = np.clip(poly[:, 0], 0, FRAME_WIDTH - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, FRAME_HEIGHT - 1)
        cv2.fillConvexPoly(overlay, poly, (30, 30, 230), cv2.LINE_AA)
        outlines.append(poly.reshape((-1, 1, 2)))

    cv2.addWeighted(overlay, ROBOT_PATH_BOX_ALPHA, image, 1.0 - ROBOT_PATH_BOX_ALPHA, 0, image)
    for poly in outlines:
        cv2.polylines(image, [poly], True, (80, 80, 255), 2, cv2.LINE_AA)


def draw_debug_status(image, floor_mask, safe_mask, grid_safe, path, pitch_degrees, servo_status, mask_change_held):
    floor_pixels = int(np.count_nonzero(floor_mask))
    safe_pixels = int(np.count_nonzero(safe_mask))
    grid_cells = int(np.count_nonzero(grid_safe)) if grid_safe is not None else 0
    path_points = 0 if path is None else len(path)
    hold_text = " HOLD" if mask_change_held else ""
    text = f"floor:{floor_pixels} safe:{safe_pixels} grid:{grid_cells} path:{path_points}{hold_text}"
    cv2.putText(image, text, (18, FRAME_HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if servo_status is not None:
        servo_text = (
            f"{servo_status.get('mode', 'PATH')} servo L:{servo_status['left']} R:{servo_status['right']} "
            f"A:{servo_status.get('axis', SERVO_STOP)} "
            f"err:{servo_status['error']:+.2f} corr:{servo_status['correction']:+.1f}"
        )
        cv2.putText(image, servo_text, (18, FRAME_HEIGHT - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def grid_cell_corners_to_pixels(row, col, step_cells, pitch_degrees):
    z0 = row * GRID_CELL_SIZE_METERS
    z1 = (row + step_cells) * GRID_CELL_SIZE_METERS
    x0 = col * GRID_CELL_SIZE_METERS - GRID_WIDTH_METERS / 2.0
    x1 = (col + step_cells) * GRID_CELL_SIZE_METERS - GRID_WIDTH_METERS / 2.0
    corners = [
        ground_to_pixel(x0, z0, pitch_degrees),
        ground_to_pixel(x1, z0, pitch_degrees),
        ground_to_pixel(x1, z1, pitch_degrees),
        ground_to_pixel(x0, z1, pitch_degrees),
    ]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners, dtype=np.int32)


def draw_perspective_floor_rectangles(image, safe_grid, obstacle_grid, pitch_degrees):
    if safe_grid is None:
        return

    overlay = image.copy()
    outlines = []
    rows, cols = safe_grid.shape
    step = max(1, PERSPECTIVE_RECTANGLE_STEP_CELLS)

    for row in range(0, rows - step, step):
        for col in range(0, cols - step, step):
            cell_block = safe_grid[row : row + step, col : col + step]
            obstacle_block = obstacle_grid[row : row + step, col : col + step]
            if np.count_nonzero(cell_block) < PERSPECTIVE_RECTANGLE_MIN_SAFE_CELLS:
                continue
            if obstacle_grid is not None and np.count_nonzero(obstacle_block) > 0:
                continue

            corners = grid_cell_corners_to_pixels(row, col, step, pitch_degrees)
            if corners is None:
                continue

            cv2.fillConvexPoly(overlay, corners, (35, 170, 80), cv2.LINE_AA)
            outlines.append(corners.reshape((-1, 1, 2)))

    cv2.addWeighted(overlay, PERSPECTIVE_RECTANGLE_ALPHA, image, 1.0 - PERSPECTIVE_RECTANGLE_ALPHA, 0, image)
    for corners in outlines:
        cv2.polylines(image, [corners], True, (210, 255, 210), 1, cv2.LINE_AA)


def grid_view_to_bgr(safe_grid, obstacle_grid, grid_path):
    view = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    if safe_grid is None:
        return view

    safe_view = mask_to_bgr(safe_grid, (60, 180, 60))
    if obstacle_grid is not None:
        safe_view[obstacle_grid > 0] = (40, 40, 220)
    safe_view = cv2.flip(safe_view, 0)
    view = cv2.resize(safe_view, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)

    if grid_path is not None:
        rows, cols = safe_grid.shape
        scale_x = FRAME_WIDTH / cols
        scale_y = FRAME_HEIGHT / rows
        points = []
        for row, col in grid_path:
            x = int(round(col * scale_x))
            y = int(round(FRAME_HEIGHT - 1 - row * scale_y))
            points.append([x, y])
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(view, [pts], False, (255, 255, 255), 8, cv2.LINE_AA)
        cv2.polylines(view, [pts], False, (40, 220, 40), 4, cv2.LINE_AA)

    return view


def build_display(frame, depth, floor_mask, safe_mask, obstacle_mask, path, grid_safe, grid_obstacle, grid_path, label, pitch_degrees, servo_status, mask_change_held, target_view=None):
    camera_view = frame.copy()
    if USE_IMAGE_SPACE_PERSPECTIVE_GRID:
        draw_floor_perspective_grid(camera_view, safe_mask)
        draw_side_edge_guides(camera_view, frame)
    else:
        draw_perspective_floor_rectangles(camera_view, grid_safe, grid_obstacle, pitch_degrees)
    draw_path(camera_view, path, label)
    draw_debug_status(camera_view, floor_mask, safe_mask, grid_safe, path, pitch_degrees, servo_status, mask_change_held)
    cv2.putText(camera_view, "FLOOR / PATH CAMERA", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    depth_view = colorize_depth(depth)
    cv2.putText(depth_view, "DEPTH CLEARANCE", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    floor_view = mask_to_bgr(floor_mask, (80, 200, 80))
    cv2.putText(floor_view, "FLOOR MASK", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if USE_IMAGE_SPACE_PERSPECTIVE_GRID:
        safe_view = mask_to_bgr(grid_safe, (60, 180, 60))
        safe_view[grid_obstacle > 0] = (40, 40, 220)
        cv2.putText(safe_view, "SAFE FLOOR MASK", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        safe_view = grid_view_to_bgr(grid_safe, grid_obstacle, grid_path)
        cv2.putText(safe_view, "TOP-DOWN FLOOR SPACE", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if target_view is None:
        target_view = np.zeros_like(camera_view)
        cv2.putText(target_view, "TARGET CAMERA OFF", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2, cv2.LINE_AA)
    else:
        target_view = cv2.resize(target_view, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

    camera_pair = np.hstack([camera_view, target_view])
    model_pair = np.hstack([depth_view, floor_view])
    safety_pair = np.hstack([safe_view, np.zeros_like(safe_view)])
    cv2.putText(safety_pair[:, FRAME_WIDTH:], "RESERVED / ARM STATUS", (FRAME_WIDTH + 18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return np.vstack([camera_pair, model_pair, safety_pair])


def main():
    args = parse_args()
    mission_mode = bool(args.mission.strip())
    mission_bottle_stable_frames = 0
    mission_state = MissionState.PATH_FOLLOWING
    mission_complete = False
    if mission_mode:
        print(f"Mission mode enabled for: {args.mission}")
        print(f"Physical payload sequence enabled: {args.enable_payload_sequence}")
        emit_mission_event(
            mission_state,
            args.mission,
            f"following the camera path toward the {args.mission}",
            0.15,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        depth_processor, depth_model = load_depth_model(device)
        seg_processor, seg_model, traversable_ids = load_segmentation_model(device)
        capture, floor_camera_index = open_camera("floor", CAMERA_INDEX)
        target_capture = None
        if ENABLE_TARGET_CAMERA:
            try:
                target_capture, _ = open_camera(
                    "target",
                    TARGET_CAMERA_INDEX,
                    excluded_indices=(floor_camera_index,),
                )
            except Exception as exc:
                print(f"Target camera disabled: {exc}")
        servo_connection = open_arduino_serial()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    smoothed_depth = None
    smoothed_floor = None
    smoothed_obstacle = None
    smoothed_path = None
    smoothed_grid_path = None
    selected_pitch_degrees = CAMERA_PITCH_DEGREES
    latest_floor_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    frame_index = 0
    steering_controller = SteeringPid()
    safe_mask_filter = StableMaskFilter()
    last_servo_send_time = 0.0
    servo_status = None
    mask_change_held = False
    target_detection = None
    target_view = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("ERROR: Could not read a frame from the webcam.", file=sys.stderr)
                return 1

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            target_view = None
            target_detection = None
            if target_capture is not None:
                target_ok, target_frame = target_capture.read()
                if target_ok and target_frame is not None:
                    target_frame = cv2.resize(
                        target_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA
                    )
                    if args.target_rotate_180:
                        target_frame = cv2.rotate(target_frame, cv2.ROTATE_180)
                    target_detection, target_mask = detect_orange_bottle(target_frame)
                    target_view = draw_target_view(target_frame, target_detection, target_mask)

            should_infer = frame_index % max(1, INFERENCE_EVERY_N_FRAMES) == 0
            if should_infer or smoothed_depth is None:
                depth = estimate_depth(frame, depth_processor, depth_model, device)
                floor_probability = estimate_floor_mask(
                    frame, seg_processor, seg_model, device, traversable_ids
                )
                smoothed_depth = smooth_array(smoothed_depth, depth, DEPTH_EMA_ALPHA)
                smoothed_floor = smooth_array(smoothed_floor, floor_probability, FLOOR_MASK_EMA_ALPHA)
                latest_floor_mask = smoothed_floor

            floor_mask, obstacle_mask, safe_mask = build_masks(smoothed_depth, latest_floor_mask)
            smoothed_obstacle = smooth_array(
                smoothed_obstacle, obstacle_mask.astype(np.float32), OBSTACLE_MASK_EMA_ALPHA
            )
            obstacle_mask = (smoothed_obstacle >= OBSTACLE_MASK_THRESHOLD).astype(np.uint8)
            safe_mask = ((safe_mask == 1) & (obstacle_mask == 0)).astype(np.uint8)
            safe_mask = keep_bottom_center_component(safe_mask)
            safe_mask, mask_change_held = safe_mask_filter.update(safe_mask)
            safe_mask = keep_bottom_center_component(safe_mask)

            if USE_IMAGE_SPACE_PERSPECTIVE_GRID:
                grid_floor = floor_mask
                grid_obstacle = obstacle_mask
                grid_safe = safe_mask
                selected_path = image_space_path_from_floor(safe_mask)
                selected_grid_path = None
            else:
                selected_pitch_degrees, grid_floor, grid_obstacle, grid_safe = choose_pitch_and_grid(
                    floor_mask, obstacle_mask, smoothed_depth, selected_pitch_degrees
                )
                selected_path, selected_grid_path = path_from_floor_grid(grid_safe, selected_pitch_degrees)
            if selected_path is None:
                smoothed_path = None
                smoothed_grid_path = None
                label = STOP_TEXT
            else:
                smoothed_path = smooth_path(smoothed_path, selected_path)
                if selected_grid_path is None:
                    smoothed_grid_path = None
                else:
                    smoothed_grid_path = smooth_path(smoothed_grid_path, selected_grid_path)
                label = direction_from_path(smoothed_path, frame.shape[1])

            if mission_mode and mission_state == MissionState.PATH_FOLLOWING and target_detection is not None:
                mission_state = MissionState.TARGET_CENTERING
                steering_controller.reset()
                emit_mission_event(
                    mission_state, args.mission, f"found the {args.mission}; centering it", 0.45
                )

            if mission_mode and mission_state == MissionState.TARGET_CENTERING:
                steering_controller.reset()
                if target_detection is None:
                    left_value = right_value = SERVO_STOP
                    error = correction = 0.0
                    mode = "TARGET_LOST_STOP"
                    mission_bottle_stable_frames = 0
                else:
                    error_pixels = float(target_detection["error_pixels"])
                    error = float(np.clip(error_pixels / (FRAME_WIDTH / 2.0), -1.0, 1.0))
                    if abs(error_pixels) <= MISSION_TARGET_CENTER_DEADBAND_PIXELS:
                        left_value = right_value = SERVO_STOP
                        correction = 0.0
                        mode = "TARGET_CENTERED_STOP"
                        mission_bottle_stable_frames += 1
                    else:
                        mission_bottle_stable_frames = 0
                        speed = float(np.clip(
                            abs(error) * MISSION_TARGET_FB_KP,
                            MISSION_TARGET_FB_MIN_SPEED,
                            MISSION_TARGET_FB_MAX_SPEED,
                        ))
                        speed *= MISSION_TARGET_FB_SIGN * (1.0 if error > 0.0 else -1.0)
                        left_value, right_value = servo_values_for_tank_speeds(speed, speed)
                        correction = speed
                        mode = "TARGET_FB_CENTER"

                if mission_bottle_stable_frames >= MISSION_TARGET_CENTER_STABLE_FRAMES:
                    mission_state = MissionState.TARGET_LOCKED
                    send_servo_values(servo_connection, SERVO_STOP, SERVO_STOP, SERVO_STOP)
                    emit_mission_event(
                        mission_state, args.mission, f"centered on the {args.mission}", 0.58
                    )
                    mission_state = MissionState.ARM_PICK_PLACEHOLDER
                    emit_mission_event(
                        mission_state, args.mission,
                        "arm pickup placeholder; physical arm is not installed", 0.66, simulated=True
                    )
                    time.sleep(MISSION_ARM_PICK_SECONDS)
                    if args.enable_payload_sequence:
                        mission_state = MissionState.WINCH_UP
                        emit_mission_event(mission_state, args.mission, "raising the arm", 0.74)
                        run_timed_servo_phase(
                            servo_connection, MISSION_WINCH_UP_SECONDS,
                            axis=MISSION_AXIS_UP_VALUE, label="Winch up"
                        )
                    else:
                        emit_mission_event(
                            MissionState.WINCH_UP, args.mission,
                            "winch-up dry run; payload sequence is disabled", 0.74, simulated=True
                        )
                    mission_state = MissionState.BASKET_DROP_PLACEHOLDER
                    emit_mission_event(
                        mission_state, args.mission,
                        "basket drop placeholder; physical arm is not installed", 0.82, simulated=True
                    )
                    time.sleep(MISSION_BASKET_DROP_SECONDS)
                    if args.enable_payload_sequence:
                        mission_state = MissionState.WINCH_DOWN
                        emit_mission_event(mission_state, args.mission, "lowering the arm", 0.9)
                        run_timed_servo_phase(
                            servo_connection, MISSION_WINCH_DOWN_SECONDS,
                            axis=MISSION_AXIS_DOWN_VALUE, label="Winch down"
                        )
                    else:
                        emit_mission_event(
                            MissionState.WINCH_DOWN, args.mission,
                            "winch-down dry run; payload sequence is disabled", 0.9, simulated=True
                        )
                    mission_state = MissionState.RESUMING_PATH
                    mission_resume_until = time.monotonic() + MISSION_RESUME_PATH_SECONDS
                    emit_mission_event(
                        mission_state, args.mission, "resuming the planned path", 0.96
                    )
                    mission_bottle_stable_frames = 0

            if mission_mode and mission_state in (
                MissionState.TARGET_LOCKED,
                MissionState.ARM_PICK_PLACEHOLDER,
                MissionState.WINCH_UP,
                MissionState.BASKET_DROP_PLACEHOLDER,
                MissionState.WINCH_DOWN,
            ):
                left_value = right_value = SERVO_STOP
                error = correction = 0.0
                mode = mission_state.value
            elif mission_mode and mission_state == MissionState.RESUMING_PATH:
                left_value, right_value, error, correction = servo_values_from_path(
                    smoothed_path, steering_controller, frame.shape[1]
                )
                mode = "PATH"
                if time.monotonic() >= mission_resume_until:
                    mission_state = MissionState.COMPLETE
                    mission_complete = True
                    emit_mission_event(
                        mission_state, args.mission, f"camera mission for {args.mission} complete", 1.0
                    )
            elif not (mission_mode and mission_state == MissionState.TARGET_CENTERING):
                left_value, right_value, error, correction = servo_values_from_path(
                    smoothed_path, steering_controller, frame.shape[1]
                )
                mode = "PATH"
            axis_value = SERVO_STOP
            servo_status = {
                "left": left_value,
                "right": right_value,
                "axis": axis_value,
                "error": error,
                "correction": correction,
                "mode": mode,
            }

            now = time.monotonic()
            if now - last_servo_send_time >= SERIAL_SEND_INTERVAL_SECONDS:
                send_servo_values(servo_connection, left_value, right_value, axis_value)
                last_servo_send_time = now

            display = build_display(
                frame,
                smoothed_depth,
                floor_mask,
                safe_mask,
                obstacle_mask,
                smoothed_path,
                grid_safe,
                grid_obstacle,
                smoothed_grid_path,
                label,
                selected_pitch_degrees,
                servo_status,
                mask_change_held,
                target_view,
            )
            cv2.imshow("Depth + SegFormer Safe Path - press Q or Esc to quit", display)

            if mission_complete:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            frame_index += 1

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if mission_mode:
            emit_mission_event(MissionState.FAILED, args.mission, str(exc), 1.0)
        print(f"ERROR: Runtime failure: {exc}", file=sys.stderr)
        return 1
    finally:
        if "servo_connection" in locals() and servo_connection is not None:
            send_servo_values(servo_connection, SERVO_STOP, SERVO_STOP, SERVO_STOP)
            servo_connection.close()
        if "target_capture" in locals() and target_capture is not None:
            target_capture.release()
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
