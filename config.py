# ============================================
# STATIC ENVIRONMENT SYSTEM - CONFIGURATION
# ============================================

# --- Camera Settings ---
CAMERA_SOURCE = 0  # 0 = Webcam
# CAMERA_SOURCE = "rtsp://username:password@ip:554/stream"  # IP Camera
# CAMERA_SOURCE = "video.mp4"  # Video File

# --- YOLO Model ---
MODEL_PATH = "yolov8n.pt"  # Nano = Lightweight, fits in 4GB VRAM easily
VISION_DEVICE = "cpu"  # Leave accelerator memory for local text reasoning.

# --- Static Detection Settings ---
MOTION_THRESHOLD    = 0.10   # If 10% of object area is moving = moving object
STATIC_CONFIRM_FRAMES = 30   # Frames before confirming object is static
MIN_CONFIDENCE      = 0.40   # Minimum detection confidence

# --- Background Subtractor Settings ---
BG_HISTORY          = 500    # Frames for background model
BG_THRESHOLD        = 50     # Sensitivity of motion detection
BG_DETECT_SHADOWS   = False  # Ignore shadows

# --- Display ---
SHOW_MOTION_MASK    = True   # Show motion mask window
SHOW_COORDS         = True   # Show coordinates on frame

# --- Context / LLM Reasoning / Alerts ---
LLM_ENABLED = True
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_BASE_URL = OLLAMA_HOST  # Backward-compatible alias.
OLLAMA_MODEL = "phi3:mini"
OLLAMA_TIMEOUT = 30
OLLAMA_TIMEOUT_SECONDS = OLLAMA_TIMEOUT  # Backward-compatible alias.
LLM_ANALYSIS_INTERVAL_FRAMES = 30
LLM_UNCERTAIN_CONFIDENCE_THRESHOLD = 0.75

# --- Scene Context ---
CAMERA_LOCATION = "Entrance"
# Normalized [x1, y1, x2, y2] regions, e.g.
# {"name": "restricted_area", "bbox": [0.7, 0.0, 1.0, 1.0]}
RESTRICTED_ZONES = []
HISTORY_FRAMES = 10
HISTORY_WINDOW_FRAMES = HISTORY_FRAMES
IDLE_SPEED_BODY_LENGTHS_PER_SECOND = 0.15
RUNNING_SPEED_BODY_LENGTHS_PER_SECOND = 2.0
SUDDEN_SPEEDUP_RATIO = 1.8
VELOCITY_THRESHOLD = 5.0
LOITER_RADIUS = 50.0
LOITER_TIME_SECONDS = 30.0
LOITER_TRACK_TIMEOUT_SECONDS = 2.0

# --- Rule-Based Alerts ---
ABANDONED_ITEM_CLASSES = ["backpack", "handbag", "suitcase", "package", "box"]
ABANDONED_ITEM_SECONDS = 30
PERSON_ITEM_DISTANCE_NORMALIZED = 0.12
RAPID_APPROACH_DISTANCE_BODY_LENGTHS = 2.0
RAPID_APPROACH_SPEED_BODY_LENGTHS_PER_SECOND = 1.5
ABANDONED_OBJECT_SECONDS = ABANDONED_ITEM_SECONDS
AFTER_HOURS_START = "22:00"
AFTER_HOURS_END = "06:00"
LOITER_MIN_OBSERVATIONS = 6
CROWDING_MIN_PEOPLE = 5
CROWDING_RADIUS_NORMALIZED = 0.15
CROSS_CAMERA_WINDOW_SECONDS = 60.0

# Camera entries may use an integer camera index, file path, or stream URL.
# An empty list falls back to the legacy single-camera settings above.
CAMERAS = []

# --- Shift Reports ---
SHIFT_REPORT_INTERVAL_MINUTES = 60
SHIFT_REPORT_HOOKS = []

# --- Alert Actions ---
ALERT_DATABASE_PATH = "alerts.sqlite3"
ALERT_FRAME_DIRECTORY = "alert_frames"
# Optional callbacks keyed by "push", "email", "buzzer", or "notification".
# Each callback receives (alert, context); context includes the alert level,
# frame number, related coordinates, and the saved frame path when available.
ALERT_HOOKS = {}

# --- Push Provider ---
PUSH_PROVIDER = "ntfy"
NTFY_TOPIC = ""
PUSHOVER_TOKEN = ""
PUSHOVER_USER_KEY = ""

# --- Email Provider ---
SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASS = ""
ALERT_EMAIL_TO = ""

# --- Buzzer Provider ---
BUZZER_MODE = "none"  # "gpio", "sound", or "none"
SOUND_FILE_PATH = ""
BUZZER_GPIO_PIN = 18
BUZZER_DURATION_SECONDS = 1.0