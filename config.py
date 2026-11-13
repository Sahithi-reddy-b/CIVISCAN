"""
=============================================================
  Pothole Detection — Configuration
  Centralized hyperparameters, paths, and device settings
=============================================================
"""
import os
from pathlib import Path
import torch

# ─── Device ───────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Paths ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "CiviScan"
APP_TAGLINE = "AI-powered civic infrastructure intelligence"
DATABASE_PATH = Path(os.getenv("CIVISCAN_DATABASE_PATH", os.path.join(ROOT_DIR, "civiscan.db")))
MODEL_PATH = Path(os.getenv("CIVISCAN_MODEL_PATH", os.path.join(ROOT_DIR, "model.pt")))
YOLOV8_FALLBACK_PATH = Path(
    os.getenv(
        "CIVISCAN_YOLOV8_FALLBACK_PATH",
        os.path.join(ROOT_DIR, "yolov8_pothole_best.pt"),
    )
)
INCOMING_MEDIA_DIR = Path(os.getenv("CIVISCAN_INCOMING_MEDIA_DIR", os.path.join(ROOT_DIR, "incoming_media")))
PROCESSED_MEDIA_DIR = Path(os.getenv("CIVISCAN_PROCESSED_MEDIA_DIR", os.path.join(ROOT_DIR, "processed_media")))
ANNOTATED_RESULTS_DIR = Path(os.getenv("CIVISCAN_ANNOTATED_RESULTS_DIR", os.path.join(ROOT_DIR, "results", "annotated")))
UPLOAD_DIR = Path(os.getenv("CIVISCAN_UPLOAD_DIR", os.path.join(ROOT_DIR, "uploads")))
DATA_DIR = Path(os.getenv("CIVISCAN_DATA_DIR", os.path.join(ROOT_DIR, "data")))
INCOMING_IMAGE_DIR = Path(os.getenv("CIVISCAN_INCOMING_IMAGE_DIR", str(DATA_DIR / "incoming_images")))
PROCESSING_DIR = Path(os.getenv("CIVISCAN_PROCESSING_DIR", str(DATA_DIR / "processing")))
DATA_PROCESSED_DIR = Path(os.getenv("CIVISCAN_DATA_PROCESSED_DIR", str(DATA_DIR / "processed")))
FAILED_MEDIA_DIR = Path(os.getenv("CIVISCAN_FAILED_MEDIA_DIR", str(DATA_DIR / "failed")))
EXTRACTED_FRAMES_DIR = Path(os.getenv("CIVISCAN_EXTRACTED_FRAMES_DIR", str(DATA_DIR / "extracted_frames")))
ANNOTATED_MEDIA_DIR = Path(os.getenv("CIVISCAN_ANNOTATED_MEDIA_DIR", str(DATA_DIR / "annotated_results")))
FRAME_INTERVAL_SECONDS = float(os.getenv("CIVISCAN_FRAME_INTERVAL_SECONDS", "2"))
MAX_VIDEO_FRAMES = int(os.getenv("CIVISCAN_MAX_VIDEO_FRAMES", "100"))
INCOMING_POLL_SECONDS = float(os.getenv("CIVISCAN_INCOMING_POLL_SECONDS", "3"))
DEFAULT_THEME = os.getenv("CIVISCAN_THEME", "Light")
ACTIVE_CONFIDENCE_THRESHOLD = float(
    os.getenv("CIVISCAN_CONFIDENCE_THRESHOLD", "0.44")
)
AUTH_MAX_LOGIN_ATTEMPTS = int(os.getenv("CIVISCAN_AUTH_MAX_LOGIN_ATTEMPTS", "5"))
AUTH_LOCKOUT_MINUTES = int(os.getenv("CIVISCAN_AUTH_LOCKOUT_MINUTES", "15"))
PASSWORD_RESET_TTL_MINUTES = int(os.getenv("CIVISCAN_PASSWORD_RESET_TTL_MINUTES", "30"))
PASSWORD_RESET_PUBLIC_URL = os.getenv("CIVISCAN_PUBLIC_URL", "http://localhost:8501")
SMTP_HOST = os.getenv("CIVISCAN_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("CIVISCAN_SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("CIVISCAN_SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("CIVISCAN_SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("CIVISCAN_SMTP_FROM_EMAIL", "")

DATASET_DIR = os.path.join(ROOT_DIR, "dataset-pothole-main", "dataset")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train")
TEST_IMAGES = os.path.join(DATASET_DIR, "test")

# Pre-trained weight files
WEIGHTS_PATH_1 = os.path.join(ROOT_DIR, "yolo_rdd_1.pth")
WEIGHTS_PATH_3 = os.path.join(ROOT_DIR, "yolo_rdd_3.pth")
DEFAULT_WEIGHTS = WEIGHTS_PATH_1


def ensure_app_directories():
    """Create only the runtime media directories used by the web application."""
    for directory in (
        INCOMING_MEDIA_DIR,
        PROCESSED_MEDIA_DIR,
        ANNOTATED_RESULTS_DIR,
        UPLOAD_DIR,
        INCOMING_IMAGE_DIR,
        PROCESSING_DIR,
        DATA_PROCESSED_DIR,
        FAILED_MEDIA_DIR,
        EXTRACTED_FRAMES_DIR,
        ANNOTATED_MEDIA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

# Output directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DETECTION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "sample_detections")
GRADCAM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "gradcam_outputs")
METRICS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "metrics")

for d in [RESULTS_DIR, DETECTION_OUTPUT_DIR, GRADCAM_OUTPUT_DIR, METRICS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Model Architecture ──────────────────────────────────
NUM_CLASSES = 10        # As defined in the original notebook training
NUM_ANCHORS = 3         # 3 anchor boxes per grid cell
IMAGE_SIZE = 416        # Input image resolution (416×416)

# ─── Class Names ──────────────────────────────────────────
# The pothole dataset has 1 actual class, but the model was
# trained with num_classes=10 in the notebook. The first class
# (index 0) is "pothole". Remaining slots are unused.
CLASS_NAMES = [
    "pothole",
    "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9",
]

# ─── Inference Settings ───────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25     # Minimum confidence to keep a detection
NMS_IOU_THRESHOLD = 0.45        # IoU threshold for Non-Maximum Suppression
MAX_DETECTIONS = 100            # Maximum detections per image

# ─── YOLOv8 No-Retrain Deployment Settings ────────────────
# Raw prediction threshold tuned from the F1-confidence curve.
YOLOV8_CONFIDENCE_THRESHOLD = ACTIVE_CONFIDENCE_THRESHOLD
# Final display threshold keeps borderline boxes out of the UI.
YOLOV8_POST_FILTER_CONFIDENCE = 0.45
YOLOV8_NMS_IOU_THRESHOLD = 0.55
# Allow a few extra boxes before filtering, then cap final output harder.
YOLOV8_MAX_PREDICTIONS = 8
YOLOV8_MAX_DISPLAY_DETECTIONS = 5
YOLOV8_MIN_BOX_AREA = 800
YOLOV8_MIN_ASPECT_RATIO = 0.3
YOLOV8_MAX_ASPECT_RATIO = 3.0
YOLOV8_AGNOSTIC_NMS = True
YOLOV8_TTA = True

# ─── Training Hyperparameters (for reference / retraining) ─
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4

# ─── Data Augmentation ────────────────────────────────────
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.0,
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.1,
    },
    "random_rotation_degrees": 5,
}

# ─── Severity Thresholds (based on pothole area %) ────────
# Relative to image area
SEVERITY_THRESHOLDS = {
    "small": 0.005,      # < 0.5% of image area
    "medium": 0.02,      # 0.5% – 2%
    "large": 0.05,       # 2% – 5%
    "critical": 0.05,    # > 5%
}

# ─── Visualization ────────────────────────────────────────
BOX_COLOR = (0, 255, 0)         # Green bounding boxes
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)    # White text

# ─── Color palette for severity levels ────────────────────
SEVERITY_COLORS = {
    "Small":    (0, 255, 0),      # Green
    "Medium":   (0, 255, 255),    # Yellow
    "Large":    (0, 165, 255),    # Orange
    "Critical": (0, 0, 255),      # Red
}
