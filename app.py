"""
=============================================================
  Pothole Detection — Streamlit Web Application
  Interactive pothole detection with real-time inference
=============================================================

Features:
  🔍 Upload & detect potholes in images (YOLOv8 + YOLOv3-Tiny)
  📊 Confidence scores, severity analysis, pothole count
  🔥 Grad-CAM model explainability
  📈 Model evaluation metrics
  🏗️ Architecture visualization
"""

import os
import sys
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import io

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEVICE, IMAGE_SIZE, NUM_CLASSES, DEFAULT_WEIGHTS,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    YOLOV8_CONFIDENCE_THRESHOLD, YOLOV8_POST_FILTER_CONFIDENCE,
    YOLOV8_NMS_IOU_THRESHOLD, YOLOV8_MAX_PREDICTIONS,
    YOLOV8_MAX_DISPLAY_DETECTIONS, YOLOV8_MIN_BOX_AREA,
    YOLOV8_MIN_ASPECT_RATIO, YOLOV8_MAX_ASPECT_RATIO,
    YOLOV8_AGNOSTIC_NMS, YOLOV8_TTA,
    CLASS_NAMES, SEVERITY_COLORS,
)
from utils import classify_severity

# ─── Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title="Pothole Detection — YOLOv8 + YOLOv3",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0c0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d2ff;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        margin-top: 0.3rem;
    }

    .severity-small { color: #51cf66; }
    .severity-medium { color: #ffd43b; }
    .severity-large { color: #ff922b; }
    .severity-critical { color: #ff6b6b; }

    .detection-box {
        background: rgba(0, 210, 255, 0.05);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .concept-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0, 210, 255, 0.4);
    }

    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    .badge-v8 {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        color: white;
    }

    .badge-v3 {
        background: linear-gradient(135deg, #748ffc, #4c6ef5);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (cached) ──────────────────────────────

def _check_yolov8_weights():
    """Check if fine-tuned YOLOv8 weights exist."""
    candidates = [
        os.path.join(PROJECT_ROOT, "yolov8_pothole_best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "pothole_yolov8", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "pothole_yolov8", "weights", "last.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@st.cache_resource
def get_yolov8_model():
    """Load fine-tuned YOLOv8 model."""
    from ultralytics import YOLO
    weights = _check_yolov8_weights()
    if weights:
        model = YOLO(weights)
        return model, weights
    return None, None


@st.cache_resource
def get_yolov3_model():
    """Load custom YOLOv3-Tiny model."""
    from model import load_model
    model = load_model(DEFAULT_WEIGHTS, DEVICE)
    return model


@st.cache_resource
def get_gradcam(_model):
    """Initialize Grad-CAM once and cache it."""
    from gradcam import GradCAM
    return GradCAM(_model)


# ─── Detection Functions ─────────────────────────────────

def _filter_detections(boxes, img_w, img_h):
    """False positive filtering tuned for deployment-time cleanup."""
    filtered = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        if conf < YOLOV8_POST_FILTER_CONFIDENCE:
            continue

        area = w * h
        aspect_ratio = w / h

        if area < YOLOV8_MIN_BOX_AREA:
            continue
        if not (YOLOV8_MIN_ASPECT_RATIO <= aspect_ratio <= YOLOV8_MAX_ASPECT_RATIO):
            continue

        filtered.append(box)

    filtered.sort(key=lambda det: det[4], reverse=True)
    return filtered[:YOLOV8_MAX_DISPLAY_DETECTIONS]

def detect_with_yolov8(uploaded_file, model,
                       conf_thresh=YOLOV8_CONFIDENCE_THRESHOLD,
                       iou_thresh=YOLOV8_NMS_IOU_THRESHOLD):
    """Run detection using Optimized YOLOv8 predict pipeline."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, []

    temp_path = os.path.join(PROJECT_ROOT, "_temp_upload.jpg")
    cv2.imwrite(temp_path, image)

    # 1. Inference-time strict optimizations
    results = model.predict(
        temp_path, 
        conf=conf_thresh,
        iou=iou_thresh,
        agnostic_nms=YOLOV8_AGNOSTIC_NMS,
        max_det=YOLOV8_MAX_PREDICTIONS,
        augment=YOLOV8_TTA,
        verbose=False
    )

    original = image.copy()
    h, w = original.shape[:2]
    raw_detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_detections.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])

    # 2. Filter False Positives
    detections = _filter_detections(raw_detections, w, h)

    # 3. Clean Output
    annotated = _draw_v8(original, detections)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return original, annotated, detections


def detect_with_yolov3(uploaded_file, model, conf_thresh, nms_thresh):
    """Run detection using custom YOLOv3-Tiny."""
    from inference import preprocess_image, decode_predictions
    from utils import non_max_suppression, draw_detections

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, []

    temp_path = os.path.join(PROJECT_ROOT, "_temp_upload.jpg")
    cv2.imwrite(temp_path, image)

    tensor, original = preprocess_image(temp_path)
    with torch.no_grad():
        output = model(tensor)

    h, w = original.shape[:2]
    raw_dets = decode_predictions(output, (h, w), conf_thresh)
    detections = non_max_suppression(raw_dets, conf_thresh, nms_thresh)
    annotated = draw_detections(original, detections)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return original, annotated, detections


def _draw_v8(image, detections):
    """Draw bounding boxes with anti-overlap labels for YOLOv8."""
    from config import BOX_THICKNESS, FONT_SCALE
    img = image.copy()
    h, w = img.shape[:2]

    # Track used label areas to prevent overlap
    used_label_regions = []

    for det in detections:
        x1, y1, x2, y2, conf = det[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 165, 255) # Orange for pothole
        label = f"Pothole {conf:.2f}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )

        label_x = x1
        label_y = y1 - 5

        # Anti-overlap logic (push label down inside box if overlap happens)
        for (ux1, uy1, ux2, uy2) in used_label_regions:
            if not (label_x + tw < ux1 or label_x > ux2 or label_y - th - baseline > uy2 or label_y < uy1):
                label_y = y1 + th + baseline + 5
                
        used_label_regions.append((label_x, label_y - th - baseline, label_x + tw, label_y))

        cv2.rectangle(img, (label_x, label_y - th - baseline),
                      (label_x + tw, label_y), color, -1)
        cv2.putText(img, label, (label_x, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (0, 0, 0), 1, cv2.LINE_AA)

    n = len(detections)
    summary = f"YOLOv8 | Detected: {n} pothole{'s' if n != 1 else ''}"
    cv2.putText(img, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return img


def generate_gradcam_overlay(uploaded_file, model):
    """Generate Grad-CAM for uploaded image using YOLOv3-Tiny."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    temp_path = os.path.join(PROJECT_ROOT, "_temp_gradcam.jpg")
    cv2.imwrite(temp_path, image)

    gradcam = get_gradcam(model)
    heatmap, overlay, original = gradcam.generate(temp_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return heatmap, overlay, original


# ─── Main App ─────────────────────────────────────────────

def main():
    # Check available models
    v8_model, v8_weights = get_yolov8_model()
    v3_model = get_yolov3_model()

    has_v8 = v8_model is not None

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Model selector
        if has_v8:
            model_choice = st.radio(
                "🧠 Choose Model",
                ["YOLOv8 (Recommended)", "YOLOv3-Tiny (Custom)"],
                index=0,
                help="YOLOv8 uses transfer learning from COCO for much better accuracy"
            )
        else:
            model_choice = "YOLOv3-Tiny (Custom)"
            st.info("YOLOv8 weights not found. Run `python train_yolov8.py` to fine-tune.")

        if "YOLOv8" in model_choice:
            conf_thresh = YOLOV8_CONFIDENCE_THRESHOLD
            nms_thresh = YOLOV8_NMS_IOU_THRESHOLD
            st.info(
                "YOLOv8 no-retrain cleanup is enforced: "
                f"conf={YOLOV8_CONFIDENCE_THRESHOLD:.2f}, "
                f"post-filter>={YOLOV8_POST_FILTER_CONFIDENCE:.2f}, "
                f"iou={YOLOV8_NMS_IOU_THRESHOLD:.2f}, "
                f"raw max_det={YOLOV8_MAX_PREDICTIONS}, "
                f"final max boxes={YOLOV8_MAX_DISPLAY_DETECTIONS}."
            )
            st.caption(
                f"Small boxes under {YOLOV8_MIN_BOX_AREA}px^2 and aspect ratios "
                f"outside {YOLOV8_MIN_ASPECT_RATIO:.1f}-{YOLOV8_MAX_ASPECT_RATIO:.1f} are removed."
            )
        else:
            conf_thresh = st.slider(
                "Confidence Threshold",
                min_value=0.05, max_value=0.95, value=CONFIDENCE_THRESHOLD, step=0.05,
                help="Minimum confidence to show a detection"
            )

            nms_thresh = st.slider(
                "NMS IoU Threshold",
                min_value=0.1, max_value=0.9, value=NMS_IOU_THRESHOLD, step=0.05,
                help="IoU threshold for Non-Maximum Suppression"
            )

        st.markdown("---")

        # Model info based on selection
        if "YOLOv8" in model_choice:
            st.markdown("### 📐 Model: YOLOv8n")
            st.markdown('<span class="model-badge badge-v8">YOLOv8 Nano</span>',
                        unsafe_allow_html=True)
            st.markdown(f"- **Architecture:** YOLOv8n (CSPDarknet)")
            st.markdown(f"- **Parameters:** ~3.0M")
            st.markdown(f"- **Input Size:** 640 x 640")
            st.markdown(f"- **Pre-trained:** COCO (80 classes)")
            st.markdown(f"- **Fine-tuned:** Pothole dataset")
            st.markdown(f"- **FPN:** Yes (multi-scale)")
        else:
            st.markdown("### 📐 Model: YOLOv3-Tiny")
            st.markdown('<span class="model-badge badge-v3">Custom YOLOv3</span>',
                        unsafe_allow_html=True)
            st.markdown(f"- **Architecture:** YOLOv3-Tiny (Custom)")
            st.markdown(f"- **Parameters:** ~6.3M")
            st.markdown(f"- **Input Size:** {IMAGE_SIZE} x {IMAGE_SIZE}")
            st.markdown(f"- **Device:** {DEVICE}")

        st.markdown("---")
        st.markdown("### 🎓 DL Concepts Used")
        concepts = [
            "CNN Feature Extraction",
            "Single-Shot Detection",
            "Anchor Boxes",
            "Non-Maximum Suppression",
            "Batch Normalization",
            "Transfer Learning",
            "Grad-CAM Explainability",
            "Data Augmentation",
            "Feature Pyramid Network",
            "Grid-based Prediction",
        ]
        for c in concepts:
            st.markdown(f"  ✅ {c}")

    # ── Hero Section ──────────────────────────────────────
    st.markdown('<div class="hero-title">🕳️ Pothole Detection System</div>',
                unsafe_allow_html=True)

    subtitle_model = "YOLOv8" if "YOLOv8" in model_choice else "YOLOv3-Tiny"
    st.markdown(
        f'<div class="hero-subtitle">'
        f'Deep Learning-Powered Road Damage Detection using {subtitle_model} | '
        f'Transfer Learning & Computer Vision'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detection", "🔥 Grad-CAM", "📊 About Model", "🎯 Interview Q&A"
    ])

    # ── Tab 1: Detection ──────────────────────────────────
    with tab1:
        active_model = subtitle_model
        st.markdown(f"### Upload an image to detect potholes ({active_model})")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="detection_upload"
        )

        if uploaded_file is not None:
            with st.spinner(f"🔍 Detecting potholes with {active_model}..."):
                uploaded_file.seek(0)

                if "YOLOv8" in model_choice and has_v8:
                    original, annotated, detections = detect_with_yolov8(
                        uploaded_file, v8_model, conf_thresh, nms_thresh
                    )
                else:
                    original, annotated, detections = detect_with_yolov3(
                        uploaded_file, v3_model, conf_thresh, nms_thresh
                    )

            if original is not None:
                # Results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📷 Original Image")
                    st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

                with col2:
                    st.markdown(f"#### 🔍 {active_model} Detection")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

                # Metrics
                st.markdown("---")
                st.markdown("### 📊 Detection Summary")

                h, w = original.shape[:2]
                severity_counts = {"Small": 0, "Medium": 0, "Large": 0, "Critical": 0}

                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    sev = classify_severity((x2-x1, y2-y1), (w, h))
                    severity_counts[sev] += 1

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🕳️ Potholes Found", len(detections))
                m2.metric("📏 Avg Confidence",
                          f"{np.mean([d[4] for d in detections]):.2f}" if detections else "N/A")
                m3.metric("⚠️ Critical", severity_counts["Critical"])
                m4.metric("🟢 Small", severity_counts["Small"])

                if detections:
                    st.markdown("### 📋 Detection Details")
                    for i, det in enumerate(detections):
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        sev = classify_severity((x2-x1, y2-y1), (w, h))
                        sev_class = f"severity-{sev.lower()}"

                        st.markdown(f"""
                        <div class="detection-box">
                            <strong>Pothole #{i+1}</strong> |
                            Confidence: <strong>{conf:.3f}</strong> |
                            Severity: <strong class="{sev_class}">{sev}</strong> |
                            Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)}) |
                            Size: {int(x2-x1)} x {int(y2-y1)} px
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload an image above to start detecting potholes!")

    # ── Tab 2: Grad-CAM ──────────────────────────────────
    with tab2:
        st.markdown("### 🔥 Grad-CAM — Where Does the Model Look?")
        st.markdown("""
        **Gradient-weighted Class Activation Mapping** visualizes the regions
        of the image that the model focuses on when making predictions.

        - 🔴 **Red/Yellow areas**: High attention (model "looks" here)
        - 🔵 **Blue areas**: Low attention (ignored regions)

        > *Note: Grad-CAM runs on the custom YOLOv3-Tiny model backbone.*
        """)

        gradcam_file = st.file_uploader(
            "Upload image for Grad-CAM analysis",
            type=["jpg", "jpeg", "png"],
            key="gradcam_upload"
        )

        if gradcam_file is not None:
            with st.spinner("🔥 Generating Grad-CAM heatmap..."):
                gradcam_file.seek(0)
                try:
                    heatmap, overlay, original = generate_gradcam_overlay(
                        gradcam_file, v3_model
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("#### Original")
                        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                                 use_container_width=True)
                    with col2:
                        st.markdown("#### Heatmap")
                        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
                                 use_container_width=True)
                    with col3:
                        st.markdown("#### Overlay")
                        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                 use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {e}")

    # ── Tab 3: About Model ────────────────────────────────
    with tab3:
        st.markdown("### 🏗️ Model Architectures")

        st.markdown("#### YOLOv8n (Primary — High Accuracy)")
        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────┐
        │                INPUT IMAGE (3 × 640 × 640)           │
        └──────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────┐
        │  CSPDarknet BACKBONE (Pre-trained on COCO)           │
        │  ┌─────────────────────────────────────────────────┐ │
        │  │ Conv → C2f(32)  → Conv → C2f(64)               │ │
        │  │ Conv → C2f(128) → Conv → C2f(256)              │ │
        │  │ SPPF (Spatial Pyramid Pooling Fast)             │ │
        │  └─────────────────────────────────────────────────┘ │
        └──────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────┐
        │  NECK — Feature Pyramid Network (FPN + PAN)          │
        │  ┌─────────────────────────────────────────────────┐ │
        │  │ Upsample + Concat + C2f (×2)     ← FPN (top-down)│
        │  │ Conv + Concat + C2f (×2)   ← PAN (bottom-up)   │ │
        │  │ Multi-scale features: P3, P4, P5               │ │
        │  └─────────────────────────────────────────────────┘ │
        └──────────────────────┬───────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────┐
        │  DETECTION HEAD (Anchor-Free)                        │
        │  ┌─────────────────────────────────────────────────┐ │
        │  │ 3 detection scales: 80×80, 40×40, 20×20         │ │
        │  │ DFL (Distribution Focal Loss) for bbox          │ │
        │  │ 1 class: pothole                                │ │
        │  └─────────────────────────────────────────────────┘ │
        └──────────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("#### YOLOv3-Tiny (Custom — Educational)")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────┐
        │           INPUT IMAGE (3 × 416 × 416)           │
        └──────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────┐
        │  BACKBONE — 7 Conv-BN-LeakyReLU blocks          │
        │  3→16→32→64→128→256→512→1024                    │
        │  + MaxPool downsampling (6 stages)              │
        └──────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────┐
        │  DETECTION HEAD — 1×1 Conv → [3, 6, 6, 15]     │
        │  Per anchor: [10 class + 1 obj + 4 bbox]        │
        └─────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("### 📊 Model Comparison")
        st.markdown("""
        | Feature | YOLOv8n | YOLOv3-Tiny (Custom) |
        |---------|---------|---------------------|
        | Parameters | 3.0M | 6.3M |
        | Input Size | 640×640 | 416×416 |
        | Detection Scales | 3 (multi-scale) | 1 (single) |
        | Pre-trained | COCO (80 classes) | Scratch |
        | Anchor Type | Anchor-free | Anchor-based |
        | FPN | Yes (PANet) | No |
        | Augmentation | Mosaic, MixUp, etc. | Basic |
        | Loss | CIoU + DFL | MSE |
        | Accuracy | High | Low (limited training) |
        """)

    # ── Tab 4: Interview Q&A ─────────────────────────────
    with tab4:
        st.markdown("### 🎯 Interview Questions & Answers")
        st.markdown("*Key Deep Learning & Computer Vision concepts covered in this project*")

        qa_pairs = [
            ("What is YOLO and how does it differ from R-CNN?",
             "YOLO (You Only Look Once) is a single-shot detector that processes the entire image "
             "in one forward pass, dividing it into a grid and predicting boxes + classes simultaneously. "
             "R-CNN uses a two-stage approach: first proposing regions, then classifying each. "
             "YOLO is much faster (real-time) but traditionally less accurate on small objects."),

            ("Why does YOLOv8 perform better than YOLOv3-Tiny here?",
             "Three key reasons: 1) Transfer learning — YOLOv8 is pre-trained on COCO (millions of images), "
             "so it already knows visual features like edges, textures, and shapes. 2) Multi-scale detection — "
             "FPN/PAN neck provides features at 3 resolutions (80x80, 40x40, 20x20), detecting small and large "
             "potholes equally well. 3) Modern loss functions — CIoU loss and Distribution Focal Loss produce "
             "tighter, more accurate bounding boxes."),

            ("Explain Non-Maximum Suppression (NMS).",
             "NMS eliminates duplicate detections. Algorithm: 1) Sort by confidence, "
             "2) Take the highest-scoring box, 3) Remove all boxes with IoU > threshold, "
             "4) Repeat. This ensures each object gets exactly one detection box."),

            ("What is IoU and how is mAP calculated?",
             "IoU = Area of Overlap / Area of Union between predicted and ground truth boxes. "
             "mAP is calculated by: 1) Compute Precision-Recall curve per class, "
             "2) AP = area under PR curve (11-point interpolation), "
             "3) mAP = mean of AP across all classes. Standard: mAP@0.5 (PASCAL VOC)."),

            ("What is Transfer Learning and why is it critical here?",
             "Transfer learning uses a model pre-trained on a large dataset (COCO: 80 classes, "
             "millions of images) and fine-tunes it on your task. The backbone already learned general "
             "features (edges, textures, shapes). Benefits: requires less data, faster convergence (10-100x), "
             "better generalization. In our case, YOLOv8n pre-trained backbone dramatically improves "
             "pothole detection vs training from scratch."),

            ("What is Grad-CAM and why is it useful?",
             "Grad-CAM uses gradients flowing into the last conv layer to produce a heatmap "
             "showing which image regions are most important for the prediction. It enables "
             "model interpretability — crucial for trust, debugging, and deployment in safety-critical "
             "applications like autonomous driving."),

            ("Explain Feature Pyramid Network (FPN) in YOLOv8.",
             "FPN creates a multi-scale feature hierarchy by combining low-resolution, "
             "semantically-strong features with high-resolution, spatially-precise features. "
             "YOLOv8 uses PANet (Path Aggregation Network): top-down FPN + bottom-up path. "
             "This lets it detect small potholes (at 80x80) and large ones (at 20x20) equally well."),

            ("How does anchor-free detection in YOLOv8 differ from anchor-based in YOLOv3?",
             "YOLOv3 uses pre-defined anchor boxes (fixed width/height ratios) and predicts offsets. "
             "YOLOv8 is anchor-free — it directly predicts the center point and distances to the 4 edges. "
             "Anchor-free: simpler, no anchor hyperparameter tuning, better generalization to unusual shapes."),

            ("How would you deploy this model in production?",
             "1) Model optimization: ONNX export, TensorRT, INT8 quantization. "
             "2) Serving: FastAPI/Flask REST API, or edge with TensorRT on Jetson. "
             "3) Monitoring: precision/recall drift, input distribution shift. "
             "4) CI/CD: Automated retraining when new data arrives. "
             "5) YOLOv8 supports direct export: model.export(format='onnx')"),

            ("What data augmentation does YOLOv8 use?",
             "YOLOv8 uses aggressive augmentation: Mosaic (4 images combined), MixUp (image blending), "
             "HSV jitter (hue/saturation/value), Random flip, Scale, Translate, Random erasing. "
             "Mosaic is particularly powerful — it forces the model to detect objects at various scales "
             "and in different contexts, dramatically improving robustness."),
        ]

        for i, (q, a) in enumerate(qa_pairs):
            with st.expander(f"❓ Q{i+1}: {q}"):
                st.markdown(f"**Answer:** {a}")


if __name__ == "__main__":
    main()
