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
import hashlib
import cv2
import torch
import numpy as np
import streamlit as st
import folium
import time
from datetime import datetime
import math
import pandas as pd
from pathlib import Path
from streamlit import components

from auth import (
    ROLE_CITIZEN,
    ROLE_MUNICIPAL_OFFICER,
    clear_session,
    establish_session,
    get_current_user,
    initialize_users_table,
)
from database import (
    add_issue,
    get_issue_by_id,
    get_maintenance_cost_rates,
    initialize_database,
    link_detections_to_issue,
    record_inference_detections,
    update_issue_status,
)

# add postprocess utilities
from postprocess import export_geojson, generate_alerts, cluster_hotspots
from priority import calculate_and_update_priority, cluster_hotspots as analyze_hotspots
from civiscan.ui.navigation import render_navigation
from civiscan.ui.theme import apply_theme
from civiscan.pages.municipal import render_municipal_page
from civiscan.pages.citizen import render_citizen_dashboard, render_my_reports
from civiscan.pages.authentication import render_authentication_page
from civiscan.pages.profile import render_profile
from civiscan.pages.officer_admin import render_officer_admin
from civiscan.pages.incoming_media import render_incoming_media_page
from civiscan.ai.assessment import assess_severity, estimate_repair_cost
from civiscan.ai.model_service import get_model as get_configured_yolo_model, run_inference

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    APP_NAME, ACTIVE_CONFIDENCE_THRESHOLD,
    DEVICE, DEFAULT_WEIGHTS,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    YOLOV8_CONFIDENCE_THRESHOLD, YOLOV8_POST_FILTER_CONFIDENCE,
    YOLOV8_NMS_IOU_THRESHOLD, YOLOV8_MAX_PREDICTIONS,
    YOLOV8_MAX_DISPLAY_DETECTIONS, YOLOV8_MIN_BOX_AREA,
    YOLOV8_MIN_ASPECT_RATIO, YOLOV8_MAX_ASPECT_RATIO,
    YOLOV8_AGNOSTIC_NMS, YOLOV8_TTA,
    CLASS_NAMES,
    MODEL_PATH, UPLOAD_DIR, YOLOV8_FALLBACK_PATH, ensure_app_directories,
)

# ─── Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()
initialize_users_table()

# ─── Model Loading (cached) ──────────────────────────────

def _check_yolov8_weights():
    """Resolve the configured model first, then use the existing trained checkpoint."""
    candidates = [
        str(MODEL_PATH),
        str(YOLOV8_FALLBACK_PATH),
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
    try:
        model, weights = get_configured_yolo_model()
    except FileNotFoundError:
        return None, None
    return model, str(weights)


@st.cache_resource
def get_yolov3_model():
    """Load custom YOLOv3-Tiny model."""
    from model import load_model
    model = load_model(DEFAULT_WEIGHTS, DEVICE)
    return model


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
    """Run the shared CiviScan inference service for an uploaded image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, [], None
    inference = run_inference(
        image,
        source_name=getattr(uploaded_file, "name", "upload"),
        model=model,
        model_path=Path(_check_yolov8_weights()) if _check_yolov8_weights() else None,
        confidence=conf_thresh,
        iou=iou_thresh,
    )
    detections = [
        [*item["bbox"], item["confidence"], item["class_id"]]
        for item in inference.detections
    ]
    return inference.original, inference.annotated, detections, inference


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


def _class_name(class_id, model=None):
    """Resolve class labels from model metadata for future multi-class weights."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"

# ─── Main App ─────────────────────────────────────────────

def _priority_from_severity(severity):
    """Map the existing severity category to a transparent initial priority."""
    priority_mapping = {
        "low": (25.0, "Low"),
        "small": (25.0, "Low"),
        "medium": (55.0, "Moderate"),
        "high": (85.0, "Critical"),
        "large": (85.0, "Critical"),
        "critical": (100.0, "Critical"),
    }
    return priority_mapping.get(str(severity).lower(), (25.0, "Low"))


def _display_submitted_report(report):
    """Display the result of a successful issue submission."""
    st.success("Report submitted successfully")
    st.markdown(f"**Issue ID:** {report['issue_id']}")
    st.markdown(f"**Category:** {report['category']}")
    st.markdown(f"**Severity:** {report['severity']}")
    st.markdown(f"**Priority:** {report['priority_label']} ({report['priority_score']:.0f})")
    cost = report.get("estimated_repair_cost")
    st.markdown(
        f"**Indicative repair estimate:** ₹{cost:.2f}"
        if cost is not None
        else "**Indicative repair estimate:** Not configured"
    )
    st.markdown(f"**Status:** {report['status']}")

def _logout():
    """Clear authentication state for the current browser session."""
    clear_session(st.session_state)


def _render_login_page():
    """Render login and registration forms for unauthenticated users."""
    render_authentication_page()


def _render_role_navigation(user) -> str:
    """Render the shared role-aware navigation and apply the selected theme."""
    page, theme, logout = render_navigation(st, user)
    apply_theme(st, theme)
    if logout:
        _logout()
        st.rerun()
    return page


def _render_placeholder_dashboard(page: str):
    """Render navigation placeholders until later dashboard steps are implemented."""
    role = st.session_state["role"]
    st.markdown(f"## {role} Dashboard")
    st.write(f"Welcome, {st.session_state['name']}.")
    st.info(f"{page} will be implemented in a later step.")


def _normalise_value(value):
    return str(value).strip() if value is not None else ""


def _valid_coordinates(issue):
    try:
        latitude = float(issue.get("latitude"))
        longitude = float(issue.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def _render_municipal_dashboard(section="Overview"):
    """Render the Municipal Officer dashboard from persisted issue records."""
    st.markdown(f"# {section}")
    st.write(f"Welcome, {st.session_state['name']}.")
    issues = calculate_and_update_priority(st.session_state.get("current_user"))
    hotspot_result = analyze_hotspots(issues)
    hotspot_summaries = hotspot_result[0]

    priority_counts = {
        label: sum(_normalise_value(issue.get("priority_label")).lower() == label.lower() for issue in issues)
        for label in ("High", "Medium", "Low")
    }
    status_counts = {
        "NEW": sum(_normalise_value(issue.get("status")).upper() == "NEW" for issue in issues),
        "RESOLVED": sum(_normalise_value(issue.get("status")).upper() == "RESOLVED" for issue in issues),
    }

    summary_cards = [
        ("Total Issues", len(issues)),
        ("High Priority", priority_counts["High"]),
        ("Medium Priority", priority_counts["Medium"]),
        ("Low Priority", priority_counts["Low"]),
        ("New Issues", status_counts["NEW"]),
        ("Resolved Issues", status_counts["RESOLVED"]),
    ]
    columns = st.columns(len(summary_cards))
    for column, (label, value) in zip(columns, summary_cards):
        column.metric(label, value)

    st.markdown("## All Civic Issues")
    if not issues:
        st.info("No civic issues have been submitted yet.")
        st.markdown("## Analytics")
        st.info("Analytics will appear after civic issues are submitted.")
        st.markdown("## Issue Map")
        st.info("The map will appear when issues with valid coordinates are available.")
        return

    categories = sorted({_normalise_value(issue.get("category")) for issue in issues if issue.get("category")})
    priorities = sorted({_normalise_value(issue.get("priority_label")) for issue in issues if issue.get("priority_label")})
    severities = sorted({_normalise_value(issue.get("severity")) for issue in issues if issue.get("severity")})
    statuses = sorted({_normalise_value(issue.get("status")) for issue in issues if issue.get("status")})

    with st.expander("Filters", expanded=True):
        filter_columns = st.columns(4)
        category_filter = filter_columns[0].selectbox("Category", ["All"] + categories)
        priority_filter = filter_columns[1].selectbox("Priority", ["All"] + priorities)
        severity_filter = filter_columns[2].selectbox("Severity", ["All"] + severities)
        status_filter = filter_columns[3].selectbox("Status", ["All"] + statuses)

    filtered_issues = [
        issue for issue in issues
        if (category_filter == "All" or _normalise_value(issue.get("category")) == category_filter)
        and (priority_filter == "All" or _normalise_value(issue.get("priority_label")) == priority_filter)
        and (severity_filter == "All" or _normalise_value(issue.get("severity")) == severity_filter)
        and (status_filter == "All" or _normalise_value(issue.get("status")) == status_filter)
    ]

    table_rows = [
        {
            "Issue ID": issue.get("issue_id"),
            "Category": issue.get("category") or "N/A",
            "Confidence": issue.get("confidence"),
            "Severity": issue.get("severity") or "N/A",
            "Priority": issue.get("priority_label") or "N/A",
            "Priority Score": issue.get("priority_score"),
            "Estimated Cost": issue.get("estimated_repair_cost"),
            "Latitude": issue.get("latitude"),
            "Longitude": issue.get("longitude"),
            "Reported Date": issue.get("reported_at") or "N/A",
            "Status": issue.get("status") or "N/A",
        }
        for issue in filtered_issues
    ]
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No issues match the selected filters.")

    st.markdown("## Issue Details")
    if not filtered_issues:
        st.info("Select different filters to view issue details.")
    else:
        issue_options = [issue["issue_id"] for issue in filtered_issues]
        selected_issue_id = st.selectbox("Select Issue ID", issue_options)
        selected_issue = next(
            issue for issue in filtered_issues if issue["issue_id"] == selected_issue_id
        )
        detail_columns = st.columns([1, 2])
        with detail_columns[0]:
            image_path = selected_issue.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption="Uploaded issue image", use_container_width=True)
            else:
                st.info("Original uploaded image is unavailable.")
        with detail_columns[1]:
            st.markdown(f"**Detected category:** {selected_issue.get('category') or 'N/A'}")
            st.markdown(f"**Confidence:** {selected_issue.get('confidence') if selected_issue.get('confidence') is not None else 'N/A'}")
            st.markdown(f"**Severity:** {selected_issue.get('severity') or 'N/A'}")
            st.markdown(f"**Priority score:** {selected_issue.get('priority_score') if selected_issue.get('priority_score') is not None else 'N/A'}")
            st.markdown(f"**Priority label:** {selected_issue.get('priority_label') or 'N/A'}")
            st.markdown(f"**Estimated repair cost:** ₹{selected_issue.get('estimated_repair_cost') if selected_issue.get('estimated_repair_cost') is not None else 'N/A'}")
            st.markdown(f"**Latitude:** {selected_issue.get('latitude') if selected_issue.get('latitude') is not None else 'N/A'}")
            st.markdown(f"**Longitude:** {selected_issue.get('longitude') if selected_issue.get('longitude') is not None else 'N/A'}")
            st.markdown(f"**Reported date:** {selected_issue.get('reported_at') or 'N/A'}")
            st.markdown(f"**Citizen/user ID:** {selected_issue.get('user_id') or 'N/A'}")
            st.markdown(f"**Description:** {selected_issue.get('description') or 'N/A'}")
            st.markdown(f"**Current status:** {selected_issue.get('status') or 'N/A'}")

        st.markdown("### Update Status")
        status_options = ["NEW", "VERIFIED", "IN PROGRESS", "RESOLVED", "REOPENED"]
        current_status = _normalise_value(selected_issue.get("status")).upper()
        selected_status = st.selectbox(
            "New status",
            status_options,
            index=status_options.index(current_status) if current_status in status_options else 0,
        )
        if st.button("Update Status", key=f"update_status_{selected_issue_id}"):
            if update_issue_status(
                selected_issue_id,
                selected_status,
                actor=st.session_state.get("current_user"),
            ):
                calculate_and_update_priority(st.session_state.get("current_user"))
                st.success("Issue status updated successfully.")
                st.rerun()
            else:
                st.error("The selected issue could not be updated.")

    st.markdown("## Analytics")
    analytics_frame = pd.DataFrame(issues)
    chart_columns = st.columns(2)
    for column, field, title in [
        (chart_columns[0], "category", "Issues by Category"),
        (chart_columns[1], "severity", "Issues by Severity"),
        (chart_columns[0], "priority_label", "Issues by Priority"),
        (chart_columns[1], "status", "Issues by Status"),
    ]:
        values = analytics_frame[field].fillna("Unknown").replace("", "Unknown").value_counts()
        column.markdown(f"**{title}**")
        column.bar_chart(values)

    st.markdown("## Priority Analysis")
    priority_analysis_columns = st.columns(3)
    priority_groups = {
        "High Priority Issues": "High",
        "Medium Priority Issues": "Medium",
        "Low Priority Issues": "Low",
    }
    for column, (title, priority_label) in zip(priority_analysis_columns, priority_groups.items()):
        matching = [issue for issue in issues if issue.get("priority_label") == priority_label]
        column.markdown(f"**{title}: {len(matching)}**")
        if matching:
            column.dataframe(
                pd.DataFrame([
                    {"Issue ID": issue.get("issue_id"), "Category": issue.get("category"), "Score": issue.get("priority_score")}
                    for issue in matching
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            column.caption("None")

    st.markdown("## Hotspot Analysis")
    if not hotspot_summaries:
        st.info("Insufficient data for hotspot analysis.")
    else:
        hotspot_frame = pd.DataFrame([
            {
                "Hotspot ID": hotspot["hotspot_id"],
                "Issues": hotspot["count"],
                "Dominant Category": hotspot["dominant_category"],
                "Average Severity": hotspot["average_severity"],
                "Average Priority": hotspot["average_priority"],
            }
            for hotspot in hotspot_summaries
        ])
        st.dataframe(hotspot_frame, use_container_width=True, hide_index=True)
        hotspot_counts = pd.Series(
            {str(hotspot["hotspot_id"]): hotspot["count"] for hotspot in hotspot_summaries},
            name="Issues",
        )
        st.markdown("**Issues by Hotspot**")
        st.bar_chart(hotspot_counts)

    st.markdown("## Issue Map")
    mapped_issues = [(issue, _valid_coordinates(issue)) for issue in issues]
    mapped_issues = [(issue, coordinates) for issue, coordinates in mapped_issues if coordinates]
    if not mapped_issues:
        st.info("No issues have valid latitude and longitude values for mapping.")
        return

    center_latitude = sum(coordinates[0] for _, coordinates in mapped_issues) / len(mapped_issues)
    center_longitude = sum(coordinates[1] for _, coordinates in mapped_issues) / len(mapped_issues)
    issue_map = folium.Map(location=(center_latitude, center_longitude), zoom_start=12)
    marker_colors = {"high": "red", "medium": "orange", "low": "green"}
    for issue, coordinates in mapped_issues:
        priority = _normalise_value(issue.get("priority_label")).lower()
        popup = (
            f"Issue ID: {issue.get('issue_id')}<br>"
            f"Category: {issue.get('category') or 'N/A'}<br>"
            f"Severity: {issue.get('severity') or 'N/A'}<br>"
            f"Priority: {issue.get('priority_label') or 'N/A'}<br>"
            f"Cost: {issue.get('estimated_repair_cost') if issue.get('estimated_repair_cost') is not None else 'N/A'}<br>"
            f"Status: {issue.get('status') or 'N/A'}"
        )
        folium.Marker(
            location=coordinates,
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color=marker_colors.get(priority, "gray")),
        ).add_to(issue_map)
    for hotspot in hotspot_summaries:
        latitude, longitude = hotspot["center"]
        popup = (
            f"Hotspot ID: {hotspot['hotspot_id']}<br>"
            f"Issues: {hotspot['count']}<br>"
            f"Dominant category: {hotspot['dominant_category']}<br>"
            f"Average severity: {hotspot['average_severity']}<br>"
            f"Average priority: {hotspot['average_priority']}"
        )
        folium.Circle(
            location=(latitude, longitude),
            radius=50,
            color="purple",
            fill=True,
            fill_opacity=0.15,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(issue_map)
        folium.Marker(
            location=(latitude, longitude),
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color="purple", icon="info-sign"),
        ).add_to(issue_map)
    components.v1.html(issue_map._repr_html_(), height=550)

def main():
    ensure_app_directories()
    apply_theme(st, "Light")
    user = get_current_user(st.session_state)
    if user is None:
        _render_login_page()
        return

    selected_page = _render_role_navigation(user)
    if selected_page == "My Profile":
        render_profile(user)
        return
    if selected_page == "Officer Administration":
        render_officer_admin(user)
        return
    if user["role"] == ROLE_MUNICIPAL_OFFICER and selected_page == "Incoming Media":
        render_incoming_media_page(user)
        return
    if user["role"] == ROLE_MUNICIPAL_OFFICER and selected_page != "AI Detection":
        render_municipal_page(selected_page)
        return
    if user["role"] == ROLE_CITIZEN and selected_page == "My Reports":
        render_my_reports(user)
        return
    if user["role"] == ROLE_CITIZEN and selected_page == "Dashboard":
        render_citizen_dashboard(user)
        return

    # Check available models
    try:
        v8_model, v8_weights = get_yolov8_model()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        st.error(f"The configured YOLO model could not be loaded: {error}")
        return
    has_v8 = v8_model is not None
    v3_model = None if has_v8 else get_yolov3_model()

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Detection settings")

        if has_v8:
            model_choice = "YOLOv8 (Configured)"
        else:
            model_choice = "YOLOv3-Tiny (Custom)"
            st.info("Using the configured YOLOv3-Tiny fallback because YOLOv8 weights were not found.")

        if "YOLOv8" in model_choice:
            conf_thresh = ACTIVE_CONFIDENCE_THRESHOLD
            nms_thresh = YOLOV8_NMS_IOU_THRESHOLD
        else:
            conf_thresh = CONFIDENCE_THRESHOLD
            nms_thresh = NMS_IOU_THRESHOLD

        st.markdown("---")
        st.caption("Active model")
        st.markdown(os.path.basename(v8_weights or DEFAULT_WEIGHTS))
        st.caption(f"Confidence threshold: {conf_thresh:.2f}")
        st.caption(f"Processing device: {DEVICE}")

    # ── Hero Section ──────────────────────────────────────
    st.title(APP_NAME)

    subtitle_model = "YOLOv8" if "YOLOv8" in model_choice else "YOLOv3-Tiny"
    st.markdown(
        f"Detect and report civic infrastructure issues with {subtitle_model}."
    )

    # ── Tabs ──────────────────────────────────────────────
    tab1, = st.tabs(["🔍 Detection"])

    # ── Tab 1: Detection ──────────────────────────────────
    with tab1:
        active_model = subtitle_model
        st.markdown(f"### Upload an image for civic-issue detection ({active_model})")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="detection_upload"
        )

        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()
            upload_key = hashlib.sha256(upload_bytes).hexdigest()
            inference_result = None
            try:
                with st.spinner(f"🔍 Detecting civic issues with {active_model}..."):
                    uploaded_file.seek(0)

                    if "YOLOv8" in model_choice and has_v8:
                        original, annotated, detections, inference_result = detect_with_yolov8(
                            uploaded_file, v8_model, conf_thresh, nms_thresh
                        )
                    else:
                        inference_started = time.perf_counter()
                        original, annotated, detections = detect_with_yolov3(
                            uploaded_file, v3_model, conf_thresh, nms_thresh
                        )
                        inference_seconds = time.perf_counter() - inference_started
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Image inference failed: {error}")
                return
            if inference_result is not None:
                inference_seconds = inference_result.inference_seconds

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

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Issues detected", len(detections))
                m2.metric("📏 Avg Confidence",
                          f"{np.mean([d[4] for d in detections]):.2f}" if detections else "N/A")
                m3.metric("Inference time", f"{inference_seconds:.3f}s")
                m4.metric("Model version", inference_result.model_version[:12] if inference_result else "Legacy")

                if detections:
                    os.makedirs(str(UPLOAD_DIR), exist_ok=True)
                    extension = Path(uploaded_file.name).suffix.lower()
                    if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
                        extension = ".jpg"
                    source_path = os.path.join(str(UPLOAD_DIR), f"{upload_key}{extension}")
                    if not os.path.exists(source_path):
                        with open(source_path, "wb") as image_file:
                            image_file.write(upload_bytes)
                    annotated_path = os.path.join(str(UPLOAD_DIR), f"{upload_key}_annotated.jpg")
                    if not cv2.imwrite(annotated_path, annotated):
                        st.error("Could not save the annotated image.")
                        return

                    from postprocess import extract_gps_from_image
                    image_gps = extract_gps_from_image(source_path)
                    cost_rates = get_maintenance_cost_rates()
                    enriched = []
                    persisted_detections = []
                    for index, det in enumerate(detections):
                        x1, y1, x2, y2, confidence = map(float, det[:5])
                        class_id = int(det[5]) if len(det) > 5 else 0
                        category = _class_name(class_id, v8_model if has_v8 else None)
                        bbox = [x1, y1, x2, y2]
                        assessment = assess_severity(category, bbox, (w, h))
                        priority_score, priority_label = _priority_from_severity(assessment["severity"])
                        cost_estimate = estimate_repair_cost(category, cost_rates.get(category))
                        cost = cost_estimate["total_inr"] if cost_estimate else None
                        if inference_result is not None:
                            record = dict(inference_result.detections[index])
                            record["detection_id"] = hashlib.sha256(
                                f"{upload_key}:{inference_result.model_version}:{index}".encode("utf-8")
                            ).hexdigest()
                        else:
                            record = {
                                "detection_id": hashlib.sha256(
                                    f"{upload_key}:legacy:{index}".encode("utf-8")
                                ).hexdigest(),
                                "image_filename": uploaded_file.name,
                                "issue_category": category,
                                "class_id": class_id,
                                "confidence": confidence,
                                "bbox": bbox,
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "model_name": "legacy-yolov3",
                                "model_version": "legacy",
                                "inference_seconds": inference_seconds,
                            }
                        record.update(assessment)
                        record["estimated_repair_cost"] = cost
                        persisted_detections.append(record)
                        enriched.append({
                            "bbox": bbox,
                            "confidence": confidence,
                            "class": class_id,
                            "category": category,
                            "severity_category": assessment["severity"],
                            "severity_reason": assessment["severity_reason"],
                            "priority": priority_score / 100.0,
                            "priority_score": priority_score,
                            "priority_label": priority_label,
                            "estimated_repair_cost": cost,
                            "gps": image_gps,
                        })

                    if persisted_detections:
                        detection_ids = record_inference_detections(
                            persisted_detections,
                            annotated_path,
                            actor=user,
                            source_image_path=source_path,
                        )
                    else:
                        detection_ids = []

                    st.markdown("### Detection Details")
                    category_counts = {}
                    for det in enriched:
                        category_counts[det["category"]] = category_counts.get(det["category"], 0) + 1
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        cost_label = (
                            f"₹{det['estimated_repair_cost']:.2f} (estimate)"
                            if det["estimated_repair_cost"] is not None
                            else "Not configured"
                        )
                        gps = det["gps"]
                        gps_label = f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "Not available"
                        sev_class = f"severity-{det['severity_category'].lower()}"
                        st.markdown(f"""
                        <div class="detection-box">
                            <strong>{det['category']}</strong> |
                            Confidence: <strong>{det['confidence']:.3f}</strong> |
                            Bounding Box: <strong>[{x1}, {y1}, {x2}, {y2}]</strong> |
                            Severity: <strong class="{sev_class}">{det['severity_category']}</strong> |
                            Priority: <strong>{det['priority_label']} ({det['priority_score']:.0f}/100)</strong> |
                            Indicative repair estimate: <strong>{cost_label}</strong> |
                            GPS: <strong>{gps_label}</strong><br>
                            <small>{det['severity_reason']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame([{"Category": key, "Detections": value} for key, value in category_counts.items()]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    gps_points = [d for d in enriched if d.get("gps")]
                    if gps_points:
                        st.markdown("---")
                        st.markdown("### Map & Hotspots")
                        eps_m = st.sidebar.slider("Hotspot radius (meters)", 10, 200, 50)
                        min_samples = st.sidebar.slider("Hotspot min samples", 2, 10, 3)
                        labels, clusters = cluster_hotspots(enriched, eps_m=eps_m, min_samples=min_samples)

                        lats = [d["gps"][0] for d in gps_points]
                        lons = [d["gps"][1] for d in gps_points]
                        center = (float(np.mean(lats)), float(np.mean(lons)))
                        fmap = folium.Map(location=center, zoom_start=15)
                        for idx, d in enumerate(enriched):
                            g = d.get("gps")
                            if not g:
                                continue
                            color = "green" if d["severity_category"] == "Low" else ("orange" if d["severity_category"] == "Medium" else "red")
                            folium.CircleMarker(
                                location=(g[0], g[1]),
                                radius=6,
                                color=color,
                                fill=True,
                                popup=f"{d['category']} | Priority {d['priority_score']:.0f}/100",
                            ).add_to(fmap)

                        for cid, info in clusters.items():
                            c_lat, c_lon = info["center"]
                            folium.CircleMarker(
                                location=(c_lat, c_lon),
                                radius=10,
                                color="#B58A55",
                                fill=True,
                                popup=f"Hotspot {cid} ({info['count']} detections)",
                            ).add_to(fmap)

                        components.v1.html(fmap._repr_html_(), height=500)

                        st.markdown("### Alerts & Export")
                        priority_thresh = st.number_input("Alert priority threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                        if st.button("Generate Alerts"):
                            alerts = generate_alerts(enriched, priority_threshold=priority_thresh, out_csv=None)
                            st.success(f"{len(alerts)} alerts generated (priority >= {priority_thresh})")
                            geojson_path = os.path.join(PROJECT_ROOT, "results", "detections.geojson")
                            os.makedirs(os.path.dirname(geojson_path), exist_ok=True)
                            export_geojson(enriched, geojson_path)
                            with open(geojson_path, "rb") as geojson_file:
                                st.download_button("Download GeoJSON", geojson_file, file_name="detections.geojson")

                    primary_detection = max(enriched, key=lambda item: item["confidence"])
                    primary_category = primary_detection["category"]
                    primary_severity = primary_detection["severity_category"]
                    priority_score = primary_detection["priority_score"]
                    priority_label = primary_detection["priority_label"]

                    if user["role"] != ROLE_CITIZEN:
                        st.info("Detection is available to officers; issue submission is restricted to citizens.")
                    elif st.session_state.get("submitted_report_key") == upload_key:
                        _display_submitted_report(st.session_state["last_submitted_report"])
                    else:
                        st.markdown("### Submit Civic Issue")
                        gps_hint = "Image GPS found; edit coordinates if needed." if image_gps else "Location is optional; coordinates are never inferred."
                        with st.form(f"issue_submission_form_{upload_key}"):
                            st.caption(gps_hint)
                            latitude_text = st.text_input(
                                "Latitude",
                                value=f"{image_gps[0]:.6f}" if image_gps else "",
                                key=f"report_latitude_{upload_key}",
                                placeholder="Optional",
                            )
                            longitude_text = st.text_input(
                                "Longitude",
                                value=f"{image_gps[1]:.6f}" if image_gps else "",
                                key=f"report_longitude_{upload_key}",
                                placeholder="Optional",
                            )
                            description = st.text_area(
                                "Additional description",
                                key=f"report_description_{upload_key}",
                            )
                            submit_report = st.form_submit_button("Submit Civic Issue")

                        if submit_report:
                            latitude = longitude = None
                            coordinates_valid = True
                            if bool(latitude_text.strip()) != bool(longitude_text.strip()):
                                st.error("Enter both coordinates, or leave both blank.")
                                coordinates_valid = False
                            elif latitude_text.strip():
                                try:
                                    latitude = float(latitude_text.strip())
                                    longitude = float(longitude_text.strip())
                                except ValueError:
                                    st.error("Enter valid latitude and longitude values.")
                                    coordinates_valid = False
                            if coordinates_valid:
                                if latitude is not None and (
                                    not -90 <= latitude <= 90 or not -180 <= longitude <= 180
                                ):
                                    st.error("Latitude must be between -90 and 90; longitude must be between -180 and 180.")
                                else:
                                    report = {
                                        "user_id": str(user["user_id"]),
                                        "category": primary_category,
                                        "confidence": float(primary_detection["confidence"]),
                                        "severity": primary_severity,
                                        "estimated_repair_cost": primary_detection["estimated_repair_cost"],
                                        "latitude": latitude,
                                        "longitude": longitude,
                                        "reported_at": datetime.now().isoformat(timespec="seconds"),
                                        "image_path": source_path,
                                        "status": "NEW",
                                        "priority_score": priority_score,
                                        "priority_label": priority_label,
                                        "description": description.strip() or None,
                                        "severity_reason": primary_detection["severity_reason"],
                                        "annotated_image_path": annotated_path,
                                        "model_name": inference_result.model_name if inference_result else "legacy-yolov3",
                                        "model_version": inference_result.model_version if inference_result else "legacy",
                                        "inference_seconds": inference_seconds,
                                    }
                                    report["issue_id"] = add_issue(**report, actor=user)
                                    link_detections_to_issue(detection_ids, report["issue_id"], actor=user)
                                    saved_report = get_issue_by_id(report["issue_id"], actor=user)
                                    if saved_report is not None:
                                        report.update(saved_report)
                                    st.session_state["submitted_report_key"] = upload_key
                                    st.session_state["last_submitted_report"] = report
                                    _display_submitted_report(report)
                else:
                    st.info("No detections met the configured confidence threshold.")
            else:
                st.error("The uploaded file could not be decoded as an image.")
        else:
            st.info("Upload an image above to analyze civic infrastructure.")

    # (Grad-CAM, About Model, and Interview Q&A sections were removed per user request)


if __name__ == "__main__":
    main()
