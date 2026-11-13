"""Municipal incoming-media queue and processing history."""

import os
from pathlib import Path
import subprocess
import sys
import uuid

import pandas as pd
import streamlit as st

from auth import ROLE_MUNICIPAL_OFFICER, require_authenticated_role
from config import INCOMING_IMAGE_DIR, ensure_app_directories
from civiscan.ui.components import render_empty_state, render_metrics, render_page_header
from civiscan.workers.incoming_worker import read_worker_status
from database import get_all_issues, get_media_statistics, get_processing_jobs


_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov", ".mkv"}


def _launch_one_time_worker() -> None:
    status = read_worker_status()
    if status.get("is_running"):
        st.warning("An incoming-media worker is already active.")
        return
    root = Path(__file__).resolve().parents[2]
    options = {
        "cwd": str(root),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [sys.executable, "-m", "civiscan.workers.incoming_worker", "--once"],
        **options,
    )
    st.success("A one-time folder scan has started in the background.")


def render_incoming_media_page(user) -> None:
    """Render the authenticated officer queue, worker state, and recent results."""
    user = require_authenticated_role(user, {ROLE_MUNICIPAL_OFFICER})
    ensure_app_directories()
    render_page_header(
        st,
        "Incoming Media",
        "Camera and uploaded media are processed by the same configured YOLO service.",
    )
    status = read_worker_status()
    if status.get("is_running"):
        st.success(f"Worker running · {status.get('message') or status.get('mode')}")
    else:
        st.info(f"Worker stopped · {status.get('message') or 'Start it with --watch or scan the folder once.'}")

    pending_count = sum(
        1 for path in INCOMING_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    stats = get_media_statistics()
    render_metrics(st, [
        ("Incoming files", pending_count),
        ("Images processed", stats["images_processed"]),
        ("Videos processed", stats["videos_processed"]),
        ("Detections generated", stats["detections"]),
        ("Failed jobs", stats["failed_jobs"]),
    ])

    upload, scan = st.columns([2, 1])
    with upload:
        files = st.file_uploader(
            "Add image or video to the incoming queue",
            type=[suffix[1:] for suffix in sorted(_SUPPORTED_SUFFIXES)],
            accept_multiple_files=True,
        )
        if files and st.button("Add to incoming queue", key="queue_incoming_media"):
            saved = 0
            for uploaded in files:
                suffix = Path(uploaded.name).suffix.lower()
                if suffix not in _SUPPORTED_SUFFIXES:
                    st.error(f"Unsupported file type: {uploaded.name}")
                    continue
                target = INCOMING_IMAGE_DIR / f"{uuid.uuid4().hex}{suffix}"
                target.write_bytes(uploaded.getvalue())
                saved += 1
            if saved:
                st.success(f"Added {saved} file(s) to the incoming queue.")
                st.rerun()
    with scan:
        st.write("")
        st.write("")
        if st.button("Manual Scan Incoming Folder", use_container_width=True):
            _launch_one_time_worker()

    st.subheader("Recent processing jobs")
    jobs = get_processing_jobs(50)
    if jobs:
        table = [{
            "File": job["source_filename"],
            "Type": job["media_type"],
            "Status": job["status"],
            "Frames": job["frames_processed"],
            "Detections": job["detection_count"],
            "Created": job["created_at"],
            "Error": job["error_message"] or "",
        } for job in jobs]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    else:
        render_empty_state(st, "No media has been processed yet. Add files or place them in the incoming folder.")

    st.subheader("Recent annotated detections")
    reports = [
        issue for issue in get_all_issues(actor=user)
        if issue.get("source") == "incoming_media"
    ][:12]
    if not reports:
        render_empty_state(st, "Annotated previews will appear here after the worker detects an issue.")
        return

    selected_id = st.selectbox(
        "Detection issue",
        [issue["issue_id"] for issue in reports],
        format_func=lambda value: f"Issue #{value}",
    )
    selected = next(issue for issue in reports if issue["issue_id"] == selected_id)
    original, annotated = st.columns(2)
    with original:
        if selected.get("image_path") and Path(selected["image_path"]).is_file():
            st.image(selected["image_path"], caption="Original media", use_container_width=True)
        else:
            render_empty_state(st, "Original media is unavailable.")
    with annotated:
        if selected.get("annotated_image_path") and Path(selected["annotated_image_path"]).is_file():
            st.image(selected["annotated_image_path"], caption="AI annotation", use_container_width=True)
        else:
            render_empty_state(st, "Annotated image is unavailable.")
    st.caption(
        f"{selected.get('category')} · confidence {float(selected.get('confidence') or 0):.2f} · "
        f"{selected.get('severity') or 'Severity unassessed'} · "
        f"{selected.get('reported_at') or 'Timestamp unavailable'}"
    )
