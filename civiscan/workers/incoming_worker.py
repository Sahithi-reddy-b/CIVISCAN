"""Process incoming civic images and videos outside the Streamlit process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Optional, Tuple
import uuid

import cv2
import numpy as np

from config import (
    ANNOTATED_MEDIA_DIR,
    DATA_PROCESSED_DIR,
    EXTRACTED_FRAMES_DIR,
    FAILED_MEDIA_DIR,
    FRAME_INTERVAL_SECONDS,
    INCOMING_IMAGE_DIR,
    INCOMING_POLL_SECONDS,
    MAX_VIDEO_FRAMES,
    PROCESSING_DIR,
    ensure_app_directories,
)
from civiscan.ai.assessment import assess_severity, estimate_repair_cost
from civiscan.ai.model_service import get_model, run_inference
from database import (
    completed_job_for_hash,
    create_processing_job,
    get_maintenance_cost_rates,
    initialize_database,
    mark_processing_job_started,
    record_worker_detection,
    update_processing_job,
)
from postprocess import extract_gps_from_image


LOGGER = logging.getLogger("civiscan.incoming_worker")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
STATUS_PATH = INCOMING_IMAGE_DIR.parent / "worker_status.json"


def _set_worker_status(mode: str, message: str = "") -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STATUS_PATH.with_suffix(".tmp")
    payload = {
        "mode": mode,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }
    temporary_path.write_text(json.dumps(payload), encoding="utf-8")
    temporary_path.replace(STATUS_PATH)


def read_worker_status() -> dict:
    """Read the status heartbeat without treating stale files as a running worker."""
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(status["updated_at"])
        age_seconds = (datetime.now(timezone.utc) - updated).total_seconds()
        status["is_running"] = (
            status.get("mode") in {"watch", "once", "processing"}
            and age_seconds < max(30, INCOMING_POLL_SECONDS * 4)
        )
        return status
    except (OSError, ValueError, KeyError, TypeError):
        return {"mode": "stopped", "is_running": False, "message": "Worker heartbeat unavailable."}


def _wait_until_stable(path: Path, retries: int = 3, delay: float = 0.25) -> bool:
    previous_size = None
    for _ in range(retries):
        try:
            current_size = path.stat().st_size
        except OSError:
            return False
        if current_size == previous_size:
            return True
        previous_size = current_size
        time.sleep(delay)
    try:
        return path.stat().st_size == previous_size
    except OSError:
        return False


def _decode_image(path: Path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path.name}")
    return image


def _safe_write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image output: {path}")


def _finalize_original(source: Path, job_id: str, original_filename: Optional[str] = None) -> Path:
    destination = DATA_PROCESSED_DIR / f"{job_id}_{original_filename or source.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return destination


def _record_result(job_id: str, result, source_path: Path, annotated_path: Path, frame_seconds: Optional[float] = None) -> int:
    gps = extract_gps_from_image(str(source_path)) if source_path.suffix.lower() in IMAGE_EXTENSIONS else None
    rates = get_maintenance_cost_rates()
    persisted = 0
    for raw_detection in result.detections:
        detection = dict(raw_detection)
        assessment = assess_severity(
            detection["issue_category"],
            detection["bbox"],
            tuple(detection["image_size"]),
        )
        detection.update(assessment)
        detection["timestamp"] = datetime.now(timezone.utc).isoformat()
        detection["source_path"] = str(source_path)
        cost_components = estimate_repair_cost(
            detection["issue_category"],
            rates.get(detection["issue_category"]),
        )
        cost = cost_components["total_inr"] if cost_components else None
        record_worker_detection(
            job_id,
            detection,
            str(annotated_path),
            frame_timestamp_seconds=frame_seconds,
            estimated_repair_cost=cost,
            latitude=gps[0] if gps else None,
            longitude=gps[1] if gps else None,
        )
        persisted += 1
    return persisted


def _process_image(
    job_id: str,
    processing_path: Path,
    original_filename: str,
    model,
    model_path: Path,
) -> Tuple[int, int, Path]:
    image = _decode_image(processing_path)
    result = run_inference(
        image,
        source_name=processing_path.name,
        model=model,
        model_path=model_path,
    )
    annotated_path = ANNOTATED_MEDIA_DIR / f"{job_id}_annotated.jpg"
    _safe_write_image(annotated_path, result.annotated)
    # Keep the original media at its permanent path before persisting issue references.
    final_original = _finalize_original(processing_path, job_id, original_filename)
    count = _record_result(job_id, result, final_original, annotated_path)
    return count, 1, final_original


def _process_video(
    job_id: str,
    processing_path: Path,
    original_filename: str,
    model,
    model_path: Path,
) -> Tuple[int, int, Path]:
    final_original = DATA_PROCESSED_DIR / f"{job_id}_{original_filename}"
    capture = cv2.VideoCapture(str(processing_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {processing_path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    interval = max(1, int(round((fps if fps > 0 else 30.0) * FRAME_INTERVAL_SECONDS)))
    frame_index = 0
    frames_processed = 0
    detections_count = 0
    try:
        while frames_processed < max(1, MAX_VIDEO_FRAMES):
            success, frame = capture.read()
            if not success:
                break
            if frame_index % interval == 0:
                timestamp_seconds = frame_index / fps if fps > 0 else float(frame_index)
                frame_path = EXTRACTED_FRAMES_DIR / f"{job_id}_{frame_index:08d}.jpg"
                _safe_write_image(frame_path, frame)
                result = run_inference(
                    frame,
                    source_name=frame_path.name,
                    model=model,
                    model_path=model_path,
                )
                annotated_path = ANNOTATED_MEDIA_DIR / f"{job_id}_{frame_index:08d}_annotated.jpg"
                _safe_write_image(annotated_path, result.annotated)
                detections_count += _record_result(
                    job_id,
                    result,
                    final_original,
                    annotated_path,
                    frame_seconds=timestamp_seconds,
                )
                frames_processed += 1
            frame_index += 1
    finally:
        capture.release()
    if frames_processed == 0:
        raise ValueError(f"No readable frames found in video: {processing_path.name}")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(processing_path), str(final_original))
    return detections_count, frames_processed, final_original


def process_file(path: Path, model=None, model_path: Optional[Path] = None) -> dict:
    """Process one supported file, moving it to processed or failed storage."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported incoming media format: {extension or '(no extension)'}")
    stable = _wait_until_stable(path)
    if not stable:
        try:
            if path.stat().st_size > 0:
                return {"status": "pending", "filename": path.name}
        except OSError:
            return {"status": "pending", "filename": path.name}

    content = path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    media_type = "image" if extension in IMAGE_EXTENSIONS else "video"
    existing_job = completed_job_for_hash(content_hash)
    job_id = create_processing_job(path.name, str(path.resolve()), content_hash, media_type)
    processing_path = PROCESSING_DIR / f"{job_id}{extension}"
    try:
        processing_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(processing_path))
        mark_processing_job_started(job_id)
        if not stable:
            raise OSError(f"Incoming file is empty or still being written: {path.name}")
        if existing_job:
            final_original = _finalize_original(processing_path, job_id, path.name)
            update_processing_job(job_id, "completed", processed_path=str(final_original))
            LOGGER.info("Skipped duplicate media %s (previous job %s)", path.name, existing_job["job_id"])
            return {"job_id": job_id, "status": "completed", "duplicate": True, "detections": 0}

        if model is None or model_path is None:
            model, model_path = get_model()
        if media_type == "image":
            detections_count, frames_processed, final_original = _process_image(
                job_id, processing_path, path.name, model, model_path
            )
        else:
            detections_count, frames_processed, final_original = _process_video(
                job_id, processing_path, path.name, model, model_path
            )
        update_processing_job(
            job_id,
            "completed",
            processed_path=str(final_original),
            detection_count=detections_count,
            frames_processed=frames_processed,
        )
        LOGGER.info("Processed %s: %d detections across %d frames", path.name, detections_count, frames_processed)
        return {
            "job_id": job_id,
            "status": "completed",
            "duplicate": False,
            "detections": detections_count,
            "frames_processed": frames_processed,
            "processed_path": str(final_original),
        }
    except Exception as error:
        failure_path = FAILED_MEDIA_DIR / f"{job_id}_{path.name}"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if processing_path.exists():
                shutil.move(str(processing_path), str(failure_path))
            elif path.exists():
                shutil.move(str(path), str(failure_path))
        except OSError:
            LOGGER.exception("Could not move failed input %s to the failed folder", path.name)
        update_processing_job(job_id, "failed", error_message=str(error))
        LOGGER.exception("Failed to process incoming media %s", path.name)
        return {"job_id": job_id, "status": "failed", "error": str(error)}


def scan_incoming_folder() -> list:
    """Discover and process supported files currently in the incoming folder."""
    ensure_app_directories()
    results = []
    for path in sorted(INCOMING_IMAGE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            _set_worker_status("processing", f"Processing {path.name}")
            results.append(process_file(path))
    _set_worker_status("idle", f"Last scan processed {len(results)} file(s).")
    return results


def run_watch() -> None:
    """Poll the incoming directory until interrupted by the operator."""
    ensure_app_directories()
    LOGGER.info("Watching incoming media directory: %s", INCOMING_IMAGE_DIR)
    _set_worker_status("watch", "Watching for incoming media.")
    while True:
        try:
            for path in sorted(INCOMING_IMAGE_DIR.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                    _set_worker_status("watch", f"Processing {path.name}")
                    process_file(path)
            _set_worker_status("watch", "Watching for incoming media.")
            time.sleep(max(0.25, INCOMING_POLL_SECONDS))
        except KeyboardInterrupt:
            _set_worker_status("stopped", "Stopped by operator.")
            raise
        except OSError:
            LOGGER.exception("Incoming folder scan failed")
            _set_worker_status("watch", "Folder scan failed; see worker log.")
            time.sleep(max(0.25, INCOMING_POLL_SECONDS))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Process CiviScan incoming civic media.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Scan and process the incoming folder once.")
    mode.add_argument("--watch", action="store_true", help="Continuously monitor the incoming folder.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_app_directories()
    initialize_database()
    if args.once:
        try:
            _set_worker_status("once", "Starting one-time scan.")
            results = scan_incoming_folder()
        except Exception:
            LOGGER.exception("Incoming folder scan could not be completed")
            _set_worker_status("error", "Scan failed; see worker log.")
            return 1
        return 1 if any(result["status"] == "failed" for result in results) else 0
    try:
        run_watch()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
