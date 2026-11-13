"""Shared, configurable Ultralytics inference service for images and video frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ACTIVE_CONFIDENCE_THRESHOLD,
    DEVICE,
    MODEL_PATH,
    YOLOV8_FALLBACK_PATH,
    YOLOV8_AGNOSTIC_NMS,
    YOLOV8_MAX_PREDICTIONS,
    YOLOV8_NMS_IOU_THRESHOLD,
    YOLOV8_TTA,
)


@dataclass
class InferenceResult:
    """Inference output with model metadata and image-space detections."""

    original: np.ndarray
    annotated: np.ndarray
    detections: List[Dict[str, Any]]
    inference_seconds: float
    model_name: str
    model_version: str
    model_path: str

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("original")
        value.pop("annotated")
        return value


def resolve_model_path() -> Path:
    """Choose the explicitly configured model, then the existing trained checkpoint."""
    candidates = (
        Path(MODEL_PATH),
        Path(YOLOV8_FALLBACK_PATH),
        Path(__file__).resolve().parents[2] / "runs" / "pothole_yolov8" / "weights" / "best.pt",
        Path(__file__).resolve().parents[2] / "runs" / "pothole_yolov8" / "weights" / "last.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No configured YOLO weights were found. Checked: {expected}")


@lru_cache(maxsize=3)
def load_model(model_path: str):
    """Load one configured Ultralytics model, cached per resolved checkpoint path."""
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO weights do not exist: {path}")
    return YOLO(str(path))


def get_model():
    path = resolve_model_path()
    return load_model(str(path.resolve())), path


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        name = names.get(class_id, names.get(str(class_id)))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        name = names[class_id]
    else:
        name = None
    return str(name if name is not None else f"class_{class_id}")


def _model_version(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"


def run_inference(
    image: np.ndarray,
    *,
    source_name: str = "image",
    model=None,
    model_path: Optional[Path] = None,
    confidence: float = ACTIVE_CONFIDENCE_THRESHOLD,
    iou: float = YOLOV8_NMS_IOU_THRESHOLD,
) -> InferenceResult:
    """Run YOLO on a decoded BGR image and return stable, class-aware output."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The supplied image is empty or invalid.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a decoded three-channel color image.")
    if not 0 <= float(confidence) <= 1 or not 0 <= float(iou) <= 1:
        raise ValueError("Confidence and IoU thresholds must be between 0 and 1.")

    if model is None:
        model, model_path = get_model()
    if model_path is None:
        model_path = Path(getattr(model, "ckpt_path", None) or "configured-model.pt")
    model_path = Path(model_path)

    started = time.perf_counter()
    results = model.predict(
        source=image,
        conf=float(confidence),
        iou=float(iou),
        agnostic_nms=YOLOV8_AGNOSTIC_NMS,
        max_det=YOLOV8_MAX_PREDICTIONS,
        augment=YOLOV8_TTA,
        device=str(DEVICE),
        verbose=False,
        save=False,
    )
    elapsed = time.perf_counter() - started

    result = results[0] if results else None
    annotated = result.plot() if result is not None else image.copy()
    boxes = getattr(result, "boxes", None) if result is not None else None
    names = getattr(model, "names", {})
    height, width = image.shape[:2]
    detections = []
    if boxes is not None:
        for box in boxes:
            bbox = [float(item) for item in box.xyxy[0].detach().cpu().tolist()]
            class_id = int(box.cls[0].detach().cpu().item())
            category = _class_name(names, class_id)
            detections.append({
                "detection_id": uuid.uuid4().hex,
                "image_filename": os.path.basename(source_name),
                "issue_category": category,
                "class_id": class_id,
                "confidence": float(box.conf[0].detach().cpu().item()),
                "bbox": bbox,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_path.stem,
                "model_version": _model_version(model_path) if model_path.is_file() else "injected",
                "inference_seconds": elapsed,
                "image_size": [width, height],
            })

    return InferenceResult(
        original=image.copy(),
        annotated=annotated,
        detections=detections,
        inference_seconds=elapsed,
        model_name=model_path.stem,
        model_version=_model_version(model_path) if model_path.is_file() else "injected",
        model_path=str(model_path),
    )
