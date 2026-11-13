"""
Post-processing utilities for PotholeGuard-AI
- severity estimation
- priority scoring
- repair cost estimation
- GPS extraction from image EXIF
- hotspot clustering (DBSCAN/haversine)
- alert/GeoJSON export

Expect detection entries of form (list/dict):
- dicts with keys: 'bbox' (xyxy or xywh normalized or absolute), 'confidence', 'class', 'image_path' (optional), 'gps' (optional)
- or tuples/lists: (x1,y1,x2,y2, confidence, class)

These functions make conservative fallbacks when absolute geo/scale data is unavailable.
"""
from typing import List, Dict, Optional, Tuple, Any
import math
import json
import os
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6371000.0


def _bbox_xyxy(det: Dict[str, Any], image_size: Optional[Tuple[int, int]] = None) -> Tuple[float, float, float, float]:
    """Return bbox as absolute xyxy in pixels. Accepts several common formats.
    det can contain 'bbox' as [x1,y1,x2,y2] or [x_center,y_center,w,h] (either normalized or absolute).
    If values look normalized (<=1) and image_size provided, they are scaled.
    """
    if isinstance(det, (list, tuple)):
        if len(det) >= 6:
            # (x1,y1,x2,y2,conf,cls)
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            return x1, y1, x2, y2
    if 'bbox' not in det:
        raise ValueError('detection has no bbox')
    b = det['bbox']
    W, H = None, None
    if image_size is not None:
        W, H = image_size
    # xyxy
    if len(b) == 4:
        x0, y0, x1, y1 = b
        # detect if xywh center format by seeing if coords are <=1 but sum looks like center/width
        if 0 < x0 <= 1 and 0 < y0 <= 1 and 0 < x1 <= 1 and 0 < y1 <= 1 and W and H:
            # ambiguous: could be normalized xyxy or normalized xywh; assume xyxy normalized
            x0 *= W; y0 *= H; x1 *= W; y1 *= H
        return float(x0), float(y0), float(x1), float(y1)
    raise ValueError('unsupported bbox format')


def estimate_severity(det: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    """Estimate severity [0..1] and category from bbox size and confidence.

    severity_score = normalized_area * confidence
    category thresholds (tunable):
      - low: < 0.05
      - medium: < 0.15
      - high: >= 0.15
    """
    x1, y1, x2, y2 = _bbox_xyxy(det, image_size)
    W, H = image_size
    area = max(0.0, (x2 - x1) * (y2 - y1))
    area_ratio = area / (W * H) if W and H else 0.0
    conf = float(det.get('confidence', det.get('score', 1.0)))
    severity = float(area_ratio) * float(conf)
    # clamp
    severity = max(0.0, min(1.0, severity))
    if severity < 0.05:
        cat = 'low'
    elif severity < 0.15:
        cat = 'medium'
    else:
        cat = 'high'
    out = det.copy()
    out['severity'] = severity
    out['severity_category'] = cat
    out['area_ratio'] = area_ratio
    return out


def compute_priority(det: Dict[str, Any], traffic_factor: Optional[float] = None, proximity_factor: Optional[float] = None) -> Dict[str, Any]:
    """Compute priority score [0..1] using severity and optional external factors.

    By default: priority = 0.8*severity + 0.2*confidence
    If traffic_factor (0..1) provided, it replaces the confidence component.
    proximity_factor (0..1) can boost priority (closer to critical infra -> higher).
    """
    severity = float(det.get('severity', 0.0))
    conf = float(det.get('confidence', det.get('score', 0.0)))
    factor = traffic_factor if traffic_factor is not None else conf
    base = 0.8 * severity + 0.2 * factor
    if proximity_factor is not None:
        # boost up to +10% if proximity high
        base = base + 0.1 * proximity_factor
    priority = max(0.0, min(1.0, base))
    out = det.copy()
    out['priority'] = priority
    return out


def estimate_repair_cost(det: Dict[str, Any], image_size: Tuple[int, int], area_m2_estimate: Optional[float] = None, base_cost_per_m2: float = 120.0) -> Dict[str, Any]:
    """Estimate repair cost in currency units.

    If `area_m2_estimate` provided, uses that. Otherwise heuristically converts image-area ratio
    into a rough area in m^2 by assuming the image frame covers `default_coverage_m2` (default 10 m^2).

    cost = base_cost_per_m2 * estimated_area_m2 * severity_multiplier
    """
    out = det.copy()
    x1, y1, x2, y2 = _bbox_xyxy(det, image_size)
    W, H = image_size
    area_px = max(0.0, (x2 - x1) * (y2 - y1))
    area_ratio = area_px / (W * H) if W and H else 0.0
    if area_m2_estimate is None:
        default_coverage_m2 = 10.0
        estimated_area_m2 = area_ratio * default_coverage_m2
    else:
        estimated_area_m2 = area_m2_estimate
    severity = float(det.get('severity', 0.0))
    severity_multiplier = 0.5 + severity  # between 0.5 and 1.5
    cost = base_cost_per_m2 * estimated_area_m2 * severity_multiplier
    out['estimated_area_m2'] = estimated_area_m2
    out['estimated_repair_cost'] = float(round(cost, 2))
    return out


def extract_gps_from_image(image_path: str) -> Optional[Tuple[float, float]]:
    """Try to extract GPS (lat, lon) from image EXIF. Returns (lat, lon) or None."""
    if not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None
            gps_info = {}
            for key, val in exif.items():
                name = ExifTags.TAGS.get(key)
                if name == 'GPSInfo':
                    for tag, value in val.items():
                        subname = ExifTags.GPSTAGS.get(tag)
                        gps_info[subname] = value
            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                def _convert(coord):
                    degrees = coord[0][0] / coord[0][1]
                    minutes = coord[1][0] / coord[1][1]
                    seconds = coord[2][0] / coord[2][1]
                    return degrees + (minutes / 60.0) + (seconds / 3600.0)
                latitude = _convert(gps_info['GPSLatitude'])
                longitude = _convert(gps_info['GPSLongitude'])
                if gps_info.get('GPSLatitudeRef') in ['S', b'S']:
                    latitude = -latitude
                if gps_info.get('GPSLongitudeRef') in ['W', b'W']:
                    longitude = -longitude
                return latitude, longitude
    except Exception:
        return None
    return None


def _to_radians(coords: np.ndarray) -> np.ndarray:
    return np.radians(coords)


def cluster_hotspots(detections: List[Dict[str, Any]], eps_m: float = 50.0, min_samples: int = 3) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    """Cluster detections by geographic proximity using DBSCAN with haversine metric.

    Returns (labels, cluster_summary) where cluster_summary maps label->summary dict
    """
    coords = []
    indices = []
    for i, d in enumerate(detections):
        gps = d.get('gps')
        if gps is None:
            continue
        coords.append([gps[0], gps[1]])
        indices.append(i)
    if len(coords) == 0:
        return [], {}
    arr = np.array(coords)
    radians = np.radians(arr)
    eps_rad = eps_m / EARTH_RADIUS_M
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(radians)
    # map labels back to original detection indices
    label_map = {}
    for idx, lbl in zip(indices, labels):
        label_map.setdefault(int(lbl), []).append(idx)
    cluster_summary = {}
    for lbl, inds in label_map.items():
        if lbl == -1:
            continue
        pts = [detections[i]['gps'] for i in inds]
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        center = (float(np.mean(lats)), float(np.mean(lons)))
        cluster_summary[int(lbl)] = {
            'count': len(inds),
            'members': inds,
            'center': center,
        }
    return labels.tolist(), cluster_summary


def export_geojson(detections: List[Dict[str, Any]], out_path: str) -> None:
    """Export detections with GPS to a GeoJSON FeatureCollection."""
    features = []
    for d in detections:
        gps = d.get('gps')
        if gps is None:
            continue
        props = {k: v for k, v in d.items() if k not in ('gps',)}
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [gps[1], gps[0]]
            },
            'properties': props
        }
        features.append(feature)
    coll = {'type': 'FeatureCollection', 'features': features}
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(coll, f, ensure_ascii=False, indent=2)


def generate_alerts(detections: List[Dict[str, Any]], priority_threshold: float = 0.7, out_csv: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return list of detections that exceed `priority_threshold`. Optionally write CSV."""
    alerts = [d for d in detections if d.get('priority', 0.0) >= priority_threshold]
    if out_csv:
        df = pd.DataFrame(alerts)
        df.to_csv(out_csv, index=False)
    return alerts


def process_detections(detections: List[Dict[str, Any]], image_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """Process list of detections in-place and return enriched list.

    - ensures bbox normalized to pixels if image_size known
    - tries EXIF GPS if image_path provided and detection lacks `gps`
    - computes severity, priority, cost
    """
    enriched = []
    for d in detections:
        det = dict(d)
        # try to pull image size if missing
        if image_size is None and 'image_path' in det and os.path.exists(det['image_path']):
            try:
                with Image.open(det['image_path']) as im:
                    image_size = det.get('image_size', (im.width, im.height))
            except Exception:
                pass
        if image_size is None:
            # default safe values
            image_size = (1280, 720)
        # ensure gps
        if 'gps' not in det and 'image_path' in det:
            gps = extract_gps_from_image(det['image_path'])
            if gps:
                det['gps'] = gps
        det = estimate_severity(det, image_size)
        det = compute_priority(det)
        det = estimate_repair_cost(det, image_size)
        enriched.append(det)
    return enriched


if __name__ == '__main__':
    print('postprocess module — import functions in your pipeline')
