"""Transparent priority scoring and geographic hotspot analysis."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from database import get_all_issues, update_issue_priority


# Keep the scoring policy in one place so it can be tuned without changing the UI.
PRIORITY_WEIGHTS = {
    "severity": 0.40,
    "report_frequency": 0.20,
    "location_risk": 0.20,
    "persistence": 0.20,
}
SEVERITY_SCORES = {
    "low": 30.0,
    "small": 30.0,
    "medium": 60.0,
    "high": 90.0,
    "large": 90.0,
    "critical": 100.0,
}
NEARBY_DISTANCE_METERS = 50.0
HOTSPOT_MIN_ISSUES = 2
DEFAULT_LOCATION_RISK = 50.0
PERSISTENCE_WINDOW_DAYS = 30.0
EARTH_RADIUS_METERS = 6_371_000.0


def _coordinates(issue: Dict[str, Any]) -> Optional[Tuple[float, float]]:
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


def geographic_distance_meters(first: Dict[str, Any], second: Dict[str, Any]) -> Optional[float]:
    """Return haversine distance, or None when either issue lacks valid coordinates."""
    first_coordinates = _coordinates(first)
    second_coordinates = _coordinates(second)
    if first_coordinates is None or second_coordinates is None:
        return None

    first_latitude, first_longitude = map(math.radians, first_coordinates)
    second_latitude, second_longitude = map(math.radians, second_coordinates)
    delta_latitude = second_latitude - first_latitude
    delta_longitude = second_longitude - first_longitude
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(haversine))


def _parse_reported_at(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _same_category(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    return str(first.get("category") or "").strip().lower() == str(second.get("category") or "").strip().lower()


def _related_reports(issue: Dict[str, Any], issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    related = []
    for candidate in issues:
        if not _same_category(issue, candidate):
            continue
        distance = geographic_distance_meters(issue, candidate)
        if distance is not None and distance <= NEARBY_DISTANCE_METERS:
            related.append(candidate)
    return related


def _severity_score(issue: Dict[str, Any]) -> float:
    return SEVERITY_SCORES.get(str(issue.get("severity") or "").strip().lower(), 0.0)


def _report_frequency_score(related_count: int) -> float:
    # The first observation is the baseline; four additional nearby reports reach 100.
    repeated_count = max(0, related_count - 1)
    return min(100.0, repeated_count / 4.0 * 100.0)


def _persistence_score(issue: Dict[str, Any], related: List[Dict[str, Any]]) -> float:
    if str(issue.get("status") or "").upper() == "RESOLVED":
        return 0.0

    observations_score = min(100.0, max(0, len(related) - 1) / 2.0 * 100.0)
    dates = [_parse_reported_at(item.get("reported_at")) for item in related]
    dates = [date for date in dates if date is not None]
    if not dates:
        age_score = 0.0
    else:
        age_days = max(0.0, (datetime.now(timezone.utc) - min(dates)).total_seconds() / 86_400)
        age_score = min(100.0, age_days / PERSISTENCE_WINDOW_DAYS * 100.0)
    return (observations_score + age_score) / 2.0


def _priority_components(issue: Dict[str, Any], issues: List[Dict[str, Any]]):
    related = _related_reports(issue, issues)
    return {
        "severity": _severity_score(issue),
        "report_frequency": _report_frequency_score(len(related)),
        "location_risk": DEFAULT_LOCATION_RISK,
        "persistence": _persistence_score(issue, related),
    }, related


def _score_and_label(components: Dict[str, float]) -> Tuple[float, str]:
    score = sum(PRIORITY_WEIGHTS[name] * value for name, value in components.items())
    score = round(max(0.0, min(100.0, score)), 2)
    if score >= 70:
        label = "Critical"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Low"
    return score, label


def calculate_priority(issue: Dict[str, Any], issues: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Calculate a 0-100 priority score and Low/Medium/High label."""
    components, _ = _priority_components(issue, issues)
    return _score_and_label(components)


def explain_priority(issue: Dict[str, Any], issues: List[Dict[str, Any]]) -> Tuple[float, str, str]:
    """Return a score, label, and human-readable weighted factor explanation."""
    components, related = _priority_components(issue, issues)
    score, label = _score_and_label(components)
    parts = [
        f"severity {components['severity']:.0f}×{PRIORITY_WEIGHTS['severity']:.0%}",
        f"nearby recurrence {components['report_frequency']:.0f}×{PRIORITY_WEIGHTS['report_frequency']:.0%} "
        f"({len(related)} same-category report(s) within {NEARBY_DISTANCE_METERS:.0f} m)",
        f"location context baseline {components['location_risk']:.0f}×{PRIORITY_WEIGHTS['location_risk']:.0%}",
        f"unresolved persistence {components['persistence']:.0f}×{PRIORITY_WEIGHTS['persistence']:.0%}",
    ]
    return score, label, "; ".join(parts) + "."


def calculate_and_update_priority(actor) -> List[Dict[str, Any]]:
    """Recalculate and persist priority for every existing issue."""
    issues = get_all_issues(actor)
    for issue in issues:
        score, label, reason = explain_priority(issue, issues)
        update_issue_priority(issue["issue_id"], score, label, actor=actor, reason=reason)
        issue["priority_score"] = score
        issue["priority_label"] = label
        issue["priority_reason"] = reason
    return issues


def cluster_hotspots(
    issues: List[Dict[str, Any]],
    distance_meters: float = NEARBY_DISTANCE_METERS,
    min_issues: int = HOTSPOT_MIN_ISSUES,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Cluster valid issue coordinates and summarize clusters with at least min_issues."""
    located = [(issue, _coordinates(issue)) for issue in issues]
    located = [(issue, coordinates) for issue, coordinates in located if coordinates is not None]
    if len(located) < min_issues:
        return [], {}

    coordinates = np.radians(np.array([coordinates for _, coordinates in located]))
    labels = DBSCAN(
        eps=distance_meters / EARTH_RADIUS_METERS,
        min_samples=min_issues,
        metric="haversine",
    ).fit_predict(coordinates)

    clusters = []
    issue_cluster_labels: Dict[int, int] = {}
    for cluster_id in sorted(set(int(label) for label in labels) - {-1}):
        members = [located[index][0] for index, label in enumerate(labels) if label == cluster_id]
        if len(members) < min_issues:
            continue
        for member in members:
            issue_cluster_labels[int(member["issue_id"])] = cluster_id
        category_counts = Counter(str(member.get("category") or "Unknown") for member in members)
        average_severity = sum(_severity_score(member) for member in members) / len(members)
        average_priority = sum(float(member.get("priority_score") or 0.0) for member in members) / len(members)
        cluster_coordinates = [_coordinates(member) for member in members]
        clusters.append({
            "hotspot_id": cluster_id + 1,
            "count": len(members),
            "center": (
                sum(point[0] for point in cluster_coordinates) / len(cluster_coordinates),
                sum(point[1] for point in cluster_coordinates) / len(cluster_coordinates),
            ),
            "dominant_category": category_counts.most_common(1)[0][0],
            "average_severity": round(average_severity, 2),
            "average_priority": round(average_priority, 2),
        })
    return clusters, issue_cluster_labels
