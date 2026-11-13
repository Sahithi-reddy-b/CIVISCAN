"""Explainable civic issue severity and configurable indicative cost estimates."""

from __future__ import annotations

from typing import Dict, Optional


_SEVERITY_THRESHOLDS = {
    "pothole": (0.025, 0.09),
    "garbage_dumping": (0.04, 0.16),
    "damaged_streetlight": (0.035, 0.12),
}


def assess_severity(category: str, bbox: list, image_size: tuple) -> Dict[str, object]:
    """Assess visible extent; bounding-box area is not a physical-size measurement."""
    width, height = image_size
    image_area = max(1, int(width) * int(height))
    x1, y1, x2, y2 = (float(value) for value in bbox)
    visible_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
    normalized_category = str(category).strip().lower().replace(" ", "_").replace("-", "_")
    low_limit, high_limit = _SEVERITY_THRESHOLDS.get(normalized_category, (0.04, 0.14))

    if visible_ratio < low_limit:
        severity = "Low"
    elif visible_ratio < high_limit:
        severity = "Medium"
    else:
        severity = "High"

    reasons = {
        "pothole": "Based on the visible road-surface extent in the image; road exposure and physical dimensions require field review.",
        "garbage_dumping": "Based on the visible accumulation extent; obstruction and environmental exposure require field review.",
        "damaged_streetlight": "Based on the visible damaged-fixture extent; public-safety impact requires field review.",
    }
    return {
        "severity": severity,
        "severity_reason": (
            f"{reasons.get(normalized_category, 'Based on visible issue extent; field review is recommended')} "
            f"Visible box covers {visible_ratio:.1%} of the image."
        ),
        "visible_area_ratio": visible_ratio,
    }


def estimate_repair_cost(
    category: str,
    rates: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """Estimate INR from configured material, labour, equipment and transport rates."""
    if not rates:
        return None
    keys = ("material", "labour", "equipment", "transportation")
    if any(key not in rates for key in keys):
        return None
    components = {key: float(rates[key]) for key in keys}
    if any(value < 0 for value in components.values()):
        raise ValueError("Maintenance cost rates cannot be negative.")
    return {
        **components,
        "total_inr": round(sum(components.values()), 2),
    }
