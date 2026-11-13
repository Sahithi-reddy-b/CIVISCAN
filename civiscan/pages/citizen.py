"""Citizen-facing issue report history."""

import os

import pandas as pd
import streamlit as st

from civiscan.ui.components import render_empty_state, render_metrics, render_page_header
from auth import ROLE_CITIZEN, require_authenticated_role
from database import get_issues_by_user


def render_citizen_dashboard(user):
    """Show the signed-in citizen's current report and repair status summary."""
    user = require_authenticated_role(user, {ROLE_CITIZEN})
    render_page_header(
        st,
        "Citizen Dashboard",
        "A personal overview of your submitted civic reports and repair progress.",
    )
    reports = get_issues_by_user(str(user["user_id"]), actor=user)
    normalized = [str(report.get("status") or "NEW").upper() for report in reports]
    render_metrics(st, [
        ("Total reports", len(reports)),
        ("Pending", sum(status in {"NEW", "VERIFIED", "ASSIGNED"} for status in normalized)),
        ("In progress", sum(status in {"IN PROGRESS", "AWAITING VERIFICATION"} for status in normalized)),
        ("Resolved", sum(status == "RESOLVED" for status in normalized)),
    ])
    st.subheader("Recent reports")
    if not reports:
        render_empty_state(st, "No reports yet. Use AI Detection to analyze an image and submit a report.")
        return
    st.dataframe(pd.DataFrame([{
        "Issue": report.get("issue_id"),
        "Category": report.get("category") or "Unknown",
        "Severity": report.get("severity") or "Unreviewed",
        "Reported": report.get("reported_at") or "Unknown",
        "Status": report.get("status") or "NEW",
        "Repair estimate (₹)": report.get("estimated_repair_cost"),
    } for report in reports[:10]]), use_container_width=True, hide_index=True)


def render_my_reports(user):
    """Show a citizen's own issue history and selected report details."""
    user = require_authenticated_role(user, {ROLE_CITIZEN})
    render_page_header(st, "My Reports", "Track reports submitted from your account.")
    reports = get_issues_by_user(str(user["user_id"]), actor=user)
    if not reports:
        render_empty_state(st, "You have not submitted any civic issues yet.")
        return

    rows = [{
        "Issue": report.get("issue_id"),
        "Category": report.get("category") or "Unknown",
        "Severity": report.get("severity") or "Unknown",
        "Priority": report.get("priority_label") or "Unscored",
        "Repair estimate": report.get("estimated_repair_cost"),
        "Location": (
            f"{report.get('latitude')}, {report.get('longitude')}"
            if report.get("latitude") is not None and report.get("longitude") is not None
            else "Not provided"
        ),
        "Reported": report.get("reported_at") or "Unknown",
        "Status": report.get("status") or "NEW",
    } for report in reports]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    report_ids = [report["issue_id"] for report in reports]
    selected_id = st.selectbox(
        "Report details",
        report_ids,
        format_func=lambda value: f"Issue #{value}",
    )
    selected = next(report for report in reports if report["issue_id"] == selected_id)
    image_column, details_column = st.columns([1, 2])
    with image_column:
        image_path = selected.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="Submitted image", use_container_width=True)
        else:
            render_empty_state(st, "The submitted image is unavailable.")
    with details_column:
        for label, key in (
            ("Category", "category"),
            ("Detection confidence", "confidence"),
            ("Severity", "severity"),
            ("Priority score", "priority_score"),
            ("Priority label", "priority_label"),
            ("Estimated repair cost", "estimated_repair_cost"),
            ("Latitude", "latitude"),
            ("Longitude", "longitude"),
            ("Description", "description"),
            ("Reported date", "reported_at"),
            ("Status", "status"),
        ):
            st.write(f"**{label}:** {selected.get(key) if selected.get(key) is not None else 'Not available'}")
