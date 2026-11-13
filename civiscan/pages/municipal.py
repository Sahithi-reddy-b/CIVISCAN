"""Municipal-officer pages backed by existing issue and priority services."""

import math
import os

import folium
import pandas as pd
import streamlit as st
from streamlit import components

from civiscan.ui.components import render_empty_state, render_metrics, render_page_header
from auth import ROLE_MUNICIPAL_OFFICER, require_authenticated_role
from database import (
    get_maintenance_cost_rates,
    set_maintenance_cost_rates,
    update_issue_status,
)
from priority import calculate_and_update_priority, cluster_hotspots


STATUS_OPTIONS = [
    "NEW",
    "VERIFIED",
    "ASSIGNED",
    "IN PROGRESS",
    "AWAITING VERIFICATION",
    "RESOLVED",
    "REJECTED",
    "REOPENED",
]


def _text(value):
    return str(value).strip() if value is not None else ""


def _coordinates(issue):
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


def _render_overview(issues):
    render_page_header(st, "Overview", "Current civic issue workload and response status.")
    open_count = sum(_text(issue.get("status")).upper() != "RESOLVED" for issue in issues)
    high_count = sum(_text(issue.get("priority_label")).lower() in {"critical", "high"} for issue in issues)
    mapped_count = sum(_coordinates(issue) is not None for issue in issues)
    render_metrics(st, [
        ("Total issues", len(issues)),
        ("Open issues", open_count),
        ("High priority", high_count),
        ("Mapped issues", mapped_count),
    ])
    st.subheader("Recent reports")
    if not issues:
        render_empty_state(st, "No civic reports have been submitted yet.")
        return
    recent = [{
        "Issue": issue.get("issue_id"),
        "Category": issue.get("category") or "Unknown",
        "Priority": issue.get("priority_label") or "Unscored",
        "Reported": issue.get("reported_at") or "Unknown",
        "Status": issue.get("status") or "NEW",
    } for issue in issues[:10]]
    st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)


def _render_issue_management(issues):
    render_page_header(st, "Issue Management", "Review reports, inspect evidence, and update work status.")
    if not issues:
        render_empty_state(st, "No civic reports are available to manage.")
        return

    categories = sorted({_text(issue.get("category")) for issue in issues if issue.get("category")})
    priorities = sorted({_text(issue.get("priority_label")) for issue in issues if issue.get("priority_label")})
    statuses = sorted({_text(issue.get("status")) for issue in issues if issue.get("status")})
    filters = st.columns(3)
    category_filter = filters[0].selectbox("Category", ["All"] + categories)
    priority_filter = filters[1].selectbox("Priority", ["All"] + priorities)
    status_filter = filters[2].selectbox("Status", ["All"] + statuses)
    filtered = [
        issue for issue in issues
        if (category_filter == "All" or _text(issue.get("category")) == category_filter)
        and (priority_filter == "All" or _text(issue.get("priority_label")) == priority_filter)
        and (status_filter == "All" or _text(issue.get("status")) == status_filter)
    ]
    table_rows = [{
        "Issue": issue.get("issue_id"),
        "Category": issue.get("category") or "Unknown",
        "Confidence": issue.get("confidence"),
        "Severity": issue.get("severity") or "Unknown",
        "Priority": issue.get("priority_label") or "Unscored",
        "Score": issue.get("priority_score"),
        "Repair estimate": issue.get("estimated_repair_cost"),
        "Reported": issue.get("reported_at") or "Unknown",
        "Status": issue.get("status") or "NEW",
    } for issue in filtered]
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        render_empty_state(st, "No reports match the selected filters.")
        return

    issue_ids = [issue["issue_id"] for issue in filtered]
    selected_id = st.selectbox("Report details", issue_ids, format_func=lambda value: f"Issue #{value}")
    issue = next(item for item in filtered if item["issue_id"] == selected_id)
    left, right = st.columns([1, 2])
    with left:
        image_path = issue.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=f"Evidence for issue #{selected_id}", use_container_width=True)
        else:
            render_empty_state(st, "No report image is available.")
    with right:
        st.write(f"**Category:** {issue.get('category') or 'Unknown'}")
        st.write(f"**Severity:** {issue.get('severity') or 'Unknown'}")
        st.write(f"**Priority:** {issue.get('priority_label') or 'Unscored'} ({issue.get('priority_score') or 0})")
        if issue.get("priority_reason"):
            st.caption(f"Priority rationale: {issue['priority_reason']}")
        st.write(f"**Estimated repair cost:** {issue.get('estimated_repair_cost') if issue.get('estimated_repair_cost') is not None else 'Not estimated'}")
        st.write(f"**Location:** {_coordinates(issue) or 'Not provided'}")
        st.write(f"**Description:** {issue.get('description') or 'No description'}")

    current_status = _text(issue.get("status")).upper()
    status_index = STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0
    next_status = st.selectbox("Update status", STATUS_OPTIONS, index=status_index, key=f"issue_status_{selected_id}")
    if st.button("Save status", key=f"save_status_{selected_id}"):
        if update_issue_status(selected_id, next_status, actor=st.session_state.get("current_user")):
            st.success("Issue status updated.")
            st.rerun()
        st.error("The report could not be updated.")


def _render_gis_map(issues, hotspots):
    render_page_header(st, "GIS Map", "Geolocated civic reports and detected issue clusters.")
    mapped = [(issue, _coordinates(issue)) for issue in issues]
    mapped = [(issue, coordinates) for issue, coordinates in mapped if coordinates]
    if not mapped:
        render_empty_state(st, "No reports have valid coordinates yet. Add latitude and longitude when submitting a report.")
        return

    center = (
        sum(point[0] for _, point in mapped) / len(mapped),
        sum(point[1] for _, point in mapped) / len(mapped),
    )
    issue_map = folium.Map(location=center, zoom_start=12, control_scale=True)
    marker_colors = {"critical": "red", "high": "red", "moderate": "orange", "medium": "orange", "low": "green"}
    for issue, coordinates in mapped:
        priority = _text(issue.get("priority_label")).lower()
        popup = (
            f"Issue #{issue.get('issue_id')}<br>"
            f"Category: {issue.get('category') or 'Unknown'}<br>"
            f"Severity: {issue.get('severity') or 'Unknown'}<br>"
            f"Priority: {issue.get('priority_label') or 'Unscored'}<br>"
            f"Status: {issue.get('status') or 'NEW'}"
        )
        folium.Marker(
            location=coordinates,
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color=marker_colors.get(priority, "gray")),
        ).add_to(issue_map)

    for hotspot in hotspots:
        coordinates = hotspot["center"]
        folium.Circle(
            location=coordinates,
            radius=50,
            color="#B58A55",
            fill=True,
            fill_color="#B58A55",
            fill_opacity=0.18,
            popup=f"Hotspot {hotspot['hotspot_id']} · {hotspot['count']} issues",
        ).add_to(issue_map)
    components.v1.html(issue_map._repr_html_(), height=580)


def _render_predictive_analytics(issues, hotspots):
    render_page_header(st, "Predictive Analytics", "Priority and recurrence indicators derived from submitted reports.")
    if not issues:
        render_empty_state(st, "Analytics will be available when civic reports are submitted.")
        return

    data = pd.DataFrame(issues)
    chart_columns = st.columns(2)
    for column, field, title in [
        (chart_columns[0], "category", "Issues by category"),
        (chart_columns[1], "severity", "Issues by severity"),
        (chart_columns[0], "priority_label", "Issues by priority"),
        (chart_columns[1], "status", "Issues by status"),
    ]:
        values = data[field].fillna("Unknown").replace("", "Unknown").value_counts()
        column.markdown(f"**{title}**")
        column.bar_chart(values)

    st.subheader("Geographic hotspots")
    if not hotspots:
        render_empty_state(st, "No recurring geographic clusters meet the current minimum report count.")
        return
    hotspot_rows = [{
        "Hotspot": item["hotspot_id"],
        "Reports": item["count"],
        "Dominant category": item["dominant_category"],
        "Average severity score": item["average_severity"],
        "Average priority": item["average_priority"],
    } for item in hotspots]
    st.dataframe(pd.DataFrame(hotspot_rows), use_container_width=True, hide_index=True)


def _render_settings(user):
    render_page_header(st, "Settings", "Configure indicative maintenance cost assumptions.")
    st.info("These values are editable assumptions in INR, not official municipal quotations.")
    categories = ("pothole", "garbage_dumping", "damaged_streetlight")
    rates = get_maintenance_cost_rates()
    with st.form("maintenance_cost_rates_form"):
        category = st.selectbox("Issue category", categories)
        current = rates.get(category, {})
        columns = st.columns(4)
        values = {
            "material": columns[0].number_input(
                "Material (₹)", min_value=0.0, value=float(current.get("material", 0.0)), step=100.0
            ),
            "labour": columns[1].number_input(
                "Labour (₹)", min_value=0.0, value=float(current.get("labour", 0.0)), step=100.0
            ),
            "equipment": columns[2].number_input(
                "Equipment (₹)", min_value=0.0, value=float(current.get("equipment", 0.0)), step=100.0
            ),
            "transportation": columns[3].number_input(
                "Transportation (₹)", min_value=0.0, value=float(current.get("transportation", 0.0)), step=100.0
            ),
        }
        submitted = st.form_submit_button("Save category cost assumptions")
    if submitted:
        try:
            set_maintenance_cost_rates(category, values, actor=user)
        except (ValueError, PermissionError) as error:
            st.error(str(error))
        else:
            st.success(f"Indicative cost assumptions saved for {category}.")
            st.rerun()


def render_municipal_page(section):
    """Render one implemented municipal section using persisted reports."""
    current_user = require_authenticated_role(
        st.session_state.get("current_user"),
        {ROLE_MUNICIPAL_OFFICER},
    )
    actor = current_user
    if section == "Settings":
        _render_settings(actor)
        return
    issues = calculate_and_update_priority(actor)
    hotspots, _ = cluster_hotspots(issues)
    if section == "Overview":
        _render_overview(issues)
    elif section == "Issue Management":
        _render_issue_management(issues)
    elif section == "GIS Map":
        _render_gis_map(issues, hotspots)
    elif section == "Predictive Analytics":
        _render_predictive_analytics(issues, hotspots)
