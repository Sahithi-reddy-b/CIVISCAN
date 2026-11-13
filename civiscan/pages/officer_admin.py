"""Administrator-only municipal officer account management."""

import pandas as pd
import streamlit as st

from auth import create_officer, list_officers, require_authenticated_role, update_officer
from civiscan.ui.components import render_empty_state, render_page_header
from auth import ROLE_MUNICIPAL_OFFICER


def render_officer_admin(user):
    user = require_authenticated_role(user, {ROLE_MUNICIPAL_OFFICER})
    if not user.get("is_admin"):
        st.error("Administrator permission is required to manage officer accounts.")
        st.stop()

    render_page_header(st, "Officer Administration", "Create officer accounts and manage access status.")
    st.subheader("Create municipal officer")
    show_password = st.checkbox("Show password", key="officer_admin_show_password")
    password_type = "default" if show_password else "password"
    with st.form("create_officer_form"):
        name = st.text_input("Full name")
        email = st.text_input("Official email")
        employee_id = st.text_input("Employee identifier")
        department = st.text_input("Department")
        designation = st.text_input("Designation")
        jurisdiction = st.text_input("Assigned jurisdiction")
        password = st.text_input("Temporary password", type=password_type, help="At least 12 characters.")
        confirmation = st.text_input("Confirm password", type=password_type)
        submitted = st.form_submit_button("Create officer account")
    if submitted:
        if password != confirmation:
            st.error("Password confirmation does not match.")
        else:
            try:
                create_officer(
                    user,
                    name=name,
                    email=email,
                    employee_id=employee_id,
                    department=department,
                    designation=designation,
                    jurisdiction=jurisdiction,
                    password=password,
                )
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("Municipal Officer account created.")
                st.rerun()

    st.subheader("Officer accounts")
    officers = list_officers(user)
    if not officers:
        render_empty_state(st, "No municipal officer accounts were found.")
        return
    visible_fields = (
        "user_id", "name", "email", "employee_id", "department", "designation",
        "jurisdiction", "account_status", "is_admin",
    )
    st.dataframe(
        pd.DataFrame([{field: officer.get(field) for field in visible_fields} for officer in officers]),
        use_container_width=True,
        hide_index=True,
    )
    selected_id = st.selectbox(
        "Select officer",
        [officer["user_id"] for officer in officers],
        format_func=lambda value: next(
            f"{item['name']} · {item.get('employee_id') or item['user_id']}"
            for item in officers if item["user_id"] == value
        ),
    )
    selected = next(item for item in officers if item["user_id"] == selected_id)
    with st.form(f"manage_officer_{selected_id}"):
        account_status = st.selectbox(
            "Account status",
            ("active", "inactive"),
            index=0 if selected.get("account_status") == "active" else 1,
        )
        is_admin = st.checkbox("Administrator access", value=selected.get("is_admin", False))
        save = st.form_submit_button("Save officer access")
    if save:
        try:
            update_officer(
                user,
                selected_id,
                account_status=account_status,
                is_admin=is_admin,
            )
        except (ValueError, PermissionError) as error:
            st.error(str(error))
        else:
            st.success("Officer access updated.")
            st.rerun()
