"""Self-service profile and password management."""

import streamlit as st

from auth import (
    change_password,
    get_user_by_id,
    require_authenticated_role,
    update_user_profile,
)
from civiscan.ui.components import render_page_header
from auth import ROLE_CITIZEN, ROLE_MUNICIPAL_OFFICER


PROFILE_ROLES = {ROLE_CITIZEN, ROLE_MUNICIPAL_OFFICER}


def render_profile(user):
    user = require_authenticated_role(user, PROFILE_ROLES)
    fresh_user = get_user_by_id(user["user_id"], actor=user)
    render_page_header(st, "My Profile", "Manage your account details and password.")

    with st.form("profile_form"):
        full_name = st.text_input("Full name", value=fresh_user.get("name") or "")
        email = st.text_input("Email", value=fresh_user.get("email") or "")
        phone_number = st.text_input("Phone number", value=fresh_user.get("phone_number") or "")
        profile_submitted = st.form_submit_button("Save profile")
    if profile_submitted:
        try:
            update_user_profile(
                user,
                user["user_id"],
                full_name=full_name,
                email=email,
                phone_number=phone_number,
            )
        except (ValueError, PermissionError) as error:
            st.error(str(error))
        else:
            st.success("Profile updated.")
            st.rerun()

    st.subheader("Change password")
    show_password = st.checkbox("Show password fields", key="profile_show_password")
    password_type = "default" if show_password else "password"
    with st.form("change_password_form"):
        current_password = st.text_input("Current password", type=password_type)
        new_password = st.text_input("New password", type=password_type, help="Use at least 12 characters.")
        confirm_password = st.text_input("Confirm new password", type=password_type)
        password_submitted = st.form_submit_button("Change password")
    if password_submitted:
        if new_password != confirm_password:
            st.error("New password confirmation does not match.")
        else:
            try:
                change_password(user, current_password, new_password)
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("Password changed. Sign in again on your next session.")
