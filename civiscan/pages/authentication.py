"""Public CiviScan welcome, login, registration, and password-reset page."""

import streamlit as st

from auth import (
    ROLE_CITIZEN,
    ROLE_MUNICIPAL_OFFICER,
    authenticate_user,
    establish_session,
    password_reset_delivery_configured,
    register_citizen,
    request_password_reset,
    reset_password,
)
from civiscan.ui.components import render_brand
from civiscan.ui.theme import apply_theme


def _login_form(tab, role, key):
    with tab:
        show_password = st.checkbox("Show password", key=f"{key}_show_password")
        with st.form(f"{key}_login_form"):
            identifier = st.text_input("Email or username", key=f"{key}_identifier")
            password = st.text_input(
                "Password",
                type="default" if show_password else "password",
                key=f"{key}_password",
            )
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user = authenticate_user(identifier, password, required_role=role)
            if user is None:
                st.error("Credentials are invalid, the account is unavailable, or sign-in is temporarily locked.")
                return
            establish_session(st.session_state, user)
            st.success("Signed in successfully.")
            st.rerun()


def _render_reset_form(token):
    st.subheader("Choose a new password")
    show_password = st.checkbox("Show password", key="reset_show_password")
    password_type = "default" if show_password else "password"
    with st.form("password_reset_form"):
        password = st.text_input("New password", type=password_type, help="Use at least 12 characters.")
        confirmation = st.text_input("Confirm new password", type=password_type)
        submitted = st.form_submit_button("Reset password")
    if submitted:
        if password != confirmation:
            st.error("Password confirmation does not match.")
            return
        try:
            completed = reset_password(token, password)
        except ValueError as error:
            st.error(str(error))
            return
        if not completed:
            st.error("This reset link is invalid or has expired. Request a new one.")
            return
        st.success("Password updated. Sign in with your new password.")
        st.query_params.clear()


def render_authentication_page():
    """Render the public authentication workflow with no officer registration path."""
    with st.sidebar:
        mode = st.radio("Appearance", ("Light", "Dark"), key="civiscan_public_theme")
    apply_theme(st, mode)

    content, auth_panel = st.columns([1, 1.2], gap="large")
    with content:
        render_brand(st, "Civic infrastructure intelligence")
        st.title("A clearer view of civic issues")
        st.write(
            "Report local infrastructure concerns, follow their progress, and help municipal "
            "teams prioritize repairs with location-aware evidence."
        )
        st.markdown("---")
        st.markdown("**For residents**")
        st.caption("Submit issue reports, review status updates, and manage your account.")
        st.markdown("**For municipal teams**")
        st.caption("Review incoming reports, manage issue status, and inspect geographic patterns.")

    with auth_panel:
        reset_token = st.query_params.get("reset")
        if reset_token:
            _render_reset_form(reset_token)
            st.markdown("---")

        citizen_login, citizen_registration, officer_login, forgot_password = st.tabs([
            "Citizen Login",
            "Citizen Registration",
            "Officer Login",
            "Forgot Password",
        ])
        _login_form(citizen_login, ROLE_CITIZEN, "citizen")
        _login_form(officer_login, ROLE_MUNICIPAL_OFFICER, "officer")

        with citizen_registration:
            show_password = st.checkbox("Show password", key="citizen_register_show_password")
            password_type = "default" if show_password else "password"
            with st.form("citizen_registration_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("Email address")
                phone = st.text_input("Phone number", placeholder="+1 555 010 1234")
                password = st.text_input(
                    "Password", type=password_type, help="Use at least 12 characters."
                )
                confirmation = st.text_input("Confirm password", type=password_type)
                submitted = st.form_submit_button("Create citizen account", use_container_width=True)
            if submitted:
                if password != confirmation:
                    st.error("Password confirmation does not match.")
                else:
                    try:
                        register_citizen(full_name, email, phone, password)
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.success("Registration complete. You can now sign in.")

        with forgot_password:
            st.caption("Enter the email address linked to your account.")
            with st.form("forgot_password_form"):
                email = st.text_input("Account email")
                submitted = st.form_submit_button("Send reset link", use_container_width=True)
            if submitted:
                if not password_reset_delivery_configured():
                    st.info("Password reset email is not configured. Contact your CiviScan administrator.")
                else:
                    request_password_reset(email)
                    st.success("If an active account matches that email, a reset link has been sent.")
