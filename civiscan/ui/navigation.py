"""Role-aware navigation for the currently implemented application pages."""

from auth import ROLE_CITIZEN, ROLE_MUNICIPAL_OFFICER
from config import APP_TAGLINE, DEFAULT_THEME
from civiscan.ui.components import render_brand


ROLE_PAGES = {
    ROLE_CITIZEN: ("Dashboard", "AI Detection", "My Reports", "My Profile"),
    ROLE_MUNICIPAL_OFFICER: (
        "Overview",
        "AI Detection",
        "Incoming Media",
        "Issue Management",
        "GIS Map",
        "Predictive Analytics",
        "Settings",
        "My Profile",
    ),
}


def render_navigation(st, user):
    """Render navigation and theme controls; return (page, theme, logout)."""
    with st.sidebar:
        render_brand(st, APP_TAGLINE)
        st.caption(f"Signed in as {user['name']} · {user['role']}")
        pages = ROLE_PAGES.get(user["role"], ())
        if user["role"] == ROLE_MUNICIPAL_OFFICER and user.get("is_admin"):
            pages = (*pages, "Officer Administration")
        page = st.radio("Workspace", pages, key=f"navigation_{user['role']}")
        theme = st.radio(
            "Appearance",
            ("Light", "Dark"),
            index=0 if DEFAULT_THEME.lower() != "dark" else 1,
            horizontal=True,
            key="civiscan_theme",
        )
        st.markdown("---")
        logout = st.button("Sign out", use_container_width=True)
    return page, theme, logout