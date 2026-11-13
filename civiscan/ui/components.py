"""Reusable, small Streamlit components shared across CiviScan pages."""


def render_brand(st, subtitle="Civic infrastructure intelligence"):
    st.markdown(
        "<div class='civiscan-brand'><div class='civiscan-mark'>C</div>"
        "<div><div class='civiscan-name'>CiviScan</div>"
        f"<div class='civiscan-subtitle'>{subtitle}</div></div></div>",
        unsafe_allow_html=True,
    )


def render_page_header(st, title, description=None):
    st.title(title)
    if description:
        st.caption(description)


def render_empty_state(st, message):
    st.info(message)


def render_metrics(st, metrics):
    """Render a compact row of (label, value, optional delta) metric tuples."""
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        label, value, *delta = metric
        column.metric(label, value, delta[0] if delta else None)