"""Shared CiviScan light and dark themes."""

PALETTE = {
    "forest": "#173C32",
    "green": "#315B48",
    "bronze": "#B58A55",
    "charcoal": "#141B18",
    "ivory": "#F4F2EA",
    "sage": "#DCE5DE",
}


def apply_theme(st, mode="Light"):
    """Apply the shared CiviScan design tokens to a Streamlit page."""
    dark_mode = mode == "Dark"
    background = PALETTE["charcoal"] if dark_mode else PALETTE["ivory"]
    surface = "#202925" if dark_mode else "#FFFFFF"
    text = PALETTE["ivory"] if dark_mode else PALETTE["charcoal"]
    muted = "#AEBBB3" if dark_mode else "#65736B"
    border = "#3B4841" if dark_mode else "#D7DED8"
    sidebar = "#18221D" if dark_mode else "#E8EDE7"

    st.markdown(
        f"""
<style>
    :root {{
        --civiscan-forest: {PALETTE['forest']};
        --civiscan-green: {PALETTE['green']};
        --civiscan-bronze: {PALETTE['bronze']};
        --civiscan-charcoal: {PALETTE['charcoal']};
        --civiscan-ivory: {PALETTE['ivory']};
        --civiscan-sage: {PALETTE['sage']};
        --civiscan-background: {background};
        --civiscan-surface: {surface};
        --civiscan-text: {text};
        --civiscan-muted: {muted};
        --civiscan-border: {border};
    }}
    html, body, [class*="css"] {{ color: var(--civiscan-text); }}
    .stApp {{ background: var(--civiscan-background); color: var(--civiscan-text); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {sidebar}; border-right: 1px solid var(--civiscan-border); }}
    [data-testid="stSidebar"] * {{ color: var(--civiscan-text); }}
    .block-container {{ max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }}
    h1, h2, h3 {{ color: var(--civiscan-forest); font-weight: 700; }}
    {'h1, h2, h3 { color: #DCE5DE; }' if dark_mode else ''}
    [data-testid="stMetric"] {{ background: var(--civiscan-surface); border: 1px solid var(--civiscan-border); border-radius: 8px; padding: 1rem; }}
    [data-testid="stMetricLabel"] {{ color: var(--civiscan-muted); }}
    [data-testid="stMetricValue"] {{ color: var(--civiscan-forest); }}
    {'[data-testid="stMetricValue"] { color: #DCE5DE; }' if dark_mode else ''}
    .stButton > button, .stFormSubmitButton > button {{
        background: var(--civiscan-forest); color: #FFFFFF; border: 1px solid var(--civiscan-forest);
        border-radius: 6px; min-height: 2.5rem; font-weight: 600;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: var(--civiscan-green); border-color: var(--civiscan-green); color: #FFFFFF;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--civiscan-border); border-radius: 6px; }}
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div {{
        background: var(--civiscan-surface); border-color: var(--civiscan-border);
    }}
    .civiscan-brand {{ display:flex; align-items:center; gap:0.65rem; margin:0.2rem 0 1.2rem; }}
    .civiscan-mark {{ width:2.1rem; height:2.1rem; display:grid; place-items:center; border-radius:6px; background:{PALETTE['forest']}; color:#fff; font-weight:800; }}
    .civiscan-name {{ color:var(--civiscan-text); font-size:1.15rem; font-weight:750; }}
    .civiscan-subtitle {{ color:var(--civiscan-muted); font-size:0.88rem; margin-top:0.15rem; }}
    .civiscan-card {{ background:var(--civiscan-surface); border:1px solid var(--civiscan-border); border-radius:8px; padding:1rem 1.15rem; }}
    .detection-box {{ background:var(--civiscan-surface); border:1px solid var(--civiscan-border); border-left:4px solid var(--civiscan-bronze); border-radius:6px; padding:0.9rem 1rem; margin:0.55rem 0; }}
    .model-badge {{ display:inline-block; padding:0.25rem 0.6rem; border-radius:4px; background:var(--civiscan-sage); color:var(--civiscan-forest); font-size:0.78rem; font-weight:700; }}
    .badge-v3 {{ background:var(--civiscan-green); color:#FFFFFF; }}
    .severity-low {{ color:var(--civiscan-green); }}
    .severity-medium {{ color:var(--civiscan-bronze); }}
    .severity-high, .severity-critical {{ color:#B54736; }}
</style>
""",
        unsafe_allow_html=True,
    )