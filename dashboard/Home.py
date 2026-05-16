"""EpiPulse AI multipage home screen."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="EpiPulse AI", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "disease_data.csv"


st.markdown(
    """
    <style>
    .block-container{padding-top:1.6rem;padding-bottom:2.5rem}
    .hero{
        border:1px solid #d9e2ec;
        border-radius:8px;
        padding:1.4rem 1.5rem;
        background:linear-gradient(135deg,#f8fbff 0%,#eef8fb 100%);
        margin-bottom:1.2rem;
    }
    .hero-title{color:#102a43;font-size:2.35rem;font-weight:800;line-height:1.12;margin:0}
    .hero-subtitle{color:#52606d;font-size:1.05rem;margin-top:.45rem;max-width:900px}
    .section-title{color:#102a43;font-size:1.2rem;font-weight:750;margin:.35rem 0 .65rem}
    .feature-card{
        border:1px solid #d9e2ec;
        border-radius:8px;
        padding:1rem 1.05rem;
        background:#fff;
        min-height:184px;
    }
    .feature-kicker{color:#0e7490;font-size:.78rem;font-weight:800;text-transform:uppercase}
    .feature-title{color:#102a43;font-size:1.35rem;font-weight:760;margin:.2rem 0 .45rem}
    .feature-copy{color:#52606d;font-size:.95rem;line-height:1.48;margin-bottom:.75rem}
    .metric-card{
        border:1px solid #d9e2ec;
        border-radius:8px;
        padding:.9rem 1rem;
        background:#fff;
    }
    .metric-label{color:#627d98;font-size:.76rem;text-transform:uppercase;font-weight:750}
    .metric-value{color:#102a43;font-size:1.7rem;line-height:1.15;font-weight:800;margin-top:.25rem}
    .hint-panel{
        border-left:4px solid #0e7490;
        background:#f7fbfc;
        border-radius:6px;
        padding:1rem 1.1rem;
        color:#334e68;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_home_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    return df


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(kicker: str, title: str, copy: str, page: str, button: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-kicker">{kicker}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(page, label=button, use_container_width=True)


try:
    df = load_home_data()
except Exception as exc:
    st.error(f"Could not load the dashboard dataset: {exc}")
    st.stop()


regions = df["region"].nunique() if "region" in df.columns else 0
diseases = df["disease"].nunique() if "disease" in df.columns else 0
years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
year_label = f"{int(min(years))}-{int(max(years))}" if years else "N/A"
total_cases = int(df["cases"].sum()) if "cases" in df.columns else 0

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">Welcome to EpiPulse AI</h1>
        <div class="hero-subtitle">
            Monitor disease activity, ask grounded public-health questions, and inspect
            outbreak signals across regions from one Streamlit workspace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Regions", f"{regions:,}")
with metric_cols[1]:
    metric_card("Diseases", f"{diseases:,}")
with metric_cols[2]:
    metric_card("Years", year_label)
with metric_cols[3]:
    metric_card("Total Cases", f"{total_cases:,}")

st.write("")
st.markdown('<div class="section-title">Choose Where To Start</div>', unsafe_allow_html=True)

feature_cols = st.columns(2)
with feature_cols[0]:
    feature_card(
        "Ask and explain",
        "Chat",
        "Use the AI epidemiology assistant to ask about cases, risk levels, trends, "
        "WHO-style interventions, IDSP response steps, and source-backed guidance. "
        "It also shows the selected region snapshot beside the conversation.",
        "Pages/1_chat.py",
        "Open Chat",
    )

with feature_cols[1]:
    feature_card(
        "Inspect and compare",
        "Analytics",
        "Explore the outbreak dashboard with filters, risk leaderboards, case trends, "
        "spike detection, disease breakdowns, year-over-year comparison, and a 7-day "
        "case projection.",
        "Pages/2_Analytics.py",
        "Open Analytics",
    )

st.write("")
chart_col, note_col = st.columns([1.65, 1])

with chart_col:
    st.markdown('<div class="section-title">Dataset Snapshot</div>', unsafe_allow_html=True)
    if {"disease", "cases"}.issubset(df.columns):
        disease_totals = (
            df.groupby("disease", as_index=False)["cases"]
            .sum()
            .sort_values("cases", ascending=True)
        )
        fig = px.bar(
            disease_totals,
            x="cases",
            y="disease",
            orientation="h",
            color="disease",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text="cases",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis_title="Total cases",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add disease and cases columns to show the dataset snapshot chart.")

with note_col:
    st.markdown('<div class="section-title">What This App Does</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hint-panel">
            <strong>Chat</strong> is for plain-English investigation and guideline-aware
            answers. Start there when you want interpretation or recommendations.<br><br>
            <strong>Analytics</strong> is for visual monitoring. Start there when you
            want to compare regions, spot spikes, filter dates, or export filtered data.<br><br>
            Use the sidebar to move between Home, Chat, and Analytics at any time.
        </div>
        """,
        unsafe_allow_html=True,
    )
