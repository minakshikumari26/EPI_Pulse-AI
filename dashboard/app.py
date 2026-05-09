from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import zscore


st.set_page_config(
    page_title="EpiPulse AI",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "disease_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.drop_duplicates()
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    numeric_cols = ["cases", "temperature", "humidity", "rainfall"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in ["cases", "humidity", "rainfall"]:
        value_range = df.groupby("region")[col].transform(lambda values: values.max() - values.min())
        min_value = df.groupby("region")[col].transform("min")
        df[f"{col}_scaled"] = ((df[col] - min_value) / value_range).fillna(0)

    df["risk_score"] = (
        df["cases_scaled"] * 60
        + df["humidity_scaled"] * 25
        + df["rainfall_scaled"] * 15
    ).round(2)
    df["risk_level"] = df["risk_score"].apply(
        lambda score: "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    )
    df["z_score"] = df.groupby("region")["cases"].transform(zscore)
    df["is_spike"] = df["z_score"] > 1.5

    return df


def build_projection(df, periods=7):
    projections = []

    for region, region_df in df.groupby("region"):
        region_df = region_df.sort_values("date")
        last_date = region_df["date"].max()
        last_cases = float(region_df.iloc[-1]["cases"])
        daily_change = get_daily_change(region_df)

        for day in range(1, periods + 1):
            projections.append(
                {
                    "region": region,
                    "date": last_date + pd.Timedelta(days=day),
                    "projected_cases": max(0, round(last_cases + daily_change * day, 1)),
                }
            )

    return pd.DataFrame(projections)


def get_daily_change(region_df):
    region_df = region_df.sort_values("date")
    if len(region_df) < 8:
        return 0

    recent_avg = region_df.tail(7)["cases"].mean()
    previous_window = region_df.iloc[-14:-7] if len(region_df) >= 14 else region_df.iloc[:-7]
    previous_avg = previous_window["cases"].mean()

    if pd.isna(previous_avg):
        return 0

    return (recent_avg - previous_avg) / 7


def metric_card(label, value, helper):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level):
    return f'<span class="risk-badge risk-{level.lower()}">{level}</span>'


df = load_data()

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .app-header {
            border-bottom: 1px solid #d9e2ec;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
        }
        .app-title {
            color: #102a43;
            font-size: 2.15rem;
            font-weight: 750;
            line-height: 1.15;
            margin: 0;
        }
        .app-subtitle {
            color: #52606d;
            font-size: 1rem;
            margin-top: 0.35rem;
            max-width: 780px;
        }
        .metric-card {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            min-height: 116px;
        }
        .metric-label {
            color: #627d98;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 700;
        }
        .metric-value {
            color: #102a43;
            font-size: 1.95rem;
            line-height: 1.1;
            font-weight: 750;
            margin-top: 0.35rem;
        }
        .metric-helper {
            color: #52606d;
            font-size: 0.86rem;
            margin-top: 0.4rem;
        }
        .report-panel {
            border-left: 4px solid #0e7490;
            background: #f7fbfc;
            padding: 1rem 1.1rem;
            border-radius: 6px;
            color: #243b53;
        }
        .risk-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .risk-high {
            color: #7f1d1d;
            background: #fee2e2;
        }
        .risk-medium {
            color: #7c2d12;
            background: #ffedd5;
        }
        .risk-low {
            color: #14532d;
            background: #dcfce7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">EpiPulse AI</h1>
        <div class="app-subtitle">
            Regional outbreak monitoring with trend signals, spike detection, risk scoring, and short-term projections.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

regions = sorted(df["region"].unique())
region_search = st.sidebar.text_input(
    "Search Region",
    placeholder="Type a region or state name to filter",
)
selected_regions = st.sidebar.multiselect(
    "Regions",
    regions,
    default=regions[:4],
    help="Choose one or more regions from the dataset, or use the search box above.",
)

if region_search:
    matched_regions = [r for r in regions if region_search.lower() in r.lower()]
    if matched_regions:
        selected_regions = matched_regions
    else:
        st.sidebar.warning(
            "No regions match that search. Please enter a valid region name present in the dataset."
        )

date_min = df["date"].min().date()
date_max = df["date"].max().date()
selected_dates = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = date_min, date_max

if not selected_regions:
    st.warning("Select at least one region to view the dashboard.")
    st.stop()

filtered_df = df[
    (df["region"].isin(selected_regions))
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

latest_rows = filtered_df.sort_values("date").groupby("region").tail(1)
total_cases = int(filtered_df["cases"].sum())
peak_row = filtered_df.loc[filtered_df["cases"].idxmax()]
spike_count = int(filtered_df["is_spike"].sum())
high_risk_count = int((latest_rows["risk_level"] == "High").sum())
top_risk_row = latest_rows.loc[latest_rows["risk_score"].idxmax()]

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Total Cases", f"{total_cases:,}", f"{len(selected_regions)} selected regions")
with metric_cols[1]:
    metric_card("Peak Daily Cases", int(peak_row["cases"]), f"{peak_row['region']} on {peak_row['date'].date()}")
with metric_cols[2]:
    metric_card("Spike Days", spike_count, "Region-wise z-score above 1.5")
with metric_cols[3]:
    metric_card("High Risk Regions", high_risk_count, f"Highest: {top_risk_row['region']}")

st.write("")

chart_col, side_col = st.columns([2.2, 1])

with chart_col:
    st.subheader("Case Trend")
    trend_fig = px.line(
        filtered_df,
        x="date",
        y="cases",
        color="region",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    spike_df = filtered_df[filtered_df["is_spike"]]
    if not spike_df.empty:
        trend_fig.add_trace(
            go.Scatter(
                x=spike_df["date"],
                y=spike_df["cases"],
                mode="markers",
                marker={"size": 9, "color": "#dc2626", "symbol": "x"},
                name="Spike",
                text=spike_df["region"],
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Cases: %{y}<extra></extra>",
            )
        )
    trend_fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend_title_text="Region",
        xaxis_title=None,
        yaxis_title="Cases",
    )
    st.plotly_chart(trend_fig, use_container_width=True)

with side_col:
    st.subheader("Latest Risk")
    risk_fig = px.bar(
        latest_rows.sort_values("risk_score", ascending=True),
        x="risk_score",
        y="region",
        color="risk_level",
        orientation="h",
        color_discrete_map={"Low": "#16a34a", "Medium": "#f97316", "High": "#dc2626"},
        text="risk_score",
    )
    risk_fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Risk Score",
        yaxis_title=None,
        legend_title_text="Level",
    )
    st.plotly_chart(risk_fig, use_container_width=True)

st.subheader("7-Day Case Projection")
projection_df = build_projection(filtered_df)
projection_fig = go.Figure()

for region, region_df in filtered_df.groupby("region"):
    history = region_df.sort_values("date").tail(21)
    projected = projection_df[projection_df["region"] == region]

    projection_fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["cases"],
            mode="lines+markers",
            name=f"{region} history",
        )
    )
    projection_fig.add_trace(
        go.Scatter(
            x=projected["date"],
            y=projected["projected_cases"],
            mode="lines+markers",
            line={"dash": "dash"},
            name=f"{region} projection",
        )
    )

projection_fig.update_layout(
    height=380,
    margin={"l": 10, "r": 10, "t": 20, "b": 10},
    xaxis_title=None,
    yaxis_title="Cases",
    legend_title_text="Series",
)
st.plotly_chart(projection_fig, use_container_width=True)

st.subheader("Report")

top_regions = (
    filtered_df.groupby("region")["cases"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
)
declining_regions = []
rising_regions = []
for region, region_df in filtered_df.groupby("region"):
    daily_change = get_daily_change(region_df)
    if daily_change > 0:
        rising_regions.append(region)
    else:
        declining_regions.append(region)

top_region_text = ", ".join([f"{region} ({int(cases):,})" for region, cases in top_regions.items()])
rising_text = ", ".join(rising_regions) if rising_regions else "none"
declining_text = ", ".join(declining_regions) if declining_regions else "none"
top_risk_badge = risk_badge(top_risk_row["risk_level"])

st.markdown(
    f"""
    <div class="report-panel">
        <strong>Summary:</strong> The selected view contains <strong>{total_cases:,}</strong> reported cases.
        The highest daily value is <strong>{int(peak_row["cases"])}</strong> in <strong>{peak_row["region"]}</strong>
        on <strong>{peak_row["date"].date()}</strong>. Region-wise spike detection found
        <strong>{spike_count}</strong> possible outbreak spike days.
        <br><br>
        <strong>Highest burden:</strong> {top_region_text}.
        <br>
        <strong>Current highest risk:</strong> {top_risk_row["region"]} with score
        <strong>{top_risk_row["risk_score"]}</strong> {top_risk_badge}.
        <br>
        <strong>Recent direction:</strong> Rising regions: {rising_text}. Declining or stable regions: {declining_text}.
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Conclusion")

if high_risk_count > 0 or spike_count > 0:
    conclusion = (
        "The selected regions show active outbreak signals. Monitoring should prioritize "
        f"{top_risk_row['region']} and any regions with repeated spike days, while tracking whether "
        "the next 7-day projection continues upward."
    )
else:
    conclusion = (
        "The selected regions are currently in a lower-risk state. Continued monitoring is still useful, "
        "especially if cases begin rising above each region's recent baseline."
    )

st.info(conclusion)

with st.expander("View Raw Data"):
    st.dataframe(
        filtered_df[
            ["date", "region", "cases", "temperature", "humidity", "rainfall", "risk_score", "risk_level", "is_spike"]
        ],
        use_container_width=True,
        hide_index=True,
    )
