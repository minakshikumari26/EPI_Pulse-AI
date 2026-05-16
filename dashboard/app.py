"""EpiPulse AI — Analytics Dashboard (Dynamic, cross-regional risk scoring)."""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import zscore as scipy_zscore

st.set_page_config(page_title="EpiPulse AI — Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "disease_data.csv"

st.markdown("""
<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem}
.app-header{border-bottom:1px solid #d9e2ec;padding-bottom:1rem;margin-bottom:1.2rem}
.app-title{color:#102a43;font-size:2.15rem;font-weight:750;line-height:1.15;margin:0}
.app-subtitle{color:#52606d;font-size:1rem;margin-top:.35rem;max-width:780px}
.metric-card{border:1px solid #d9e2ec;border-radius:8px;padding:1rem;background:#fff;min-height:116px}
.metric-label{color:#627d98;font-size:.78rem;text-transform:uppercase;font-weight:700}
.metric-value{color:#102a43;font-size:1.95rem;line-height:1.1;font-weight:750;margin-top:.35rem}
.metric-helper{color:#52606d;font-size:.86rem;margin-top:.4rem}
.report-panel{border-left:4px solid #0e7490;background:#f7fbfc;padding:1rem 1.1rem;border-radius:6px;color:#243b53}
.risk-badge{display:inline-block;border-radius:999px;padding:.15rem .55rem;font-size:.8rem;font-weight:700}
.risk-high{color:#7f1d1d;background:#fee2e2}
.risk-medium{color:#7c2d12;background:#ffedd5}
.risk-low{color:#14532d;background:#dcfce7}
.filter-badge{background:#e0f2fe;color:#0369a1;border-radius:6px;padding:.2rem .6rem;font-size:.82rem;font-weight:600}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return _enrich(df)


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["date"])
    missing = {"date","region","cases"} - set(df.columns)
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()
    return _enrich(df)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().sort_values(["region","date"]).reset_index(drop=True)
    for col in ["temperature","humidity","rainfall"]:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].mean())
    if "disease" not in df.columns: df["disease"] = "Unknown"
    if "year"    not in df.columns: df["year"]    = df["date"].dt.year
    if "month"   not in df.columns: df["month"]   = df["date"].dt.month
    df["cases"] = df["cases"].fillna(0).astype(int)

    # ── Z-score: computed WITHIN each region+disease group ────────────────────
    # This captures "unusual for THIS region" — outbreak detection
    df["z_score"] = df.groupby(["region","disease"])["cases"].transform(
        lambda v: scipy_zscore(v) if len(v) > 1 else 0
    )
    df["is_spike"] = df["z_score"] > 1.5

    # ── Risk score: CROSS-REGIONAL (not normalized per region) ────────────────
    # Use global min/max so high-case regions score higher than low-case regions
    for col in ["cases","humidity","rainfall"]:
        g_min = df[col].min()
        g_max = df[col].max()
        denom = g_max - g_min if g_max > g_min else 1
        df[f"{col}_scaled"] = (df[col] - g_min) / denom

    df["risk_score"] = (
        df["cases_scaled"]   * 60 +
        df["humidity_scaled"] * 25 +
        df["rainfall_scaled"] * 15
    ).round(2)

    df["risk_level"] = df["risk_score"].apply(
        lambda s: "High" if s >= 0.55 else "Medium" if s >= 0.25 else "Low"
    )
    return df


def metric_card(label, value, helper):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-helper">{helper}</div></div>',
        unsafe_allow_html=True)


def risk_badge(level):
    return f'<span class="risk-badge risk-{level.lower()}">{level}</span>'


def get_daily_change(grp):
    grp = grp.sort_values("date")
    if len(grp) < 8: return 0.0
    recent = grp.tail(7)["cases"].mean()
    prev   = grp.iloc[-14:-7]["cases"].mean() if len(grp) >= 14 else grp.iloc[:-7]["cases"].mean()
    return 0.0 if pd.isna(prev) else (recent - prev) / 7


def build_projection(df, periods=7):
    rows = []
    for (region, disease), grp in df.groupby(["region","disease"]):
        grp = grp.sort_values("date")
        last_date  = grp["date"].max()
        last_cases = float(grp.iloc[-1]["cases"])
        delta      = get_daily_change(grp)
        for day in range(1, periods+1):
            rows.append({"region":region,"disease":disease,
                         "date":last_date+pd.Timedelta(days=day),
                         "projected_cases":max(0,round(last_cases+delta*day,1))})
    return pd.DataFrame(rows)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
<h1 class="app-title">🩺 EpiPulse AI</h1>
<div class="app-subtitle">Dynamic outbreak monitoring — filter by year, disease, state, or upload your own data.</div>
</div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📂 Data Source")
src = st.sidebar.radio("Source", ["📊 Built-in Dataset","📁 Upload CSV"], label_visibility="collapsed")

if src == "📁 Upload CSV":
    up = st.sidebar.file_uploader("CSV: date, region, cases", type=["csv"],
                                   help="Optional: temperature, humidity, rainfall, disease")
    df = load_uploaded(up) if up else (st.sidebar.info("Using built-in data."), load_base_data())[1]
else:
    df = load_base_data()

st.sidebar.markdown("---\n## 🔍 Filters")

diseases = sorted(df["disease"].unique())
multi_d  = len(diseases) > 1
selected_diseases = (
    st.sidebar.multiselect("🦠 Disease", diseases, default=diseases)
    if multi_d else diseases
)
if not selected_diseases: selected_diseases = diseases

years = sorted(df["year"].unique(), reverse=True)
selected_years = st.sidebar.multiselect("📅 Year", years, default=years)
if not selected_years: selected_years = years

region_search   = st.sidebar.text_input("🔎 Search region", placeholder="e.g. Delhi")
all_regions     = sorted(df["region"].unique())
default_regions = ([r for r in all_regions if region_search.lower() in r.lower()]
                   if region_search else all_regions[:4]) or all_regions[:4]
selected_regions = st.sidebar.multiselect("🗺️ Regions", all_regions, default=default_regions)
if not selected_regions: selected_regions = all_regions

ydf      = df[df["year"].isin(selected_years)]
date_min = ydf["date"].min().date()
date_max = ydf["date"].max().date()
sel_dates = st.sidebar.date_input("📆 Date Range", (date_min, date_max),
                                   min_value=date_min, max_value=date_max)
start_date, end_date = sel_dates if len(sel_dates) == 2 else (date_min, date_max)


# ── Filter ─────────────────────────────────────────────────────────────────────
filtered = df[
    df["region"].isin(selected_regions) &
    df["disease"].isin(selected_diseases) &
    df["year"].isin(selected_years) &
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if filtered.empty:
    st.warning("⚠️ No records match filters."); st.stop()

# Filter badges
badges = [f"📅 {min(selected_years)}–{max(selected_years)}",
          f"🗺️ {len(selected_regions)} region(s)"]
if multi_d:
    badges.append(f"🦠 {', '.join(selected_diseases[:3])}{'…' if len(selected_diseases)>3 else ''}")
st.markdown(" &nbsp;".join(f'<span class="filter-badge">{b}</span>' for b in badges),
            unsafe_allow_html=True)
st.write("")


# ── KPIs ───────────────────────────────────────────────────────────────────────
latest_rows = filtered.sort_values("date").groupby(["region","disease"]).tail(1)

# ← ADD THESE
h_thresh = latest_rows["risk_score"].quantile(0.75)
m_thresh = latest_rows["risk_score"].quantile(0.40)
latest_rows["risk_level"] = latest_rows["risk_score"].apply(
    lambda s: "High" if s >= h_thresh else "Medium" if s >= m_thresh else "Low"
)
total_cases     = int(filtered["cases"].sum())
peak_row        = filtered.loc[filtered["cases"].idxmax()]
spike_count     = int(filtered["is_spike"].sum())
high_risk_count = int((latest_rows["risk_level"] == "High").sum())
top_risk_row    = latest_rows.loc[latest_rows["risk_score"].idxmax()]

cols = st.columns(4)
with cols[0]: metric_card("Total Cases",      f"{total_cases:,}",      f"{len(selected_regions)} region(s)")
with cols[1]: metric_card("Peak Daily Cases", int(peak_row["cases"]),  f"{peak_row['region']} · {peak_row['date'].date()}")
with cols[2]: metric_card("Spike Days",       spike_count,             "z-score > 1.5 within region")
with cols[3]: metric_card("High-Risk Combos", high_risk_count,         f"Top: {top_risk_row['region']}")
st.write("")


# ── Risk leaderboard ───────────────────────────────────────────────────────────
st.subheader("🏆 Region Risk Leaderboard (Cross-Regional)")
leaderboard = (latest_rows.groupby("region")["risk_score"].max()
               .reset_index().sort_values("risk_score", ascending=False))
leaderboard["risk_level"] = leaderboard["risk_score"].apply(
    lambda s: "High" if s >= 0.55 else "Medium" if s >= 0.25 else "Low"
)
leaderboard["rank"] = range(1, len(leaderboard)+1)

lb_fig = px.bar(
    leaderboard.sort_values("risk_score"),
    x="risk_score", y="region", color="risk_level", orientation="h",
    color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#16a34a"},
    text=leaderboard.sort_values("risk_score")["risk_score"].round(3),
)
lb_fig.update_layout(height=max(300, len(selected_regions)*28),
                     margin=dict(l=10,r=10,t=10,b=10),
                     xaxis_title="Risk Score (0–1, cross-regional)",
                     yaxis_title=None, legend_title_text="Risk Level")
st.plotly_chart(lb_fig, use_container_width=True)


# ── Trend + Risk charts ────────────────────────────────────────────────────────
chart_col, side_col = st.columns([2.2, 1])

with chart_col:
    st.subheader("📈 Case Trend")
    color_col = ("label" if (multi_d and len(selected_diseases)>1) else "region")
    if color_col == "label":
        filtered["label"] = filtered["region"] + " — " + filtered["disease"]
    trend_fig = px.line(filtered, x="date", y="cases", color=color_col,
                        markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    spikes = filtered[filtered["is_spike"]]
    if not spikes.empty:
        trend_fig.add_trace(go.Scatter(
            x=spikes["date"], y=spikes["cases"], mode="markers",
            marker={"size":9,"color":"#dc2626","symbol":"x"}, name="Spike",
            text=spikes["region"],
            hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Cases:%{y}<extra></extra>"))
    trend_fig.update_layout(height=430, margin=dict(l=10,r=10,t=20,b=10),
                            xaxis_title=None, yaxis_title="Cases")
    st.plotly_chart(trend_fig, use_container_width=True)

with side_col:
    st.subheader("⚠️ Latest Risk Score")
    risk_fig = px.bar(
        latest_rows.sort_values("risk_score", ascending=True),
        x="risk_score", y="region", color="risk_level", orientation="h",
        color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#16a34a"},
        text=latest_rows.sort_values("risk_score")["risk_score"].round(3),
    )
    risk_fig.update_layout(height=430, margin=dict(l=10,r=10,t=20,b=10),
                           xaxis_title="Risk Score", yaxis_title=None)
    st.plotly_chart(risk_fig, use_container_width=True)


# ── Year-over-year ─────────────────────────────────────────────────────────────
if len(selected_years) > 1:
    st.subheader("📊 Year-over-Year Comparison")
    yoy = filtered.groupby(["year","region"])["cases"].sum().reset_index()
    yoy_fig = px.bar(yoy, x="region", y="cases", color="year",
                     barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    yoy_fig.update_layout(height=350, margin=dict(l=10,r=10,t=20,b=10),
                          xaxis_title="Region", yaxis_title="Total Cases")
    st.plotly_chart(yoy_fig, use_container_width=True)


# ── Disease breakdown ──────────────────────────────────────────────────────────
if multi_d and len(selected_diseases) > 1:
    st.subheader("🦠 Cases by Disease")
    d_tot = filtered.groupby("disease")["cases"].sum().reset_index()
    pie   = px.pie(d_tot, names="disease", values="cases",
                   color_discrete_sequence=px.colors.qualitative.Bold)
    pie.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
    p1, p2 = st.columns([1,2])
    with p1: st.plotly_chart(pie, use_container_width=True)
    with p2:
        dr = (filtered.groupby(["disease","region"])["cases"].sum()
              .reset_index().sort_values("cases", ascending=False))
        st.dataframe(dr, use_container_width=True, hide_index=True)


# ── 7-day projection ───────────────────────────────────────────────────────────
st.subheader("🔮 7-Day Case Projection")
proj_df  = build_projection(filtered)
proj_fig = go.Figure()
for (region, disease), grp in filtered.groupby(["region","disease"]):
    history = grp.sort_values("date").tail(21)
    fcast   = proj_df[(proj_df["region"]==region)&(proj_df["disease"]==disease)]
    label   = f"{region} — {disease}" if multi_d else region
    proj_fig.add_trace(go.Scatter(x=history["date"],y=history["cases"],
                                   mode="lines+markers",name=f"{label} (actual)"))
    proj_fig.add_trace(go.Scatter(x=fcast["date"],y=fcast["projected_cases"],
                                   mode="lines+markers",line={"dash":"dash"},
                                   name=f"{label} (forecast)"))
proj_fig.update_layout(height=380,margin=dict(l=10,r=10,t=20,b=10),
                       xaxis_title=None,yaxis_title="Cases")
st.plotly_chart(proj_fig, use_container_width=True)


# ── Summary ────────────────────────────────────────────────────────────────────
st.subheader("📋 Summary Report")
top_regions  = filtered.groupby("region")["cases"].sum().sort_values(ascending=False).head(3)
rising       = [r for r,grp in filtered.groupby("region") if get_daily_change(grp)>0]
declining    = [r for r,grp in filtered.groupby("region") if get_daily_change(grp)<=0]
top_text     = ", ".join(f"{r} ({int(c):,})" for r,c in top_regions.items())
yr_range     = f"{min(selected_years)}–{max(selected_years)}" if len(selected_years)>1 else str(selected_years[0])

st.markdown(f"""
<div class="report-panel">
<strong>Period:</strong> {yr_range} &nbsp;|&nbsp;
<strong>Regions:</strong> {len(selected_regions)} &nbsp;|&nbsp;
<strong>Records:</strong> {len(filtered):,}<br><br>
<strong>Total cases:</strong> {total_cases:,}<br>
<strong>Peak day:</strong> {int(peak_row['cases'])} in <strong>{peak_row['region']}</strong> on {peak_row['date'].date()}<br>
<strong>Spike days:</strong> {spike_count}<br>
<strong>Highest burden:</strong> {top_text}<br>
<strong>Top risk:</strong> {top_risk_row['region']} — score {top_risk_row['risk_score']:.3f} {risk_badge(top_risk_row['risk_level'])}<br><br>
<strong>Rising:</strong> {', '.join(rising) or 'none'} &nbsp;|&nbsp;
<strong>Declining:</strong> {', '.join(declining) or 'none'}
</div>""", unsafe_allow_html=True)

st.subheader("🎯 Conclusion")
if high_risk_count > 0 or spike_count > 0:
    st.warning(
        f"Active outbreak signals detected. Prioritize **{top_risk_row['region']}** "
        f"and regions with repeated spike days. Monitor the 7-day projection closely.")
else:
    st.success("All selected regions are currently in a lower-risk state. Continue routine monitoring.")

with st.expander("🔍 View Raw Data"):
    show_cols = [c for c in ["date","region","disease","cases","temperature",
                              "humidity","rainfall","risk_score","risk_level","is_spike"]
                 if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered CSV",
                       filtered[show_cols].to_csv(index=False).encode(),
                       "filtered_data.csv","text/csv")
