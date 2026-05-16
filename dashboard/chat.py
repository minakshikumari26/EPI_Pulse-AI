"""EpiPulse AI — Chat Interface (Fixed: cross-regional risk, rich LLM context)."""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import zscore as scipy_zscore

import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import get_llm_client
from src.utils.config import load_config

DATA_PATH = PROJECT_ROOT / "data" / "disease_data.csv"


def inject_css():
    st.markdown("""
    <style>
    .main-header{background:linear-gradient(135deg,#1e3a8a 0%,#0ea5e9 100%);
        color:white;padding:2rem;border-radius:10px;margin-bottom:2rem;
        box-shadow:0 4px 6px rgba(0,0,0,.1)}
    .main-header h1{margin:0;font-size:2.5rem;font-weight:800}
    .main-header p{margin:.5rem 0 0 0;font-size:1.1rem;opacity:.9}
    .status-card{background:white;border-left:4px solid #0ea5e9;
        padding:1.5rem;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.05);
        margin-bottom:1rem}
    .status-card.success{border-left-color:#10b981}
    .status-card.warning{border-left-color:#f59e0b}
    .status-card.danger {border-left-color:#ef4444}
    .stButton>button{background:linear-gradient(135deg,#1e3a8a 0%,#0ea5e9 100%);
        color:white;border:none;border-radius:6px;
        padding:.5rem 1.5rem;font-weight:600;transition:all .3s ease}
    .context-pill{background:#f0f9ff;border:1px solid #bae6fd;border-radius:20px;
        padding:.3rem .9rem;font-size:.85rem;color:#0369a1;font-weight:600;
        display:inline-block;margin:.2rem}
    .risk-high  {color:#991b1b;font-weight:700}
    .risk-medium{color:#92400e;font-weight:700}
    .risk-low   {color:#14532d;font-weight:700}
    </style>""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return _enrich(df)


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["date"])
    missing = {"date","region","cases"} - set(df.columns)
    if missing:
        st.error(f"❌ Missing columns: {missing}"); st.stop()
    return _enrich(df)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().sort_values(["region","date"]).reset_index(drop=True)
    for col in ["temperature","humidity","rainfall"]:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(df[col].mean())
    if "disease" not in df.columns: df["disease"] = "Unknown"
    if "year"    not in df.columns: df["year"]    = df["date"].dt.year
    df["cases"] = df["cases"].fillna(0).astype(int)

    # Z-score within region+disease (anomaly vs own baseline)
    df["z_score"] = df.groupby(["region","disease"])["cases"].transform(
        lambda v: scipy_zscore(v) if len(v) > 1 else 0
    )
    df["is_spike"] = df["z_score"] > 1.5

    # Risk score: CROSS-REGIONAL using global min/max
    for col in ["cases","humidity","rainfall"]:
        g_min, g_max = df[col].min(), df[col].max()
        denom = g_max - g_min if g_max > g_min else 1
        df[f"{col}_scaled"] = (df[col] - g_min) / denom

    df["risk_score"] = (
        df["cases_scaled"] * 60 +
        df["humidity_scaled"] * 25 +
        df["rainfall_scaled"] * 15
    ).round(3)
    df["risk_level"] = df["risk_score"].apply(
        lambda s: "High" if s >= 0.55 else "Medium" if s >= 0.25 else "Low"
    )
    return df


def get_rich_context(df: pd.DataFrame, region: str, disease: str, year) -> dict:
    """Build rich, specific context for the LLM — actual numbers, comparisons, trends."""
    mask = (df["region"] == region) & (df["disease"] == disease)
    if year != "All":
        mask = mask & (df["year"] == int(year))
    rdf = df[mask].sort_values("date")

    if rdf.empty:
        return {"region": region, "disease": disease, "error": "No data for this selection"}

    latest      = rdf.iloc[-1]
    last_7      = rdf.tail(7)
    last_30     = rdf.tail(30)
    prev_7      = rdf.iloc[-14:-7] if len(rdf) >= 14 else rdf.iloc[:-7]

    avg_7d      = float(last_7["cases"].mean())
    avg_prev_7d = float(prev_7["cases"].mean()) if not prev_7.empty else avg_7d
    growth_rate = (avg_7d - avg_prev_7d) / avg_prev_7d if avg_prev_7d > 0 else 0
    trend_dir   = "📈 Rising" if growth_rate > 0.05 else "📉 Falling" if growth_rate < -0.05 else "➡️ Stable"

    # Cross-regional comparison
    all_latest = df.groupby(["region","disease"]).last().reset_index()
    dis_latest = all_latest[all_latest["disease"] == disease].sort_values("cases", ascending=False)
    rank        = int((dis_latest["region"] == region).argmax()) + 1
    total_r     = len(dis_latest)

    # Global avg for this disease
    global_avg = float(dis_latest["cases"].mean())
    vs_global  = ((float(latest["cases"]) - global_avg) / global_avg * 100) if global_avg > 0 else 0

    return {
        "region":             region,
        "disease":            disease,
        "year_filter":        str(year),
        "latest_date":        latest["date"].strftime("%Y-%m-%d"),
        "latest_cases":       int(latest["cases"]),
        "latest_risk_score":  float(latest["risk_score"]),
        "latest_risk_level":  latest["risk_level"],
        "latest_z_score":     round(float(latest["z_score"]), 2),
        "is_spike_today":     bool(latest["is_spike"]),
        "avg_7d_cases":       round(avg_7d, 1),
        "avg_prev_7d_cases":  round(avg_prev_7d, 1),
        "growth_rate_pct":    round(growth_rate * 100, 1),
        "trend_direction":    trend_dir,
        "peak_cases_period":  int(rdf["cases"].max()),
        "total_cases_period": int(rdf["cases"].sum()),
        "spike_days":         int(rdf["is_spike"].sum()),
        "avg_temperature":    round(float(rdf["temperature"].mean()), 1),
        "avg_humidity":       round(float(rdf["humidity"].mean()), 1),
        "rank_among_regions": f"{rank} of {total_r} (1=highest burden)",
        "vs_national_avg_pct":round(vs_global, 1),
        "national_avg_cases": round(global_avg, 1),
        "top_3_regions":      dis_latest.head(3)["region"].tolist(),
        "bottom_3_regions":   dis_latest.tail(3)["region"].tolist(),
    }


def plot_trend(df, region, disease):
    rdf = df[(df["region"]==region)&(df["disease"]==disease)].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rdf["date"], y=rdf["cases"],
                              mode="lines+markers", name="Cases",
                              line=dict(color="#0ea5e9",width=3),
                              fill="tozeroy", fillcolor="rgba(14,165,233,.1)"))
    spikes = rdf[rdf["is_spike"]]
    if not spikes.empty:
        fig.add_trace(go.Scatter(x=spikes["date"], y=spikes["cases"],
                                  mode="markers", name="Spike",
                                  marker=dict(size=10,color="#ef4444",symbol="x")))
    fig.update_layout(title=f"{region} — {disease}",
                      xaxis_title="Date", yaxis_title="Cases",
                      template="plotly_white", height=260,
                      margin=dict(l=0,r=0,t=30,b=0))
    return fig


def risk_css_class(level):
    return {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(level,"risk-low")


def main():
    st.set_page_config(page_title="EpiPulse AI Chat", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()

    st.markdown("""
    <div class="main-header">
        <h1>🩺 EpiPulse AI</h1>
        <p>AI-Powered Disease Outbreak Intelligence & Analysis</p>
    </div>""", unsafe_allow_html=True)

    # ── LLM ───────────────────────────────────────────────────────────────────
    try:
        config = load_config()
        if not config.get("llm", {}).get("enabled", False):
            st.error("⚠️ LLM disabled — set `enabled: true` in configs/config.yaml"); return
        llm            = get_llm_client()
        provider_label = "Groq ☁️" if "groq" in str(type(llm)).lower() else "Ollama 🖥️"
        conn_status    = f"✅ Connected to **{llm.model}** on `{llm.base_url}`"
        avail_models   = getattr(llm, "available_models", [])
    except (ConnectionError, ValueError) as e:
        err = str(e)
        if "GROQ_API_KEY" in err or "groq" in err.lower():
            st.error("❌ **Groq API key missing.**\n\n"
                     "1. https://console.groq.com → free key\n"
                     "2. Add `GROQ_API_KEY=gsk_...` to `.env`\n3. Restart")
        else:
            st.error(f"❌ Cannot connect: {err}")
        return
    except Exception as e:
        st.error(f"❌ LLM init error: `{e}`"); return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Data source
        st.markdown("## 📂 Data Source")
        src = st.radio("src", ["📊 Built-in","📁 Upload CSV"], label_visibility="collapsed")
        if src == "📁 Upload CSV":
            up = st.file_uploader("CSV: date, region, cases", type=["csv"])
            df = load_uploaded(up) if up else (st.info("Using built-in."), load_data())[1]
        else:
            df = load_data()

        st.markdown("---\n## 🔍 Filters")

        diseases     = sorted(df["disease"].unique())
        multi_d      = len(diseases) > 1
        sel_disease  = st.selectbox("🦠 Disease", diseases) if multi_d else diseases[0]

        years        = sorted(df["year"].unique(), reverse=True)
        sel_year     = st.selectbox("📅 Year", ["All"] + [str(y) for y in years])

        regions      = sorted(df["region"].unique())
        sel_region   = st.selectbox("🗺️ Region", regions)

        # Apply year filter for display
        disp_df = df.copy()
        if sel_year != "All":
            disp_df = disp_df[disp_df["year"] == int(sel_year)]

        st.markdown("---")

        # LLM status
        st.markdown(f"""
        <div class="status-card success">
            <div style="font-weight:600;margin-bottom:.5rem">🔌 LLM</div>
            <div>{conn_status}</div>
            <div style="margin-top:.3rem;font-size:.83rem;color:#6b7280">Provider: {provider_label}</div>
        </div>""", unsafe_allow_html=True)
        if avail_models:
            st.caption("Models: " + ", ".join(avail_models[:4]))
        if st.button("🔄 Reconnect LLM"):
            try:
                from src.llm import get_llm_client as _g
                llm = _g(force_new=True); st.success("✅ Reconnected!"); st.rerun()
            except Exception as _e:
                st.error(f"Failed: {_e}")

        # Regional snapshot
        st.markdown("---")
        ctx = get_rich_context(disp_df, sel_region, sel_disease, sel_year)

        if "error" not in ctx:
            risk_cls = risk_css_class(ctx["latest_risk_level"])
            st.markdown(f"#### {sel_region} · {sel_disease}")
            st.markdown(f"""
            <div class="status-card {'danger' if ctx['latest_risk_level']=='High' else 'warning' if ctx['latest_risk_level']=='Medium' else 'success'}">
                <div class="{risk_cls}" style="font-size:1.2rem">
                    {'🔴 HIGH RISK' if ctx['latest_risk_level']=='High' else '🟠 MEDIUM RISK' if ctx['latest_risk_level']=='Medium' else '🟢 LOW RISK'}
                </div>
                <div style="font-size:.85rem;margin-top:.4rem;color:#374151">
                    {ctx['trend_direction']} · {ctx['growth_rate_pct']:+.1f}% vs prev week
                </div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Cases Today", f"{ctx['latest_cases']:,}")
                st.metric("7d Avg",      f"{ctx['avg_7d_cases']:.0f}")
            with c2:
                st.metric("Z-Score",     f"{ctx['latest_z_score']:.2f}")
                st.metric("Rank",        ctx['rank_among_regions'])

            if ctx.get("vs_national_avg_pct") is not None:
                delta_color = "inverse" if ctx["vs_national_avg_pct"] > 0 else "normal"
                st.metric("vs National Avg",
                          f"{ctx['vs_national_avg_pct']:+.1f}%",
                          delta=f"Nat avg: {ctx['national_avg_cases']:.0f}",
                          delta_color=delta_color)

            if ctx["is_spike_today"]:
                st.error("🚨 Active outbreak spike detected!")
            if ctx["spike_days"] > 0:
                st.warning(f"⚠️ {ctx['spike_days']} spike days in selected period")

            st.markdown("---")
            st.plotly_chart(plot_trend(disp_df, sel_region, sel_disease),
                            use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analysis", "⚙️ Tools"])

    # ── TAB 1: CHAT ───────────────────────────────────────────────────────────
    with tab1:
        yr_note = f" ({sel_year})" if sel_year != "All" else " (all years)"

        # Context pills
        ctx_pills = [f"📍 {sel_region}", f"🦠 {sel_disease}", f"📅 {sel_year}"]
        if "error" not in ctx:
            ctx_pills.append(f"📊 {ctx['latest_cases']} cases today")
            ctx_pills.append(f"{'🔴' if ctx['latest_risk_level']=='High' else '🟠' if ctx['latest_risk_level']=='Medium' else '🟢'} {ctx['latest_risk_level']} risk")
            ctx_pills.append(ctx["trend_direction"])

        st.markdown(
            " ".join(f'<span class="context-pill">{p}</span>' for p in ctx_pills),
            unsafe_allow_html=True
        )
        st.markdown("")

        if "messages" not in st.session_state:
            region_list  = ", ".join(regions[:6]) + ("…" if len(regions) > 6 else "")
            disease_list = ", ".join(diseases)
            yr_list      = f"{min(years)}–{max(years)}"

            if "error" not in ctx:
                intro = (
                    f"👋 Hello! I'm your AI epidemiologist.\n\n"
                    f"**Currently analysing:** {sel_region} → {sel_disease}{yr_note}\n\n"
                    f"Here's what I see right now:\n"
                    f"- **{ctx['latest_cases']:,} cases** as of {ctx['latest_date']}\n"
                    f"- **{ctx['latest_risk_level']} risk** (score: {ctx['latest_risk_score']:.3f})\n"
                    f"- **{ctx['trend_direction']}** — {ctx['growth_rate_pct']:+.1f}% vs last week\n"
                    f"- Ranked **{ctx['rank_among_regions']}** for {sel_disease}\n"
                    f"- **{ctx['spike_days']} spike days** in selected period\n\n"
                    f"**Data covers:** {region_list} · {disease_list} · {yr_list}\n\n"
                    f"Ask me anything — cases, risk levels, comparisons, trends or intervention strategies!"
                )
            else:
                intro = (
                    f"👋 Hello! I'm your AI epidemiologist.\n\n"
                    f"Available regions: {region_list}\n"
                    f"Diseases: {disease_list} · Years: {yr_list}\n\n"
                    f"Ask me anything about the outbreak data!"
                )
            st.session_state.messages = [{"role": "assistant", "content": intro}]

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask your AI epidemiologist...")
        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
            try:
                full_ctx = get_rich_context(disp_df, sel_region, sel_disease, sel_year)
                full_ctx["available_regions"]  = regions
                full_ctx["available_diseases"] = diseases
                full_ctx["available_years"]    = years

                with st.spinner("🤔 Analysing..."):
                    response = llm.answer_question(user_input, context=full_ctx)
                st.session_state.messages.append({"role":"assistant","content":response})
            except ConnectionError as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ LLM not reachable: `{e}`"})
            except RuntimeError as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ Generation failed: `{e}`"})
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ Error: `{e}`"})
            st.rerun()

    # ── TAB 2: ANALYSIS ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 Regional Analysis Dashboard")
        fdf = disp_df[disp_df["disease"] == sel_disease]

        # Cross-regional risk leaderboard
        st.markdown("#### 🏆 Risk Leaderboard (Cross-Regional)")
        latest_all = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1]).reset_index(drop=True)
        latest_all = latest_all.sort_values("risk_score", ascending=False)

        # ← ADD THESE 3 LINES — recompute risk levels on this snapshot
        h_thresh = latest_all["risk_score"].quantile(0.75)
        m_thresh = latest_all["risk_score"].quantile(0.40)
        latest_all["risk_level"] = latest_all["risk_score"].apply(
            lambda s: "High" if s >= h_thresh else "Medium" if s >= m_thresh else "Low"
        )
        

        lb = px.bar(latest_all.sort_values("risk_score"),
                    x="risk_score", y="region", color="risk_level",
                    orientation="h",
                    color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#16a34a"},
                    text=latest_all.sort_values("risk_score")["risk_score"].round(3))
        lb.update_layout(height=max(300,len(regions)*26),
                         margin=dict(l=10,r=10,t=10,b=10),
                         xaxis_title="Risk Score (0–1 cross-regional)",
                         yaxis_title=None)
        st.plotly_chart(lb, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Risk Distribution")
            high   = int((latest_all["risk_level"]=="High").sum())
            medium = int((latest_all["risk_level"]=="Medium").sum())
            low    = int((latest_all["risk_level"]=="Low").sum())
            pie = go.Figure(data=[go.Pie(
                labels=["🔴 High","🟠 Medium","🟢 Low"],
                values=[high, medium, low],
                marker=dict(colors=["#ef4444","#f97316","#16a34a"]))])
            pie.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(pie, use_container_width=True)
        with c2:
            st.markdown("#### Cases by Region")
            bar = go.Figure(data=[go.Bar(
                y=latest_all["region"], x=latest_all["cases"],
                orientation="h",
                marker=dict(color=latest_all["risk_score"],
                            colorscale="RdYlGn_r", showscale=True))])
            bar.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                              xaxis_title="Cases")
            st.plotly_chart(bar, use_container_width=True)

        # Year-over-year for selected region
        if len(years) > 1 and sel_year == "All":
            st.markdown(f"---\n#### 📅 Year-over-Year — {sel_region} — {sel_disease}")
            yoy = df[(df["region"]==sel_region)&(df["disease"]==sel_disease)].groupby("year")["cases"].agg(["sum","max","mean"]).reset_index()
            yoy.columns = ["Year","Total","Peak","Avg Daily"]
            yoy_fig = px.bar(yoy, x="Year", y="Total",
                             color="Total", color_continuous_scale="RdYlGn_r",
                             text="Total")
            yoy_fig.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(yoy_fig, use_container_width=True)
            st.dataframe(yoy.style.format({"Total":"{:,.0f}","Peak":"{:,.0f}","Avg Daily":"{:.1f}"}),
                         use_container_width=True, hide_index=True)

        # Full region status table
        st.markdown("---\n#### 📋 All Regions — Current Status")
        table = latest_all[["region","cases","z_score","risk_score","risk_level","is_spike"]].copy()
        table["z_score"]    = table["z_score"].round(2)
        table["risk_score"] = table["risk_score"].round(3)
        table["Spike"]      = table["is_spike"].map({True:"🔴 Yes",False:"🟢 No"})
        st.dataframe(
            table[["region","cases","z_score","risk_score","risk_level","Spike"]]
            .sort_values("risk_score", ascending=False),
            use_container_width=True, hide_index=True)

    # ── TAB 3: TOOLS ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### ⚙️ Quick Analysis Tools")
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📊 Situation Report", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        fdf   = disp_df[disp_df["disease"]==sel_disease]
                        lates = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                        ctx_r = {
                            "disease": sel_disease, "year": sel_year,
                            "total_regions": len(regions),
                            "high_risk_regions": lates[lates["risk_level"]=="High"]["region"].tolist(),
                            "medium_risk_regions": lates[lates["risk_level"]=="Medium"]["region"].tolist(),
                            "total_cases_today": int(lates["cases"].sum()),
                            "national_avg_cases": round(float(lates["cases"].mean()),1),
                            "max_cases_region": lates.loc[lates["cases"].idxmax()]["region"],
                            "max_cases_value": int(lates["cases"].max()),
                        }
                        resp = llm.generate_report(ctx_r)
                        st.markdown(resp)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c2:
            if st.button("⚠️ High-Risk Alert", use_container_width=True):
                with st.spinner("Scanning..."):
                    try:
                        fdf   = disp_df[disp_df["disease"]==sel_disease]
                        lates = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                        high  = lates[lates["risk_level"]=="High"][["region","cases","risk_score","z_score"]]
                        if not high.empty:
                            st.error(f"🔴 **{len(high)} High-Risk Region(s) detected!**")
                            for _, row in high.iterrows():
                                st.markdown(f"""
                                **{row['region']}** — {int(row['cases'])} cases
                                · Risk: {row['risk_score']:.3f} · Z-score: {row['z_score']:.2f}
                                """)
                            resp = llm.answer_question(
                                "These regions are HIGH RISK right now. Give urgent public health recommendations for each.",
                                context={"high_risk_regions": high.to_dict("records"), "disease": sel_disease})
                            st.markdown(resp)
                        else:
                            st.success("✅ No high-risk regions detected.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c3:
            if st.button("🔄 Compare Regions", use_container_width=True):
                with st.spinner("Comparing..."):
                    try:
                        fdf   = disp_df[disp_df["disease"]==sel_disease]
                        lates = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                        comp  = {
                            "disease":  sel_disease,
                            "year":     sel_year,
                            "regions":  lates[["region","cases","risk_score","risk_level","z_score"]].to_dict("records"),
                            "highest_burden": lates.loc[lates["cases"].idxmax()]["region"],
                            "lowest_burden":  lates.loc[lates["cases"].idxmin()]["region"],
                            "national_avg":   round(float(lates["cases"].mean()),1),
                        }
                        resp = llm.answer_question(
                            "Compare all regions. Which are most/least affected? "
                            "What patterns do you see? Give specific numbers.",
                            context=comp)
                        st.markdown(resp)
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---\n### 🎯 Custom Analysis")
        opts = {
            "Intervention Recommendations": "Based on current risk data, what specific interventions do you recommend for each high-risk region? Be concrete and region-specific.",
            "Trend Analysis":               "Analyse trends in detail. Which regions are worsening fastest? Which are improving? Use the numbers provided.",
            "Root Cause Analysis":          "Based on climate data (temperature, humidity, rainfall) and case counts, what environmental factors may be driving outbreaks in the highest-risk regions?",
            "Year-over-Year Analysis":      "Compare disease burden across years. Is the situation improving or worsening? Which years had the worst outbreaks?",
            "Resource Allocation Advice":   "If you had to prioritise limited health resources across these regions, which 3 regions need immediate attention and why?",
        }
        analysis_type = st.selectbox("Select analysis:", list(opts.keys()))
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    fdf   = disp_df[disp_df["disease"]==sel_disease]
                    lates = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                    run_ctx = {
                        "disease":          sel_disease,
                        "year":             sel_year,
                        "regions_data":     lates[["region","cases","risk_score","risk_level","z_score","temperature","humidity","rainfall"]].to_dict("records"),
                        "national_avg":     round(float(lates["cases"].mean()),1),
                        "high_risk_count":  int((lates["risk_level"]=="High").sum()),
                        "total_cases_today":int(lates["cases"].sum()),
                    }
                    resp = llm.answer_question(opts[analysis_type], context=run_ctx)
                    st.markdown(resp)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("""
        <div style="text-align:center;color:#9ca3af;font-size:.85rem;padding:2rem 0">
            🩺 <strong>EpiPulse AI v2.1</strong> · Powered by Groq + Llama
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
