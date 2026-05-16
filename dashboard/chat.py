"""EpiPulse AI — RAG-Powered Chat Interface."""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import zscore as scipy_zscore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import get_llm_client
from src.utils.config import load_config

DATA_PATH = PROJECT_ROOT / "data" / "disease_data.csv"


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    .main-header{background:linear-gradient(135deg,#1e3a8a 0%,#0ea5e9 100%);
        color:white;padding:2rem;border-radius:10px;margin-bottom:2rem}
    .main-header h1{margin:0;font-size:2.3rem;font-weight:800}
    .main-header p{margin:.4rem 0 0;font-size:1rem;opacity:.9}
    .status-card{background:white;border-left:4px solid #0ea5e9;
        padding:1rem 1.2rem;border-radius:8px;margin-bottom:.8rem}
    .status-card.success{border-left-color:#10b981}
    .status-card.warning{border-left-color:#f59e0b}
    .status-card.danger {border-left-color:#ef4444}
    .rag-badge{background:#f0fdf4;border:1px solid #86efac;border-radius:6px;
        padding:.2rem .6rem;font-size:.8rem;color:#166534;font-weight:600;display:inline-block}
    .citation-box{background:#f8fafc;border-left:3px solid #0891b2;
        padding:.6rem .9rem;border-radius:4px;font-size:.82rem;color:#334155;margin-top:.5rem}
    .context-pill{background:#f0f9ff;border:1px solid #bae6fd;border-radius:20px;
        padding:.2rem .7rem;font-size:.82rem;color:#0369a1;font-weight:600;
        display:inline-block;margin:.15rem}
    .stButton>button{background:linear-gradient(135deg,#1e3a8a 0%,#0ea5e9 100%);
        color:white;border:none;border-radius:6px;padding:.45rem 1.3rem;font-weight:600}
    .kb-stat{background:#eff6ff;border-radius:8px;padding:.7rem 1rem;
        border:1px solid #bfdbfe;font-size:.88rem;color:#1e40af}
    </style>""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...", ttl=300)
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH)
        return _enrich(df)
    except FileNotFoundError:
        st.error("❌ Run `python scripts/generate_dataset.py` first."); st.stop()


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = {"date","region","cases"} - set(df.columns)
    if missing:
        st.error(f"❌ Missing columns: {missing}"); st.stop()
    return _enrich(df)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates()
    if df.empty:
        st.error("No valid dates found in the dataset.")
        st.stop()
    df = df.sort_values(["region","date"]).reset_index(drop=True)
    for col in ["temperature","humidity","rainfall"]:
        if col not in df.columns: df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "disease" not in df.columns: df["disease"] = "Unknown"
    if "year"    not in df.columns: df["year"]    = df["date"].dt.year
    df["cases"]   = pd.to_numeric(df["cases"], errors="coerce").fillna(0).astype(int)
    df["z_score"] = df.groupby(["region","disease"])["cases"].transform(
        lambda v: scipy_zscore(v) if len(v) > 1 else 0)
    df["is_spike"] = df["z_score"] > 1.5
    for col in ["cases","humidity","rainfall"]:
        g_min, g_max = df[col].min(), df[col].max()
        df[f"{col}_scaled"] = (df[col] - g_min) / (g_max - g_min if g_max > g_min else 1)
    df["risk_score"] = (df["cases_scaled"]*60 + df["humidity_scaled"]*25 + df["rainfall_scaled"]*15).round(4)
    h_t = df["risk_score"].quantile(0.75)
    m_t = df["risk_score"].quantile(0.40)
    df["risk_level"] = df["risk_score"].apply(
        lambda s: "High" if s >= h_t else "Medium" if s >= m_t else "Low")
    return df


def get_rich_context(df, region, disease, year, conversation_history=None):
    mask = (df["region"]==region) & (df["disease"]==disease)
    if year != "All": mask = mask & (df["year"]==int(year))
    rdf = df[mask].sort_values("date")
    if rdf.empty:
        return {"region":region,"disease":disease,"note":"No data"}
    latest  = rdf.iloc[-1]
    last7   = rdf.tail(7)
    prev7   = rdf.iloc[-14:-7] if len(rdf)>=14 else rdf.head(7)
    avg7    = float(last7["cases"].mean())
    avgp7   = float(prev7["cases"].mean()) if not prev7.empty else avg7
    growth  = (avg7-avgp7)/avgp7 if avgp7>0 else 0
    all_lat = df.groupby(["region","disease"]).last().reset_index()
    dis_lat = all_lat[all_lat["disease"]==disease].sort_values("cases",ascending=False)
    rank    = int(dis_lat[dis_lat["region"]==region].index[0] - dis_lat.index[0]+1) if region in dis_lat["region"].values else "N/A"
    g_avg   = float(dis_lat["cases"].mean())
    vs_g    = ((float(latest["cases"])-g_avg)/g_avg*100) if g_avg>0 else 0
    ctx = {
        "region":region,"disease":disease,"year_filter":str(year),
        "latest_date":latest["date"].strftime("%Y-%m-%d"),
        "latest_cases":int(latest["cases"]),"risk_score":round(float(latest["risk_score"]),4),
        "risk_level":latest["risk_level"],"z_score":round(float(latest["z_score"]),2),
        "is_spike":bool(latest["is_spike"]),
        "last_7_days_cases":last7["cases"].tolist(),
        "last_7_days_dates":last7["date"].dt.strftime("%Y-%m-%d").tolist(),
        "avg_7d":round(avg7,1),"avg_prev_7d":round(avgp7,1),
        "growth_rate_pct":round(growth*100,1),
        "trend":"Rising" if growth>0.05 else "Falling" if growth<-0.05 else "Stable",
        "peak_cases":int(rdf["cases"].max()),"total_cases":int(rdf["cases"].sum()),
        "spike_days":int(rdf["is_spike"].sum()),
        "avg_temp":round(float(rdf["temperature"].mean()),1),
        "avg_humidity":round(float(rdf["humidity"].mean()),1),
        "rank_of_total":f"{rank} of {len(dis_lat)}",
        "vs_national_avg_pct":round(vs_g,1),"national_avg_cases":round(g_avg,1),
        "top_3_regions":dis_lat.head(3)["region"].tolist(),
        "available_regions":sorted(df["region"].unique().tolist()),
        "available_diseases":sorted(df["disease"].unique().tolist()),
    }
    if conversation_history:
        ctx["conversation_history"] = [
            {"role":m["role"],"content":m["content"][:250]}
            for m in conversation_history[-6:]
        ]
    return ctx


def plot_trend(df, region, disease):
    rdf = df[(df["region"]==region)&(df["disease"]==disease)].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rdf["date"],y=rdf["cases"],mode="lines+markers",
        line=dict(color="#0ea5e9",width=2),fill="tozeroy",
        fillcolor="rgba(14,165,233,.08)",name="Cases"))
    spikes = rdf[rdf["is_spike"]]
    if not spikes.empty:
        fig.add_trace(go.Scatter(x=spikes["date"],y=spikes["cases"],mode="markers",
            marker=dict(size=9,color="#ef4444",symbol="x"),name="Spike"))
    fig.update_layout(height=240,margin=dict(l=0,r=0,t=20,b=0),
        title=f"{region} — {disease}",template="plotly_white",
        xaxis_title=None,yaxis_title="Cases")
    return fig


# ── RAG init (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG knowledge base...")
def init_rag():
    """Initialize RAG knowledge base and retriever (cached across sessions)."""
    try:
        from src.rag.knowledge_base import get_knowledge_base
        from src.rag.retriever import RAGRetriever
        kb  = get_knowledge_base(auto_seed=True)
        rag = RAGRetriever(knowledge_base=kb)
        return rag, None
    except Exception as e:
        return None, str(e)


def display_citations(chunks: list[dict]):
    """Render retrieved source citations in an expander."""
    if not chunks:
        return
    with st.expander(f"📚 Sources ({len(chunks)} retrieved)", expanded=False):
        for i, c in enumerate(chunks):
            st.markdown(f"""
<div class="citation-box">
<strong>[{i+1}] {c['source']}</strong>
<span style="color:#64748b;font-size:.78rem"> — relevance: {c['score']:.0%}</span><br>
<em style="font-size:.82rem">{c['text'][:280]}{'...' if len(c['text'])>280 else ''}</em>
</div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="EpiPulse AI — RAG Chat",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.markdown("""
    <div class="main-header">
        <h1>🩺 EpiPulse AI</h1>
        <p>RAG-Powered Disease Outbreak Intelligence — grounded in WHO guidelines & IDSP protocols</p>
    </div>""", unsafe_allow_html=True)

    # ── LLM ──────────────────────────────────────────────────────────────────
    try:
        config = load_config()
        if not config.get("llm",{}).get("enabled",False):
            st.error("⚠️ Set `enabled: true` in configs/config.yaml"); return
        llm            = get_llm_client()
        provider_label = "Groq ☁️" if "groq" in str(type(llm)).lower() else "Ollama 🖥️"
        conn_status    = f"✅ Connected to **{llm.model}**"
    except (ConnectionError,ValueError) as e:
        err = str(e)
        if "GROQ_API_KEY" in err or "groq" in err.lower():
            st.error("❌ Add `GROQ_API_KEY=gsk_...` to your `.env` file")
        else:
            st.error(f"❌ LLM error: {err}")
        return
    except Exception as e:
        st.error(f"❌ LLM init: `{e}`"); return

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag, rag_error = init_rag()
    rag_ready = rag is not None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📂 Data Source")
        src = st.radio("src",["📊 Built-in","📁 Upload CSV"],label_visibility="collapsed")
        if src == "📁 Upload CSV":
            up = st.file_uploader("CSV: date, region, cases",type=["csv"])
            df = load_uploaded(up) if up else (st.info("Using built-in."), load_data())[1]
        else:
            df = load_data()

        st.markdown("---\n## 🔍 Filters")
        diseases    = sorted(df["disease"].unique())
        multi_d     = len(diseases)>1
        sel_disease = st.selectbox("🦠 Disease",diseases) if multi_d else diseases[0]
        years       = sorted(df["year"].unique(),reverse=True)
        sel_year    = st.selectbox("📅 Year",["All"]+[str(y) for y in years])
        regions     = sorted(df["region"].unique())
        sel_region  = st.selectbox("🗺️ Region",regions)

        disp_df = df.copy()
        if sel_year != "All":
            disp_df = disp_df[disp_df["year"]==int(sel_year)]

        st.markdown("---")

        # LLM status
        st.markdown(f"""
        <div class="status-card success">
            <div style="font-weight:600;margin-bottom:.4rem">🔌 LLM — {provider_label}</div>
            <div style="font-size:.9rem">{conn_status}</div>
        </div>""", unsafe_allow_html=True)

        # RAG status
        if rag_ready:
            stats = rag.get_kb_stats()
            st.markdown(f"""
            <div class="status-card success">
                <div style="font-weight:600;margin-bottom:.4rem">🧠 RAG Knowledge Base</div>
                <div class="rag-badge">✅ Active</div>
                <div style="font-size:.85rem;margin-top:.5rem;color:#374151">
                    📄 <strong>{stats['total_chunks']}</strong> chunks indexed<br>
                    📚 <strong>{len(stats['sources'])}</strong> sources
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("📚 View sources"):
                for src_name in stats["sources"]:
                    st.markdown(f"• {src_name}")
        else:
            st.markdown(f"""
            <div class="status-card warning">
                <div style="font-weight:600">⚠️ RAG Unavailable</div>
                <div style="font-size:.85rem">{rag_error}</div>
            </div>""", unsafe_allow_html=True)

        # Upload documents to knowledge base
        if rag_ready:
            st.markdown("---\n## 📤 Add to Knowledge Base")
            doc_upload = st.file_uploader(
                "Upload PDF/TXT to knowledge base",
                type=["pdf","txt","md"],
                help="WHO reports, IDSP bulletins, research papers — any health document",
                key="kb_upload",
            )
            if doc_upload:
                if st.button("📥 Index Document"):
                    with st.spinner("Indexing..."):
                        try:
                            n = rag.kb.add_uploaded_file(
                                doc_upload, doc_upload.name,
                                source=doc_upload.name.replace("_"," ").replace("-"," ").split(".")[0]
                            )
                            st.success(f"✅ Added {n} chunks from '{doc_upload.name}'")
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to index: {e}")

        if st.button("🔄 Reconnect LLM"):
            try:
                from src.llm import get_llm_client as _g
                _g(force_new=True); st.success("✅ Reconnected!"); st.rerun()
            except Exception as _e:
                st.error(f"Failed: {_e}")

        # Regional snapshot
        st.markdown("---")
        ctx = get_rich_context(disp_df, sel_region, sel_disease, sel_year)
        if "note" not in ctx:
            rl  = ctx["risk_level"]
            css = {"High":"danger","Medium":"warning","Low":"success"}.get(rl,"success")
            st.markdown(f"#### {sel_region} · {sel_disease}")
            st.markdown(f"""
            <div class="status-card {css}">
                <div style="font-size:1.1rem;font-weight:700">
                    {'🔴 HIGH' if rl=='High' else '🟠 MEDIUM' if rl=='Medium' else '🟢 LOW'} RISK
                </div>
                <div style="font-size:.83rem;margin-top:.3rem;color:#374151">
                    {ctx['trend']} · {ctx['growth_rate_pct']:+.1f}% vs prev week
                </div>
            </div>""", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: st.metric("Cases",f"{ctx['latest_cases']:,}"); st.metric("Z-Score",f"{ctx['z_score']:.2f}")
            with c2: st.metric("7d Avg",f"{ctx['avg_7d']:.0f}"); st.metric("Rank",ctx["rank_of_total"])
            if ctx["is_spike"]: st.error("🚨 Active spike!")
            st.markdown("---")
            st.plotly_chart(plot_trend(disp_df,sel_region,sel_disease),use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["💬 Chat","📊 Analysis","⚙️ Tools"])

    # ── TAB 1: RAG CHAT ───────────────────────────────────────────────────────
    with tab1:
        # Context pills
        ctx = get_rich_context(disp_df, sel_region, sel_disease, sel_year)
        pills = [f"📍 {sel_region}", f"🦠 {sel_disease}", f"📅 {sel_year}"]
        if "note" not in ctx:
            pills += [
                f"📊 {ctx['latest_cases']} cases",
                f"{'🔴' if ctx['risk_level']=='High' else '🟠' if ctx['risk_level']=='Medium' else '🟢'} {ctx['risk_level']}",
                ctx["trend"],
            ]
        if rag_ready:
            pills.append("🧠 RAG Active")

        st.markdown(
            " ".join(f'<span class="context-pill">{p}</span>' for p in pills),
            unsafe_allow_html=True)
        st.markdown("")

        # Init messages
        if "messages" not in st.session_state:
            kb_note = (f"grounded in **{rag.get_kb_stats()['total_chunks']} chunks** "
                       f"from {len(rag.get_kb_stats()['sources'])} sources "
                       f"(WHO guidelines, IDSP protocols)"
                       if rag_ready else "LLM-only mode (RAG unavailable)")
            intro = (
                f"👋 Hello! I'm your AI epidemiologist — {kb_note}.\n\n"
                f"**Currently watching:** {sel_region} → {sel_disease} ({sel_year})\n\n"
            )
            if "note" not in ctx:
                intro += (
                    f"Here's what I see right now:\n"
                    f"- **{ctx['latest_cases']:,} cases** as of {ctx['latest_date']}\n"
                    f"- **{ctx['risk_level']} risk** (score: {ctx['risk_score']:.3f})\n"
                    f"- **{ctx['trend']}** — {ctx['growth_rate_pct']:+.1f}% vs last week\n"
                    f"- Ranked **{ctx['rank_of_total']}** for {sel_disease}\n\n"
                )
            intro += (
                "I can answer questions about:\n"
                "- Current cases, risk levels, outbreak trends\n"
                "- **WHO-recommended interventions** for this disease\n"
                "- **IDSP outbreak response protocols**\n"
                "- Cross-regional comparisons and forecasts\n\n"
                "_Every answer is grounded in real guidelines — check the 📚 Sources below each reply._"
            )
            st.session_state.messages = [{"role":"assistant","content":intro,"chunks":[]}]

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
                st.markdown(msg["content"])
                if msg.get("chunks"):
                    display_citations(msg["chunks"])

        # Input
        user_input = st.chat_input("Ask your AI epidemiologist...")
        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input,"chunks":[]})
            try:
                full_ctx = get_rich_context(
                    disp_df, sel_region, sel_disease, sel_year,
                    conversation_history=st.session_state.messages[-10:]
                )

                with st.spinner("🔍 Searching knowledge base + generating answer..."):
                    if rag_ready:
                        answer, chunks = rag.answer_with_rag(
                            question=user_input,
                            live_context=full_ctx,
                            llm_client=llm,
                            top_k=4,
                            conversation_history=st.session_state.messages[-8:],
                        )
                    else:
                        # Fallback to plain LLM
                        answer = llm.answer_question(user_input, context=full_ctx)
                        chunks = []

                st.session_state.messages.append({
                    "role":"assistant","content":answer,"chunks":chunks
                })
            except ConnectionError as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ LLM not reachable: `{e}`","chunks":[]})
            except RuntimeError as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ Generation failed: `{e}`","chunks":[]})
            except Exception as e:
                st.session_state.messages.append({"role":"assistant","content":f"❌ Error: `{e}`","chunks":[]})
            st.rerun()

    # ── TAB 2: ANALYSIS ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 Regional Analysis Dashboard")
        fdf = disp_df[disp_df["disease"]==sel_disease]
        lat = fdf.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1]).reset_index(drop=True)

        # Recompute risk on snapshot
        ht = lat["risk_score"].quantile(0.75)
        mt = lat["risk_score"].quantile(0.40)
        lat["risk_level"] = lat["risk_score"].apply(
            lambda s: "High" if s>=ht else "Medium" if s>=mt else "Low")

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Risk Distribution")
            pie = go.Figure(data=[go.Pie(
                labels=["🔴 High","🟠 Medium","🟢 Low"],
                values=[(lat["risk_level"]=="High").sum(),(lat["risk_level"]=="Medium").sum(),(lat["risk_level"]=="Low").sum()],
                marker=dict(colors=["#ef4444","#f97316","#16a34a"]))])
            pie.update_layout(height=280,margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(pie,use_container_width=True)
        with c2:
            st.markdown("#### Risk Leaderboard")
            lb = px.bar(lat.sort_values("risk_score"),x="risk_score",y="region",
                color="risk_level",orientation="h",
                color_discrete_map={"High":"#ef4444","Medium":"#f97316","Low":"#16a34a"})
            lb.update_layout(height=280,margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(lb,use_container_width=True)

        # YoY
        if len(years)>1 and sel_year=="All":
            st.markdown(f"---\n#### 📅 Year-over-Year — {sel_region} — {sel_disease}")
            yoy = df[(df["region"]==sel_region)&(df["disease"]==sel_disease)].groupby("year")["cases"].agg(["sum","max","mean"]).reset_index()
            yoy.columns=["Year","Total","Peak","Avg Daily"]
            yf = px.bar(yoy,x="Year",y="Total",color="Total",color_continuous_scale="RdYlGn_r",text="Total")
            yf.update_layout(height=280,margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(yf,use_container_width=True)

        st.markdown("---\n#### 📋 Current Status — All Regions")
        tbl = lat[["region","cases","z_score","risk_score","risk_level","is_spike"]].copy()
        tbl["Spike"] = tbl["is_spike"].map({True:"🔴 Yes",False:"🟢 No"})
        st.dataframe(tbl[["region","cases","z_score","risk_score","risk_level","Spike"]].sort_values("risk_score",ascending=False),
                     use_container_width=True,hide_index=True)

    # ── TAB 3: TOOLS ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### ⚙️ Quick Analysis Tools")

        if rag_ready:
            st.markdown(
                '<span class="rag-badge">🧠 RAG Active — all analyses grounded in WHO/IDSP guidelines</span>',
                unsafe_allow_html=True)
            st.markdown("")

        c1,c2,c3 = st.columns(3)

        with c1:
            if st.button("📊 Situation Report",use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        fdf2 = disp_df[disp_df["disease"]==sel_disease]
                        lts  = fdf2.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                        q    = f"Generate a comprehensive situation report for {sel_disease} across all regions. Include total cases, high-risk regions, trends, and recommendations."
                        ctx2 = get_rich_context(disp_df,sel_region,sel_disease,sel_year)
                        ctx2["all_regions_summary"] = lts[["cases","risk_score"]].to_dict()
                        if rag_ready:
                            ans,cks = rag.answer_with_rag(q,ctx2,llm,top_k=5)
                        else:
                            ans,cks = llm.generate_report(ctx2), []
                        st.markdown(ans)
                        display_citations(cks)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c2:
            if st.button("⚠️ High-Risk Regions",use_container_width=True):
                with st.spinner("Scanning..."):
                    try:
                        fdf2 = disp_df[disp_df["disease"]==sel_disease]
                        lts  = fdf2.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                        ht2  = lts["risk_score"].quantile(0.75)
                        high = lts[lts["risk_score"]>=ht2]
                        q    = f"These regions have HIGH risk for {sel_disease}: {high['region'].tolist() if 'region' in high.columns else list(high.index)}. What immediate interventions does WHO/IDSP recommend?"
                        ctx2 = get_rich_context(disp_df,sel_region,sel_disease,sel_year)
                        ctx2["high_risk_regions"] = high[["cases","risk_score"]].to_dict()
                        if rag_ready:
                            ans,cks = rag.answer_with_rag(q,ctx2,llm,top_k=5)
                        else:
                            ans,cks = llm.answer_question(q,context=ctx2), []
                        st.markdown(ans)
                        display_citations(cks)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c3:
            if st.button("💊 Intervention Guide",use_container_width=True):
                with st.spinner("Fetching guidelines..."):
                    try:
                        q   = f"Based on WHO and IDSP guidelines, what are the specific intervention steps for a {sel_disease} outbreak with z-score {ctx.get('z_score','N/A')} in {sel_region}? Include vector control, treatment, and public health measures."
                        ctx2 = get_rich_context(disp_df,sel_region,sel_disease,sel_year)
                        if rag_ready:
                            ans,cks = rag.answer_with_rag(q,ctx2,llm,top_k=5)
                        else:
                            ans,cks = llm.answer_question(q,context=ctx2), []
                        st.markdown(ans)
                        display_citations(cks)
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---\n### 🎯 Custom Analysis")
        opts = {
            "WHO Intervention Recommendations": f"Based on WHO guidelines, what interventions are recommended for {sel_disease} at the current risk level in {sel_region}?",
            "Climate-Disease Correlation Analysis": f"Based on the climate data (temperature {ctx.get('avg_temp','N/A')}°C, humidity {ctx.get('avg_humidity','N/A')}%), how does this affect {sel_disease} transmission risk?",
            "Outbreak Response Protocol": f"Walk me through the step-by-step IDSP outbreak response protocol for {sel_disease} in {sel_region} given z-score {ctx.get('z_score','N/A')}.",
            "Resource Allocation Advice": "Based on risk scores, how should we prioritize health resources across all monitored regions?",
            "Year-over-Year Analysis": f"Analyse disease burden trends for {sel_disease} across all available years. Is the situation improving?",
            "Compare with Similar Past Outbreaks": f"Have there been similar {sel_disease} outbreaks in the data? What happened and what was effective?",
        }
        analysis_type = st.selectbox("Select analysis:",list(opts.keys()))
        if st.button("🚀 Run Analysis",use_container_width=True):
            with st.spinner("Running RAG analysis..."):
                try:
                    q    = opts[analysis_type]
                    ctx2 = get_rich_context(disp_df,sel_region,sel_disease,sel_year)
                    fdf2 = disp_df[disp_df["disease"]==sel_disease]
                    lts  = fdf2.groupby("region").apply(lambda x: x.sort_values("date").iloc[-1])
                    ctx2["all_regions_risk"] = lts[["cases","risk_score","z_score"]].to_dict()
                    if rag_ready:
                        ans,cks = rag.answer_with_rag(q,ctx2,llm,top_k=5)
                    else:
                        ans,cks = llm.answer_question(q,context=ctx2), []
                    st.markdown(ans)
                    display_citations(cks)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("""
        <div style="text-align:center;color:#9ca3af;font-size:.85rem;padding:2rem 0">
            🩺 <strong>EpiPulse AI v3.0</strong> · RAG + Groq + Llama · Grounded in WHO & IDSP guidelines
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
