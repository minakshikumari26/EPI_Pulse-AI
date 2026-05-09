"""Streamlit chat interface for LLM-powered disease outbreak analysis."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm import get_llm_client
from src.utils.config import get_data_path, load_config


# Custom CSS for enhanced styling
def inject_custom_css():
    """Inject custom CSS for better UI."""
    st.markdown(
        """
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #1e3a8a;
            --secondary-color: #0ea5e9;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }

        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
        }

        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }

        /* Card styling */
        .status-card {
            background: white;
            border-left: 4px solid #0ea5e9;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }

        .status-card.success {
            border-left-color: #10b981;
        }

        .status-card.warning {
            border-left-color: #f59e0b;
        }

        .status-card.danger {
            border-left-color: #ef4444;
        }

        /* Region selector */
        .region-selector {
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }

        /* Chat message styling */
        .chat-user {
            background: #dbeafe;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            margin-left: 2rem;
        }

        .chat-assistant {
            background: #f0f9ff;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            margin-right: 2rem;
            border-left: 3px solid #0ea5e9;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        /* Metric card */
        .metric-container {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #1e3a8a;
            margin: 0.5rem 0;
        }

        /* Alert styling */
        .alert-success {
            background: #d1fae5;
            border-left: 4px solid #10b981;
            padding: 1rem;
            border-radius: 6px;
            color: #065f46;
        }

        .alert-warning {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 6px;
            color: #78350f;
        }

        .alert-danger {
            background: #fee2e2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 6px;
            color: #7f1d1d;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] button {
            border-radius: 6px 6px 0 0;
            background: #f3f4f6;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
            color: white;
        }

        /* Spinner customization */
        .stSpinner {
            color: #0ea5e9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_disease_data():
    """Load and preprocess disease data."""
    df = pd.read_csv(get_data_path(), parse_dates=["date"])
    df = df.drop_duplicates().sort_values(["region", "date"]).reset_index(drop=True)

    numeric_cols = ["cases", "temperature", "humidity", "rainfall"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    df["z_score"] = df.groupby("region")["cases"].transform(zscore)
    df["is_spike"] = df["z_score"] > 1.5

    return df


def get_region_summary(df, region):
    """Get summary statistics for a region."""
    region_df = df[df["region"] == region].sort_values("date")

    if region_df.empty:
        return None

    latest = region_df.iloc[-1]
    prev_week_avg = region_df.iloc[-8:-1]["cases"].mean() if len(region_df) > 7 else 0

    return {
        "region": region,
        "latest_cases": int(latest["cases"]),
        "latest_z_score": float(latest["z_score"]),
        "is_spike": bool(latest["is_spike"]),
        "prev_week_avg": int(prev_week_avg),
        "growth_rate": (
            (latest["cases"] - prev_week_avg) / prev_week_avg
            if prev_week_avg > 0
            else 0
        ),
        "last_date": latest["date"].strftime("%Y-%m-%d"),
    }


def get_risk_level_color(z_score):
    """Get color based on risk level."""
    if z_score > 2.0:
        return "🔴"  # Critical
    elif z_score > 1.5:
        return "🟠"  # High
    elif z_score > 1.0:
        return "🟡"  # Medium
    else:
        return "🟢"  # Low


def plot_regional_trend(df, region):
    """Create interactive plot for regional trend."""
    region_df = df[df["region"] == region].sort_values("date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=region_df["date"],
            y=region_df["cases"],
            mode="lines+markers",
            name="Cases",
            line=dict(color="#0ea5e9", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.1)",
        )
    )

    fig.update_layout(
        title=f"📈 {region} - Case Trend",
        xaxis_title="Date",
        yaxis_title="Number of Cases",
        template="plotly_white",
        hovermode="x unified",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="EpiPulse AI Chat",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "EpiPulse AI - Disease Intelligence Platform v1.0"},
    )

    # Inject custom CSS
    inject_custom_css()

    # Header with gradient
    st.markdown(
        """
        <div class="main-header">
            <h1>🩺 EpiPulse AI</h1>
            <p>AI-Powered Disease Outbreak Intelligence & Analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load configuration and data
    try:
        config = load_config()
        llm_config = config.get("llm", {})

        if not llm_config.get("enabled", False):
            st.error("⚠️ LLM is disabled in configuration. Enable it in config.yaml")
            return

        llm = get_llm_client()
        connection_status = f"✅ Connected to **{llm.model}** on `{llm.base_url}`"
        available_models = getattr(llm, "available_models", [])

    except ConnectionError as e:
        st.error(
            f"❌ Cannot connect to Ollama: {str(e)}\n\n"
            "**Setup Instructions:**\n"
            "1. Install Ollama from https://ollama.ai\n"
            "2. Run: `ollama pull llama3.2:1b` (recommended for lower RAM)\n"
            "3. Start Ollama: `ollama serve`\n"
            "4. Refresh this page"
        )
        return
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return

    df = load_disease_data()
    regions = sorted(df["region"].unique())

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("---")
        
        # Connection status
        st.markdown(
            f"""
            <div class="status-card success">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">🔌 LLM Connection</div>
                <div>{connection_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if available_models:
            st.markdown(
                f"**Available models:** {', '.join(available_models)}"
            )

        if getattr(llm, "fallback_used", False) and getattr(llm, "fallback_model", None):
            st.info(
                f"⚠️ Using fallback model `{llm.fallback_model}` because the configured model was not available or failed during generation."
            )

        st.markdown("---")
        st.markdown("### 📍 Select Region")

        selected_region = st.selectbox(
            "Choose a region for contextual analysis:",
            regions,
            label_visibility="collapsed",
        )

        # Regional status dashboard
        if selected_region:
            summary = get_region_summary(df, selected_region)
            if summary:
                st.markdown("---")
                st.markdown(f"#### {selected_region} - Snapshot")

                # Risk indicator
                risk_color = get_risk_level_color(summary["latest_z_score"])
                risk_text = (
                    "🔴 CRITICAL"
                    if summary["latest_z_score"] > 2.0
                    else "🟠 HIGH"
                    if summary["latest_z_score"] > 1.5
                    else "🟡 MEDIUM"
                    if summary["latest_z_score"] > 1.0
                    else "🟢 LOW"
                )

                st.markdown(
                    f"""
                    <div class="status-card {'danger' if summary['latest_z_score'] > 1.5 else 'warning' if summary['latest_z_score'] > 1.0 else 'success'}">
                        <div class="metric-label">Risk Level</div>
                        <div style="font-size: 1.5rem; margin: 0.5rem 0;">{risk_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metrics in 2x2 grid
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Cases", f"{summary['latest_cases']:,}")

                with col2:
                    st.metric("Z-Score", f"{summary['latest_z_score']:.2f}")

                col3, col4 = st.columns(2)

                with col3:
                    st.metric(
                        "Growth",
                        f"{summary['growth_rate']:.1%}",
                        delta=f"vs {summary['prev_week_avg']} avg",
                    )

                with col4:
                    spike_indicator = "🔴 SPIKE" if summary["is_spike"] else "🟢 Normal"
                    st.metric("Status", spike_indicator)

                # Trend chart
                st.markdown("---")
                st.markdown("#### 📈 Trend")
                fig = plot_regional_trend(df, selected_region)
                st.plotly_chart(fig, use_container_width=True)

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analysis", "⚙️ Tools"])

    # ==================== TAB 1: CHAT ====================
    with tab1:
        st.markdown(
            """
            <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #0ea5e9; margin-bottom: 1.5rem;">
                <p style="margin: 0; color: #1e3a8a; font-weight: 600;">💡 Tip: Ask about cases, trends, risk levels, regional comparisons, or outbreak strategies.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 Hello! I'm your AI epidemiologist assistant. I can help you analyze "
                        f"disease outbreak data for **{', '.join(regions)}**.\n\n"
                        f"**I can help with:**\n"
                        f"• 📍 Current outbreak status in any region\n"
                        f"• 📈 Trends and growth rates\n"
                        f"• ⚠️ Risk assessments and recommendations\n"
                        f"• 🔄 Comparisons between regions\n"
                        f"• 💊 Intervention strategies\n\n"
                        f"**What would you like to know about the current disease situation?**"
                    ),
                }
            ]

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar="👤" if message["role"] == "user" else "🤖",
            ):
                st.markdown(message["content"])

        # Chat input
        user_input = st.chat_input(
            placeholder="Ask your AI epidemiologist..."
        )

        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤔 Analyzing..."):
                    try:
                        context = {
                            "selected_region": selected_region,
                            "available_regions": regions,
                        }

                        if selected_region:
                            summary = get_region_summary(df, selected_region)
                            if summary:
                                context["region_summary"] = summary

                        response = llm.answer_question(user_input, context=context)

                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    # ==================== TAB 2: ANALYSIS ====================
    with tab2:
        st.markdown("### 📊 Regional Analysis Dashboard")

        # Get latest data for all regions
        latest_data = df.groupby("region").apply(
            lambda x: x.sort_values("date").iloc[-1]
        )
        latest_data = latest_data.sort_values("z_score", ascending=False)

        # Risk distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Risk Distribution")
            high_risk = len(latest_data[latest_data["z_score"] > 1.5])
            medium_risk = len(
                latest_data[(latest_data["z_score"] > 1.0) & (latest_data["z_score"] <= 1.5)]
            )
            low_risk = len(latest_data[latest_data["z_score"] <= 1.0])

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["🔴 High", "🟡 Medium", "🟢 Low"],
                        values=[high_risk, medium_risk, low_risk],
                        marker=dict(colors=["#ef4444", "#f59e0b", "#10b981"]),
                    )
                ]
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Cases by Region")
            region_cases = latest_data.sort_values("cases", ascending=True).tail(8)

            fig = go.Figure(
                data=[
                    go.Bar(
                        y=region_cases.index,
                        x=region_cases["cases"],
                        orientation="h",
                        marker=dict(
                            color=region_cases["z_score"],
                            colorscale="RdYlGn_r",
                            showscale=False,
                        ),
                    )
                ]
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="Cases",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Regional table
        st.markdown("---")
        st.markdown("#### 📋 All Regions Status")

        if "region" in latest_data.columns:
            table_data = latest_data[["region", "cases", "z_score", "is_spike"]].copy()
        else:
            table_data = latest_data.reset_index()[["region", "cases", "z_score", "is_spike"]].copy()

        table_data["Risk"] = table_data["z_score"].apply(
            lambda x: "🔴 CRITICAL"
            if x > 2.0
            else "🟠 HIGH"
            if x > 1.5
            else "🟡 MEDIUM"
            if x > 1.0
            else "🟢 LOW"
        )
        table_data = table_data.sort_values("z_score", ascending=False)

        st.dataframe(
            table_data[["region", "cases", "z_score", "Risk"]],
            use_container_width=True,
            hide_index=True,
        )

    # ==================== TAB 3: QUICK TOOLS ====================
    with tab3:
        st.markdown("### ⚙️ Quick Analysis Tools")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Summarize Current Situation", use_container_width=True):
                with st.spinner("📝 Generating comprehensive summary..."):
                    try:
                        latest_data = df.groupby("region").apply(
                            lambda x: x.sort_values("date").iloc[-1]
                        )
                        summary_text = llm.generate_report(
                            {"regions": len(regions), "data": latest_data.to_dict()}
                        )

                        st.markdown(
                            """
                            <div class="status-card">
                                <div style="font-weight: 600; margin-bottom: 1rem; color: #1e3a8a;">📋 Executive Summary</div>
                                <div style="line-height: 1.6; color: #374151;">
                            """
                            + summary_text.replace("\n", "<br>")
                            + """
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        with col2:
            if st.button("⚠️ Identify High-Risk Regions", use_container_width=True):
                with st.spinner("🔍 Analyzing risk levels..."):
                    try:
                        latest_data = df.groupby("region").apply(
                            lambda x: x.sort_values("date").iloc[-1]
                        )
                        high_risk = latest_data[latest_data["is_spike"]]["region"].tolist()

                        if high_risk:
                            st.markdown(
                                f"""
                                <div class="alert-danger">
                                    <strong>⚠️ High-Risk Regions Detected</strong><br>
                                    {', '.join([f"🔴 <strong>{r}</strong>" for r in high_risk])}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="alert-success">
                                    <strong>✅ All Clear!</strong><br>
                                    No high-risk regions detected at this time.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        with col3:
            if st.button("🔄 Compare All Regions", use_container_width=True):
                with st.spinner("📊 Generating regional comparison..."):
                    try:
                        latest_data = df.groupby("region").apply(
                            lambda x: x.sort_values("date").iloc[-1]
                        )
                        comparison = {
                            "regions": list(latest_data.index),
                            "cases": latest_data["cases"].to_dict(),
                            "z_scores": latest_data["z_score"].to_dict(),
                        }
                        response = llm.answer_question(
                            "Provide a detailed comparison of the disease situation across all regions. "
                            "Highlight which regions are most affected and any notable patterns.",
                            context=comparison,
                        )
                        st.markdown(
                            """
                            <div class="status-card">
                                <div style="font-weight: 600; margin-bottom: 1rem; color: #1e3a8a;">🔄 Regional Comparison Analysis</div>
                                <div style="line-height: 1.6; color: #374151;">
                            """
                            + response.replace("\n", "<br>")
                            + """
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.markdown("---")
        st.markdown("### 🎯 Custom Analysis")

        analysis_type = st.selectbox(
            "Select analysis type:",
            [
                "Generate Situation Report",
                "Risk Assessment",
                "Trend Analysis",
                "Intervention Recommendations",
            ],
        )

        custom_question_map = {
            "Generate Situation Report": (
                "Generate a comprehensive situation report summarizing the current disease "
                "outbreak status including total cases, affected regions, and key findings."
            ),
            "Risk Assessment": (
                "Conduct a detailed risk assessment identifying which regions are at highest "
                "risk and what factors contribute to their risk levels."
            ),
            "Trend Analysis": (
                "Analyze current trends in the disease data. Are cases increasing or decreasing? "
                "Which regions are showing concerning trends?"
            ),
            "Intervention Recommendations": (
                "Based on the current outbreak data, what interventions would you recommend "
                "for each high-risk region?"
            ),
        }

        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("⏳ Running analysis..."):
                try:
                    latest_data = df.groupby("region").apply(
                        lambda x: x.sort_values("date").iloc[-1]
                    )
                    context = {
                        "total_regions": len(regions),
                        "total_cases": int(latest_data["cases"].sum()),
                        "high_risk_count": len(latest_data[latest_data["is_spike"]]),
                        "avg_z_score": float(latest_data["z_score"].mean()),
                    }

                    response = llm.answer_question(
                        custom_question_map[analysis_type], context=context
                    )

                    st.markdown(
                        f"""
                        <div class="status-card">
                            <div style="font-weight: 600; margin-bottom: 1rem; color: #1e3a8a;">📈 {analysis_type}</div>
                            <div style="line-height: 1.6; color: #374151;">
                        """
                        + response.replace("\n", "<br>")
                        + """
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #9ca3af; font-size: 0.9rem; padding: 2rem 0;">
                <p>🩺 <strong>EpiPulse AI v1.0</strong> | Powered by Ollama Local LLM</p>
                <p>For issues or feedback, contact your EpiPulse admin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
