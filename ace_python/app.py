"""
ACE - Autonomous Cognitive Engine
Streamlit User Interface
Run with: streamlit run app.py
"""
import os
import sys
import time
import json
import threading
import queue
from typing import Optional
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACE — Autonomous Cognitive Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background: #0d0d14; }
.stApp { background: #0d0d14; color: #e2e8f0; }

.ace-header {
    background: linear-gradient(135deg, #1a0a2e, #0d0d1f);
    border: 1px solid #3d1d6e;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.ace-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: #c4b5fd;
    letter-spacing: 0.15em;
}

.ace-subtitle {
    font-size: 12px;
    color: #5b4a7c;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
}

.task-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 12px;
    margin: 6px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}

.task-card.active {
    border-color: #7c3aed;
    background: #1a1028;
}

.task-card.done {
    border-color: #064e3b;
    background: #0c1f18;
}

.agent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace;
}

.log-panel {
    background: #0a0a12;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 12px;
    height: 400px;
    overflow-y: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}

.report-panel {
    background: #0f0f1a;
    border: 1px solid #2d1d4e;
    border-radius: 8px;
    padding: 24px;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.7;
}

.metric-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 14px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #c4b5fd;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-label {
    font-size: 11px;
    color: #5b4a7c;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.vfs-item {
    display: flex;
    align-items: center;
    padding: 6px 8px;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    cursor: pointer;
}

.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

/* Streamlit overrides */
.stTextArea textarea {
    background: #1a1a2e !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}

.stButton button {
    background: #2d1058 !important;
    color: #c4b5fd !important;
    border: 1px solid #5b21b6 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stSelectbox select, .stTextInput input {
    background: #1a1a2e !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d2d4e !important;
}

.stProgress .st-bo { background: #7c3aed !important; }

div[data-testid="metric-container"] {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "engine_state": None,
        "is_running": False,
        "logs": [],
        "tasks": [],
        "vfs": {},
        "final_report": "",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-opus-4-5",
        "active_agent": None,
        "progress": 0.0,
        "status": "idle",
        "stream_buffer": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ace-header"><div class="ace-title">⬡ ACE</div><div class="ace-subtitle">Autonomous Cognitive Engine</div></div>', unsafe_allow_html=True)

    st.markdown("### ⚙ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        help="Your Anthropic API key"
    )
    if api_key:
        st.session_state.api_key = api_key

    model = st.selectbox(
        "Model",
        ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5-20251001"],
        index=0,
        help="Select the Claude model to use"
    )
    st.session_state.model = model

    enable_search = st.toggle("Enable Web Search", value=True, help="Use DuckDuckGo for research tasks")

    st.divider()

    st.markdown("### 📊 Pipeline Status")

    status_colors = {
        "idle": "#6b7280",
        "planning": "#2563eb",
        "running": "#f59e0b",
        "reporting": "#e11d48",
        "complete": "#10b981",
        "error": "#ef4444",
    }
    status = st.session_state.status
    color = status_colors.get(status, "#6b7280")
    st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-size:13px;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{status.upper()}</div>', unsafe_allow_html=True)

    if st.session_state.tasks:
        completed = sum(1 for t in st.session_state.tasks if t.get("status") == "completed")
        total = len(st.session_state.tasks)
        st.progress(completed / total if total > 0 else 0)
        st.caption(f"{completed}/{total} tasks complete")

    st.divider()

    st.markdown("### 📂 Agent Pipeline")
    agents = ["🔷 Supervisor", "📋 Planner", "🔬 Research", "📝 Summarizer", "📊 Reporter"]
    for agent in agents:
        agent_name = agent.split()[-1].lower()
        is_active = st.session_state.active_agent == agent_name
        dot_color = "#10b981" if is_active else "#3d3d5c"
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:{"#c4b5fd" if is_active else "#6b7280"};padding:3px 0;"><span style="width:6px;height:6px;border-radius:50%;background:{dot_color};"></span>{agent}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### ℹ Architecture")
    st.caption("""
**LangGraph-style Pipeline:**
- State machine with typed nodes
- Conditional routing between agents
- Shared state object
- VFS for intermediate storage
- Memory context management

**Agents:**
- Supervisor: orchestration
- Planner: task decomposition
- Research: information gathering
- Summarizer: synthesis
- Reporter: report generation
    """)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<div class="ace-header"><div class="ace-title">⬡ Autonomous Cognitive Engine</div><div class="ace-subtitle">Deep Research & Long-Horizon Task System</div></div>', unsafe_allow_html=True)

# Query Input
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_area(
        "Research Query",
        placeholder="Enter a complex research question or task...\n\nExample: 'Analyze the current state and future trajectory of quantum computing, focusing on near-term applications, technical challenges, and strategic implications for cybersecurity'",
        height=120,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.is_running:
        if st.button("▶ Launch\nEngine", use_container_width=True):
            if not st.session_state.api_key:
                st.error("API key required")
            elif not query.strip():
                st.warning("Enter a query")
            else:
                st.session_state.is_running = True
                st.session_state.status = "planning"
                st.session_state.logs = []
                st.session_state.tasks = []
                st.session_state.vfs = {}
                st.session_state.final_report = ""
                # Trigger rerun to start engine
                st.rerun()
    else:
        if st.button("✕ Stop", use_container_width=True):
            st.session_state.is_running = False
            st.session_state.status = "idle"
            st.rerun()

# Quick examples
st.markdown('<p style="font-size:11px;color:#3d3d5c;margin-bottom:6px;">Quick Examples:</p>', unsafe_allow_html=True)
example_cols = st.columns(3)
examples = [
    "Analyze the current state of quantum computing and its impact on cryptography",
    "What are the key technical and economic challenges facing renewable energy adoption?",
    "How is artificial intelligence transforming drug discovery and clinical trials?",
]
for i, (col, ex) in enumerate(zip(example_cols, examples)):
    with col:
        if st.button(ex[:50] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["_set_query"] = ex

# Run Engine
if st.session_state.is_running and query.strip():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from engine import build_engine
        from core.state import TaskStatus

        log_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()

        logs_display = []
        tasks_display = []

        def on_update(state):
            st.session_state.active_agent = state.active_agent
            st.session_state.status = state.status
            st.session_state.tasks = [t.to_dict() for t in state.tasks]
            st.session_state.vfs = {k: {"size": len(v["content"])} for k, v in state.vfs.items()}
            st.session_state.logs = [
                {"agent": l.agent, "message": l.message, "level": l.level, "time": l.timestamp}
                for l in state.logs[-50:]
            ]
            if state.tasks:
                completed = len([t for t in state.tasks if t.status == TaskStatus.COMPLETED])
                progress = completed / len(state.tasks)
                progress_bar.progress(progress)
                status_text.caption(f"Executing: {completed}/{len(state.tasks)} tasks complete")

        engine = build_engine(
            api_key=st.session_state.api_key,
            model=st.session_state.model,
            enable_search=enable_search,
            on_update=on_update,
        )

        with st.spinner("🔬 Cognitive engine running..."):
            final_state = engine.run(query)

        st.session_state.final_report = final_state.final_report
        st.session_state.tasks = [t.to_dict() for t in final_state.tasks]
        st.session_state.vfs = {k: {"size": len(v["content"])} for k, v in final_state.vfs.items()}
        st.session_state.logs = [
            {"agent": l.agent, "message": l.message, "level": l.level, "time": l.timestamp}
            for l in final_state.logs
        ]
        st.session_state.status = "complete"
        st.session_state.is_running = False
        st.rerun()

    except ImportError as e:
        st.error(f"Missing dependency: {e}\n\nRun: pip install anthropic langchain-anthropic langchain-community")
        st.session_state.is_running = False
    except Exception as e:
        st.error(f"Engine error: {e}")
        st.session_state.is_running = False

# ─── Results Display ──────────────────────────────────────────────────────────

if st.session_state.tasks or st.session_state.final_report:
    st.divider()

    # Metrics
    if st.session_state.tasks:
        tasks = st.session_state.tasks
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Tasks", len(tasks))
        with m2:
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            st.metric("Completed", completed)
        with m3:
            st.metric("VFS Files", len(st.session_state.vfs))
        with m4:
            report_len = len(st.session_state.final_report)
            st.metric("Report Size", f"{report_len // 1000}k chars" if report_len > 0 else "—")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Task Board", "📝 Agent Logs", "📂 File System", "📊 Final Report"])

    with tab1:
        if st.session_state.tasks:
            agent_colors = {
                "research": ("#064e3b", "#059669"),
                "summarizer": ("#451a03", "#d97706"),
                "reporter": ("#4c0519", "#e11d48"),
                "supervisor": ("#1a1a2e", "#7c3aed"),
                "planner": ("#1e3a5f", "#2563eb"),
            }
            status_icons = {"pending": "○", "in_progress": "◐", "completed": "●", "failed": "✕"}
            status_colors = {"pending": "#6b7280", "in_progress": "#f59e0b", "completed": "#10b981", "failed": "#ef4444"}

            for task in st.session_state.tasks:
                agent = task.get("agent", "research")
                status = task.get("status", "pending")
                bg, accent = agent_colors.get(agent, ("#13131f", "#6b7280"))
                scolor = status_colors.get(status, "#6b7280")
                icon = status_icons.get(status, "○")

                is_active = st.session_state.active_agent == agent and status == "in_progress"
                border = accent if is_active else ("#064e3b" if status == "completed" else "#1e1e30")

                st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:6px;padding:12px;margin:6px 0;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="color:{scolor};font-size:14px;">{icon}</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500;color:#e2e8f0;">{task.get('title', '')}</span>
    <span style="margin-left:auto;background:{bg};color:{accent};padding:2px 8px;border-radius:3px;font-size:10px;text-transform:uppercase;letter-spacing:0.06em;border:1px solid {accent}40;">{agent}</span>
  </div>
  <div style="font-size:11px;color:#6b7280;margin-bottom:6px;">{task.get('description','')[:100]}...</div>
  <div style="font-size:10px;color:{scolor};">{status.replace('_',' ').upper()}</div>
</div>""", unsafe_allow_html=True)

    with tab2:
        level_colors = {"info": "#94a3b8", "success": "#10b981", "warning": "#f59e0b", "error": "#ef4444"}
        agent_colors_text = {
            "supervisor": "#7c3aed", "planner": "#2563eb", "research": "#059669",
            "summarizer": "#d97706", "reporter": "#e11d48", "memory": "#6366f1"
        }

        log_html = '<div style="background:#0a0a12;border:1px solid #1e1e30;border-radius:6px;padding:12px;font-family:\'IBM Plex Mono\',monospace;font-size:11px;max-height:400px;overflow-y:auto;">'
        for log in st.session_state.logs:
            agent = log.get("agent", "")
            msg = log.get("message", "")
            level = log.get("level", "info")
            time_str = log.get("time", "")
            acolor = agent_colors_text.get(agent, "#6b7280")
            lcolor = level_colors.get(level, "#94a3b8")
            log_html += f'<div style="display:flex;gap:10px;padding:4px 0;border-bottom:1px solid #1e1e30;"><span style="color:#3d3d5c;min-width:55px;">{time_str}</span><span style="color:{acolor};min-width:80px;text-transform:uppercase;">{agent}</span><span style="color:{lcolor};flex:1;">{msg}</span></div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with tab3:
        if st.session_state.vfs:
            for filename, info in st.session_state.vfs.items():
                icon = "◈" if filename == "FINAL_REPORT.md" else ("◎" if filename.endswith(".json") else "◌")
                color = "#c4b5fd" if filename == "FINAL_REPORT.md" else "#6b7280"
                size_kb = info.get("size", 0) / 1024
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:3px;background:#13131f;margin:3px 0;font-family:\'IBM Plex Mono\',monospace;font-size:11px;"><span style="color:{color};">{icon}</span><span style="flex:1;color:{color};">{filename}</span><span style="color:#3d3d5c;">{size_kb:.1f}kb</span></div>', unsafe_allow_html=True)
        else:
            st.caption("No files stored yet")

    with tab4:
        if st.session_state.final_report:
            col_a, col_b = st.columns([1, 6])
            with col_a:
                if st.button("⬇ Download"):
                    st.download_button(
                        "📄 Download Report",
                        st.session_state.final_report,
                        file_name="research_report.md",
                        mime="text/markdown",
                    )
            st.markdown("---")
            st.markdown(st.session_state.final_report)
        else:
            st.info("Report will appear here when complete")

# ─── Empty State ──────────────────────────────────────────────────────────────
if not st.session_state.tasks and not st.session_state.is_running:
    st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#3d3d5c;">
    <div style="font-size:48px;margin-bottom:16px;opacity:0.4;">⬡</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;">Ready</div>
    <div style="font-size:13px;line-height:1.8;max-width:500px;margin:0 auto;">
        Enter a complex research query above to activate the multi-agent cognitive pipeline.<br><br>
        <span style="color:#5b4a7c;">Supervisor → Planner → Research → Memory → Summarizer → Reporter</span>
    </div>
</div>
""", unsafe_allow_html=True)
