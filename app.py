"""
app.py

"""

__version__ = "2.0"

import io
import random

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

from utils import SearchSimulator, average_witness_score
from visualization import (
    plot_comparison,
    plot_fuzzy_membership,
    plot_fuzzy_age_membership,
    plot_fuzzy_output,
    plot_monte_carlo,
    plot_single,
)

# ═══════════════════════════════════════════════════════════════
# Light theme palette (Torneo-inspired)
# ═══════════════════════════════════════════════════════════════
_BG       = "#f4f8ff"
_WHITE    = "#ffffff"
_SURFACE  = "#ffffff"
_SURFACE2 = "#f0f4fb"
_BORDER   = "#d0dff0"
_ACCENT   = "#1565C0"   # deep blue (primary)
_ACCENT_L = "#1976D2"   # medium blue (hover)
_ACCENT2  = "#0D47A1"   # navy blue (secondary)
_GREEN    = "#00897b"
_AMBER    = "#f57c00"
_TEXT     = "#0d1b2e"
_MUTED    = "#607d9b"

# ─── matplotlib light style ──────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  _WHITE,
    "axes.facecolor":    "#f5f9ff",
    "axes.edgecolor":    "#ccdaf0",
    "axes.labelcolor":   "#334e68",
    "axes.titlecolor":   _TEXT,
    "xtick.color":       _MUTED,
    "ytick.color":       _MUTED,
    "text.color":        "#334e68",
    "grid.color":        "#ddeeff",
    "grid.alpha":        0.8,
    "legend.facecolor":  _WHITE,
    "legend.edgecolor":  "#ccdaf0",
    "legend.labelcolor": "#334e68",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.grid":         True,
})

# ═══════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="jatAIyu: AI-Assisted Missing Person Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# Global CSS
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {_TEXT} !important;
}}
/* force all Streamlit markdown content to dark text */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"],
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: {_TEXT} !important;
}}
.stApp {{
    background-color: {_BG};
}}
.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1360px;
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {_WHITE};
    border-right: 1px solid {_BORDER};
    box-shadow: 2px 0 12px rgba(0,0,0,0.06);
}}
[data-testid="stSidebar"] * {{ color: {_TEXT} !important; }}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] .bar {{
    background: {_ACCENT};
}}

/* ── top-level buttons ── */
.stButton > button {{
    background: {_ACCENT};
    color: white !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.55rem 1.4rem;
    letter-spacing: 0.04em;
    transition: background 0.2s, transform 0.15s, box-shadow 0.15s;
    box-shadow: 0 2px 8px rgba(21,101,192,0.25);
}}
.stButton > button:hover {{
    background: {_ACCENT_L} !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(21,101,192,0.35);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* ── download button ── */
[data-testid="stDownloadButton"] > button {{
    background: {_ACCENT2} !important;
    color: white !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: background 0.2s, transform 0.15s;
    box-shadow: 0 2px 8px rgba(13,71,161,0.20);
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: #1565C0 !important;
    transform: translateY(-1px);
}}

/* ── metric cards ── */
[data-testid="stMetric"] {{
    background: {_WHITE};
    border: 1px solid {_BORDER};
    border-top: 3px solid {_ACCENT};
    border-radius: 8px;
    padding: 1rem 1.1rem 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeInUp 0.5s ease both;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.10);
}}
[data-testid="stMetricLabel"] {{
    color: {_MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
[data-testid="stMetricValue"] {{
    color: {_TEXT} !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}}

/* ── tabs ── */
[data-testid="stTabs"],
[data-testid="stTabs"] > div:first-child {{
    background: {_WHITE} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-baseweb="tab-list"] {{
    background: {_WHITE} !important;
    border-bottom: 2px solid {_BORDER} !important;
    gap: 0 !important;
    padding: 0 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"],
[data-baseweb="tab"] {{
    background: {_WHITE} !important;
    color: {_MUTED} !important;
    border-radius: 0 !important;
    padding: 10px 18px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    transition: color 0.2s, border-color 0.2s;
}}
[data-testid="stTabs"] [aria-selected="true"],
[data-baseweb="tab"][aria-selected="true"] {{
    background: {_WHITE} !important;
    color: {_ACCENT} !important;
    border-bottom: 3px solid {_ACCENT} !important;
    font-weight: 700 !important;
}}
/* tab panel content area */
[data-baseweb="tab-panel"] {{
    background: transparent !important;
}}

/* ── pyplot wrapper — remove white box ── */
[data-testid="stPyplotFigure"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stPyplotFigure"] > div {{
    background: transparent !important;
}}

/* ── expander ── */
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] > details {{
    background: {_WHITE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}}
/* the clickable summary/header bar */
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary,
details[data-testid] > summary {{
    background: {_WHITE} !important;
    font-weight: 600 !important;
    color: {_TEXT} !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
}}
[data-testid="stExpander"] details > summary:hover {{
    background: {_SURFACE2} !important;
}}
/* expanded state — remove dark inner bg */
[data-testid="stExpander"] details[open] > summary {{
    border-bottom: 1px solid {_BORDER} !important;
    border-radius: 8px 8px 0 0 !important;
}}
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] .streamlit-expanderContent {{
    background: {_WHITE} !important;
}}
/* toolbar strip that appears as a dark bar */
[data-testid="stElementToolbar"],
[data-testid="stElementToolbarButton"] {{
    background: {_SURFACE2} !important;
    color: {_TEXT} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}

/* ── alert boxes ── */
[data-testid="stAlert"] {{
    border-radius: 8px;
}}

/* ── divider ── */
hr {{ border-color: {_BORDER}; }}

/* ── card class ── */
.card {{
    background: {_WHITE};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease both;
}}
.card:hover {{
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    transition: box-shadow 0.25s;
}}

/* ── section header ── */
.section-header {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {_TEXT};
    border-left: 4px solid {_ACCENT};
    padding-left: 10px;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}}

/* ── stat row ── */
.stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: {_SURFACE2};
    border-radius: 6px;
    margin: 4px 0;
    border: 1px solid {_BORDER};
    transition: background 0.2s;
}}
.stat-row:hover {{ background: #f0f0f0; }}
.stat-label {{ color: {_MUTED}; font-size: 0.82rem; }}
.stat-value {{ color: {_TEXT}; font-weight: 600; font-size: 0.88rem; }}

/* ── score bar ── */
.bar-wrap {{
    background: #eeeeee;
    border-radius: 3px;
    height: 6px;
    margin: 2px 0 6px;
    overflow: hidden;
}}
.bar-fill {{
    height: 6px;
    border-radius: 3px;
    background: {_ACCENT};
    transition: width 0.6s ease;
}}

/* ── zone card ── */
.zone-card {{
    background: {_WHITE};
    border: 1px solid {_BORDER};
    border-left: 4px solid {_AMBER};
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    color: {_TEXT} !important;
    transition: box-shadow 0.2s;
}}
.zone-card:hover {{ box-shadow: 0 3px 10px rgba(0,0,0,0.08); }}
.zone-card * {{ color: inherit !important; }}

/* ── constraint cards ── */
.hard-card {{
    background: {_WHITE};
    border: 1px solid {_BORDER};
    border-left: 4px solid {_ACCENT};
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    color: {_TEXT} !important;
}}
.soft-card {{
    background: {_WHITE};
    border: 1px solid {_BORDER};
    border-left: 4px solid {_ACCENT2};
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    color: {_TEXT} !important;
}}

/* ── risk pills ── */
.risk-low    {{ color: {_GREEN};  background: #eafaf1; border: 1px solid #b7dfc7;
               display:inline-block;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.78rem; }}
.risk-medium {{ color: {_AMBER};  background: #fef9e7; border: 1px solid #f7dc6f;
               display:inline-block;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.78rem; }}
.risk-high   {{ color: {_ACCENT}; background: #e3f2fd; border: 1px solid #90caf9;
               display:inline-block;padding:3px 12px;border-radius:20px;font-weight:700;font-size:0.78rem; }}

/* ── animations ── */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.6; }}
}}
.animate-fade  {{ animation: fadeIn 0.6s ease both; }}
.animate-up    {{ animation: fadeInUp 0.5s ease both; }}
.animate-pulse {{ animation: pulse 2s ease infinite; }}

/* ── selectbox ── */
[data-baseweb="select"] > div,
[data-baseweb="select"] > div:hover {{
    background-color: {_WHITE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
    color: {_TEXT} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] [data-testid="stSelectbox"] {{
    color: {_TEXT} !important;
}}
/* dropdown menu list */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {{
    background-color: {_WHITE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}
[data-baseweb="popover"] li {{
    background-color: {_WHITE} !important;
    color: {_TEXT} !important;
}}
[data-baseweb="popover"] li:hover {{
    background-color: {_SURFACE2} !important;
}}
[data-baseweb="option"][aria-selected="true"] {{
    background-color: #e3f2fd !important;
    color: {_ACCENT} !important;
}}

/* ── number input — field + stepper buttons ── */
[data-baseweb="input"],
[data-baseweb="base-input"] {{
    background-color: {_WHITE} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
    color: {_TEXT} !important;
}}
[data-baseweb="input"] input,
[data-baseweb="base-input"] input {{
    background-color: {_WHITE} !important;
    color: {_TEXT} !important;
}}
/* stepper +/- buttons — target by data-testid and generic button */
[data-testid="stNumberInput-StepUp"],
[data-testid="stNumberInput-StepDown"],
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"],
[data-baseweb="input"] button,
[data-baseweb="base-input"] button,
[data-testid="stNumberInput"] button {{
    background-color: {_SURFACE2} !important;
    color: {_TEXT} !important;
    border-left: 1px solid {_BORDER} !important;
    border-top: none !important;
    border-right: none !important;
    border-bottom: none !important;
}}
[data-testid="stNumberInput-StepUp"]:hover,
[data-testid="stNumberInput-StepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover,
[data-testid="stNumberInputStepDown"]:hover,
[data-baseweb="input"] button:hover,
[data-baseweb="base-input"] button:hover {{
    background-color: {_BORDER} !important;
}}
/* ensure SVG icons inside buttons are visible */
[data-testid="stNumberInput"] button svg,
[data-baseweb="input"] button svg {{
    fill: {_TEXT} !important;
    stroke: {_TEXT} !important;
}}

/* ── checkbox — the visual indicator box ── */
[data-baseweb="checkbox"] {{
    background: transparent !important;
}}
/* label text */
[data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p,
[data-baseweb="checkbox"] label p,
[data-testid="stCheckbox"] label p {{
    color: {_TEXT} !important;
}}
/* the actual visual square box rendered by BaseWeb */
[data-baseweb="checkbox"] label > span:first-of-type,
[data-baseweb="checkbox"] [role="checkbox"],
[data-testid="stCheckbox"] label > span:first-of-type {{
    background-color: {_WHITE} !important;
    border: 2px solid {_BORDER} !important;
    border-radius: 3px !important;
    min-width: 16px !important;
    min-height: 16px !important;
}}
/* checked state */
[data-baseweb="checkbox"] label [aria-checked="true"] > span:first-of-type,
[data-testid="stCheckbox"] label [aria-checked="true"] > span:first-of-type,
[data-baseweb="checkbox"] input:checked + span,
input[type="checkbox"]:checked + span {{
    background-color: {_ACCENT} !important;
    border-color: {_ACCENT} !important;
}}
/* checkmark SVG inside box */
[data-baseweb="checkbox"] label > span svg,
[data-testid="stCheckbox"] label > span svg {{
    fill: white !important;
    stroke: white !important;
}}
/* fallback: any dark div used as checkbox body */
[data-testid="stCheckbox"] div[class] {{
    background-color: transparent !important;
}}

/* ── text & number input wrappers ── */
.stTextInput input, .stNumberInput input {{
    background-color: {_WHITE} !important;
    color: {_TEXT} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}

/* ── multiselect chips ── */
[data-baseweb="tag"] {{
    background-color: #e3f2fd !important;
    color: {_ACCENT} !important;
}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def _show_fig(fig: plt.Figure) -> None:
    """Render a matplotlib figure styled for the light theme."""
    fig.patch.set_facecolor(_WHITE)
    for ax in fig.axes:
        ax.set_facecolor("#f5f9ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccdaf0")
        ax.tick_params(colors=_MUTED)
        ax.xaxis.label.set_color("#334e68")
        ax.yaxis.label.set_color("#334e68")
        try:
            ax.title.set_color(_TEXT)
        except Exception:
            pass
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _stat(label: str, value: str, colour: str = _TEXT) -> str:
    return (f'<div class="stat-row">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-value" style="color:{colour}">{value}</span>'
            f'</div>')


def _bar(mu: float, colour: str = _ACCENT) -> str:
    pct = max(0, min(100, int(mu * 100)))
    return (f'<div class="bar-wrap">'
            f'<div class="bar-fill" style="width:{pct}%;background:{colour}"></div>'
            f'</div>')


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _build_pdf(r, mc_runs: int, mc_steps: int) -> bytes:
    """Generate all PDF pages and return bytes. Call once, store in session_state."""
    buf = io.BytesIO()
    with pdf_backend.PdfPages(buf) as pdf_pages:
        _pages = [
            lambda: plot_fuzzy_membership(r.height),
            lambda: plot_fuzzy_age_membership(r.age),
            lambda: plot_fuzzy_output(r.fuzzy_breakdown),
            lambda: plot_comparison(
                grid=r.grid, start=r.start, goal=r.goal,
                visited_bfs=r.bfs_visited,
                path_bfs=list(r.bfs_path) if r.bfs_path else None,
                visited_astar=r.astar_visited,
                path_astar=list(r.astar_path) if r.astar_path else None,
                restricted_zones=list(r.hp_zones),
                risk_level=r.risk_level, suspicion_score=r.suspicion_score,
                prob_matrix=r.prob_matrix, show_prob_overlay=True,
            ),
        ]
        for page_fn in _pages:
            fig = page_fn()
            _stylise_for_pdf(fig)
            pdf_pages.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
        if r.mc_density is not None:
            fig = plot_monte_carlo(
                grid=r.grid, density=r.mc_density, start=r.start,
                n_runs=mc_runs, steps=mc_steps)
            _stylise_for_pdf(fig)
            pdf_pages.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)
    buf.seek(0)
    return buf.read()


def _stylise_for_pdf(fig: plt.Figure) -> None:
    """Apply clean white styling for PDF export."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("#fafafa")


# ═══════════════════════════════════════════════════════════════
# Animated particle-network hero (canvas JS via components.html)
# ═══════════════════════════════════════════════════════════════
_HERO_HTML = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#f9f9f9; overflow:hidden; font-family:'Inter',sans-serif; }}
  canvas {{ position:absolute; top:0; left:0; width:100%; height:100%; }}
  .content {{
      position:relative; z-index:2;
      display:flex; flex-direction:column; align-items:center;
      justify-content:center; height:220px; text-align:center;
      padding: 0 20px;
  }}
  .badge {{
      display:inline-block; margin-bottom:10px;
      background:#e3f2fd; color:{_ACCENT};
      border:1px solid #90caf9; border-radius:20px;
      padding:3px 14px; font-size:0.72rem; font-weight:700;
      letter-spacing:.08em; text-transform:uppercase;
  }}
  h1 {{
      font-size:1.8rem; font-weight:800; color:#1a1a2e;
      letter-spacing:-0.03em; line-height:1.2; margin-bottom:6px;
  }}
  h1 span {{ color:{_ACCENT}; }}
  p {{
      color:#888; font-size:0.82rem; font-weight:400;
      max-width:480px; line-height:1.6;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="content">
  <div class="badge">🔍 jatAIyu · v{__version__}</div>
  <h1>jat<span>AI</span>yu — AI-Assisted <span>Missing Person</span> Finder</h1>
  <p>Fuzzy Logic · BFS &amp; A* · CSP Probability Grid · Monte-Carlo Agent</p>
</div>
<script>
(function(){{
  const canvas = document.getElementById('c');
  const ctx    = canvas.getContext('2d');
  let W, H, pts;

  function resize(){{
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    init();
  }}

  function init(){{
    pts = Array.from({{length: 60}}, () => ({{
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      r:  2 + Math.random() * 3,
    }}));
  }}

  function draw(){{
    ctx.clearRect(0, 0, W, H);
    // Draw edges
    for(let i=0;i<pts.length;i++){{
      for(let j=i+1;j<pts.length;j++){{
        const dx = pts[i].x - pts[j].x;
        const dy = pts[i].y - pts[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if(d < 130){{
          ctx.beginPath();
          ctx.globalAlpha = (1 - d/130) * 0.22;
          ctx.strokeStyle = '#1565C0';
          ctx.lineWidth   = 1;
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }}
      }}
    }}
    // Draw nodes
    ctx.globalAlpha = 1;
    pts.forEach(p => {{
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(21,101,192,0.28)';
      ctx.fill();
      // update
      p.x += p.vx; p.y += p.vy;
      if(p.x < 0 || p.x > W) p.vx *= -1;
      if(p.y < 0 || p.y > H) p.vy *= -1;
    }});
    requestAnimationFrame(draw);
  }}

  window.addEventListener('resize', resize);
  resize();
  draw();
}})();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown(f"""
<div style="padding:14px 0 10px;text-align:center;border-bottom:1px solid {_BORDER};margin-bottom:12px">
  <span style="font-size:1.6rem">🔍</span><br>
  <span style="font-weight:700;font-size:0.95rem;color:{_TEXT}">jatAIyu — Search Parameters</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"<p style='font-size:0.78rem;font-weight:700;color:{_ACCENT};text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>Witness Description</p>", unsafe_allow_html=True)
multi_witness = st.sidebar.checkbox("Multi-Witness Mode (up to 3)", value=False)

if multi_witness:
    n_witnesses = st.sidebar.slider("Number of witnesses", 2, 3, 2)
    st.sidebar.caption("Fuzzy scores from each witness are averaged.")
    witness_inputs = []
    for i in range(n_witnesses):
        st.sidebar.markdown(f"**Witness {i+1}**")
        w_height   = st.sidebar.slider(f"Height #{i+1} (cm)",  140, 195, 175, 1, key=f"wh{i}")
        w_age      = st.sidebar.slider(f"Age #{i+1} (years)",   15,  70,  28, 1, key=f"wa{i}")
        w_behavior = st.sidebar.selectbox(f"Behavior #{i+1}",
            ["calm","normal","nervous","very nervous"], index=2, key=f"wb{i}")
        w_clothing = st.sidebar.selectbox(f"Clothing #{i+1}",
            ["black","blue","brown","gray","green","red","white","yellow"], key=f"wc{i}")
        witness_inputs.append(dict(height=w_height, age=w_age, behavior=w_behavior, clothing=w_clothing))
        st.sidebar.markdown("---")
    height=witness_inputs[0]["height"]; age=witness_inputs[0]["age"]
    behavior=witness_inputs[0]["behavior"]; clothing=witness_inputs[0]["clothing"]
else:
    witness_inputs = None
    height   = st.sidebar.slider("Height (cm)",  140, 195, 175, 1)
    age      = st.sidebar.slider("Age (years)",   15,  70,  28, 1)
    behavior = st.sidebar.selectbox("Observed Behavior",
        ["calm","normal","nervous","very nervous"], index=2)
    clothing = st.sidebar.selectbox("Clothing Color",
        ["black","blue","brown","gray","green","red","white","yellow"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='font-size:0.78rem;font-weight:700;color:{_ACCENT};text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>Search Environment</p>", unsafe_allow_html=True)
grid_size     = st.sidebar.slider("Grid Size (N×N)", 5, 25, 10, 1)
obstacle_pct  = st.sidebar.slider("Obstacle Density (%)", 5, 30, 12, 1)
use_diagonals = st.sidebar.checkbox("8-Directional Movement", value=False)
random_seed   = st.sidebar.number_input("Random Seed (0 = random)", 0, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='font-size:0.78rem;font-weight:700;color:{_ACCENT};text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>Last Seen Location</p>", unsafe_allow_html=True)
use_last_seen    = st.sidebar.checkbox("Use last-seen cell", value=False)
ls_row           = st.sidebar.number_input("Row", 0, grid_size-1, 0) if use_last_seen else 0
ls_col           = st.sidebar.number_input("Col", 0, grid_size-1, 0) if use_last_seen else 0
last_seen        = (int(ls_row), int(ls_col)) if use_last_seen else None
hours_since_seen = st.sidebar.slider("Hours since last seen", 0, 48, 6, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='font-size:0.78rem;font-weight:700;color:{_ACCENT};text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px'>Monte-Carlo</p>", unsafe_allow_html=True)
run_mc   = st.sidebar.checkbox("Run Monte-Carlo", value=True)
mc_runs  = st.sidebar.slider("Number of runs",  10, 200, 50, 10)
mc_steps = st.sidebar.slider("Steps per run",    5,  60, 20,  5)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Hero header (animated particle network)
# ═══════════════════════════════════════════════════════════════
components.html(_HERO_HTML, height=220, scrolling=False)

# ─── "ready" screen ───────────────────────────────────────────
if not run_btn:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem;animation:fadeInUp .5s ease">
      <p style="color:{_MUTED};font-size:0.9rem">
        Configure parameters in the sidebar and press
        <strong style="color:{_ACCENT}">▶ Run Simulation</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════
errors = []
if not (5 <= grid_size <= 25):   errors.append("Grid size must be 5–25.")
if not (100 <= height <= 220):   errors.append("Height must be 100–220 cm.")
if not (10 <= age <= 90):        errors.append("Age must be 10–90 years.")
if errors:
    [st.error(f"⛔ {e}") for e in errors]
    st.stop()


# ═══════════════════════════════════════════════════════════════
# Simulation (session_state cache)
# ═══════════════════════════════════════════════════════════════
_cache_key = (height, age, behavior, clothing, grid_size, obstacle_pct,
              use_diagonals, int(random_seed), last_seen, hours_since_seen,
              run_mc, mc_runs, mc_steps)

if st.session_state.get("_cache_key") != _cache_key:
    # Invalidate PDF bytes whenever inputs change
    st.session_state.pop("pdf_bytes", None)
    with st.spinner("⚙️ Running AI pipeline…"):
        if random_seed > 0:
            random.seed(int(random_seed)); np.random.seed(int(random_seed))
        sim = SearchSimulator(
            height=height, age=age, behavior=behavior, clothing=clothing,
            grid_rows=grid_size, grid_cols=grid_size,
            obstacle_ratio=obstacle_pct/100, use_diagonals=use_diagonals,
            random_seed=int(random_seed) if random_seed > 0 else None,
            last_seen=last_seen, hours_since_seen=float(hours_since_seen),
            mc_runs=mc_runs, mc_steps=mc_steps,
        )
        st.session_state["result"]     = sim.run(run_mc=run_mc)
        st.session_state["_cache_key"] = _cache_key

r = st.session_state["result"]
multi_witness_score = average_witness_score(witness_inputs) if witness_inputs else None

# ─── KPI strip ──────────────────────────────────────────────
RISK_C = {"high": _ACCENT, "medium": _AMBER, "low": _GREEN}
rc     = RISK_C.get(r.risk_level, _TEXT)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Suspicion Score",  f"{r.suspicion_score:.2f}", help="Defuzzified crisp output ∈ [0,1]")
k2.metric("Risk Level",       r.risk_level.upper())
k3.metric("A* Weight",        f"{r.suspicion_weight:.2f}")
k4.metric("Hotspot Zones",    len(r.hp_zones))
k5.metric("Hours Since Seen", f"{hours_since_seen}h")

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🧠 Fuzzy Inference",
    "📅 Age MF",
    "🗺 Probability Map",
    "🔄 BFS vs A*",
    "🎲 Monte-Carlo",
    "📊 Performance",
    "⚙️ Constraints",
])


# ───────────────────────────────────────────────────────────────
# TAB 1 — Fuzzy Inference
# ───────────────────────────────────────────────────────────────
with tabs[0]:
    _section("Fuzzy Logic Inference — Mamdani + Centroid Defuzzification")

    if multi_witness_score is not None:
        st.info(f"👥 **Multi-Witness Mode** — {len(witness_inputs)} witnesses · "
                f"averaged score: **{multi_witness_score:.2f}**")

    col_mf, col_out = st.columns(2)
    with col_mf:
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{_MUTED};text-transform:uppercase;letter-spacing:.06em'>Height Membership Functions</p>", unsafe_allow_html=True)
        _show_fig(plot_fuzzy_membership(r.height))
    with col_out:
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{_MUTED};text-transform:uppercase;letter-spacing:.06em'>Aggregated Output + Centroid Defuzz</p>", unsafe_allow_html=True)
        _show_fig(plot_fuzzy_output(r.fuzzy_breakdown))

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Full Score Breakdown", expanded=False):
        bd = r.fuzzy_breakdown
        c1, c2, c3, c4 = st.columns(4)
        def _breakdown_col(col, title, items: dict, score: float, accent: str):
            with col:
                st.markdown(f"<p style='font-weight:700;color:{accent};font-size:0.85rem;margin-bottom:8px'>{title}</p>", unsafe_allow_html=True)
                html = ""
                for k, v in items.items():
                    html += (f"<div style='margin-bottom:7px'>"
                             f"<div style='display:flex;justify-content:space-between;font-size:0.79rem'>"
                             f"<span style='color:{_TEXT}'>{k.title()}</span>"
                             f"<span style='color:{accent};font-weight:700'>{v:.3f}</span></div>"
                             f"{_bar(v, accent)}</div>")
                html += f"<div style='font-size:0.75rem;color:{_MUTED};margin-top:4px'>Contribution: <strong>{score:.3f}</strong></div>"
                st.markdown(html, unsafe_allow_html=True)

        _breakdown_col(c1, "Height Sets",             bd["height_sets"],   bd["h_score"], "#1565C0")
        _breakdown_col(c2, "Age Sets",                bd["age_sets"],      bd["a_score"], "#00897b")
        _breakdown_col(c3, "Behavior → Output Terms", bd["behavior_sets"], bd["b_score"], "#f57c00")
        _breakdown_col(c4, "Clothing → Output Terms", bd["clothing_sets"], bd["c_score"], "#1976D2")


# ───────────────────────────────────────────────────────────────
# TAB 2 — Age MF
# ───────────────────────────────────────────────────────────────
with tabs[1]:
    _section("Age Membership Functions")
    st.caption("Young + nervous → higher suspicion. Senior + calm → lower suspicion.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        _show_fig(plot_fuzzy_age_membership(r.age))
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        bd = r.fuzzy_breakdown
        colours = {"young": "#00897b", "adult": "#f57c00", "senior": "#6a1b9a"}
        html = ""
        for k, v in bd["age_sets"].items():
            c = colours.get(k, _TEXT)
            html += (f"<div style='margin-bottom:10px'>"
                     f"<div style='display:flex;justify-content:space-between;font-size:0.85rem'>"
                     f"<span style='font-weight:500;color:{_TEXT}'>{k.title()}</span>"
                     f"<span style='font-weight:700;color:{c}'>{v:.3f}</span></div>"
                     f"{_bar(v, c)}</div>")
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(_stat("Age Input",      f"{r.age} years"),       unsafe_allow_html=True)
        st.markdown(_stat("Contribution",   f"{bd['a_score']:.3f}", "#1565C0"), unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 3 — Probability Map
# ───────────────────────────────────────────────────────────────
with tabs[2]:
    _section("Probability Heatmap & Hotspot Zones")
    st.caption(f"Higher intensity = more likely location. Amber = hotspot attractor. Time elapsed: {hours_since_seen}h.")

    col_m, col_c = st.columns([2, 1])
    with col_m:
        _show_fig(plot_single(
            grid=r.grid, start=r.start, goal=r.goal,
            visited=set(), path=None,
            restricted_zones=list(r.hp_zones),
            algorithm_name="Probability Heatmap",
            prob_matrix=r.prob_matrix, show_prob_overlay=True,
        ))
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        g = r.grid
        cells = sorted(
            [(float(r.prob_matrix[row, col]), row, col)
             for row in range(g.rows) for col in range(g.cols)
             if g.grid[row][col] == 0],
            reverse=True)
        st.markdown(f"<p style='font-weight:700;color:{_ACCENT};margin-bottom:8px'>Top-5 Probable Cells</p>", unsafe_allow_html=True)
        for rank, (prob, row, col) in enumerate(cells[:5], 1):
            is_g   = (row, col) == r.goal
            badge  = f" <span style='color:{_AMBER};font-size:0.7rem'>★ GOAL</span>" if is_g else ""
            accent = _AMBER if is_g else _ACCENT
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.82rem'>"
                f"<span style='color:{_TEXT}'>{rank}. ({row},{col}){badge}</span>"
                f"<span style='color:{accent};font-weight:700'>P={prob:.3f}</span></div>"
                f"{_bar(prob, accent)}</div>",
                unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 4 — BFS vs A*
# ───────────────────────────────────────────────────────────────
with tabs[3]:
    _section("BFS vs A* Search Comparison")

    bfs_path_list   = list(r.bfs_path)   if r.bfs_path   else None
    astar_path_list = list(r.astar_path) if r.astar_path else None

    _show_fig(plot_comparison(
        grid=r.grid, start=r.start, goal=r.goal,
        visited_bfs=r.bfs_visited, path_bfs=bfs_path_list,
        visited_astar=r.astar_visited, path_astar=astar_path_list,
        restricted_zones=list(r.hp_zones),
        risk_level=r.risk_level, suspicion_score=r.suspicion_score,
        prob_matrix=r.prob_matrix, show_prob_overlay=True,
    ))

    imp = r.improvement_pct
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"<p style='font-weight:700;color:{_ACCENT2}'>BFS — Breadth-First Search</p>", unsafe_allow_html=True)
        st.markdown(_stat("Nodes Explored", str(r.bfs_nodes)), unsafe_allow_html=True)
        st.markdown(_stat("Path Length",    str(r.bfs_path_len or "No path")), unsafe_allow_html=True)
        st.markdown(_stat("Time",           f"{r.bfs_time_s:.4f} s"), unsafe_allow_html=True)
        if bfs_path_list:
            st.caption("Path: " + " → ".join(str(c) for c in bfs_path_list))
    with cb:
        st.markdown(f"<p style='font-weight:700;color:{_ACCENT}'>A*  (w = {r.suspicion_weight:.2f})</p>", unsafe_allow_html=True)
        imp_c = _GREEN if imp > 0 else _ACCENT
        imp_s = f"{abs(imp):.1f}% {'fewer' if imp > 0 else 'more'}"
        st.markdown(_stat("Nodes Explored",
            f"{r.astar_nodes} <small style='color:{imp_c}'>({imp_s})</small>"), unsafe_allow_html=True)
        st.markdown(_stat("Path Length",    str(r.astar_path_len or "No path")), unsafe_allow_html=True)
        st.markdown(_stat("Time",           f"{r.astar_time_s:.4f} s"), unsafe_allow_html=True)
        if astar_path_list:
            st.caption("Path: " + " → ".join(str(c) for c in astar_path_list))

    st.markdown("<br>", unsafe_allow_html=True)
    if imp > 0:
        st.success(f"✅ A* explored **{imp:.1f}% fewer nodes** than BFS.")
    elif imp < 0:
        st.warning(f"⚠️ A* explored **{abs(imp):.1f}% more nodes** — heuristic less effective here.")
    else:
        st.info("BFS and A* explored the same number of nodes.")

    if not r.bfs_path and not r.astar_path:
        st.error("⛔ No path found by either algorithm — reduce obstacle density or change seed.")
    elif not r.astar_path:
        st.warning("⚠️ A* found no path, but BFS succeeded.")

    with st.expander("🎬 Step-by-step BFS Explorer", expanded=False):
        visited_list = list(r.bfs_visited)
        total = len(visited_list)
        if total > 0:
            frame_idx = st.slider("Exploration step", 1, total, min(10, total))
            _show_fig(plot_single(
                grid=r.grid, start=r.start, goal=r.goal,
                visited=set(visited_list[:frame_idx]),
                path=bfs_path_list if frame_idx == total else None,
                restricted_zones=list(r.hp_zones),
                algorithm_name=f"BFS — Step {frame_idx} / {total}",
                prob_matrix=r.prob_matrix, show_prob_overlay=True,
            ))


# ───────────────────────────────────────────────────────────────
# TAB 5 — Monte-Carlo
# ───────────────────────────────────────────────────────────────
with tabs[4]:
    _section("Monte-Carlo Agent Simulation")

    if not run_mc or r.mc_density is None:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:2.5rem">
          <span style="font-size:2rem">🎲</span>
          <p style="color:{_MUTED};margin-top:8px">
            Enable <strong>Run Monte-Carlo</strong> in the sidebar, then re-run.
          </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.caption(f"{mc_runs} agent simulations × {mc_steps} steps each. Brighter = more visits.")
        col_mc, col_stats = st.columns([2, 1])
        with col_mc:
            _show_fig(plot_monte_carlo(grid=r.grid, density=r.mc_density,
                                       start=r.start, n_runs=mc_runs, steps=mc_steps))
        with col_stats:
            flat_idx = int(r.mc_density.argmax())
            top_r    = flat_idx // r.grid.cols
            top_c    = flat_idx  % r.grid.cols
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-weight:700;color:{_ACCENT};margin-bottom:8px'>Simulation Stats</p>", unsafe_allow_html=True)
            st.markdown(_stat("Runs",               str(mc_runs)), unsafe_allow_html=True)
            st.markdown(_stat("Steps / run",        str(mc_steps)), unsafe_allow_html=True)
            st.markdown(_stat("Peak density cell",  f"({top_r},{top_c})"), unsafe_allow_html=True)
            st.markdown(_stat("Peak density",       f"{r.mc_density[top_r, top_c]:.2f}", _ACCENT), unsafe_allow_html=True)

            st.markdown(f"<p style='font-weight:700;color:{_ACCENT};margin:14px 0 8px'>Top-5 Hotspots</p>", unsafe_allow_html=True)
            count = 0
            for idx in np.argsort(r.mc_density.flatten())[::-1]:
                rr = idx // r.grid.cols;  cc = idx % r.grid.cols
                if r.grid.grid[rr][cc] == 0:
                    d    = r.mc_density[rr, cc]
                    is_g = (rr, cc) == r.goal
                    c    = _AMBER if is_g else _ACCENT
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px'>"
                        f"<span style='color:{_TEXT}'>({rr},{cc}){' ★' if is_g else ''}</span>"
                        f"<span style='color:{c};font-weight:700'>{d:.2f}</span></div>"
                        f"{_bar(d, c)}",
                        unsafe_allow_html=True)
                    count += 1
                    if count >= 5: break


# ───────────────────────────────────────────────────────────────
# TAB 6 — Performance + PDF Export
# ───────────────────────────────────────────────────────────────
with tabs[5]:
    _section("Performance Comparison Dashboard")

    # Bar charts
    fig_perf, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, title, vals, ylabel, cols in [
        (axes[0], "Nodes Explored",      [r.bfs_nodes, r.astar_nodes],                 "Count",   ["#0D47A1","#1976D2"]),
        (axes[1], "Path Length (steps)", [r.bfs_path_len or 0, r.astar_path_len or 0], "Steps",   ["#0D47A1","#1976D2"]),
        (axes[2], "Search Time (s)",     [r.bfs_time_s, r.astar_time_s],               "Seconds", ["#0D47A1","#1976D2"]),
    ]:
        bars = ax.bar(["BFS","A*"], vals, color=cols, width=0.45, zorder=3)
        fmt  = "%.4f" if "Time" in title else "%d"
        ax.bar_label(bars, fmt=fmt, fontsize=9, fontweight="bold", padding=3)
        ax.set_title(title, fontsize=10, fontweight="700")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    plt.tight_layout(pad=2.0)
    _show_fig(fig_perf)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("BFS Nodes",  r.bfs_nodes)
    p2.metric("A* Nodes",   r.astar_nodes,
              delta=f"{r.improvement_pct:+.1f}% vs BFS",
              delta_color="normal" if r.improvement_pct >= 0 else "inverse")
    p3.metric("BFS Path",   r.bfs_path_len or "No path")
    p4.metric("A* Path",    r.astar_path_len or "No path")

    st.markdown("---")
    st.markdown(f"<p style='font-weight:700;color:{_TEXT};margin-bottom:6px'>Simulation Parameters</p>", unsafe_allow_html=True)
    st.json({
        "version": __version__, "height_cm": r.height, "age": r.age,
        "behavior": r.behavior, "clothing": r.clothing,
        "hours_since_seen": hours_since_seen,
        "suspicion_score": r.suspicion_score, "risk_level": r.risk_level,
        "suspicion_weight": r.suspicion_weight,
        "grid_size": f"{r.grid.rows}×{r.grid.cols}",
        "obstacle_ratio": round(obstacle_pct/100, 2),
        "8dir_movement": use_diagonals,
        "goal_cell": list(r.goal), "start_cell": list(r.start),
    })

    # ── PDF Export — FIXED with session_state ─────────────────
    st.markdown("---")
    _section("📄 Export PDF Report")
    st.caption("Generates a multi-page PDF containing all charts and simulation results.")

    gen_col, dl_col = st.columns([1, 1])
    with gen_col:
        if st.button("🖨️  Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF — this may take a moment…"):
                try:
                    st.session_state["pdf_bytes"] = _build_pdf(r, mc_runs, mc_steps)
                    st.session_state["pdf_inputs"] = _cache_key  # track which run
                    st.success("✅ PDF is ready — click Download below.")
                except Exception as exc:
                    st.error(f"⛔ PDF generation failed: {exc}")

    with dl_col:
        # Always show download button if PDF bytes exist (survives reruns)
        if "pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️  Download PDF",
                data=st.session_state["pdf_bytes"],
                file_name="jataiyu_missing_person_search_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ───────────────────────────────────────────────────────────────
# TAB 7 — Constraints
# ───────────────────────────────────────────────────────────────
with tabs[6]:
    _section("Constraint Engine Summary")

    cs = r.constraint_summary
    col_h, col_s = st.columns(2)

    with col_h:
        st.markdown(f"<p style='font-weight:700;color:{_ACCENT};margin-bottom:6px'>🔒 Hard Constraints</p>", unsafe_allow_html=True)
        st.caption("Cells failing any hard constraint are infeasible (P = 0).")
        for hc in cs["hard_constraints"]:
            st.markdown(
                f'<div class="hard-card">'
                f'<span style="font-weight:600;color:{_TEXT}">{hc["name"]}</span>'
                f'<span style="color:{_MUTED};font-size:0.78rem;margin-left:8px">binary gate</span>'
                f'</div>', unsafe_allow_html=True)

    with col_s:
        st.markdown(f"<p style='font-weight:700;color:{_ACCENT2};margin-bottom:6px'>🧲 Soft Constraints</p>", unsafe_allow_html=True)
        st.caption("High score = more likely location for the missing person.")
        for sc in cs["soft_constraints"]:
            st.markdown(
                f'<div class="soft-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:600;color:{_TEXT}">{sc["name"]}</span>'
                f'<span style="color:{_ACCENT2};font-weight:700">w = {sc["weight"]:.3f}</span></div>'
                f'{_bar(sc["weight"], _ACCENT2)}'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='font-weight:700;color:{_AMBER};margin-bottom:6px'>🎯 Hotspot Zones (Attractors — not blocked)</p>", unsafe_allow_html=True)
    st.caption("These zones bias the probability grid toward likely hiding spots.")

    from csp import get_zone_coverage
    for i, (z1, z2, z3, z4) in enumerate(r.hp_zones, 1):
        cov = get_zone_coverage([(z1,z2,z3,z4)], r.grid.rows, r.grid.cols)
        st.markdown(
            f'<div class="zone-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:600;color:{_TEXT}">Zone {i}</span>'
            f'<span style="color:{_MUTED};font-size:0.8rem">'
            f'rows [{z1}–{z3}] · cols [{z2}–{z4}] · {cov*100:.1f}% of grid</span>'
            f'</div></div>', unsafe_allow_html=True)

    total_cov = get_zone_coverage(list(r.hp_zones), r.grid.rows, r.grid.cols)
    st.metric("Total hotspot coverage", f"{total_cov*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="margin-top:3rem;padding:1.2rem 1.6rem;background:{_WHITE};
            border:1px solid {_BORDER};border-radius:10px;
            display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:8px">
  <span style="color:{_MUTED};font-size:0.78rem">
    🎓 jatAIyu: AI-Assisted Missing Person Finder · v{__version__}
  </span>
  <span style="color:{_MUTED};font-size:0.78rem">
    Fuzzy (Mamdani) · BFS + A* · CSP · Monte-Carlo Agent
  </span>
</div>
""", unsafe_allow_html=True)