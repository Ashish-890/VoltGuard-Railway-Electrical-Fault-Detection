from pathlib import Path
from datetime import datetime
from typing import Dict, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

# ================== CONFIG & MODEL LOADING ==================

ROOT = Path(__file__).resolve().parents[1]   # .../railway fault detection
MODEL_PATH = ROOT / "models" / "fault_detector.pkl"


def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="VoltGuard – Railway Electrical Fault Detection",
    page_icon="⚡",
    layout="wide",
)

model = load_model()

# ================== GLOBAL STYLING (CSS) ==================

st.markdown(
    """
    <style>
    /* App background (main area) */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #020617 0, #020617 35%, #020617 55%, #020617 80%, #000000 100%);
        color: #e5e7eb;
    }

    /* Push content down so header is fully visible */
    .block-container {
        padding-top: 2.3rem;      /* more top padding */
        padding-bottom: 1.6rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #020617 30%, #020617 60%, #020617 100%);
        border-right: 1px solid rgba(148,163,184,0.4);
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h4 {
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }

    /* Sidebar nav – hide radio dots */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Sidebar nav row styling */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        border-radius: 0.55rem;
        padding: 0.28rem 0.55rem;
        margin-bottom: 0.15rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(37,99,235,0.35);
    }

    /* Make label text flex so icon + text align in one column */
    [data-testid="stSidebar"] div[role="radiogroup"] > label p {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.9rem;
    }

    /* Selected nav item */
    [data-testid="stSidebar"] div[aria-checked="true"] {
        background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(59,130,246,0.8)) !important;
        box-shadow: 0 0 0 1px rgba(191,219,254,0.8);
        border-radius: 0.55rem;
    }
    [data-testid="stSidebar"] div[aria-checked="true"] p {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    /* Add icons in a fixed column using ::before so all are aligned */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) p::before { content: "🏠"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2) p::before { content: "📡"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3) p::before { content: "🚋"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4) p::before { content: "📷"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5) p::before { content: "📥"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(6) p::before { content: "📶"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(7) p::before { content: "📜"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(8) p::before { content: "🗂"; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(9) p::before { content: "ℹ"; }

    /* Icons column spacing */
    [data-testid="stSidebar"] div[role="radiogroup"] > label p::before {
        width: 1.2rem;
        text-align: center;
    }

    /* Top header bar */
    .top-header {
        margin-top: 0.3rem;
        margin-bottom: 1.0rem;
        padding: 0.9rem 1.2rem;
        border-radius: 1.1rem;
        background: radial-gradient(circle at top left, rgba(59,130,246,0.26), rgba(15,23,42,0.97));
        border: 1px solid rgba(96,165,250,0.7);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .top-left {
        display: flex;
        flex-direction: column;
        gap: 0.18rem;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        color: #cbd5f5;
    }
    .app-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.7);
        color: #e5e7eb;
    }

    .github-pill {
        font-size: 0.8rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(191,219,254,0.8);
        background: rgba(15,23,42,0.92);
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        text-decoration: none;
        color: #e5e7eb;
    }
    .github-pill:hover {
        background: rgba(37,99,235,0.8);
        border-color: rgba(191,219,254,1);
    }

    /* Glass card style */
    .glass-card {
        margin-top: 0.9rem;
        padding: 1.4rem 1.6rem;
        border-radius: 1.2rem;
        background: rgba(15,23,42,0.97);
        border: 1px solid rgba(148,163,184,0.6);
        box-shadow: 0 22px 55px rgba(15,23,42,0.95);
        backdrop-filter: blur(18px);
    }

    /* Section label */
    .section-label {
        font-size: 0.8rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.15rem;
    }

    /* Status badges */
    .badge-ok {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(22,163,74,0.18);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.5);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.75rem;
    }
    .badge-warning {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(249,115,22,0.18);
        color: #fdba74;
        border: 1px solid rgba(251,146,60,0.6);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.75rem;
    }
    .badge-fault {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(239,68,68,0.18);
        color: #fca5a5;
        border: 1px solid rgba(248,113,113,0.6);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.75rem;
    }

    /* Dataframe tweaks */
    .dataframe tbody tr th {
        background-color: transparent;
    }

    /* Small caption under header */
    .crumb {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== NAV STATE ==================

NAV_OPTIONS = [
    "Dashboard",
    "Quick Scan",
    "OHE Diagnostics",
    "Photo Analysis",
    "Bulk Upload",
    "Live Telemetry",
    "History & Analytics",
    "Fault Register",
    "About / GitHub",
]

if "nav" not in st.session_state:
    st.session_state["nav"] = NAV_OPTIONS[0]  # default = Dashboard

# ================== SIDEBAR ==================

with st.sidebar:
    st.markdown("#### 📂 VOLTGUARD NAVIGATION")
    st.write("")

    section = st.radio(
        "",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state["nav"]),
    )
    st.session_state["nav"] = section

    st.markdown("---")
    st.markdown("**Operator Profile**")
    st.markdown("👤 Technician: `Railway OHE / TRD`")
    st.markdown("🔗 [github.com/Ashish-890](https://github.com/Ashish-890)")

    st.caption("Tip: Start with **Quick Scan**, then explore **OHE Diagnostics** and **History**.")

# ================== HISTORY STATE ==================

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts


def add_reading_to_history(
    line_voltage,
    line_current,
    transformer_temp,
    vibration,
    power_factor,
    frequency,
    pred,
    proba,
    source="manual",
):
    history = st.session_state["history"]
    reading_no = len(history) + 1
    history.append(
        {
            "reading_no": reading_no,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "line_voltage_kV": line_voltage,
            "line_current_A": line_current,
            "transformer_temp_C": transformer_temp,
            "vibration_g": vibration,
            "power_factor": power_factor,
            "frequency_Hz": frequency,
            "fault_pred": int(pred),
            "fault_prob": float(proba),
        }
    )

# ================== IMAGE "AI" ANALYZER ==================


def analyze_component_image(img: Image.Image, component_type: str) -> Dict:
    """
    Heuristic 'AI' that:
    - Detects bill / document-like images.
    - Estimates condition using brightness, contrast, edge strength.
    """
    gray = img.convert("L").resize((256, 256))
    arr = np.array(gray) / 255.0

    mean_brightness = float(arr.mean())
    contrast = float(arr.std())

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges) / 255.0
    edge_strength = float(edges_arr.mean())

    white_fraction = float((arr > 0.9).mean())

    looks_like_document = (
        white_fraction > 0.5
        and contrast < 0.28
        and 0.08 < edge_strength < 0.35
    )

    if looks_like_document:
        return {
            "status": "Not a component image",
            "severity": 0.0,
            "summary": "This image looks more like a **document or bill** than a physical railway component.",
            "issues": [
                "Large bright/white background like paper",
                "Edge patterns resemble text/lines, not hardware geometry",
            ],
            "recommendation": (
                "Upload a close-up photo of the actual component "
                f"(e.g., {component_type.lower()}, clamp, bushing, insulator) instead of a document."
            ),
            "metrics": {
                "mean_brightness": mean_brightness,
                "contrast": contrast,
                "edge_strength": edge_strength,
                "white_fraction": white_fraction,
                "detected_type": "document-like",
            },
        }

    issues: List[str] = []
    severity = 0.0

    if mean_brightness < 0.25:
        issues.append("Dark / burnt-looking zones detected on surface")
        severity += 0.45
    elif mean_brightness > 0.85:
        issues.append("Very bright / washed-out image (flash or glare)")
        severity += 0.2

    if contrast > 0.28:
        issues.append("High contrast – stains, deposits or hotspot patterns")
        severity += 0.25

    if edge_strength > 0.22:
        issues.append("Strong edges – possible cracks, chipped parts or frayed strands")
        severity += 0.25

    if not issues:
        issues.append("No obvious damage patterns detected from this image")

    severity = max(0.0, min(severity, 1.0))

    if severity < 0.3:
        status = "Normal"
        summary = f"The **{component_type}** appears visually healthy."
        recommendation = "Keep under routine inspection. No immediate action needed."
    elif severity < 0.6:
        status = "Needs Attention"
        summary = f"The **{component_type}** shows minor anomalies worth checking."
        recommendation = "Plan a closer inspection during the next maintenance block."
    else:
        status = "Faulty"
        summary = f"The **{component_type}** likely has significant damage / contamination."
        recommendation = (
            "Treat as a high-priority check. Follow OHE/traction safety SOP before any intervention."
        )

    return {
        "status": status,
        "severity": severity,
        "summary": summary,
        "issues": issues,
        "recommendation": recommendation,
        "metrics": {
            "mean_brightness": mean_brightness,
            "contrast": contrast,
            "edge_strength": edge_strength,
            "white_fraction": white_fraction,
            "detected_type": "component-like",
        },
    }

# ================== TOP HEADER ==================

st.markdown(
    f"""
    <div class="top-header">
        <div class="top-left">
            <div class="app-title">VoltGuard – Railway Electrical Fault Detection</div>
            <div class="app-subtitle">
                Condition monitoring for OHE, transformers & traction equipment – ML + visual inspection + analytics.
            </div>
            <div class="crumb">
                Current view: <strong>{st.session_state["nav"]}</strong>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.6rem;">
            <div class="app-pill">
                <span>🧠 Model:</span><span>Random Forest · Demo Data</span>
            </div>
            <a class="github-pill" href="https://github.com/Ashish-890" target="_blank">
                <span>🐙 GitHub</span>
                <span>/Ashish-890</span>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section = st.session_state["nav"]

# ================== SECTION: DASHBOARD ==================

if section == "Dashboard":
    st.markdown(
        """
        VoltGuard gives a **control-room style** overview of your simulated railway electrical health.

        • Use **Quick Scan** for point readings.  
        • Use **Photo Analysis** to visually inspect components.  
        • Use **OHE Diagnostics** & **History** to understand trends.
        """
    )

    hist = st.session_state["history"]
    if hist:
        df_hist = pd.DataFrame(hist)
        total = len(df_hist)
        fault_count = int((df_hist["fault_pred"] == 1).sum())
        warning_count = int(((df_hist["fault_prob"] >= 0.4) & (df_hist["fault_pred"] == 0)).sum())
        ok_count = total - fault_count - warning_count
        avg_prob = float(df_hist["fault_prob"].mean())
    else:
        total = fault_count = warning_count = ok_count = 0
        avg_prob = 0.0

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>SYSTEM SUMMARY</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Readings", total)
    c2.metric("OK", ok_count)
    c3.metric("Warning (prob ≥ 0.4)", warning_count)
    c4.metric("Faults", fault_count)
    st.progress(avg_prob if avg_prob else 0.01)
    st.caption(f"Average fault probability across all readings: `{avg_prob:.2f}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>NAVIGATION SHORTCUTS</div>", unsafe_allow_html=True)
    st.subheader("⚡ Quick Actions")

    col_q1, col_q2, col_q3 = st.columns(3)

    with col_q1:
        st.write("📡 **Quick Scan**")
        st.caption("Set sensor readings and classify condition instantly.")
        if st.button("Open Quick Scan", key="btn_open_quickscan"):
            st.session_state["nav"] = "Quick Scan"
            st.rerun()

    with col_q2:
        st.write("📷 **Photo Analysis**")
        st.caption("Upload insulators, clamps, bushings, etc. for visual assessment.")
        if st.button("Go to Photo Analysis", key="btn_open_photo"):
            st.session_state["nav"] = "Photo Analysis"
            st.rerun()

    with col_q3:
        st.write("📥 **Bulk Upload**")
        st.caption("Run batch health checks from a CSV log file.")
        if st.button("Open Bulk Upload", key="btn_open_bulk"):
            st.session_state["nav"] = "Bulk Upload"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: QUICK SCAN ==================

elif section == "Quick Scan":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>LIVE CHECK</div>", unsafe_allow_html=True)
    st.subheader("📡 Quick Scan – Simulated Sensor Readings")

    col1, col2 = st.columns(2)

    with col1:
        line_voltage = st.slider(
            "Line Voltage (kV)",
            min_value=20.0,
            max_value=30.0,
            value=25.0,
            step=0.1,
        )
        line_current = st.slider(
            "Line Current (A)",
            min_value=100,
            max_value=800,
            value=320,
            step=10,
        )
        transformer_temp = st.slider(
            "Transformer Temperature (°C)",
            min_value=30,
            max_value=120,
            value=65,
            step=1,
        )

    with col2:
        vibration = st.slider(
            "Vibration (g)",
            min_value=0.2,
            max_value=3.0,
            value=0.9,
            step=0.1,
        )
        power_factor = st.slider(
            "Power Factor",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.01,
        )
        frequency = st.slider(
            "Frequency (Hz)",
            min_value=48.0,
            max_value=52.0,
            value=50.0,
            step=0.1,
        )

    col_analyze, col_simulate = st.columns(2)

    with col_analyze:
        analyze_clicked = st.button("🔍 Analyze Fault Risk", use_container_width=True)

    with col_simulate:
        simulate_clicked = st.button("▶ Simulate Live Data (20 readings)", use_container_width=True)

    if analyze_clicked:
        input_df = pd.DataFrame(
            [
                {
                    "line_voltage_kV": line_voltage,
                    "line_current_A": line_current,
                    "transformer_temp_C": transformer_temp,
                    "vibration_g": vibration,
                    "power_factor": power_factor,
                    "frequency_Hz": frequency,
                }
            ]
        )

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        add_reading_to_history(
            line_voltage,
            line_current,
            transformer_temp,
            vibration,
            power_factor,
            frequency,
            pred,
            proba,
            source="quick_scan",
        )

        st.markdown("---")
        st.markdown("<div class='section-label'>RESULT</div>", unsafe_allow_html=True)
        st.subheader("🔎 VoltGuard Assessment")

        if pred == 0 and proba < 0.4:
            st.markdown("<span class='badge-ok'>OK</span>", unsafe_allow_html=True)
            st.write("No critical electrical fault detected for this operating point.")
        elif pred == 0 and proba >= 0.4:
            st.markdown("<span class='badge-warning'>WARNING</span>", unsafe_allow_html=True)
            st.write("Readings are within limits, but fault risk indicators are elevated.")
        else:
            st.markdown("<span class='badge-fault'>FAULT</span>", unsafe_allow_html=True)
            st.write(
                "- Inspect traction transformer and OHE section.\n"
                "- Check for overcurrent / overheating / abnormal vibration.\n"
                "- Verify relay and protection settings."
            )

        st.write("")
        st.write("**Fault probability (model confidence):**")
        st.progress(float(proba))
        st.write(f"`{proba:.2f}`  (0 = no fault, 1 = high fault risk)")

        st.markdown("### 🔧 Sensor Health Gauges")

        voltage_score = (line_voltage - 20.0) / (30.0 - 20.0)
        voltage_score = float(min(max(voltage_score, 0.0), 1.0))

        temp_score = (transformer_temp - 30.0) / (80.0 - 30.0)
        temp_score = float(min(max(temp_score, 0.0), 1.0))

        vib_score = vibration / 2.5
        vib_score = float(min(max(vib_score, 0.0), 1.0))

        g1, g2, g3 = st.columns(3)
        with g1:
            st.caption("Line Voltage Utilization")
            st.progress(int(voltage_score * 100))
            st.write(f"{line_voltage:.1f} kV")

        with g2:
            st.caption("Transformer Temperature Stress")
            st.progress(int(temp_score * 100))
            st.write(f"{transformer_temp:.1f} °C")

        with g3:
            st.caption("Vibration Level")
            st.progress(int(vib_score * 100))
            st.write(f"{vibration:.2f} g")

    if simulate_clicked:
        base = {
            "line_voltage_kV": line_voltage,
            "line_current_A": line_current,
            "transformer_temp_C": transformer_temp,
            "vibration_g": vibration,
            "power_factor": power_factor,
            "frequency_Hz": frequency,
        }

        for _ in range(20):
            sim_values = {
                "line_voltage_kV": base["line_voltage_kV"] + np.random.normal(0, 0.3),
                "line_current_A": base["line_current_A"] + np.random.normal(0, 40),
                "transformer_temp_C": base["transformer_temp_C"] + np.random.normal(0, 3),
                "vibration_g": base["vibration_g"] + np.random.normal(0, 0.15),
                "power_factor": np.clip(
                    base["power_factor"] + np.random.normal(0, 0.02), 0.5, 1.0
                ),
                "frequency_Hz": base["frequency_Hz"] + np.random.normal(0, 0.15),
            }

            sim_df = pd.DataFrame([sim_values])
            sim_pred = model.predict(sim_df)[0]
            sim_proba = model.predict_proba(sim_df)[0][1]

            add_reading_to_history(
                sim_values["line_voltage_kV"],
                sim_values["line_current_A"],
                sim_values["transformer_temp_C"],
                sim_values["vibration_g"],
                sim_values["power_factor"],
                sim_values["frequency_Hz"],
                sim_pred,
                sim_proba,
                source="simulation",
            )

        st.success("✅ Simulated 20 live readings and added them to history.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: OHE DIAGNOSTICS ==================

elif section == "OHE Diagnostics":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>OHE HEALTH</div>", unsafe_allow_html=True)
    st.subheader("🚋 OHE Diagnostics – Overhead Equipment Health")

    st.markdown(
        """
        This view interprets your sensor history as **Overhead Equipment (OHE)** health:

        - **OHE Voltage Health** – stability around nominal 25 kV  
        - **Catenary Load Index** – average loading of contact wire  
        - **Insulator Condition Index** – stress from temperature & vibration  
        - **Clamp / Bracket Stress** – current + mechanical vibration impact  
        - **Flashover Risk Score** – pollution, overvoltage and heat combined  
        """
    )

    history = st.session_state["history"]

    if not history:
        st.info("No readings yet. Use **Quick Scan** a few times, then return here.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(history)

        N = 50
        df_tail = df.tail(N).copy()

        nominal_kv = 25.0
        df_tail["voltage_dev"] = np.abs(df_tail["line_voltage_kV"] - nominal_kv)
        voltage_dev_mean = float(df_tail["voltage_dev"].mean())
        voltage_dev_norm = min(voltage_dev_mean / 2.0, 1.0)
        ohe_voltage_health = 1.0 - voltage_dev_norm

        df_tail["current_norm"] = df_tail["line_current_A"] / 800.0
        catenary_load_index = float(np.clip(df_tail["current_norm"].mean(), 0.0, 1.0))

        temp_norm = np.clip((df_tail["transformer_temp_C"] - 40) / 50.0, 0.0, 1.0)
        vib_norm = np.clip(df_tail["vibration_g"] / 2.5, 0.0, 1.0)
        insulator_stress = 0.6 * temp_norm + 0.4 * vib_norm
        insulator_condition_index = float(1.0 - np.clip(insulator_stress.mean(), 0.0, 1.0))

        clamp_stress_raw = 0.7 * df_tail["current_norm"] + 0.3 * vib_norm
        clamp_stress_index = float(np.clip(clamp_stress_raw.mean(), 0.0, 1.0))

        pf_penalty = np.clip(0.9 - df_tail["power_factor"], 0.0, 0.4) * 2.5
        temp_penalty = np.clip((df_tail["transformer_temp_C"] - 70) / 30.0, 0.0, 1.0)
        volt_penalty = np.clip(df_tail["voltage_dev"] / 2.0, 0.0, 1.0)
        prob_penalty = np.clip(df_tail["fault_prob"], 0.0, 1.0)
        flashover_raw = (
            0.3 * temp_penalty
            + 0.25 * volt_penalty
            + 0.25 * pf_penalty
            + 0.2 * prob_penalty
        )
        flashover_risk = float(np.clip(flashover_raw.mean(), 0.0, 1.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("OHE Voltage Health", f"{int(ohe_voltage_health * 100)}%", "Higher is better")
        c2.metric("Catenary Load Index", f"{int(catenary_load_index * 100)}%", "Target ~40–70%")
        c3.metric("Flashover Risk", f"{int(flashover_risk * 100)}%", "Lower is better")

        st.markdown("---")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.caption("Insulator Condition Index")
            st.progress(int(insulator_condition_index * 100))
            st.write(f"{insulator_condition_index:.2f}")
        with g2:
            st.caption("Clamp / Bracket Stress")
            st.progress(int(clamp_stress_index * 100))
            st.write(f"{clamp_stress_index:.2f}")
        with g3:
            st.caption("Avg Fault Probability (last 50)")
            avg_prob = float(df_tail["fault_prob"].mean())
            st.progress(int(avg_prob * 100))
            st.write(f"{avg_prob:.2f}")

        st.markdown("### Recent OHE Voltage & Current")
        chart_df = df_tail.set_index("reading_no")[
            ["line_voltage_kV", "line_current_A"]
        ]
        st.line_chart(chart_df, use_container_width=True)

        st.markdown("### Interpretation Notes")
        st.write(
            "- **OHE Voltage Health** < 80% → investigate voltage regulation or feeder switching patterns.\n"
            "- **Catenary Load Index** > 80% → heavy loading, check timetable peaks and feeder balancing.\n"
            "- **Insulator Condition Index** < 70% → consider insulator washing or replacement planning.\n"
            "- **Clamp / Bracket Stress** high → inspect fittings, droppers and contact wire tension.\n"
            "- **Flashover Risk** > 40% → check for pollution, bird droppings and tracking marks."
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: PHOTO ANALYSIS ==================

elif section == "Photo Analysis":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>VISUAL INSPECTION</div>", unsafe_allow_html=True)
    st.subheader("📷 Photo Analysis – Component Visual Inspection")

    left, right = st.columns([1.1, 1])

    with left:
        image_file = st.file_uploader(
            "Upload a close-up photo of a railway electrical component",
            type=["png", "jpg", "jpeg"],
        )

        component_type = st.selectbox(
            "Component type",
            [
                "Insulator",
                "Transformer bushing",
                "Contact wire / catenary clamp",
                "Pantograph strip",
                "Cable termination / lug",
                "Switchgear / panel surface",
                "Other railway electrical component",
            ],
        )

        run_inspection = st.button("🧠 Run Visual Analysis", use_container_width=True)

    with right:
        if image_file is not None:
            try:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded image", use_column_width=True)
            except Exception:
                st.warning("Could not open image – please upload a valid PNG/JPG.")
        else:
            st.info("Upload a component image on the left to start analysis.")

    st.markdown("---")

    if run_inspection:
        if image_file is None:
            st.warning("Please upload an image first.")
        else:
            img = Image.open(image_file)
            result = analyze_component_image(img, component_type)

            st.subheader("🔎 VoltGuard Visual Assessment")

            if result["status"] == "Not a component image":
                st.warning("This looks like a document or bill, not an electrical component.")
                st.write(result["summary"])
            else:
                if result["status"] == "Normal":
                    st.markdown("<span class='badge-ok'>NORMAL</span>", unsafe_allow_html=True)
                elif result["status"] == "Faulty":
                    st.markdown("<span class='badge-fault'>FAULTY</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-warning'>NEEDS ATTENTION</span>", unsafe_allow_html=True)

                st.write(result["summary"])

            st.write("")
            st.write("**Severity estimate:**")
            st.progress(float(result["severity"]))
            st.write(f"`{result['severity']:.2f}`  (0 = clean, 1 = high risk)")

            st.write("**Observed visual cues:**")
            for issue in result["issues"]:
                st.write(f"- {issue}")

            st.write("")
            st.write("**Maintenance recommendation:**")
            st.write(result["recommendation"])

            st.write("")
            with st.expander("Show raw analysis metrics"):
                st.json(result["metrics"])

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: BULK UPLOAD ==================

elif section == "Bulk Upload":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>BATCH MODE</div>", unsafe_allow_html=True)
    st.subheader("📥 Bulk Upload – Batch Analysis")

    st.write(
        "Upload a CSV of multiple readings (e.g. from loggers). "
        "VoltGuard will classify each row as **OK / Warning / Fault** using rule-based logic."
    )

    # Sample CSV download
    st.markdown("#### 📎 Need a sample file?")
    sample_df = pd.DataFrame(
        [
            [25.0, 320, 65, 0.9, 0.96, 50.0],
            [24.6, 410, 72, 1.1, 0.93, 49.8],
            [25.3, 580, 78, 1.4, 0.91, 49.6],
            [26.1, 730, 88, 1.6, 0.89, 50.5],
            [23.9, 290, 60, 0.7, 0.97, 50.1],
            [24.8, 640, 82, 1.3, 0.92, 49.3],
            [25.5, 510, 76, 1.2, 0.94, 50.2],
            [27.0, 760, 90, 1.8, 0.88, 51.0],
            [24.2, 350, 69, 1.0, 0.95, 50.4],
            [25.1, 430, 73, 1.1, 0.93, 49.9],
        ],
        columns=[
            "line_voltage_kV",
            "line_current_A",
            "transformer_temp_C",
            "vibration_g",
            "power_factor",
            "frequency_Hz",
        ],
    )

    sample_csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download sample CSV (bulk_sample.csv)",
        data=sample_csv_bytes,
        file_name="bulk_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    csv_file = st.file_uploader("Upload CSV file", type=["csv"])

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)

            required_cols = {
                "line_voltage_kV",
                "line_current_A",
                "transformer_temp_C",
                "vibration_g",
                "power_factor",
                "frequency_Hz",
            }

            if not required_cols.issubset(df.columns):
                st.error(
                    "CSV is missing required columns. "
                    f"Expected at least: {', '.join(required_cols)}"
                )
            else:
                if st.button("🔍 Run Bulk Analysis"):
                    def classify_row(row):
                        status = "OK"
                        reason = "Within normal limits"

                        if row["transformer_temp_C"] > 85 or row["line_current_A"] > 720:
                            status = "Fault"
                            reason = "Overheat / Overcurrent"
                        elif (
                            row["transformer_temp_C"] > 75
                            or row["line_current_A"] > 600
                            or abs(row["frequency_Hz"] - 50) > 1.5
                        ):
                            status = "Warning"
                            reason = "Parameters close to limit"

                        return pd.Series({"bulk_status": status, "bulk_reason": reason})

                    results = df.apply(classify_row, axis=1)
                    df_out = pd.concat([df, results], axis=1)

                    st.success("✅ Bulk analysis completed.")
                    st.dataframe(df_out, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Upload a CSV file to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: LIVE TELEMETRY ==================

elif section == "Live Telemetry":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>IOT VIEW (DEMO)</div>", unsafe_allow_html=True)
    st.subheader("📶 Live Telemetry – Simulated IoT Feed")

    st.write(
        "In a real deployment, VoltGuard would read data via Modbus/MQTT APIs. "
        "Here we simulate a short time-series for voltage, current and temperature."
    )

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        equipment_id = st.text_input("Equipment ID", "TRF-01")
        location = st.text_input("Location", "Yard 1, Pune")
        connect_btn = st.button("🔌 Simulate Connection")

    if connect_btn:
        st.success(f"Connected to {equipment_id} at {location} (simulated).")

    st.markdown("---")
    st.markdown("### Sample Telemetry Snapshot")

    time_index = pd.RangeIndex(start=0, stop=30)
    voltage_series = 25 + np.random.normal(0, 0.3, size=len(time_index))
    current_series = 350 + np.random.normal(0, 40, size=len(time_index))
    temp_series = 70 + np.random.normal(0, 3, size=len(time_index))

    df_stream = pd.DataFrame(
        {
            "t": time_index,
            "line_voltage_kV": voltage_series,
            "line_current_A": current_series,
            "transformer_temp_C": temp_series,
        }
    ).set_index("t")

    st.line_chart(df_stream, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: HISTORY & ANALYTICS ==================

elif section == "History & Analytics":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>TIME SERIES</div>", unsafe_allow_html=True)
    st.subheader("📜 History & Analytics")

    history = st.session_state["history"]

    if history:
        df_hist = pd.DataFrame(history)

        st.write("Recent readings:")
        st.dataframe(
            df_hist[
                [
                    "reading_no",
                    "timestamp",
                    "source",
                    "line_voltage_kV",
                    "line_current_A",
                    "transformer_temp_C",
                    "vibration_g",
                    "power_factor",
                    "frequency_Hz",
                    "fault_pred",
                    "fault_prob",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("### Trends: Voltage, Current & Temperature")
        chart_df = df_hist.set_index("reading_no")[
            ["line_voltage_kV", "line_current_A", "transformer_temp_C"]
        ]
        st.line_chart(chart_df, use_container_width=True)

        st.markdown("### Vibration & Power Factor")
        vib_pf_df = df_hist.set_index("reading_no")[
            ["vibration_g", "power_factor"]
        ]
        st.line_chart(vib_pf_df, use_container_width=True)

        st.markdown("### Fault Probability Over Readings")
        prob_df = df_hist.set_index("reading_no")[["fault_prob"]]
        st.line_chart(prob_df, use_container_width=True)
    else:
        st.info("No readings yet. Use **Quick Scan** a few times, then revisit this page.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: FAULT REGISTER ==================

elif section == "Fault Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>FAULT LOG</div>", unsafe_allow_html=True)
    st.subheader("🗂 Fault Register")

    history = st.session_state["history"]

    if history:
        df_hist = pd.DataFrame(history)
        df_faults = df_hist[df_hist["fault_pred"] == 1]

        if df_faults.empty:
            st.success("No fault events recorded yet. ✅ System looks healthy in this session.")
        else:
            st.write("Open fault-like events (based on model prediction):")
            st.dataframe(
                df_faults[
                    [
                        "reading_no",
                        "timestamp",
                        "source",
                        "line_voltage_kV",
                        "line_current_A",
                        "transformer_temp_C",
                        "vibration_g",
                        "power_factor",
                        "frequency_Hz",
                        "fault_prob",
                    ]
                ],
                use_container_width=True,
            )
            st.caption(
                "In a full system, each entry would be assignable to technicians with notes and close-out workflow."
            )
    else:
        st.info("No history yet. Use **Quick Scan** to generate some readings.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================== SECTION: ABOUT / GITHUB ==================

else:  # "About / GitHub"
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<div class='section-label'>PROJECT INFO</div>", unsafe_allow_html=True)
    st.subheader("ℹ About VoltGuard")

    st.markdown(
        """
        **VoltGuard** is a concept application for **Railway Electrical Fault Detection** with:

        - 🔬 ML-based classification of sensor readings (Random Forest demo model).  
        - 📷 On-device visual inspection using simple image heuristics.  
        - 🚋 OHE diagnostics: voltage health, catenary load, insulator & clamp stress, flashover risk.  
        - 📡 Quick Scan, bulk CSV analysis & simulated IoT telemetry.  
        - 📜 History, analytics and a basic fault register view.  

        You can present this as:

        > “End-to-end railway electrical fault detection demo with VoltGuard branding –  
        > ML classification + image inspection + OHE diagnostics + telemetry dashboard, built in Streamlit.”
        """
    )

    st.markdown("### 👨‍💻 Made by **ASHISH TRIPATHI**")
    st.markdown("GitHub: 🔗 [github.com/Ashish-890](https://github.com/Ashish-890)")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("⚡ VoltGuard demo – uses synthetic & heuristic data only. Not for real-world critical safety decisions.")
