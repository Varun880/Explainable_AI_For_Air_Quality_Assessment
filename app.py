"""
=============================================================
  Explainable AI for Air Quality Assessment — Delhi
  Streamlit Dashboard  (Dark Theme)
=============================================================
  Setup (one-time):
      pip install streamlit xgboost shap scikit-learn joblib pandas numpy matplotlib

  Run:
      streamlit run app.py

  Required files in the SAME folder as app.py:
      outputs/xgboost_tuned_model.pkl
      outputs/feature_scaler.pkl
      outputs/feature_columns.pkl
      Delhi_Air_Quality_Dataset.csv   (optional — needed for dependence plots)
=============================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

# ── Write .streamlit/config.toml so Streamlit uses dark base theme ─────────────
_base     = os.path.dirname(os.path.abspath(__file__))
_cfg_dir  = os.path.join(_base, ".streamlit")
_cfg_path = os.path.join(_cfg_dir, "config.toml")
os.makedirs(_cfg_dir, exist_ok=True)
with open(_cfg_path, "w") as _f:
    _f.write("""[theme]
base = "dark"
backgroundColor = "#0D1B2A"
secondaryBackgroundColor = "#162535"
textColor = "#E8F4F8"
primaryColor = "#00B4D8"
font = "sans serif"
""")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi AQI — Explainable AI Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: lock every element to dark theme ──────────────────────────────────────
st.markdown("""
<style>
html, body, #root,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container,
section.main                                 { background-color: #0D1B2A !important; }

html, body, .stApp, .stApp *,
p, li, span, div, label, small,
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
[data-testid="stText"] *                     { color: #E8F4F8 !important; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child  { background-color: #0A1520 !important; }
[data-testid="stSidebar"] *                  { color: #E8F4F8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3                 { color: #00B4D8 !important; }

div[data-baseweb="select"] > div             { background-color: #162535 !important;
                                               border-color: #2A4060 !important; }
div[data-baseweb="select"] *                 { color: #E8F4F8 !important; }
div[data-baseweb="popover"],
div[data-baseweb="popover"] *                { background-color: #162535 !important;
                                               color: #E8F4F8 !important; }
li[role="option"]:hover                      { background-color: #1E3A50 !important; }

[data-testid="stMetric"]                     { background-color: #162535 !important;
                                               border: 1px solid #2A4060 !important;
                                               border-radius: 10px !important;
                                               padding: 16px !important; }
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"] *              { color: #E8F4F8 !important; }

[data-testid="stDataFrame"]                  { background-color: #162535 !important;
                                               border-radius: 8px !important; }
.dvn-scroller                                { background-color: #162535 !important; }

hr                                           { border-color: #2A4060 !important; }

.stButton > button                           { background-color: #00B4D8 !important;
                                               color: #0D1B2A !important;
                                               font-weight: 700 !important;
                                               border: none !important;
                                               border-radius: 8px !important; }
.stButton > button:hover                     { background-color: #0099BB !important; }

header[data-testid="stHeader"]               { background-color: #0D1B2A !important; }
[data-testid="stInfo"]                       { background-color: #162535 !important;
                                               border-color: #00B4D8 !important; }

::-webkit-scrollbar                          { width: 6px; height: 6px; }
::-webkit-scrollbar-track                    { background: #0D1B2A; }
::-webkit-scrollbar-thumb                    { background: #2A4060; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark rcParams ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "#162535",
    "axes.facecolor":     "#162535",
    "axes.edgecolor":     "#6A8AAA",
    "axes.labelcolor":    "#C8D8E8",
    "text.color":         "#C8D8E8",
    "xtick.color":        "#A0B8CC",
    "ytick.color":        "#A0B8CC",
    "xtick.labelcolor":   "#A0B8CC",
    "ytick.labelcolor":   "#A0B8CC",
    "grid.color":         "#2A4060",
    "legend.facecolor":   "#1A2E44",
    "legend.edgecolor":   "#2A4060",
    "legend.labelcolor":  "#C8D8E8",
    "figure.dpi":         120,
    "savefig.facecolor":  "#162535",
    "savefig.edgecolor":  "#162535",
    "axes.titlecolor":    "#E8F4F8",
})

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    out       = os.path.join(_base, "outputs")
    model     = joblib.load(os.path.join(out, "xgboost_tuned_model.pkl"))
    scaler    = joblib.load(os.path.join(out, "feature_scaler.pkl"))
    columns   = joblib.load(os.path.join(out, "feature_columns.pkl"))
    explainer = shap.TreeExplainer(model)
    return model, scaler, columns, explainer

try:
    model, scaler, FEATURE_COLS, explainer = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Could not load model artifacts.\n{e}\n\n"
             "Make sure the `outputs/` folder is in the same directory as `app.py`.")
    st.stop()

# ── Utility functions ──────────────────────────────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:   return "Good",         "#27AE60", "😊"
    if aqi <= 100:  return "Satisfactory", "#2ECC71", "🙂"
    if aqi <= 200:  return "Moderate",     "#F39C12", "😐"
    if aqi <= 300:  return "Poor",         "#E67E22", "😷"
    if aqi <= 400:  return "Very Poor",    "#C0392B", "🤧"
    return               "Severe",         "#7B241C", "☠️"

def policy_message(shap_vals, feature_names):
    sv  = dict(zip(feature_names, shap_vals))
    top = max(sv, key=lambda k: sv[k])
    msgs = {
        "PM2.5":       ("Fine Particulate Matter (PM2.5)",
                        "Impose immediate construction bans and restrict heavy vehicle movement. "
                        "Coordinate crop stubble burning bans in neighboring states."),
        "PM10":        ("Coarse Particulate Matter (PM10)",
                        "Enforce dust suppression on construction sites and major roads. "
                        "Deploy water sprinklers. Restrict open waste burning."),
        "AQI_Lag_1":   ("Legacy Pollution — Yesterday's AQI",
                        "Multi-day smog event in progress. Short-term single-day interventions are "
                        "insufficient. Sustained emission controls required to break the pollution cycle."),
        "NO2":         ("Nitrogen Dioxide (NO₂)",
                        "Vehicular & industrial emission levels elevated. "
                        "Consider odd-even traffic scheme and industrial activity curfews."),
        "CO":          ("Carbon Monoxide (CO)",
                        "Combustion sources elevated. Restrict biomass burning and industrial furnaces."),
        "Month_Cos":   ("Winter Seasonal Conditions",
                        "Temperature inversions trapping pollutants near ground level. "
                        "Issue public health advisories and pre-emptive school closures."),
        "Month_Sin":   ("Seasonal Transition Period",
                        "Seasonal atmospheric conditions amplifying pollution impact. "
                        "Enhanced monitoring and early warning systems recommended."),
        "PM2.5_Lag_1": ("Residual Fine Particles from Yesterday",
                        "Carryover PM2.5 compounding today's readings. "
                        "Sustained PM2.5 source controls required."),
    }
    label, action = msgs.get(top, (top, "Review pollutant levels and implement targeted controls."))
    return top, label, action

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Input Pollutant Levels")
    st.markdown("Adjust sliders to simulate a pollution scenario.")
    st.markdown("---")

    pm25 = st.slider("PM2.5 (µg/m³)",                0.0,  500.0,  80.0, 1.0)
    pm10 = st.slider("PM10 (µg/m³)",                 0.0,  700.0, 150.0, 1.0)
    no2  = st.slider("NO₂ (µg/m³)",                  0.0,  250.0,  45.0, 1.0)
    co   = st.slider("CO (mg/m³)",                    0.0,   30.0,   1.2, 0.1)
    st.markdown("---")
    month    = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
        index=10,
    )
    aqi_lag  = st.slider("Yesterday's AQI  (AQI_Lag_1)",     0,   500, 200,  1)
    pm25_lag = st.slider("Yesterday's PM2.5  (PM2.5_Lag_1)", 0.0, 500.0, 70.0, 1.0)
    st.markdown("---")
    st.markdown(
        "<small style='color:#4A6080;'>Model: XGBoost Tuned&nbsp;&nbsp;R²=0.941<br>"
        "Test set: full year 2024 (366 days)</small>",
        unsafe_allow_html=True,
    )

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#E8F4F8;margin-bottom:2px;'>🌫️ Delhi AQI — Explainable AI Dashboard</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='color:#94A3B8;font-size:15px;margin-top:0;'>"
    "Predict Delhi's Air Quality Index using <b style='color:#00B4D8;'>XGBoost (R²=0.941)</b> "
    "and understand <i>why</i> with <b style='color:#00B4D8;'>SHAP explainability</b>. "
    "Adjust inputs in the sidebar — results update instantly.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Run inference ──────────────────────────────────────────────────────────────
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

input_df = pd.DataFrame([{
    "PM2.5": pm25, "PM10": pm10, "NO2": no2, "CO": co,
    "Month_Sin": month_sin, "Month_Cos": month_cos,
    "AQI_Lag_1": float(aqi_lag), "PM2.5_Lag_1": pm25_lag,
}])[FEATURE_COLS]

scaled    = scaler.transform(input_df)
aqi_pred  = float(model.predict(scaled)[0])
shap_vals = explainer.shap_values(scaled)[0]
base_val  = float(explainer.expected_value)

category, cat_color, emoji = aqi_category(aqi_pred)
top_feat, top_label, top_action = policy_message(shap_vals, FEATURE_COLS)

# ── Row 1: AQI badge + metric cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    st.markdown(
        f'<div style="background:{cat_color};border-radius:14px;padding:24px 28px;'
        f'text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.5);">'
        f'<div style="font-size:58px;font-weight:900;color:white;line-height:1.1;">{aqi_pred:.0f}</div>'
        f'<div style="font-size:24px;color:white;font-weight:700;margin-top:6px;">{emoji} {category}</div>'
        f'<div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:4px;">Predicted AQI</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.metric("PM2.5", f"{pm25:.0f} µg/m³",
              delta=f"{pm25-60:.0f} vs safe (60)", delta_color="inverse")
with c3:
    st.metric("PM10", f"{pm10:.0f} µg/m³",
              delta=f"{pm10-100:.0f} vs safe (100)", delta_color="inverse")
with c4:
    st.metric("Baseline AQI", f"{base_val:.0f}",
              help="Average AQI across training data — the SHAP baseline.")

st.markdown("---")

# ── Row 2: SHAP Waterfall + Policy panel ──────────────────────────────────────
c_shap, c_policy = st.columns([3, 2])

with c_shap:
    st.markdown("<h3 style='color:#00B4D8;'>📊 SHAP Explanation — Why this prediction?</h3>",
                unsafe_allow_html=True)
    st.caption("Each bar shows how much a feature pushed the AQI ↑ above or ↓ below the baseline.")

    try:
        shap_exp = shap.Explanation(
            values=shap_vals, base_values=base_val,
            data=scaled[0],   feature_names=FEATURE_COLS,
        )
        fig, _ = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_exp, show=False, max_display=8)
        plt.title(f"SHAP Waterfall  |  Predicted AQI: {aqi_pred:.0f}",
                  fontsize=13, fontweight="bold", color="#E8F4F8")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as err:
        st.warning(f"Could not render waterfall plot: {err}")

with c_policy:
    st.markdown("<h3 style='color:#00B4D8;'>💬 Policy Recommendation</h3>",
                unsafe_allow_html=True)

    health_info = {
        "Good":         ("No restrictions needed. Air quality is satisfactory.",          "#27AE60"),
        "Satisfactory": ("Acceptable for most. Sensitive individuals should take care.",   "#2ECC71"),
        "Moderate":     ("Sensitive groups may be affected. Reduce outdoor exertion.",     "#F39C12"),
        "Poor":         ("Everyone may experience effects. Limit prolonged outdoor time.", "#E67E22"),
        "Very Poor":    ("Health alert — serious effects for all. Avoid outdoor activity.","#C0392B"),
        "Severe":       ("EMERGENCY. Remain indoors. Immediate government action needed.", "#7B241C"),
    }
    health_msg, h_color = health_info[category]

    st.markdown(
        f'<div style="background:{h_color}30;border-left:4px solid {h_color};'
        f'padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:12px;">'
        f'<b style="color:{h_color};font-size:15px;">{emoji} {category}</b><br>'
        f'<span style="font-size:13px;color:#C8D8E8;">{health_msg}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#0A1F35;border-left:4px solid #00B4D8;'
        f'padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:12px;">'
        f'<b style="color:#00B4D8;font-size:13px;">🎯 Primary Driver: {top_label}</b><br>'
        f'<span style="font-size:12px;color:#B0C8DC;">{top_action}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<p style='color:#94A3B8;font-size:13px;margin-bottom:4px;'>"
                "<b>Feature Contributions (SHAP values)</b></p>", unsafe_allow_html=True)
    shap_series  = pd.Series(dict(zip(FEATURE_COLS, shap_vals))).sort_values(ascending=False)

    rows_html = ""
    for feat, val in shap_series.items():
        direction = "⬆ Push Up" if val > 0 else "⬇ Push Down"
        val_color = "#E74C3C" if val > 0 else "#3498DB"
        rows_html += (
            f"<tr style='border-bottom:1px solid #2A4060;'>"
            f"<td style='padding:7px 10px;color:#E8F4F8;'>{feat}</td>"
            f"<td style='padding:7px 10px;text-align:right;color:{val_color};font-weight:600;'>{val:.1f}</td>"
            f"<td style='padding:7px 10px;color:{val_color};'>{direction}</td>"
            f"</tr>"
        )
    st.markdown(
        f"""<div style='background:#162535;border-radius:8px;border:1px solid #2A4060;overflow:hidden;'>
        <table style='width:100%;border-collapse:collapse;font-size:13px;font-family:sans-serif;'>
          <thead>
            <tr style='background:#0A1F35;'>
              <th style='padding:8px 10px;text-align:left;color:#00B4D8;font-weight:600;'>Feature</th>
              <th style='padding:8px 10px;text-align:right;color:#00B4D8;font-weight:600;'>SHAP (AQI pts)</th>
              <th style='padding:8px 10px;text-align:left;color:#00B4D8;font-weight:600;'>Direction</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Row 3: SHAP Dependence plots ──────────────────────────────────────────────
st.markdown("<h3 style='color:#00B4D8;'>🔬 SHAP Dependence — Non-linear Relationships</h3>",
            unsafe_allow_html=True)
st.caption("How each feature's value maps to its SHAP contribution across all 366 days in the 2024 test set.")

@st.cache_data
def load_test_shap():
    try:
        csv = os.path.join(_base, "Delhi_Air_Quality_Dataset.csv")
        df  = pd.read_csv(csv)
        df["Full_Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" +
            df["Month"].astype(str).str.zfill(2) + "-" +
            df["Date"].astype(str).str.zfill(2)
        )
        df.sort_values("Full_Date", inplace=True)
        df["AQI_Lag_1"]   = df["AQI"].shift(1)
        df["PM2.5_Lag_1"] = df["PM2.5"].shift(1)
        df.dropna(inplace=True)
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        test      = df[df["Full_Date"] >= "2024-01-01"]
        X_sc      = scaler.transform(test[FEATURE_COLS])
        sv        = explainer.shap_values(X_sc)
        return pd.DataFrame(X_sc, columns=FEATURE_COLS), sv
    except Exception:
        return None, None

X_test_df, test_sv = load_test_shap()

if X_test_df is not None:
    d1, d2 = st.columns(2)
    with d1:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        shap.dependence_plot("PM2.5", test_sv, X_test_df,
                             interaction_index="Month_Sin", show=False, ax=ax2)
        ax2.set_title("SHAP Dependence: PM2.5\n(coloured by Month_Sin)",
                      fontsize=12, fontweight="bold", color="#E8F4F8")
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
    with d2:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        shap.dependence_plot("AQI_Lag_1", test_sv, X_test_df,
                             interaction_index="Month_Cos", show=False, ax=ax3)
        ax3.set_title("SHAP Dependence: AQI_Lag_1\n(coloured by Month_Cos)",
                      fontsize=12, fontweight="bold", color="#E8F4F8")
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)
else:
    st.info("📂 Place `Delhi_Air_Quality_Dataset.csv` in the same folder as `app.py` "
            "to enable the SHAP dependence plots.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4A6080;font-size:12px;padding:8px 0;'>"
    "Varun Singh &nbsp;"
    "&nbsp;·&nbsp; Explainable AI for Air Quality Assessment &nbsp;·&nbsp; "
    "</div>",
    unsafe_allow_html=True,
)