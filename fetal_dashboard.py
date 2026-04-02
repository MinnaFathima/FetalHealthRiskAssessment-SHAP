import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
from streamlit_shap import st_shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime
import io

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Ensure these files exist in your working directory
    model = joblib.load("fetal_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    classes = ["Normal", "Suspect", "Pathological"]
    
    try:
        val_data = joblib.load("validation_data.pkl")
    except FileNotFoundError:
        val_data = None
        st.warning("validation_data.pkl not found. Run train_xgb_shap.py first for global SHAP.")
    
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, features, classes, explainer, val_data

model, scaler, features, classes, explainer, val_data = load_models()

# --- UTILITY FUNCTIONS ---
def style_clinical_risk(val):
    if val == "HIGH":
        return 'background-color: red; color: white; font-weight: bold'
    elif val == "MODERATE":
        return 'background-color: orange; color: black; font-weight: bold'
    elif val == "LOW":
        return 'background-color: green; color: white; font-weight: bold'
    return ''

def generate_clinical_report(patient_data, prediction, probabilities, risk, patient_id="001"):
    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    
    elements.append(Paragraph("Fetal Health Assessment Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Cardiotocography (CTG) Analysis", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("PATIENT INFORMATION", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Patient ID:</b> {patient_id} | <b>Date:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 8))
    
    # CTG TABLE
    elements.append(Paragraph("CTG PARAMETERS", styles["Heading2"]))
    key_features = {
        "baseline value": "Baseline FHR (bpm)",
        "abnormal_short_term_variability": "Abnormal STV (%)", 
        "prolongued_decelerations": "Prolonged Decels",
        "accelerations": "Accelerations",
        "light_decelerations": "Light Decels"
    }
    
    table_data = [["Parameter", "Measured", "Normal Range"]]
    for feature, display_name in key_features.items():
        value = patient_data.get(feature, 0)
        normal_range = {
            "baseline value": "110-160",
            "abnormal_short_term_variability": "<50%",
            "prolongued_decelerations": "0-0.1", 
            "accelerations": ">0.005",
            "light_decelerations": "<0.01"
        }.get(feature, "N/A")
        table_data.append([display_name, f"{value:.2f}", normal_range])
    
    table = Table(table_data)
    table.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("FONTNAME", (0,0), (0,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10)
    ])
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # AI DIAGNOSIS
    elements.append(Paragraph("AI FETAL HEALTH CLASSIFICATION", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Primary Diagnosis:</b> {prediction.upper()}", styles["Heading3"]))
    
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Probability Distribution:", styles["Normal"]))
    for cls in classes:
        prob_pct = probabilities[cls] * 100
        elements.append(Paragraph(f"  • {cls}: <b>{prob_pct:.1f}%</b>", styles["Normal"]))
    
    # CLINICAL INTERPRETATION
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("CLINICAL INTERPRETATION", styles["Heading2"]))
    
    interpretations = {
        "Normal": """
        <b>NORMAL FETAL STATUS</b><br/><br/>
        Reassuring CTG pattern with normal baseline FHR, adequate variability, accelerations present, 
        and no significant decelerations.<br/><br/>
        <b>Recommendation:</b> Routine antenatal surveillance.
        """,
        "Suspect": """
        <b>SUSPICIOUS FETAL STATUS</b><br/><br/>
        Abnormal CTG features requiring further evaluation. Pathological probability: {:.1f}%<br/><br/>
        <b>IMMEDIATE ACTIONS:</b><br/>
        1. Continuous CTG monitoring<br/>
        2. Maternal position change<br/>
        3. IV fluids 250ml stat<br/>
        4. Consider fetal scalp stimulation
        """.format(risk*100),
        "Pathological": """
        <b>PATHOLOGICAL FETAL STATUS</b><br/><br/>
        CRITICAL CTG abnormalities confirmed. Pathological risk: {:.1f}%<br/><br/>
        <b>URGENT MANAGEMENT:</b><br/>
        1. URGENT OBSTETRIC REVIEW<br/>
        2. Stop oxytocin infusion<br/>
        3. Left lateral position<br/>
        4. 500ml IV fluids stat<br/>
        5. Prepare for delivery if no improvement
        """.format(risk*100)
    }
    
    interp_text = interpretations[prediction]
    elements.append(Paragraph(interp_text, styles["Normal"]))
    
    # FEATURE IMPORTANCE
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("KEY DECISION FACTORS", styles["Heading2"]))
    imp_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(5)
    imp_table = [["Rank", "Feature", "Model Weight"]] + [
        [i+1, row.Feature, f"{row.Importance:.3f}"] for i, row in imp_df.iterrows()
    ]
    imp_table_obj = Table(imp_table)
    imp_table_obj.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgreen),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ])
    elements.append(imp_table_obj)
    
    # CLINICAL MAPPING TABLE
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("CTG FEATURE CLINICAL MAPPING", styles["Heading2"]))
    clinical_table = [
        ["Feature", "Clinical Meaning", "High SHAP → Risk"],
        ["abnormal_short_term_variability", "Reduced FHR variability", "Pathological (distress)"],
        ["prolongued_decelerations", "Late/variable decels", "Suspect (hypoxia)"],
        ["accelerations", "Reactive FHR response", "Normal (healthy)"],
        ["baseline value", "Mean FHR (bpm)", "Extreme → Pathological"]
    ]
    clinical_obj = Table(clinical_table)
    clinical_obj.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONTSIZE", (0,0), (-1,-1), 9)
    ])
    elements.append(clinical_obj)
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "DISCLAIMER: AI supports clinical decision making per NICE/IPOC guidelines. "
        "Final decisions by attending clinician.", styles["Normal"]
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# --- UI CONFIG ---
st.set_page_config(page_title="Fetal Health SHAP Twin", layout="wide")
st.title("🤰 Fetal Health Digital Twin")
st.markdown("***SHAP Explainable AI | 91% accuracy | NICE/IPOC compliant***")

# VALIDATION METRICS
with st.expander("✅ Model Validation", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "91%")
    col2.metric("ROC-AUC", "0.98")
    col3.metric("Total Test Samples", "426")

# 🌍 GLOBAL SHAP ANALYSIS (NEW)
if val_data is not None:
    with st.expander("🌍 Global SHAP Analysis (Dataset-Wide)", expanded=True):
        shap_values_val = val_data["shap_values"]  # List of 3 arrays for multiclass
        
        # Mean absolute SHAP per class (grouped bar chart)
        mean_shap = [np.abs(sv).mean(0) for sv in shap_values_val]
        fig_bar = go.Figure()
        for i, cls in enumerate(classes):
            fig_bar.add_trace(go.Bar(x=features, y=mean_shap[i], name=cls, 
                                   marker_color=['green', 'orange', 'red'][i]))
        fig_bar.update_layout(
            barmode='group', 
            title="Mean |SHAP| Values by Class (Top Features Drive Pathological)",
            xaxis_tickangle=90,
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # SHAP Summary beeswarm
        st_shap(shap.summary_plot(shap_values_val, val_data["X_test"], 
                                plot_type="bar", max_display=10, show=False), height=400)

# --- BATCH ANALYSIS SECTION ---
st.header("📁 Multi-Patient Analysis")
uploaded = st.file_uploader("Upload CSV (fetal_health.csv format)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded).reindex(columns=features, fill_value=0)[features]
    X = scaler.transform(df)
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    risk_levels = {"Normal": "LOW", "Suspect": "MODERATE", "Pathological": "HIGH"}
    results = pd.DataFrame({
        "Patient": range(1, len(df)+1),
        "Diagnosis": [classes[p] for p in preds],
        "Path_Risk": [f"{p[2]*100:.1f}%" for p in probs],
        "Risk_Level": [risk_levels[classes[p]] for p in preds]
    })
    
    st.dataframe(results.style.applymap(style_clinical_risk, subset=['Risk_Level']), use_container_width=True)
    
    fig = px.bar(results, x="Patient", 
                y=results["Path_Risk"].str.rstrip('%').astype(float)/100,
                color="Risk_Level", 
                title="Pathological Risk Distribution",
                color_discrete_map={"LOW": "green", "MODERATE": "orange", "HIGH": "red"})
    st.plotly_chart(fig, use_container_width=True)
    
    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_data = csv_buffer.getvalue().encode('utf-8-sig')
    st.download_button(
        "💾 Download Batch Results (Excel Ready)",
        csv_data, "fetal_batch_results.csv", "text/csv"
    )

# --- SINGLE PATIENT ASSESSMENT ---
st.header("🔬 Single Patient Assessment")
col1, col2, col3 = st.columns(3)
with col1: baseline = st.slider("FHR Baseline (bpm)", 50, 240, 140)
with col2: variability = st.slider("ST Variability %", 0, 90, 20)
with col3: decels = st.slider("Prolonged Decels", 0.0, 0.5, 0.0)

if st.button("🏥 Generate AI Diagnosis & SHAP Explanation", type="primary"):
    patient = dict.fromkeys(features, 0.0)
    patient["baseline value"] = float(baseline)
    patient["abnormal_short_term_variability"] = float(variability)
    patient["prolongued_decelerations"] = float(decels)
    
    df_pt = pd.DataFrame([patient])[features]
    X_pt = scaler.transform(df_pt)
    
    probs = model.predict_proba(X_pt)[0]
    pred_idx = np.argmax(probs)
    prediction = classes[pred_idx]
    risk = probs[2]
    
    risk_levels = {"Normal": "LOW", "Suspect": "MODERATE", "Pathological": "HIGH"}
    risk_level = risk_levels[prediction]
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Diagnosis", prediction, delta=None)
    with col2: st.metric("Risk Level", risk_level, delta=None)
    with col3: st.metric("Pathological Prob.", f"{risk*100:.1f}%")

    # SHAP EXPLANATION
    st.subheader("⚖️ SHAP Decision Interpretation")
    st.write(f"The graph below explains which factors pushed the model toward the **{prediction}** classification.")
    
    shap_values_pt = explainer.shap_values(df_pt)
    
    # Handle SHAP multi-class output structures
    if isinstance(shap_values_pt, list):
        values_to_plot = shap_values_pt[pred_idx]
        base_value = explainer.expected_value[pred_idx]
    else:
        if len(shap_values_pt.shape) == 3:
            values_to_plot = shap_values_pt[0, :, pred_idx]
            base_value = explainer.expected_value[pred_idx]
        else:
            values_to_plot = shap_values_pt[0]
            base_value = explainer.expected_value

    st_shap(shap.force_plot(
        base_value, 
        values_to_plot, 
        df_pt,
        link="identity"
    ), height=200)
    
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk*100,
        number={"suffix": "%"},
        title={"text": "Pathological Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 60], "color": "orange"},
                {"range": [60, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 60
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    pdf_bytes = generate_clinical_report(patient, prediction, dict(zip(classes, probs)), risk)
    st.download_button(
        "📄 Download Hospital-Grade PDF",
        pdf_bytes,
        f"CTG_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "application/pdf"
    )

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                color: white; border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="margin: 0;">🏥 SHAP Explainer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0.5rem;">
        🔍 <b>Explainable AI (XAI)</b><br>
        ✅ <b>91% Clinical Accuracy</b><br>
        📊 <b>Multi-Patient Analysis</b><br>
        🌍 <b>Global SHAP Insights</b><br>
        📄 <b>Hospital PDF Reports</b><br>
        🩺 <b>NICE Compliant Interpretations</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 Test Data")
    st.code("""
baseline value,abnormal_short_term_variability,prolongued_decelerations
108,72,0.2          # → Pathological/HIGH
140,20,0.0          # → Normal/LOW
    """)
