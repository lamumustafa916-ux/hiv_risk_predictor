import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HIV Risk Predictor",
    page_icon="🔴",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #B71C1C, #880E4F);
        color: white; padding: 22px 25px;
        border-radius: 12px; margin-bottom: 25px; text-align: center;
    }
    .main-header h1 { font-size: 1.9rem; margin: 0; }
    .main-header p  { margin: 6px 0 0 0; opacity: 0.88; font-size: 0.95rem; }

    .high-risk {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border: 2px solid #E53935; border-radius: 12px;
        padding: 22px; text-align: center; margin-top: 18px;
    }
    .high-risk h2 { color: #B71C1C; font-size: 2rem; margin: 0; }
    .high-risk p  { color: #C62828; margin: 6px 0 0 0; }

    .low-risk {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border: 2px solid #43A047; border-radius: 12px;
        padding: 22px; text-align: center; margin-top: 18px;
    }
    .low-risk h2 { color: #2E7D32; font-size: 2rem; margin: 0; }
    .low-risk p  { color: #388E3C; margin: 6px 0 0 0; }

    .info-box {
        background: #FFF3E0; border-left: 5px solid #E65100;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
        font-size: 0.93rem;
    }
    .section-header {
        background: #F3E5F5; border-left: 4px solid #7B1FA2;
        padding: 8px 14px; border-radius: 4px; margin: 16px 0 8px 0;
        font-weight: bold; color: #4A148C;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔴 HIV Risk Prediction System</h1>
    <p>Tanzania — Machine Learning Classification Model</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl",    "rb") as f: model    = pickle.load(f)
    with open("encoders.pkl", "rb") as f: encoders = pickle.load(f)
    with open("scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
    return model, encoders, scaler

try:
    model, encoders, scaler = load_artifacts()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 About This App")
    st.markdown("""
This app predicts **HIV risk level** (High / Low) using a
**Logistic Regression** model trained on 1,000 records
from Tanzania.

**Model Performance:**
- ✅ Accuracy: **84%**
- ✅ ROC-AUC: **0.917**

**Key Risk Factors:**
- Partner HIV status
- Condom use behaviour
- STI history
- IV drug use
- Number of sexual partners
- HIV knowledge score
""")
    st.divider()
    st.warning("⚠️ This tool is for **educational purposes only**. It is NOT a substitute for professional medical testing.")
    st.caption("ML Project · Tanzania · 2026")

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
    For actual HIV testing, please visit a certified healthcare facility.
    Fill in all fields honestly for the most accurate prediction.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("📋 Personal & Demographic Information")
col1, col2 = st.columns(2)

with col1:
    age              = st.number_input("🎂 Age", min_value=15, max_value=65, value=25)
    gender           = st.selectbox("⚧ Gender", list(encoders['gender'].classes_))
    marital_status   = st.selectbox("💍 Marital Status", list(encoders['marital_status'].classes_))
    education_level  = st.selectbox("🎓 Education Level", list(encoders['education_level'].classes_))

with col2:
    employment_status = st.selectbox("💼 Employment Status", list(encoders['employment_status'].classes_))
    region            = st.selectbox("📍 Region", list(encoders['region'].classes_))
    residence_type    = st.selectbox("🏘️ Residence Type", list(encoders['residence_type'].classes_))

st.divider()
st.subheader("🔬 Behavioural & Health Risk Factors")
col3, col4 = st.columns(2)

with col3:
    number_of_partners  = st.slider("👥 Number of Sexual Partners", 0, 5, 1)
    condom_use          = st.selectbox("🛡️ Condom Use", list(encoders['condom_use'].classes_))
    partner_hiv_status  = st.selectbox("❤️ Partner HIV Status", list(encoders['partner_hiv_status'].classes_))
    alcohol_use         = st.selectbox("🍺 Alcohol Use", list(encoders['alcohol_use'].classes_))
    hiv_knowledge_score = st.slider("📚 HIV Knowledge Score (0=None, 10=Expert)", 0, 10, 5)

with col4:
    hiv_tested_before  = st.selectbox("🧪 Ever Tested for HIV?", list(encoders['hiv_tested_before'].classes_))
    sti_history        = st.selectbox("🏥 STI History", list(encoders['sti_history'].classes_))
    blood_transfusion  = st.selectbox("🩸 Blood Transfusion History", list(encoders['blood_transfusion'].classes_))
    iv_drug_use        = st.selectbox("💉 IV Drug Use", list(encoders['iv_drug_use'].classes_))
    healthcare_access  = st.selectbox("🏨 Healthcare Access", list(encoders['healthcare_access'].classes_))

st.divider()

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
if st.button("🔍 Predict HIV Risk", use_container_width=True, type="primary"):

    input_dict = {
        'age':                  age,
        'gender':               encoders['gender'].transform([gender])[0],
        'marital_status':       encoders['marital_status'].transform([marital_status])[0],
        'education_level':      encoders['education_level'].transform([education_level])[0],
        'employment_status':    encoders['employment_status'].transform([employment_status])[0],
        'region':               encoders['region'].transform([region])[0],
        'residence_type':       encoders['residence_type'].transform([residence_type])[0],
        'number_of_partners':   number_of_partners,
        'condom_use':           encoders['condom_use'].transform([condom_use])[0],
        'hiv_tested_before':    encoders['hiv_tested_before'].transform([hiv_tested_before])[0],
        'sti_history':          encoders['sti_history'].transform([sti_history])[0],
        'blood_transfusion':    encoders['blood_transfusion'].transform([blood_transfusion])[0],
        'iv_drug_use':          encoders['iv_drug_use'].transform([iv_drug_use])[0],
        'alcohol_use':          encoders['alcohol_use'].transform([alcohol_use])[0],
        'hiv_knowledge_score':  hiv_knowledge_score,
        'healthcare_access':    encoders['healthcare_access'].transform([healthcare_access])[0],
        'partner_hiv_status':   encoders['partner_hiv_status'].transform([partner_hiv_status])[0],
    }

    input_df = pd.DataFrame([input_dict])
    input_sc = scaler.transform(input_df)

    prediction   = model.predict(input_sc)[0]
    probability  = model.predict_proba(input_sc)[0]
    risk_label   = encoders['hiv_risk'].inverse_transform([prediction])[0]
    risk_prob    = probability[prediction] * 100

    if risk_label == 'High Risk':
        st.markdown(f"""
        <div class="high-risk">
            <h2>🔴 HIGH RISK</h2>
            <p>Confidence: <strong>{risk_prob:.1f}%</strong></p>
            <p>This individual shows indicators associated with higher HIV exposure risk.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("🏥 **Recommendation:** Please visit a healthcare facility for HIV testing and counselling as soon as possible.")
    else:
        st.markdown(f"""
        <div class="low-risk">
            <h2>🟢 LOW RISK</h2>
            <p>Confidence: <strong>{risk_prob:.1f}%</strong></p>
            <p>This individual shows fewer indicators associated with HIV exposure risk.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ **Recommendation:** Continue practising safe behaviours. Regular HIV testing is still encouraged.")

    # Risk probability bar
    st.markdown("### 📊 Risk Probability Breakdown")
    classes = encoders['hiv_risk'].classes_
    prob_df = pd.DataFrame({'Risk Level': classes, 'Probability (%)': [p*100 for p in probability]})
    st.bar_chart(prob_df.set_index('Risk Level'))

    # Summary table
    st.markdown("### 📋 Input Summary")
    summary = pd.DataFrame({
        "Factor": list(input_dict.keys()),
        "Value":  [age, gender, marital_status, education_level, employment_status,
                   region, residence_type, number_of_partners, condom_use,
                   hiv_tested_before, sti_history, blood_transfusion, iv_drug_use,
                   alcohol_use, hiv_knowledge_score, healthcare_access, partner_hiv_status]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("🔴 HIV Risk Predictor · Tanzania · Built with Streamlit & Scikit-learn · ML Project 2026 · For Educational Purposes Only")
