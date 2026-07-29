# Student Placement Prediction System
# Using ANN (Artificial Neural Network) with Streamlit
# Technologies: Python, TensorFlow, Scikit-learn, Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import joblib
import traceback
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Placement_Prediction_data.csv")
MODEL_ZIP = os.path.join(BASE_DIR, "placement_ann_model.zip")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Colour scheme
GREEN = "#1DB954"
RED = "#E74C3C"
NAVY = "#1E3A5F"
BLUE = "#4A90D9"
GOLD = "#F5A623"
LIGHT = "#F5F7FA"

# Loading the trained ANN model
@st.cache_resource
def load_model():
    import tensorflow as tf
    from tensorflow import keras
    # The zip file is actually a .keras model file (same internal structure)
    # We copy it to .keras format and load directly
    model_path = os.path.join(BASE_DIR, "placement_ann_model.keras")
    if not os.path.exists(model_path):
        shutil.copy2(MODEL_ZIP, model_path)
    model = keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Preprocessing - scaling and encoding input
def preprocess_input(fd):
    cols = [
        "CGPA", "Major Projects", "Workshops/Certificatios",
        "Mini Projects", "Skills", "Communication Skill Rating",
        "Internship", "Hackathon", "12th Percentage",
        "10th Percentage", "backlogs"
    ]
    df = pd.DataFrame([fd])[cols]
    df["Internship"] = df["Internship"].map({"No": 0, "Yes": 1})
    df["Hackathon"] = df["Hackathon"].map({"No": 0, "Yes": 1})
    scaler = load_scaler()
    return scaler.transform(df)

# PDF report generation using ReportLab
def generate_pdf(data, result, confidence, suggestions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=18,
                                  textColor=HexColor(NAVY), spaceAfter=4, alignment=TA_CENTER)
    sub_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                                textColor=HexColor("#888888"), spaceAfter=12, alignment=TA_CENTER)
    head_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12,
                                 textColor=HexColor(NAVY), spaceBefore=10, spaceAfter=4)
    normal_style = ParagraphStyle("N", parent=styles["Normal"], fontSize=9, leading=13)

    result_color = GREEN if result == "Placed" else RED
    result_style = ParagraphStyle("R", parent=styles["Normal"], fontSize=16,
                                   textColor=HexColor(result_color), spaceAfter=2, alignment=TA_CENTER)

    elements = []
    elements.append(Paragraph("Placement Prediction Report", title_style))
    elements.append(Paragraph(f"Date: {datetime.now():%Y-%m-%d %H:%M}", sub_style))
    elements.append(HRFlowable(width="100%", color=HexColor(BLUE), thickness=1))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Prediction: {result}", result_style))
    elements.append(Paragraph(f"Confidence Score: {confidence:.1f}%", sub_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Student Details", head_style))
    table_data = [["Feature", "Value"]]
    for k, v in data.items():
        table_data.append([k, str(v)])
    table = Table(table_data, colWidths=[140, 90])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F5F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Suggestions", head_style))
    for s in suggestions:
        elements.append(Paragraph(f"- {s}", normal_style))
        elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", color=HexColor("#CCCCCC"), thickness=0.5))
    elements.append(Paragraph("Generated by Placement Prediction System",
                               ParagraphStyle("F", parent=normal_style, fontSize=7,
                                              textColor=HexColor("#AAAAAA"), alignment=TA_CENTER)))
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===== SIDEBAR NAVIGATION =====
st.sidebar.markdown(f"<h2 style='color:{NAVY}; text-align:center; font-weight:700;'>🎓<br>Placement<br>Predictor</h2>"
                    f"<p style='text-align:center; color:#555; font-size:0.8rem; font-weight:500;'>ANN | ML | Streamlit</p>",
                    unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Predict Placement"], label_visibility="collapsed")


# ===== HOME PAGE =====
if page == "Home":
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{NAVY},{BLUE});"
        f"padding:2rem; border-radius:12px; margin-bottom:1.5rem;'>"
        f"<h1 style='color:white; margin:0; font-size:1.8rem;'>Student Placement Prediction</h1>"
        f"<p style='color:rgba(255,255,255,0.85); margin-top:0.5rem;'>"
        f"Predicting placement using Artificial Neural Network (ANN)</p></div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown("### About the Project")
        st.markdown(
            "This project predicts whether a student will be placed in campus "
            "recruitment based on their academic performance, technical skills, "
            "projects, internships and other factors. The model is trained on "
            "10,000 student records and achieves 93.7% accuracy."
        )
        st.markdown("### Features")
        for item in ["Predict placement with 93% accuracy",
                      "Personalised suggestions",
                      "Charts for analysis",
                      "Download PDF report"]:
            st.markdown(f"- {item}")

        st.markdown("### Technologies")
        tech_cols = st.columns(3)
        for i, t in enumerate(["Python", "Streamlit", "TensorFlow/Keras",
                                "Scikit-learn", "Pandas/NumPy", "Matplotlib"]):
            tech_cols[i % 3].markdown(
                f"<div style='background:{NAVY}; color:white; padding:0.3rem 0.5rem; "
                f"border-radius:6px; text-align:center; font-size:0.8rem; font-weight:500;'>{t}</div>",
                unsafe_allow_html=True)

        st.markdown("### How it Works")
        steps = [
            "Student details are entered through a form",
            "Data is scaled and encoded automatically",
            "ANN model processes the input through 3 layers",
            "Result is displayed with confidence score",
            "Suggestions and PDF report are generated"
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}.** {s}")

    with col2:
        # Dataset overview
        df = load_data()
        total = len(df)
        placed = int((df["PlacementStatus"] == "Placed").sum())
        nplaced = total - placed

        st.markdown("### Dataset Overview")
        st.markdown(
            f"<div style='background:{NAVY}; padding:1rem; border-radius:10px;'>"
            f"<p style='margin:0; color:rgba(255,255,255,0.7); font-size:0.85rem;'>Total Records</p>"
            f"<p style='margin:0; font-size:1.6rem; font-weight:bold; color:white;'>{total:,}</p>"
            f"<hr style='margin:0.5rem 0; border-color:rgba(255,255,255,0.2);'>"
            f"<div style='display:flex; justify-content:space-between;'>"
            f"<div><p style='margin:0; color:rgba(255,255,255,0.7); font-size:0.85rem;'>Placed</p>"
            f"<p style='margin:0; font-size:1.2rem; font-weight:bold; color:{GREEN};'>"
            f"{placed:,} ({placed/total*100:.1f}%)</p></div>"
            f"<div><p style='margin:0; color:rgba(255,255,255,0.7); font-size:0.85rem;'>Not Placed</p>"
            f"<p style='margin:0; font-size:1.2rem; font-weight:bold; color:{RED};'>"
            f"{nplaced:,} ({nplaced/total*100:.1f}%)</p></div></div></div>",
            unsafe_allow_html=True
        )

        st.markdown("### Model Performance")
        st.markdown(
            f"<div style='background:{NAVY}; padding:1rem; border-radius:10px; text-align:center;'>"
            f"<p style='margin:0; color:rgba(255,255,255,0.7); font-size:0.85rem;'>Test Accuracy</p>"
            f"<p style='margin:0; font-size:1.8rem; font-weight:bold; color:{GREEN};'>93.67%</p>"
            f"<p style='margin:0; color:rgba(255,255,255,0.7); font-size:0.85rem;'>"
            f"Precision: 91.8% | Recall: 92.7% | F1: 92.3%</p></div>",
            unsafe_allow_html=True
        )

        st.markdown("### Model Architecture")
        st.markdown(
            f"<div style='background:{NAVY}; padding:1rem; border-radius:10px; font-size:0.85rem; color:white;'>"
            f"Input (11) → Dense(64,ReLU) → Dropout(30%) → "
            f"Dense(32,ReLU) → Dropout(30%) → Output(1,Sigmoid)<br>"
            f"<span style='color:rgba(255,255,255,0.6);'>Parameters: 2,881 | Optimizer: Adam | Epochs: 57</span></div>",
            unsafe_allow_html=True
        )


# ===== PREDICT PAGE =====
elif page == "Predict Placement":
    st.markdown(f"<h1 style='color:{NAVY};'>Predict Placement</h1>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("### Enter Student Details")

        r1 = st.columns(3)
        with r1[0]:
            cgpa = st.number_input("CGPA", 6.5, 9.1, 7.5, 0.1)
        with r1[1]:
            skills = st.slider("Skills", 6, 9, 7)
        with r1[2]:
            comm = st.slider("Communication Rating", 3.0, 4.8, 4.0, 0.1)

        r2 = st.columns(3)
        with r2[0]:
            p12 = st.number_input("12th %", 55, 90, 70)
        with r2[1]:
            p10 = st.number_input("10th %", 57, 88, 75)
        with r2[2]:
            backlogs = st.number_input("Backlogs", 0, 7, 1)

        r3 = st.columns(3)
        with r3[0]:
            maj = st.selectbox("Major Projects", [0, 1, 2])
        with r3[1]:
            mini = st.selectbox("Mini Projects", [0, 1, 2, 3])
        with r3[2]:
            ws = st.selectbox("Workshops/Certifications", [0, 1, 2, 3])

        r4 = st.columns(3)
        with r4[0]:
            intern = st.radio("Internship", ["No", "Yes"], horizontal=True)
        with r4[1]:
            hack = st.radio("Hackathon", ["No", "Yes"], horizontal=True)
        with r4[2]:
            st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Placement", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Model is predicting..."):
            try:
                # Collect all inputs
                features = {
                    "CGPA": cgpa,
                    "Major Projects": maj,
                    "Workshops/Certificatios": ws,
                    "Mini Projects": mini,
                    "Skills": skills,
                    "Communication Skill Rating": comm,
                    "Internship": intern,
                    "Hackathon": hack,
                    "12th Percentage": p12,
                    "10th Percentage": p10,
                    "backlogs": backlogs
                }

                # Preprocess and predict
                scaled = preprocess_input(features)
                model = load_model()
                prob = float(model.predict(scaled, verbose=0)[0][0])

                pred = 1 if prob > 0.5 else 0
                encoder = load_encoder()
                result = encoder.inverse_transform([pred])[0]
                confidence = prob * 100 if result == "Placed" else (1 - prob) * 100

                st.markdown("---")
                rc = GREEN if result == "Placed" else RED
                icon = "Congratulations!" if result == "Placed" else "Needs Improvement"

                # Result and charts side by side
                cres, ccharts = st.columns([1, 1.3])

                with cres:
                    st.markdown(
                        f"<div style='background:{rc}; padding:1.5rem; border-radius:12px; "
                        f"text-align:center;'>"
                        f"<h1 style='color:white; margin:0.2rem 0; font-size:2rem;'>{result}</h1>"
                        f"<p style='color:rgba(255,255,255,0.9); font-size:1rem;'>"
                        f"Confidence: <b>{confidence:.1f}%</b></p></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(int(confidence))

                    # Suggestions
                    st.markdown("### Suggestions")
                    if result == "Placed":
                        suggestions = [
                            "Continue improving technical skills",
                            "Participate in more projects",
                            "Keep practising interview preparation",
                            "Consider internships for industry exposure",
                            "Build network through hackathons and workshops"
                        ]
                        st.success("Keep growing with these tips:")
                    else:
                        suggestions = [
                            "Improve technical skills",
                            "Build more projects",
                            "Complete an internship",
                            "Improve communication skills",
                            "Reduce backlogs",
                            "Participate in hackathons"
                        ]
                        st.warning("Here's how to improve your chances:")

                    for s in suggestions:
                        st.markdown(f"- {s}")

                    # PDF download
                    st.markdown("### Download Report")
                    display = {
                        "CGPA": cgpa, "Major Projects": maj,
                        "Workshops": ws, "Mini Projects": mini,
                        "Skills": skills, "Communication": comm,
                        "Internship": intern, "Hackathon": hack,
                        "12th %": p12, "10th %": p10,
                        "Backlogs": backlogs
                    }
                    try:
                        pdf = generate_pdf(display, result, confidence, suggestions)
                        st.download_button(
                            "Download PDF Report", pdf,
                            f"Placement_Report_{datetime.now():%Y%m%d_%H%M}.pdf",
                            "application/pdf", use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Could not generate PDF: {e}")

                with ccharts:
                    # Chart 1: Probability bar
                    st.markdown("#### Placement Probability")
                    fig1, ax1 = plt.subplots(figsize=(4.5, 2.5))
                    p = [1 - prob, prob]
                    bars = ax1.barh(["Not Placed", "Placed"], p,
                                    color=[RED, GREEN], height=0.4)
                    for b, v in zip(bars, p):
                        ax1.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                                 f"{v*100:.1f}%", va="center", fontsize=11, fontweight="bold")
                    ax1.set_xlim(0, 1.3)
                    ax1.grid(axis="x", alpha=0.3)
                    sns.despine()
                    st.pyplot(fig1)
                    plt.close()

                    # Chart 2: Profile bar chart
                    st.markdown("#### Student Profile")
                    profile = {
                        "CGPA": cgpa, "Skills": skills,
                        "Comm.": comm, "Projects": maj + mini,
                        "Backlogs": backlogs
                    }
                    fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))
                    colours = [NAVY, BLUE, GOLD, GREEN, RED]
                    bars2 = ax2.bar(profile.keys(), profile.values(),
                                    color=colours, width=0.55)
                    for b, v in zip(bars2, profile.values()):
                        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                                 str(v), ha="center", fontsize=10, fontweight="bold")
                    ax2.grid(axis="y", alpha=0.3)
                    sns.despine()
                    st.pyplot(fig2)
                    plt.close()

            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
