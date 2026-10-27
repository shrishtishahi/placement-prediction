import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predictor import (
    ROOT,
    STRONG_PROFILE,
    TYPICAL_PROFILE,
    WEAK_PROFILE,
    build_suggestions,
    predict_record,
    verify_network,
)
from report_pdf import build_report

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1F4E79"
TEAL = "#1A6B7A"
GREEN = "#1E7F4F"
RED = "#C0392B"
GOLD = "#B8860B"
SLATE = "#5D6D7E"

DATA_PATH = ROOT / "Placement_Prediction_data.csv"

st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }}
    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%);
        color: white;
        border-radius: 16px;
        padding: 1.6rem 1.8rem 1.4rem 1.8rem;
        margin-bottom: 1.1rem;
    }}
    .hero h1 {{
        color: white;
        font-size: 2rem;
        margin: 0 0 0.35rem 0;
        font-weight: 700;
    }}
    .hero p {{
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.45;
    }}
    .result-banner {{
        border-radius: 14px;
        padding: 1.3rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }}
    .result-banner h2 {{
        color: white;
        margin: 0;
        font-size: 2rem;
    }}
    .result-banner p {{
        color: rgba(255,255,255,0.95);
        margin: 0.35rem 0 0 0;
        font-size: 1.05rem;
    }}
    .tech-pill {{
        background: {NAVY};
        color: white;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        text-align: center;
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
    div[data-testid="stSidebar"] {{
        background: #F8FBFD;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"
    if "student_name" not in st.session_state:
        st.session_state.student_name = ""
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    for key, value in TYPICAL_PROFILE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(page):
    st.session_state.nav = page


def make_prediction(profile):
    result = predict_record(profile)
    return {
        **result,
        "suggestions": build_suggestions(result["status"], profile),
        "profile": dict(profile),
    }


def load_example(kind):
    source = STRONG_PROFILE if kind == "strong" else WEAK_PROFILE
    for key, value in source.items():
        st.session_state[key] = value
    profile = dict(source)
    profile["student_name"] = st.session_state.get("student_name", "")
    st.session_state.prediction = make_prediction(profile)
    st.session_state.nav = "Prediction"


@st.cache_data
def load_dataset_summary():
    frame = pd.read_csv(DATA_PATH)
    total = int(len(frame))
    placed = int((frame["PlacementStatus"] == "Placed").sum())
    not_placed = total - placed
    return {"total": total, "placed": placed, "not_placed": not_placed}


def probability_chart(prob_placed, prob_not_placed):
    fig, axis = plt.subplots(figsize=(6.4, 2.6))
    labels = ["Not Placed", "Placed"]
    values = [prob_not_placed * 100, prob_placed * 100]
    colors = [RED, GREEN]
    bars = axis.barh(labels, values, color=colors, height=0.55)
    for bar, value in zip(bars, values):
        inside = value >= 18
        axis.text(
            value - 2 if inside else value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="right" if inside else "left",
            fontsize=11,
            fontweight="bold",
            color="white" if inside else "#1C2833",
        )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Probability")
    axis.grid(axis="x", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def profile_chart(profile):
    labels = ["CGPA", "Skills", "Communication", "Projects", "Backlogs"]
    values = [
        float(profile["cgpa"]),
        float(profile["skills"]),
        float(profile["communication"]),
        float(profile["major"]) + float(profile["mini"]),
        float(profile["backlogs"]),
    ]
    label_text = [
        f"{values[0]:.1f}",
        f"{values[1]:.0f}",
        f"{values[2]:.1f}",
        f"{values[3]:.0f}",
        f"{values[4]:.0f}",
    ]
    colors = [NAVY, TEAL, GOLD, GREEN, RED]
    fig, axis = plt.subplots(figsize=(6.4, 3.1))
    bars = axis.bar(labels, values, color=colors, width=0.62)
    for bar, label in zip(bars, label_text):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_ylabel("Value")
    axis.set_ylim(0, max(values) + 1.8)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Student Placement Prediction</h1>
            <p>Estimate campus placement from academics, projects, skills, internships, and backlogs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = load_dataset_summary()
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.subheader("About this system")
        st.write(
            "The system reads a student's academic and activity details and estimates "
            "whether they are likely to be placed. It uses an artificial neural network "
            "trained on 10,000 student records."
        )

        st.subheader("Features of the system")
        for item in [
            "Two pages: an overview and a prediction form",
            "Placement status with a confidence score",
            "Charts for placement probability and the student profile",
            "Suggestions based on the result",
            "A one-page PDF report you can download",
        ]:
            st.markdown(f"- {item}")

        st.subheader("Technologies used")
        technologies = [
            "Python",
            "Streamlit",
            "TensorFlow / Keras",
            "Scikit-learn",
            "Pandas",
            "NumPy",
        ]
        pills = st.columns(3)
        for index, name in enumerate(technologies):
            pills[index % 3].markdown(
                f"<div class='tech-pill'>{name}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("How the prediction works")
        steps = [
            "You enter the student details on the prediction page.",
            "Yes or No answers are converted to 1 or 0, and every value is scaled the same way as in training.",
            "The trained network passes the 11 values through two hidden layers and a sigmoid output.",
            "The output is the probability of being placed. Above 50% is Placed. Otherwise it is Not Placed.",
            "The page then shows the confidence, two charts, suggestions, and a PDF report.",
        ]
        for number, step in enumerate(steps, start=1):
            st.markdown(f"**{number}.** {step}")

        st.button(
            "Open prediction page",
            type="primary",
            on_click=go_to,
            args=("Prediction",),
            use_container_width=True,
        )

    with right:
        st.subheader("Dataset")
        total = summary["total"]
        placed = summary["placed"]
        not_placed = summary["not_placed"]
        st.markdown(
            f"""
            <div style="background:{NAVY}; color:white; border-radius:14px; padding:1rem 1.1rem;">
                <p style="margin:0; opacity:0.75; font-size:0.85rem;">Student records</p>
                <p style="margin:0; font-size:1.8rem; font-weight:700;">{total:,}</p>
                <hr style="border:0; border-top:1px solid rgba(255,255,255,0.2); margin:0.7rem 0;">
                <p style="margin:0; color:{GREEN}; font-weight:700;">Placed: {placed:,} ({placed / total:.1%})</p>
                <p style="margin:0.25rem 0 0 0; color:#F5B7B1; font-weight:700;">Not placed: {not_placed:,} ({not_placed / total:.1%})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.subheader("Model")
        st.markdown(
            f"""
            <div style="background:white; border:1px solid #E1E8EF; border-radius:14px; padding:1rem 1.1rem;">
                <p style="margin:0; color:{SLATE}; font-size:0.85rem;">Test accuracy from training</p>
                <p style="margin:0; color:{GREEN}; font-size:1.8rem; font-weight:700;">93.67%</p>
                <p style="margin:0.35rem 0 0 0; color:{SLATE};">Precision 91.8% · Recall 92.7% · F1 92.3%</p>
                <p style="margin:0.8rem 0 0 0; color:#1C2833; line-height:1.45;">
                    Input (11) → Dense 64, ReLU → Dropout 30% → Dense 32, ReLU → Dropout 30% → Output, Sigmoid
                </p>
                <p style="margin:0.45rem 0 0 0; color:{SLATE};">Optimizer: Adam · Loss: binary crossentropy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction():
    st.markdown(
        """
        <div class="hero">
            <h1>Predict placement</h1>
            <p>Enter the student details, or load an example, then predict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Examples for the presentation")
    example_cols = st.columns(2)
    example_cols[0].button(
        "Likely placed example",
        use_container_width=True,
        on_click=load_example,
        args=("strong",),
    )
    example_cols[1].button(
        "Needs improvement example",
        use_container_width=True,
        on_click=load_example,
        args=("weak",),
    )

    with st.form("prediction_form"):
        st.subheader("Student details")
        st.text_input("Student name (optional)", key="student_name", placeholder="Name for the PDF report")

        row1 = st.columns(3)
        row1[0].number_input("CGPA", min_value=6.5, max_value=9.1, step=0.1, key="cgpa")
        row1[1].number_input("10th percentage", min_value=57, max_value=88, step=1, key="tenth")
        row1[2].number_input("12th percentage", min_value=55, max_value=90, step=1, key="twelfth")

        row2 = st.columns(3)
        row2[0].selectbox("Major projects", [0, 1, 2], key="major")
        row2[1].selectbox("Mini projects", [0, 1, 2, 3], key="mini")
        row2[2].selectbox("Workshops / certifications", [0, 1, 2, 3], key="workshops")

        row3 = st.columns(3)
        row3[0].slider("Skills", min_value=6, max_value=9, step=1, key="skills")
        row3[1].slider(
            "Communication skill rating",
            min_value=3.0,
            max_value=4.8,
            step=0.1,
            key="communication",
        )
        row3[2].number_input("Backlogs", min_value=0, max_value=7, step=1, key="backlogs")

        row4 = st.columns(2)
        row4[0].radio("Internship", ["Yes", "No"], horizontal=True, key="internship")
        row4[1].radio("Hackathon", ["Yes", "No"], horizontal=True, key="hackathon")

        submitted = st.form_submit_button(
            "Predict Placement",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        profile = {
            "student_name": st.session_state.student_name,
            "cgpa": st.session_state.cgpa,
            "major": st.session_state.major,
            "workshops": st.session_state.workshops,
            "mini": st.session_state.mini,
            "skills": st.session_state.skills,
            "communication": st.session_state.communication,
            "internship": st.session_state.internship,
            "hackathon": st.session_state.hackathon,
            "twelfth": st.session_state.twelfth,
            "tenth": st.session_state.tenth,
            "backlogs": st.session_state.backlogs,
        }
        st.session_state.prediction = make_prediction(profile)

    prediction = st.session_state.prediction
    if prediction is None:
        st.info("Fill in the form and click Predict Placement. The two example buttons fill a real student and predict immediately.")
        return

    profile = prediction["profile"]
    status = prediction["status"]
    color = GREEN if status == "Placed" else RED
    confidence = prediction["confidence"] * 100
    placed_chance = prediction["prob_placed"] * 100

    st.divider()
    result_col, chart_col = st.columns([1, 1.15], gap="large")

    with result_col:
        st.markdown(
            f"""
            <div class="result-banner" style="background:{color};">
                <p style="letter-spacing:0.04em; font-size:0.85rem;">Placement status</p>
                <h2>{status}</h2>
                <p>Confidence in this result: <b>{confidence:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(2)
        metric_cols[0].metric("Chance of placement", f"{placed_chance:.1f}%")
        metric_cols[1].metric("Chance of not placed", f"{100 - placed_chance:.1f}%")
        st.progress(min(max(prediction["confidence"], 0.0), 1.0))

        st.subheader("Suggestions")
        if status == "Placed":
            st.success("The profile is in the placed range. Keep building on it.")
        else:
            st.warning("The profile is in the not placed range. These changes would help.")
        for tip in prediction["suggestions"]:
            st.markdown(f"- {tip}")

        st.subheader("Download report")
        pdf = build_report(profile, prediction, prediction["suggestions"])
        safe_name = profile.get("student_name", "").strip() or "student"
        safe_name = "".join(ch for ch in safe_name if ch.isalnum() or ch in (" ", "_", "-")).strip()
        safe_name = safe_name.replace(" ", "_") or "student"
        st.download_button(
            "Download PDF report",
            data=pdf,
            file_name=f"Placement_Report_{safe_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with chart_col:
        st.subheader("Placement probability")
        st.caption("Placed compared with Not Placed for this student.")
        fig1 = probability_chart(prediction["prob_placed"], prediction["prob_not_placed"])
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        st.subheader("Student profile summary")
        st.caption("CGPA, skills, communication, total projects, and backlogs.")
        fig2 = profile_chart(profile)
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

        name = profile.get("student_name", "").strip()
        who = f"{name}: " if name else ""
        st.caption(
            f"{who}CGPA {float(profile['cgpa']):.1f} · "
            f"Skills {int(profile['skills'])} · "
            f"Communication {float(profile['communication']):.1f} · "
            f"Projects {int(profile['major']) + int(profile['mini'])} · "
            f"Workshops {int(profile['workshops'])} · "
            f"Internship {profile['internship']} · "
            f"Hackathon {profile['hackathon']} · "
            f"10th {int(profile['tenth'])}% · "
            f"12th {int(profile['twelfth'])}% · "
            f"Backlogs {int(profile['backlogs'])}"
        )


def main():
    init_state()
    try:
        verify_network()
    except Exception as error:
        st.error(f"The prediction model could not be checked: {error}")
        st.stop()

    st.sidebar.markdown(
        f"<h2 style='color:{NAVY}; text-align:center; margin-bottom:0;'>Placement Predictor</h2>"
        f"<p style='text-align:center; color:{SLATE}; font-size:0.85rem;'>ANN · Streamlit</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.button("Home", use_container_width=True, on_click=go_to, args=("Home",))
    st.sidebar.button(
        "Prediction",
        use_container_width=True,
        on_click=go_to,
        args=("Prediction",),
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Open Home for the project overview. Open Prediction to enter a student and download the report."
    )

    if st.session_state.nav == "Home":
        render_home()
    else:
        render_prediction()


main()
