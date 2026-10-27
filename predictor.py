"""Placement prediction using the trained ANN.

The network was trained with TensorFlow/Keras:
11 inputs -> Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dropout -> Sigmoid.

Weights and the original training scaler are stored in ann_weights.npz so the
app can run on any computer without installing TensorFlow.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "ann_weights.npz"

# Column order used when the network was trained.
FEATURE_COLUMNS = [
    "CGPA",
    "Major Projects",
    "Workshops/Certificatios",
    "Mini Projects",
    "Skills",
    "Communication Skill Rating",
    "Internship",
    "Hackathon",
    "12th Percentage",
    "10th Percentage",
    "backlogs",
]

# Real students from the dataset. These two land on opposite results.
STRONG_PROFILE = {
    "cgpa": 8.9,
    "major": 0,
    "workshops": 3,
    "mini": 2,
    "skills": 9,
    "communication": 4.0,
    "internship": "Yes",
    "hackathon": "Yes",
    "twelfth": 78,
    "tenth": 82,
    "backlogs": 0,
}

WEAK_PROFILE = {
    "cgpa": 6.7,
    "major": 1,
    "workshops": 1,
    "mini": 1,
    "skills": 7,
    "communication": 4.0,
    "internship": "No",
    "hackathon": "No",
    "twelfth": 68,
    "tenth": 57,
    "backlogs": 4,
}

# A real student close to the decision line, so edits can change the result.
TYPICAL_PROFILE = {
    "cgpa": 8.3,
    "major": 1,
    "workshops": 1,
    "mini": 1,
    "skills": 8,
    "communication": 4.5,
    "internship": "Yes",
    "hackathon": "Yes",
    "twelfth": 64,
    "tenth": 65,
    "backlogs": 1,
}


@lru_cache(maxsize=1)
def load_network():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {WEIGHTS_PATH.name}")
    data = np.load(WEIGHTS_PATH)
    required = ["W1", "b1", "W2", "b2", "W3", "b3", "mean", "scale"]
    missing = [key for key in required if key not in data]
    if missing:
        raise FileNotFoundError(f"Model file is incomplete: {', '.join(missing)}")
    return data


def _yes_no(value):
    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
        return 1.0 if float(value) >= 1 else 0.0
    return 1.0 if str(value).strip().lower() == "yes" else 0.0


def profile_to_vector(profile):
    """Turn one form profile into the 11 numbers the network expects."""
    internship = _yes_no(profile["internship"])
    hackathon = _yes_no(profile["hackathon"])
    return np.array(
        [
            float(profile["cgpa"]),
            float(profile["major"]),
            float(profile["workshops"]),
            float(profile["mini"]),
            float(profile["skills"]),
            float(profile["communication"]),
            internship,
            hackathon,
            float(profile["twelfth"]),
            float(profile["tenth"]),
            float(profile["backlogs"]),
        ],
        dtype=np.float64,
    )


def _sigmoid(values):
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_matrix(raw_rows):
    """Return placement probability for each row. Shape (n,)."""
    network = load_network()
    rows = np.asarray(raw_rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    scaled = (rows - network["mean"]) / network["scale"]
    hidden = np.maximum(0.0, scaled @ network["W1"] + network["b1"])
    hidden = np.maximum(0.0, hidden @ network["W2"] + network["b2"])
    logits = (hidden @ network["W3"] + network["b3"]).reshape(-1)
    return _sigmoid(logits)


def predict_record(profile):
    """Predict one student. Probability is P(Placed)."""
    probability = float(predict_matrix(profile_to_vector(profile))[0])
    placed = probability > 0.5
    status = "Placed" if placed else "Not Placed"
    confidence = probability if placed else 1.0 - probability
    return {
        "status": status,
        "confidence": float(confidence),
        "prob_placed": probability,
        "prob_not_placed": float(1.0 - probability),
    }


def build_suggestions(status, profile):
    """Suggestions follow the prediction and the student's weak areas."""
    projects = int(profile["major"]) + int(profile["mini"])
    if status == "Placed":
        tips = [
            "Continue improving technical skills",
            "Participate in more projects",
            "Keep practicing interview preparation",
        ]
        if profile["internship"] == "No":
            tips.append("An internship will add industry experience before final placements")
        if int(profile["backlogs"]) > 0:
            tips.append("Clear the remaining backlog so it does not come up in interviews")
        if profile["hackathon"] == "No":
            tips.append("Join a hackathon to practice problem solving under a time limit")
        return tips

    tips = []
    if int(profile["skills"]) <= 7:
        tips.append("Improve technical skills")
    if projects <= 2:
        tips.append("Build more projects")
    if profile["internship"] == "No":
        tips.append("Complete internships")
    if float(profile["communication"]) < 4.2:
        tips.append("Improve communication skills")
    if int(profile["backlogs"]) >= 2:
        tips.append("Reduce backlogs before placement season")
    if float(profile["cgpa"]) < 7.5:
        tips.append("Raise your CGPA with steady academic work")
    if int(profile["tenth"]) < 70 or int(profile["twelfth"]) < 70:
        tips.append("Strengthen fundamentals and be ready to discuss school percentages")

    required = [
        "Improve technical skills",
        "Build more projects",
        "Complete internships",
        "Improve communication skills",
    ]
    for item in required:
        if item not in tips:
            tips.append(item)
    return tips


def verify_network():
    """Fail loudly if every student would receive the same status."""
    strong = predict_record(STRONG_PROFILE)
    weak = predict_record(WEAK_PROFILE)
    if strong["status"] == weak["status"]:
        raise RuntimeError(
            "The model returned the same placement status for a strong student "
            "and a weak student."
        )
    if strong["status"] != "Placed" or weak["status"] != "Not Placed":
        raise RuntimeError(
            f"Unexpected demo results: strong={strong['status']}, weak={weak['status']}"
        )
    return strong, weak


if __name__ == "__main__":
    strong, weak = verify_network()
    typical = predict_record(TYPICAL_PROFILE)
    print("strong", strong["status"], round(strong["prob_placed"], 4))
    print("weak", weak["status"], round(weak["prob_placed"], 4))
    print("typical", typical["status"], round(typical["prob_placed"], 4))
