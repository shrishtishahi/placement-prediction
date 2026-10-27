"""One-page placement report."""

from datetime import datetime
from io import BytesIO

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = "#1F4E79"
GREEN = "#1E7F4F"
RED = "#C0392B"
SLATE = "#5D6D7E"


def _clean(value):
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report(profile, result, suggestions):
    """Return a PDF as bytes."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Placement Prediction Report",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=HexColor(NAVY),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    muted = ParagraphStyle(
        "Muted",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=HexColor(SLATE),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    heading = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=HexColor(NAVY),
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=HexColor("#1C2833"),
        alignment=TA_LEFT,
    )
    result_color = GREEN if result["status"] == "Placed" else RED
    result_style = ParagraphStyle(
        "ResultLine",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=HexColor(result_color),
        alignment=TA_CENTER,
        spaceBefore=8,
        spaceAfter=2,
    )

    student = profile.get("student_name", "").strip() or "Not provided"
    rows = [
        ["Detail", "Value"],
        ["Student name", _clean(student)],
        ["CGPA", f"{float(profile['cgpa']):.1f}"],
        ["Major projects", str(int(profile["major"]))],
        ["Workshops / certifications", str(int(profile["workshops"]))],
        ["Mini projects", str(int(profile["mini"]))],
        ["Skills", str(int(profile["skills"]))],
        ["Communication skill rating", f"{float(profile['communication']):.1f}"],
        ["Internship", _clean(profile["internship"])],
        ["Hackathon", _clean(profile["hackathon"])],
        ["10th percentage", f"{int(profile['tenth'])}%"],
        ["12th percentage", f"{int(profile['twelfth'])}%"],
        ["Backlogs", str(int(profile["backlogs"]))],
    ]

    table = Table(rows, colWidths=[90 * mm, 80 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 1), (-1, -1), HexColor("#F7FAFC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F4F7FB")]),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#D5DDE5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    confidence = result["confidence"] * 100
    placed_chance = result["prob_placed"] * 100
    story = [
        Paragraph("Student Placement Prediction", title),
        Paragraph("Prediction report", muted),
        Paragraph(datetime.now().strftime("Generated on %d %B %Y, %I:%M %p"), muted),
        HRFlowable(width="100%", thickness=1.5, color=HexColor(NAVY), spaceAfter=6),
        Paragraph(f"Placement status: {result['status']}", result_style),
        Paragraph(f"Confidence in this result: {confidence:.1f}%", muted),
        Paragraph(f"Probability of being placed: {placed_chance:.1f}%", muted),
        Paragraph("Student details", heading),
        table,
        Paragraph("Suggestions", heading),
    ]
    for tip in suggestions:
        story.append(Paragraph(f"• {_clean(tip)}", body))
        story.append(Spacer(1, 2))
    story.extend(
        [
            Spacer(1, 16),
            HRFlowable(width="100%", thickness=0.4, color=HexColor("#D5DDE5")),
            Paragraph(
                "Generated by the Student Placement Prediction System. "
                "This report is a model estimate for academic demonstration.",
                ParagraphStyle(
                    "Footer",
                    parent=body,
                    fontSize=8,
                    textColor=HexColor("#95A5A6"),
                    alignment=TA_CENTER,
                    spaceBefore=8,
                ),
            ),
        ]
    )
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()
