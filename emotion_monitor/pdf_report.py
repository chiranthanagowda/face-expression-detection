import os
import io
import json
import base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


EMOTION_ORDER = ["Angry", "Disgusted", "Fearful", "Happy",
                 "Neutral", "Sad", "Surprised"]

EMOTION_COLORS_HEX = {
    "Angry": "#e74c3c",
    "Disgusted": "#8e44ad",
    "Fearful": "#34495e",
    "Happy": "#f1c40f",
    "Neutral": "#95a5a6",
    "Sad": "#3498db",
    "Surprised": "#e67e22",
}


def _b64_to_image(b64_str, max_width_cm=15):
    """Convert base64 PNG string to a ReportLab Image flowable."""
    if not b64_str:
        return None
    img_bytes = base64.b64decode(b64_str)
    img_io = io.BytesIO(img_bytes)
    img = Image(img_io)
    # Preserve aspect ratio
    w, h = img.imageWidth, img.imageHeight
    max_w = max_width_cm * cm
    scale = max_w / w
    img.drawWidth = w * scale
    img.drawHeight = h * scale
    return img


def build_session_pdf(session_row, output_path):
    """
    Build a professional PDF report for one session.

    session_row must contain:
        id, username, created_at, duration, total_faces,
        dominant_emotion, avg_confidence, transitions,
        emotion_counts (JSON), observations,
        chart (BLOB), timeline_chart (BLOB)
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"EduSense Session #{session_row['id']}",
        author="EduSense"
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                        fontSize=20, textColor=colors.HexColor('#1e3a8a'),
                        spaceAfter=10, alignment=TA_CENTER)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                        fontSize=14, textColor=colors.HexColor('#1e3a8a'),
                        spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle('Body', parent=styles['BodyText'],
                          fontSize=10, leading=14)
    small = ParagraphStyle('Small', parent=styles['BodyText'],
                           fontSize=8, textColor=colors.grey,
                           alignment=TA_CENTER)
    obs_style = ParagraphStyle('Obs', parent=styles['BodyText'],
                               fontSize=10, leading=15, spaceAfter=4)

    story = []

    # -------------------- HEADER -------------------- #
    story.append(Paragraph("EduSense", h1))
    story.append(Paragraph("Student Emotion Monitoring System — Session Report",
                           ParagraphStyle('Sub', parent=styles['BodyText'],
                                          alignment=TA_CENTER,
                                          textColor=colors.grey,
                                          fontSize=10)))
    story.append(Spacer(1, 0.5 * cm))

    # -------------------- META TABLE -------------------- #
    student = session_row['username'] if 'username' in session_row.keys() else 'N/A'
    avg_conf_pct = round((session_row['avg_confidence'] or 0) * 100, 1)

    meta_data = [
        ["Session ID", f"#{session_row['id']}"],
        ["Student", student],
        ["Date", str(session_row['created_at'])],
        ["Duration", f"{session_row['duration']} seconds"],
        ["Frames analysed", str(session_row['total_faces'])],
        ["Dominant emotion", session_row['dominant_emotion']],
        ["Average confidence", f"{avg_conf_pct}%"],
        ["Emotion transitions", str(session_row['transitions'])],
    ]
    meta_tbl = Table(meta_data, colWidths=[5 * cm, 10 * cm])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a8a')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.6 * cm))

    # -------------------- EMOTION DISTRIBUTION -------------------- #
    story.append(Paragraph("Emotion Distribution", h2))

    counts = json.loads(session_row['emotion_counts'] or "{}")
    total = sum(counts.values()) or 1

    dist_data = [["Emotion", "Frames", "Percentage"]]
    for e in EMOTION_ORDER:
        c = counts.get(e, 0)
        if c > 0:
            dist_data.append([e, str(c), f"{c / total * 100:.1f}%"])

    dist_tbl = Table(dist_data, colWidths=[6 * cm, 4 * cm, 5 * cm])
    dist_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f8fafc')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(dist_tbl)
    story.append(Spacer(1, 0.6 * cm))

    # -------------------- CHARTS -------------------- #
    story.append(PageBreak())
    story.append(Paragraph("Visual Analytics", h2))

    # Pie chart
    if session_row['chart']:
        pie_b64 = base64.b64encode(session_row['chart']).decode('utf-8')
        pie_img = _b64_to_image(pie_b64, max_width_cm=13)
        if pie_img:
            story.append(Paragraph("Distribution Chart", body))
            story.append(Spacer(1, 0.2 * cm))
            story.append(pie_img)
            story.append(Spacer(1, 0.6 * cm))

    # Timeline chart
    if session_row['timeline_chart']:
        tl_b64 = base64.b64encode(session_row['timeline_chart']).decode('utf-8')
        tl_img = _b64_to_image(tl_b64, max_width_cm=16)
        if tl_img:
            story.append(Paragraph("Emotion Timeline", body))
            story.append(Spacer(1, 0.2 * cm))
            story.append(tl_img)

    # -------------------- OBSERVATIONS -------------------- #
    story.append(PageBreak())
    story.append(Paragraph("Observations", h2))

    obs_text = session_row['observations'] or "No observations recorded."
    for line in obs_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.strip(), obs_style))

    story.append(Spacer(1, 0.5 * cm))

    # -------------------- DISCLAIMER BOX -------------------- #
    disclaimer = (
        "<b>Disclaimer:</b> These are estimates derived from observable "
        "facial-expression patterns. Facial expressions alone cannot "
        "establish attention, comprehension, or internal emotional state. "
        "This report is intended as an additional feedback channel, not as "
        "a definitive assessment of any student."
    )
    disc_tbl = Table([[Paragraph(disclaimer, body)]],
                     colWidths=[16 * cm])
    disc_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef3c7')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#f59e0b')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(disc_tbl)

    # -------------------- FOOTER -------------------- #
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated by EduSense on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        small))

    doc.build(story)
    return output_path
