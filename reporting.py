
"""
CardioScore Pro - Clinical-style PDF report and CSV export.
Model performance section removed; CSV export preserved.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

import train_model as tm


# ---------------------------------------------------------
# THEME
# ---------------------------------------------------------

RISK_COLORS = {
    "LOW": "#15803d",
    "MODERATE": "#b45309",
    "HIGH": "#b91c1c",
}

NAVY = colors.HexColor("#14213d")
LIGHT = colors.HexColor("#f1f5f9")
BORDER = colors.HexColor("#cbd5e1")
MUTED = colors.HexColor("#64748b")


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def format_datetime(iso_text: str) -> str:
    try:
        return datetime.fromisoformat(iso_text).strftime(
            "%d %b %Y, %H:%M"
        )
    except (TypeError, ValueError):
        return str(iso_text or "N/A")


def _styles() -> dict:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "clinical_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            alignment=0,
            spaceAfter=3,
        ),
        "sub": ParagraphStyle(
            "clinical_subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "clinical_heading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "clinical_body",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
        ),
        "cell": ParagraphStyle(
            "clinical_cell",
            parent=base["Normal"],
            fontSize=8.5,
            leading=10.5,
        ),
        "small": ParagraphStyle(
            "clinical_small",
            parent=base["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#475569"),
        ),
        "disclaimer": ParagraphStyle(
            "clinical_disclaimer",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#7f1d1d"),
        ),
    }


def _p(text, style) -> Paragraph:
    """Create a safely escaped paragraph."""
    return Paragraph(escape(str(text)), style)


def _grid(rows, col_widths, header=True, styles=None):
    """Create a clean clinical-style table."""
    styles = styles or _styles()

    data = [
        [
            value
            if isinstance(value, Paragraph)
            else _p(value, styles["cell"])
            for value in row
        ]
        for row in rows
    ]

    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        (
            "ROWBACKGROUNDS",
            (0, 1 if header else 0),
            (-1, -1),
            [colors.white, LIGHT],
        ),
    ]

    if header and data:
        data[0] = [
            Paragraph(
                f"<font color='white'><b>{escape(str(value))}</b></font>",
                styles["cell"],
            )
            for value in rows[0]
        ]
        commands.append(
            ("BACKGROUND", (0, 0), (-1, 0), NAVY)
        )

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle(commands))

    return table


def _disclaimer_box(width, styles):
    disclaimer = escape(
        str(
            getattr(
                tm,
                "DISCLAIMER",
                "This report is for educational purposes only and is not a medical diagnosis.",
            )
        )
    )

    box = Table(
        [[
            Paragraph(
                f"<b>IMPORTANT MEDICAL DISCLAIMER</b><br/>{disclaimer}",
                styles["disclaimer"],
            )
        ]],
        colWidths=[width],
    )

    box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1),
             colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 0.8,
             colors.HexColor("#fca5a5")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    return box


def _footer(canvas, doc):
    canvas.saveState()

    page_width, _ = A4

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(
        18 * mm,
        15 * mm,
        page_width - 18 * mm,
        15 * mm,
    )

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)

    canvas.drawString(
        18 * mm,
        10 * mm,
        "CardioScore Pro | Educational estimate, not a diagnosis",
    )

    canvas.drawRightString(
        page_width - 18 * mm,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def _result_card(width, probability, level, styles):
    color = colors.HexColor(
        RISK_COLORS.get(str(level).upper(), "#334155")
    )

    percentage_style = ParagraphStyle(
        "result_percentage",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=29,
        alignment=1,
        textColor=color,
    )

    risk_style = ParagraphStyle(
        "result_risk",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=color,
    )

    muted_style = ParagraphStyle(
        "result_note",
        parent=styles["small"],
        alignment=1,
    )

    data = [
        [
            Paragraph("ESTIMATED MODEL PROBABILITY", muted_style),
            Paragraph("ESTIMATED RISK CATEGORY", muted_style),
        ],
        [
            Paragraph(
                f"<b>{probability * 100:.0f}%</b>",
                percentage_style,
            ),
            Paragraph(
                f"<b>{escape(str(level).upper())} RISK</b>",
                risk_style,
            ),
        ],
        [
            Paragraph(
                "Relative to the training dataset.",
                muted_style,
            ),
            Paragraph(
                "Not a clinical classification.",
                muted_style,
            ),
        ],
    ]

    table = Table(
        data,
        colWidths=[width / 2, width / 2],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1),
             colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
            ("LINEBEFORE", (1, 0), (1, -1), 0.6, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    return table


# ---------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------

def build_pdf_report(
    user: dict,
    assessment: dict,
    meta: dict | None = None,
) -> bytes:
    """
    Generate a clinical-style PDF report.

    Page 1:
        Assessment overview, risk result, disclaimer,
        and patient details.

    Page 2:
        Influencing factors and limitations.

    Model performance metrics are intentionally omitted.
    """

    styles = _styles()
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=20 * mm,
        title="CardioScore Pro - Clinical Risk Assessment",
        author="CardioScore Pro",
    )

    page_width = A4[0] - 36 * mm

    level = str(assessment.get("risk_level", "UNKNOWN")).upper()
    probability = float(assessment.get("probability", 0.0))
    probability = max(0.0, min(1.0, probability))

    report_id = (
        f"CSP-{int(assessment['id']):05d}"
        if assessment.get("id") is not None
        else "N/A"
    )

    full_name = (
        user.get("full_name")
        or user.get("username")
        or "User"
    )
    username = user.get("username", "")

    story = []

    # Header
    story.append(_p("CardioScore Pro", styles["title"]))
    story.append(
        _p(
            "CLINICAL-STYLE REPORT | HEART DISEASE RISK ASSESSMENT",
            styles["sub"],
        )
    )

    # Assessment overview
    story.append(
        _p("Assessment overview", styles["h2"])
    )

    overview_rows = [
        ["Report ID", report_id],
        ["Prepared for", full_name],
        ["Username", f"@{username}" if username else "N/A"],
        [
            "Assessment date",
            format_datetime(assessment.get("created_at")),
        ],
        [
            "Model used",
            assessment.get("model_name", "N/A"),
        ],
        [
            "Report generated",
            datetime.now().strftime("%d %b %Y, %H:%M"),
        ],
        ["Developed by", "Shashant Sahu"],
    ]

    story.append(
        _grid(
            overview_rows,
            [42 * mm, page_width - 42 * mm],
            header=False,
            styles=styles,
        )
    )

    story.append(Spacer(1, 8))

    # Result card
    story.append(
        _result_card(
            page_width,
            probability,
            level,
            styles,
        )
    )

    story.append(Spacer(1, 9))
    story.append(_disclaimer_box(page_width, styles))

    # Patient details
    story.append(
        _p("Patient details entered", styles["h2"])
    )

    input_rows = tm.describe_inputs(
        assessment.get("inputs", {})
    )

    patient_table_rows = [["Patient feature", "Recorded value"]]

    for row in input_rows:
        patient_table_rows.append([
            str(row[0]),
            str(row[1]),
        ])

    story.append(
        _grid(
            patient_table_rows,
            [page_width * 0.62, page_width * 0.38],
            header=True,
            styles=styles,
        )
    )

    # Page 2: explanatory information
    story.append(PageBreak())

    story.append(
        _p("Assessment explanation", styles["title"])
    )
    story.append(
        _p(
            "Factors associated with this model estimate",
            styles["sub"],
        )
    )

    factors = assessment.get("factors") or []

    if factors:
        story.append(
            _p(
                "Factors that influenced this estimate",
                styles["h2"],
            )
        )

        story.append(
            _p(
                "The table shows how the model estimate changes "
                "when an input is compared with a typical training "
                "value. A positive effect increases the estimate; "
                "a negative effect decreases it. These effects "
                "describe model behaviour, not medical causes.",
                styles["small"],
            )
        )

        story.append(Spacer(1, 6))

        factor_rows = [[
            "Factor",
            "Your value",
            "Typical value",
            "Effect",
        ]]

        for factor in factors:
            try:
                effect = float(factor.get("influence", 0))
                effect_text = f"{effect:+.1f}"
            except (TypeError, ValueError):
                effect_text = "N/A"

            factor_rows.append([
                str(factor.get("label", "N/A")),
                str(factor.get("value", "N/A")),
                str(factor.get("typical", "N/A")),
                effect_text,
            ])

        story.append(
            _grid(
                factor_rows,
                [
                    page_width * 0.34,
                    page_width * 0.22,
                    page_width * 0.24,
                    page_width * 0.20,
                ],
                header=True,
                styles=styles,
            )
        )
    else:
        story.append(
            _p(
                "No influencing-factor details were available "
                "for this assessment.",
                styles["body"],
            )
        )

    # Limitations: retained; model performance omitted.
    story.append(
        _p("Limitations", styles["h2"])
    )

    limitations = getattr(tm, "LIMITATIONS", [])

    if limitations:
        for item in limitations:
            story.append(
                Paragraph(
                    "&#8226;&nbsp; " + escape(str(item)),
                    styles["small"],
                )
            )
            story.append(Spacer(1, 3))
    else:
        default_limitations = [
            "The model is trained on a limited dataset and may not "
            "generalise to every population.",
            "The estimate is not a confirmed diagnosis or a "
            "substitute for medical evaluation.",
            "Risk bands are display labels, not validated clinical "
            "decision thresholds.",
            "Factor explanations describe model behaviour and "
            "should not be interpreted as medical causes.",
        ]

        for item in default_limitations:
            story.append(
                Paragraph(
                    "&#8226;&nbsp; " + escape(item),
                    styles["small"],
                )
            )
            story.append(Spacer(1, 3))

    story.append(Spacer(1, 8))
    story.append(_disclaimer_box(page_width, styles))

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
    )

    return buffer.getvalue()


# ---------------------------------------------------------
# CSV EXPORT
# Kept separate from PDF layout changes.
# ---------------------------------------------------------

def assessments_to_csv(assessments: list) -> bytes:
    """Export assessments as UTF-8 CSV for Excel."""

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "assessment_id",
            "assessment_date",
            "model",
            "risk_probability_percent",
            "risk_level",
        ] + tm.FEATURE_NAMES
    )

    for assessment in assessments:
        row = [
            assessment["id"],
            format_datetime(assessment["created_at"]),
            assessment["model_name"],
            f"{assessment['probability'] * 100:.1f}",
            assessment["risk_level"],
        ]

        inputs = assessment.get("inputs", {})

        row += [
            tm.display_value(feature, inputs[feature])
            if feature in inputs
            else ""
            for feature in tm.FEATURE_NAMES
        ]

        writer.writerow(row)

    return output.getvalue().encode("utf-8-sig")
