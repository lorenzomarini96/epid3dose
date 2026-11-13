
from pathlib import Path

import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter


# =============================================================
# STATISTICS
# =============================================================

def calculate_statistics(array):
    """
    Calculate descriptive statistics for a NumPy array.
    """

    array = np.asarray(array, dtype=np.float64)
    finite_values = array[np.isfinite(array)]

    if finite_values.size == 0:
        return {
            "Shape": str(array.shape),
            "Number of elements": int(array.size),
            "Valid elements": 0,
            "Minimum": None,
            "Maximum": None,
            "Mean": None,
            "Standard deviation": None,
            "1st percentile": None,
            "5th percentile": None,
            "50th percentile": None,
            "95th percentile": None,
            "99th percentile": None,
        }

    percentiles = np.percentile(
        finite_values,
        [1, 5, 50, 95, 99]
    )

    return {
        "Shape": str(array.shape),
        "Number of elements": int(array.size),
        "Valid elements": int(finite_values.size),
        "Minimum": float(np.min(finite_values)),
        "Maximum": float(np.max(finite_values)),
        "Mean": float(np.mean(finite_values)),
        "Standard deviation": float(np.std(finite_values)),
        "1st percentile": float(percentiles[0]),
        "5th percentile": float(percentiles[1]),
        "50th percentile": float(percentiles[2]),
        "95th percentile": float(percentiles[3]),
        "99th percentile": float(percentiles[4]),
    }


# =============================================================
# COMMON REPORT DATA
# =============================================================

def _build_report_data(
    sample_name,
    ct_path,
    epid_path,
    weights_path,
    ct_array,
    epid_array,
    dose_normalized,
    dose_gy,
    model_config,
    elapsed_time,
    output_dir,
):
    """
    Prepare the information shared by the PDF and Excel reports.
    """

    output_dir = Path(output_dir)
    prediction_dir = output_dir.parent / "prediction"

    return {
        "sample_name": sample_name,
        "ct_path": str(ct_path),
        "epid_path": str(epid_path),
        "weights_path": str(weights_path),
        "model_config": model_config,
        "elapsed_time": elapsed_time,
        "ct_stats": calculate_statistics(ct_array),
        "epid_stats": calculate_statistics(epid_array),
        "normalized_stats": calculate_statistics(dose_normalized),
        "gy_stats": calculate_statistics(dose_gy),
        "generated_files": [
            (
                "Normalized dose prediction",
                str(prediction_dir / f"{sample_name}_normalized.npy"),
            ),
            (
                "Dose prediction (Gy)",
                str(prediction_dir / f"{sample_name}_Gy.npy"),
            ),
            (
                "PDF report",
                str(output_dir / f"{sample_name}_report.pdf"),
            ),
            (
                "Excel report",
                str(output_dir / f"{sample_name}_report.xlsx"),
            ),
        ],
    }


# =============================================================
# PDF REPORT
# =============================================================

def _get_pdf_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=20,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1F4E78"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
        )
    )

    return styles


def _pdf_table(data, col_widths=None):
    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2F3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#F3F6FA"),
            ]),
        ])
    )

    return table


def _format_value(value):
    if value is None:
        return "N/A"

    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"

    return str(value)


def _add_page_number(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        2 * cm,
        1 * cm,
        "EPID3DOSE - Dose Reconstruction Report"
    )

    canvas.drawRightString(
        A4[0] - 2 * cm,
        1 * cm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


def save_prediction_report(
    sample_name,
    ct_path,
    epid_path,
    weights_path,
    ct_array,
    epid_array,
    dose_normalized,
    dose_gy,
    model_config,
    output_dir,
    elapsed_time=None,
):
    """
    Generate the PDF prediction report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{sample_name}_report.pdf"

    data = _build_report_data(
        sample_name=sample_name,
        ct_path=ct_path,
        epid_path=epid_path,
        weights_path=weights_path,
        ct_array=ct_array,
        epid_array=epid_array,
        dose_normalized=dose_normalized,
        dose_gy=dose_gy,
        model_config=model_config,
        elapsed_time=elapsed_time,
        output_dir=output_dir,
    )

    styles = _get_pdf_styles()

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    story.append(
        Paragraph("EPID3DOSE", styles["ReportTitle"])
    )

    story.append(
        Paragraph(
            "3D Dose Reconstruction Report",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Sample:</b> {sample_name}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # 1. INPUT DATA
    # ---------------------------------------------------------

    story.append(
        Paragraph("1. Input data", styles["SectionTitle"])
    )

    input_data = [
        ["Parameter", "Value"],
        ["Sample", sample_name],
        ["CT file", data["ct_path"]],
        ["EPID file", data["epid_path"]],
        ["Model weights", data["weights_path"]],
    ]

    story.append(
        _pdf_table(
            input_data,
            col_widths=[4 * cm, 12 * cm]
        )
    )

    # ---------------------------------------------------------
    # 2. MODEL CONFIGURATION
    # ---------------------------------------------------------

    story.append(
        Paragraph("2. Model configuration", styles["SectionTitle"])
    )

    config = data["model_config"]

    config_data = [["Parameter", "Value"]]

    for key, value in config.items():
        config_data.append([str(key), _format_value(value)])

    if elapsed_time is not None:
        config_data.append([
            "Inference time (s)",
            _format_value(elapsed_time),
        ])

    story.append(
        _pdf_table(
            config_data,
            col_widths=[6 * cm, 10 * cm]
        )
    )

    # ---------------------------------------------------------
    # 3. PREPROCESSING AND NORMALIZATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "3. Preprocessing and normalization",
            styles["SectionTitle"]
        )
    )

    preprocessing_data = [
        ["Parameter", "Value"],
        ["CT shape", str(np.asarray(ct_array).shape)],
        ["EPID shape", str(np.asarray(epid_array).shape)],
        ["Dose shape", str(np.asarray(dose_gy).shape)],
        ["Dose normalization maximum (Gy)", "6.916368"],
        ["Dose conversion", "Normalized dose x 6.916368 Gy"],
    ]

    story.append(
        _pdf_table(
            preprocessing_data,
            col_widths=[7 * cm, 9 * cm]
        )
    )

    # ---------------------------------------------------------
    # 4. INPUT STATISTICS
    # ---------------------------------------------------------

    story.append(
        Paragraph("4. Input statistics", styles["SectionTitle"])
    )

    for input_name, stats in [
        ("CT", data["ct_stats"]),
        ("EPID", data["epid_stats"]),
    ]:
        story.append(
            Paragraph(f"<b>{input_name}</b>", styles["Normal"])
        )

        table_data = [["Statistic", "Value"]]

        for key, value in stats.items():
            table_data.append([key, _format_value(value)])

        story.append(_pdf_table(table_data))
        story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 5. PREDICTED DOSE STATISTICS
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "5. Predicted dose statistics",
            styles["SectionTitle"]
        )
    )

    for dose_name, stats in [
        ("Normalized dose", data["normalized_stats"]),
        ("Dose in Gy", data["gy_stats"]),
    ]:
        story.append(
            Paragraph(f"<b>{dose_name}</b>", styles["Normal"])
        )

        table_data = [["Statistic", "Value"]]

        for key, value in stats.items():
            table_data.append([key, _format_value(value)])

        story.append(_pdf_table(table_data))
        story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # 6. GENERATED FILES
    # ---------------------------------------------------------

    story.append(
        Paragraph("6. Generated files", styles["SectionTitle"])
    )

    files_data = [["File", "Path"]]

    for description, path in data["generated_files"]:
        files_data.append([description, path])

    story.append(
        _pdf_table(
            files_data,
            col_widths=[5 * cm, 11 * cm]
        )
    )

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Note:</b> This report documents the computational "
            "inference performed by EPID3DOSE. The reported statistics "
            "describe the predicted dose distribution and do not "
            "constitute an independent dosimetric validation.",
            styles["SmallText"]
        )
    )

    doc.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    print(f"[REPORT] PDF report saved: {report_path}")

    return report_path


# =============================================================
# EXCEL REPORT
# =============================================================

def save_excel_report(
    sample_name,
    ct_path,
    epid_path,
    weights_path,
    ct_array,
    epid_array,
    dose_normalized,
    dose_gy,
    model_config,
    output_dir,
    elapsed_time=None,
):
    """
    Generate an Excel report containing the same information
    as the PDF report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{sample_name}_report.xlsx"

    data = _build_report_data(
        sample_name=sample_name,
        ct_path=ct_path,
        epid_path=epid_path,
        weights_path=weights_path,
        ct_array=ct_array,
        epid_array=epid_array,
        dose_normalized=dose_normalized,
        dose_gy=dose_gy,
        model_config=model_config,
        elapsed_time=elapsed_time,
        output_dir=output_dir,
    )

    wb = Workbook()

    # ---------------------------------------------------------
    # STYLES
    # ---------------------------------------------------------

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )

    header_font = Font(
        bold=True,
        color="FFFFFF",
    )

    section_font = Font(
        bold=True,
        size=12,
        color="1F4E78",
    )

    thin_border = Border(
        bottom=Side(
            style="thin",
            color="D9E2F3",
        )
    )

    def style_sheet(ws):
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=True,
                )

                if cell.value is not None:
                    cell.border = thin_border

        for column_cells in ws.columns:
            max_length = 0
            column_letter = get_column_letter(
                column_cells[0].column
            )

            for cell in column_cells:
                if cell.value is not None:
                    max_length = max(
                        max_length,
                        len(str(cell.value))
                    )

            ws.column_dimensions[column_letter].width = min(
                max(max_length + 3, 14),
                60,
            )

    def write_header(ws, row=1):
        for cell in ws[row]:
            if cell.value is not None:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

    def write_statistics(ws, start_row, title, stats):
        ws.cell(start_row, 1, title)
        ws.cell(start_row, 1).font = section_font

        ws.cell(start_row + 1, 1, "Statistic")
        ws.cell(start_row + 1, 2, "Value")

        for cell in ws[start_row + 1]:
            cell.fill = header_fill
            cell.font = header_font

        row = start_row + 2

        for key, value in stats.items():
            ws.cell(row, 1, key)
            ws.cell(row, 2, value)
            row += 1

        return row + 1

    # ---------------------------------------------------------
    # 1. SUMMARY
    # ---------------------------------------------------------

    ws = wb.active
    ws.title = "Summary"

    ws.append(["Parameter", "Value"])

    summary_data = [
        ["Sample", sample_name],
        ["Model weights", data["weights_path"]],
        ["CT input", data["ct_path"]],
        ["EPID input", data["epid_path"]],
        ["Inference time (s)", elapsed_time],
    ]

    for key, value in model_config.items():
        summary_data.append([str(key), value])

    for row in summary_data:
        ws.append(row)

    write_header(ws)
    style_sheet(ws)

    # ---------------------------------------------------------
    # 2. INPUT DATA
    # ---------------------------------------------------------

    ws = wb.create_sheet("Input data")

    ws.append(["Input", "Statistic", "Value"])

    for input_name, stats in [
        ("CT", data["ct_stats"]),
        ("EPID", data["epid_stats"]),
    ]:
        for key, value in stats.items():
            ws.append([input_name, key, value])

    write_header(ws)
    style_sheet(ws)

    # ---------------------------------------------------------
    # 3. PREDICTED DOSE
    # ---------------------------------------------------------

    ws = wb.create_sheet("Predicted dose")

    ws.append([
        "Dose representation",
        "Statistic",
        "Value",
    ])

    for dose_name, stats in [
        ("Normalized dose", data["normalized_stats"]),
        ("Dose (Gy)", data["gy_stats"]),
    ]:
        for key, value in stats.items():
            ws.append([dose_name, key, value])

    write_header(ws)
    style_sheet(ws)

    # ---------------------------------------------------------
    # 4. GENERATED FILES
    # ---------------------------------------------------------

    ws = wb.create_sheet("Generated files")

    ws.append(["File", "Path"])

    for description, path in data["generated_files"]:
        ws.append([description, path])

    write_header(ws)
    style_sheet(ws)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    wb.save(report_path)

    print(f"[REPORT] Excel report saved: {report_path}")

    return report_path
