from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import os
import datetime

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically and renders header & footer."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 760, "Automated Data Analysis & ML Diagnostic Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 752, 576, 752)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 40, 576, 40)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.drawString(36, 28, f"Generated on {timestamp} | Confidential & Proprietary")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 28, page_str)
        self.restoreState()


def create_section_header(title: str, style):
    return [
        Spacer(1, 10),
        Paragraph(title, style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceBefore=4, spaceAfter=10)
    ]


def append_plot_section(story, title, plot_path, h1_style, h2_style, width, height):
    story.extend(create_section_header(title, h1_style))
    if plot_path and os.path.exists(plot_path):
        story.append(Image(plot_path, width=width, height=height))
    else:
        story.append(Paragraph("Not available for this dataset.", h2_style))
    story.append(Spacer(1, 10))


def generate_pdf_report(metrics: dict, output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=46,
        bottomMargin=46
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E3A8A")
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155")
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )

    overview = metrics.get('overview', {})
    
    # =========================================================================
    # Cover / Header Banner
    # =========================================================================
    story.append(Paragraph("Automated Data Analysis & ML Diagnostic Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Comprehensive Exploratory Data Analysis, Feature Distributions, Dimensionality Reduction & Predictive Modeling", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1E3A8A"), spaceBefore=8, spaceAfter=14))
    
    # =========================================================================
    # 0. Executive Summary (from LLM)
    # =========================================================================
    if 'llm_summary' in metrics:
        story.extend(create_section_header("Executive Summary", h1_style))
        for p in metrics['llm_summary'].split('\n'):
            p = p.strip()
            if p:
                story.append(Paragraph(p, body_style))
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))
    
    # =========================================================================
    # 1. Dataset Overview
    # =========================================================================
    story.extend(create_section_header("1. Dataset Overview", h1_style))
    
    overview_table_data = [
        [Paragraph("Metric", table_hdr_style), Paragraph("Value", table_hdr_style), Paragraph("Metric", table_hdr_style), Paragraph("Value", table_hdr_style)],
        [Paragraph("Total Rows", table_cell_style), Paragraph(f"{overview.get('rows', metrics.get('rows', 'N/A')):,}", table_cell_style),
         Paragraph("Total Columns", table_cell_style), Paragraph(f"{overview.get('cols', metrics.get('cols', 'N/A'))}", table_cell_style)],
        [Paragraph("Numeric Columns", table_cell_style), Paragraph(f"{overview.get('num_cols_count', len(metrics.get('num_cols', [])))}", table_cell_style),
         Paragraph("Categorical Columns", table_cell_style), Paragraph(f"{overview.get('cat_cols_count', len(metrics.get('cat_cols', [])))}", table_cell_style)],
        [Paragraph("Overall Missing Data Ratio", table_cell_style), Paragraph(f"{overview.get('missing_pct', metrics.get('missing_pct', 0)):.2f}%", table_cell_style),
         Paragraph("Total Missing Cells", table_cell_style), Paragraph(f"{overview.get('total_nulls', 0):,}", table_cell_style)],
        [Paragraph("Duplicate Rows", table_cell_style), Paragraph(f"{overview.get('duplicate_rows', 0):,} ({overview.get('duplicate_pct', 0):.2f}%)", table_cell_style),
         Paragraph("Memory Footprint", table_cell_style), Paragraph(f"{overview.get('memory_mb', 0):.2f} MB", table_cell_style)],
        [Paragraph("Target Column Identified", table_cell_style), Paragraph(f"{metrics.get('target_col', 'None')}", table_cell_style),
         Paragraph("Silhouette Score", table_cell_style), Paragraph(f"{metrics.get('silhouette_score', 0):.3f}", table_cell_style)]
    ]
    
    t_overview = Table(overview_table_data, colWidths=[130, 140, 130, 140])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 12))

    # =========================================================================
    # 2. Top 5 Columns by Missing Values (Numeric Columns list)
    # =========================================================================
    story.extend(create_section_header("2. Top 5 Columns by Missing Values (Numeric Columns list)", h1_style))
    
    top5 = metrics.get('top5_missing', [])
    top5_data = [
        [Paragraph("Rank", table_hdr_style), Paragraph("Column Name", table_hdr_style), Paragraph("Missing Count", table_hdr_style), Paragraph("Missing Percentage (%)", table_hdr_style)]
    ]
    for idx, item in enumerate(top5):
        top5_data.append([
            Paragraph(str(idx + 1), table_cell_style),
            Paragraph(str(item.get('column', '')), table_cell_style),
            Paragraph(f"{item.get('missing_count', 0):,}", table_cell_style),
            Paragraph(f"{item.get('missing_pct', 0):.2f}%", table_cell_style)
        ])
    if len(top5_data) == 1:
        top5_data.append([Paragraph("1", table_cell_style), Paragraph("None (No missing values detected)", table_cell_style), Paragraph("0", table_cell_style), Paragraph("0.00%", table_cell_style)])
        
    t_top5 = Table(top5_data, colWidths=[50, 220, 120, 150])
    t_top5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D9488")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
    ]))
    story.append(t_top5)
    story.append(Spacer(1, 8))
    
    num_list_str = ", ".join(metrics.get('numeric_columns_list', metrics.get('num_cols', [])))
    if not num_list_str:
        num_list_str = "None detected"
    story.append(Paragraph(f"<b>Numeric Columns list ({len(metrics.get('numeric_columns_list', []))} features):</b> <font color='#475569'>{num_list_str}</font>", body_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3-8. Numeric Summary (features 1–5 to features 26–30)
    # =========================================================================
    num_summaries = metrics.get('numeric_summaries', {})
    summary_chunks = [
        ("3. Numeric Summary (features 1–5)", "features 1–5"),
        ("4. Numeric Summary (features 6–10)", "features 6–10"),
        ("5. Numeric Summary (features 11–15)", "features 11–15"),
        ("6. Numeric Summary (features 16–20)", "features 16–20"),
        ("7. Numeric Summary (features 21–25)", "features 21–25"),
        ("8. Numeric Summary (features 26–30)", "features 26–30")
    ]

    for section_title, chunk_label in summary_chunks:
        chunk_data = num_summaries.get(chunk_label, [])
        story.append(Paragraph(section_title, h2_style))
        story.append(Spacer(1, 4))
        
        headers = [
            Paragraph("Feature", table_hdr_style),
            Paragraph("Mean", table_hdr_style),
            Paragraph("Std", table_hdr_style),
            Paragraph("Min", table_hdr_style),
            Paragraph("25%", table_hdr_style),
            Paragraph("50% (Med)", table_hdr_style),
            Paragraph("75%", table_hdr_style),
            Paragraph("Max", table_hdr_style),
            Paragraph("Skew", table_hdr_style),
            Paragraph("Kurt", table_hdr_style),
            Paragraph("Nulls", table_hdr_style),
        ]
        t_rows = [headers]
        
        if chunk_data:
            for item in chunk_data:
                t_rows.append([
                    Paragraph(str(item.get('feature', '')), table_cell_style),
                    Paragraph(f"{item.get('mean', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('std', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('min', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('25%', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('50%', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('75%', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('max', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('skewness', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('kurtosis', 0):.2f}", table_cell_style),
                    Paragraph(f"{item.get('missing_count', 0)}", table_cell_style),
                ])
        else:
            t_rows.append([Paragraph("N/A - Feature range exceeds dataset dimensions", table_cell_style)] + [Paragraph("-", table_cell_style)]*10)

        t_chunk = Table(t_rows, colWidths=[110, 43, 43, 43, 43, 45, 43, 45, 43, 43, 39])
        t_chunk.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(t_chunk)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # =========================================================================
    # 9. Correlation Heatmap
    # =========================================================================
    append_plot_section(story, "9. Correlation Heatmap", metrics.get('correlation_heatmap_plot'), h1_style, h2_style, 480, 360)

    story.append(PageBreak())

    # =========================================================================
    # 10. Categorical Columns — Value counts: diagnosis
    # =========================================================================
    story.extend(create_section_header("10. Categorical Columns — Value counts: diagnosis", h1_style))
    
    target_col_name = metrics.get('target_col', 'diagnosis')
    diag_vc = metrics.get('diagnosis_value_counts', [])
    
    cat_table_data = [
        [Paragraph("Category / Class", table_hdr_style), Paragraph("Count", table_hdr_style), Paragraph("Percentage (%)", table_hdr_style)]
    ]
    if diag_vc:
        for item in diag_vc:
            cat_table_data.append([
                Paragraph(str(item.get('category', '')), table_cell_style),
                Paragraph(f"{item.get('count', 0):,}", table_cell_style),
                Paragraph(f"{item.get('percentage', 0):.2f}%", table_cell_style)
            ])
    else:
        cat_table_data.append([Paragraph("N/A", table_cell_style), Paragraph("-", table_cell_style), Paragraph("-", table_cell_style)])
        
    t_cat = Table(cat_table_data, colWidths=[200, 160, 180])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F9FF")]),
    ]))
    story.append(t_cat)
    story.append(Spacer(1, 10))
    
    # =========================================================================
    # 11. Distribution of diagnosis
    # =========================================================================
    append_plot_section(story, f"11. Distribution of diagnosis ({target_col_name})", metrics.get('target_distribution_plot'), h1_style, h2_style, 440, 230)

    story.append(PageBreak())

    # =========================================================================
    # 12-16. KDE Plots – Page 1 to Page 5
    # =========================================================================
    for page in range(1, 6):
        plot_key = f'kde_page_{page}_plot'
        section_num = 11 + page
        append_plot_section(story, f"{section_num}. KDE Plots – Page {page}", metrics.get(plot_key), h1_style, h2_style, 520, 360)
        story.append(PageBreak())

    # =========================================================================
    # 17. Outlier Proportion per Column
    # =========================================================================
    append_plot_section(story, "17. Outlier Proportion per Column", metrics.get('outlier_plot'), h1_style, h2_style, 480, 240)

    # =========================================================================
    # 18. Duplicate Distribution per Column
    # =========================================================================
    append_plot_section(story, "18. Duplicate Distribution per Column", metrics.get('duplicate_plot'), h1_style, h2_style, 480, 240)

    story.append(PageBreak())

    # =========================================================================
    # 19. Feature Importance
    # =========================================================================
    append_plot_section(story, "19. Feature Importance", metrics.get('feature_importance_plot'), h1_style, h2_style, 480, 250)

    # =========================================================================
    # 20. PCA Clusters (Silhouette = {sil_score:.3f})
    # =========================================================================
    sil_score = metrics.get('silhouette_score', 0)
    pca_title = f"20. PCA Clusters (Silhouette = {sil_score:.3f})"
    append_plot_section(story, pca_title, metrics.get('pca_plot'), h1_style, h2_style, 480, 260)

    story.append(PageBreak())

    # =========================================================================
    # 21. Predicted Probability vs Actual
    # =========================================================================
    append_plot_section(story, "21. Predicted Probability vs Actual", metrics.get('pred_prob_vs_actual_plot'), h1_style, h2_style, 460, 230)

    # =========================================================================
    # 22. Model Summary & ROC/Residuals
    # =========================================================================
    story.extend(create_section_header("22. Model Summary & ROC/Residuals", h1_style))
    
    if metrics.get('is_classification', True):
        model_metrics_data = [
            [Paragraph("Metric", table_hdr_style), Paragraph("Score", table_hdr_style), Paragraph("Evaluation Note", table_hdr_style)],
            [Paragraph("Accuracy", table_cell_style), Paragraph(f"{metrics.get('accuracy', 0):.4f}", table_cell_style), Paragraph("Test split classification accuracy", table_cell_style)],
            [Paragraph("ROC AUC", table_cell_style), Paragraph(f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else "N/A", table_cell_style), Paragraph("Area under ROC curve", table_cell_style)],
            [Paragraph("Precision (Weighted)", table_cell_style), Paragraph(f"{metrics.get('precision', 0):.4f}", table_cell_style), Paragraph("Weighted precision score", table_cell_style)],
            [Paragraph("Recall (Weighted)", table_cell_style), Paragraph(f"{metrics.get('recall', 0):.4f}", table_cell_style), Paragraph("Weighted recall score", table_cell_style)],
            [Paragraph("F1-Score (Weighted)", table_cell_style), Paragraph(f"{metrics.get('f1_score', 0):.4f}", table_cell_style), Paragraph("Harmonic mean of precision and recall", table_cell_style)]
        ]
    else:
        model_metrics_data = [
            [Paragraph("Metric", table_hdr_style), Paragraph("Score", table_hdr_style), Paragraph("Evaluation Note", table_hdr_style)],
            [Paragraph("R² Score", table_cell_style), Paragraph(f"{metrics.get('r2_score', 0):.4f}", table_cell_style), Paragraph("Coefficient of determination", table_cell_style)],
            [Paragraph("RMSE", table_cell_style), Paragraph(f"{metrics.get('rmse', 0):.4f}", table_cell_style), Paragraph("Root Mean Squared Error", table_cell_style)],
            [Paragraph("MAE", table_cell_style), Paragraph(f"{metrics.get('mae', 0):.4f}", table_cell_style), Paragraph("Mean Absolute Error", table_cell_style)]
        ]
        
    t_model = Table(model_metrics_data, colWidths=[140, 100, 300])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4338CA")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2FF")]),
    ]))
    story.append(t_model)
    story.append(Spacer(1, 10))

    if metrics.get('roc_plot') and os.path.exists(metrics['roc_plot']):
        story.append(Image(metrics['roc_plot'], width=440, height=240))
        story.append(Spacer(1, 10))

    if metrics.get('confusion_matrix_plot') and os.path.exists(metrics['confusion_matrix_plot']):
        story.append(Paragraph("Confusion Matrix", h2_style))
        story.append(Spacer(1, 4))
        story.append(Image(metrics['confusion_matrix_plot'], width=380, height=240))

    # Build the document with custom multi-page canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_pdf_path