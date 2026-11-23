"""
IndicSent-M: PDF Report Generation Router
=========================================
Generates comprehensive PDF reports for meme sentiment predictions and explainability details.
Uses ReportLab.
"""

import io
import base64
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

router = APIRouter(prefix="/report", tags=["Reports"])


@router.post(
    "/pdf",
    summary="Download PDF Sentiment Report",
    description="Generates and streams a PDF report containing prediction data and optional explainable AI results."
)
def generate_pdf_report(
    model_name: str = Form("MuRIL + ResNet-50 (Gated Fusion)"),
    mode: str = Form("multimodal"),
    predicted_sentiment: str = Form("Neutral"),
    confidence_positive: float = Form(0.0),
    confidence_neutral: float = Form(0.0),
    confidence_negative: float = Form(0.0),
    processing_time_ms: float = Form(0.0),
    input_text: Optional[str] = Form(None),
    explanation_summary: Optional[str] = Form(None),
    gate_text_weight: Optional[float] = Form(None),
    gate_image_weight: Optional[float] = Form(None),
    top_words: Optional[str] = Form(None),
    heatmap_base64: Optional[str] = Form(None),
    dominant_emotion: Optional[str] = Form(None),
    dominant_emotion_probability: Optional[float] = Form(None),
    dominant_emotion_emoji: Optional[str] = Form(None),
    emotion_probabilities: Optional[str] = Form(None)
):
    """
    Compiles prediction results and optional explanation into a formatted PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#214B43')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#556B64')
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#214B43'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#172923')
    )

    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#214B43')
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("IndicSent-M — AI Sentiment Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Model: {model_name}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#214B43'), spaceAfter=14))

    # 2. Prediction Summary Table
    sentiment_color = '#214B43'
    if predicted_sentiment == 'Negative':
        sentiment_color = '#991B1B'
    elif predicted_sentiment == 'Neutral':
        sentiment_color = '#92400E'

    summary_data = [
        [
            Paragraph("<b>Predicted Sentiment:</b>", body_style),
            Paragraph(f"<font color='{sentiment_color}'><b>{predicted_sentiment}</b></font>", badge_style),
            Paragraph("<b>Processing Time:</b>", body_style),
            Paragraph(f"{processing_time_ms} ms", body_style)
        ],
        [
            Paragraph("<b>Analysis Mode:</b>", body_style),
            Paragraph(mode.capitalize(), body_style),
            Paragraph("<b>Model Architecture:</b>", body_style),
            Paragraph(model_name, body_style)
        ]
    ]

    t_summary = Table(summary_data, colWidths=[110, 160, 110, 160])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7F8F5')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E3E8DE')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E3E8DE')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 14))

    # 3. Confidence Distribution Table
    elements.append(Paragraph("Confidence Score Distribution", h2_style))
    conf_data = [
        [
            Paragraph("<b>Sentiment Class</b>", body_style),
            Paragraph("<b>Confidence</b>", body_style),
            Paragraph("<b>Visual Share</b>", body_style)
        ],
        [
            Paragraph("Positive", body_style),
            Paragraph(f"{confidence_positive}%", body_style),
            Paragraph("█" * int(confidence_positive / 5) if confidence_positive > 0 else "-", body_style)
        ],
        [
            Paragraph("Neutral", body_style),
            Paragraph(f"{confidence_neutral}%", body_style),
            Paragraph("█" * int(confidence_neutral / 5) if confidence_neutral > 0 else "-", body_style)
        ],
        [
            Paragraph("Negative", body_style),
            Paragraph(f"{confidence_negative}%", body_style),
            Paragraph("█" * int(confidence_negative / 5) if confidence_negative > 0 else "-", body_style)
        ],
    ]
    t_conf = Table(conf_data, colWidths=[140, 100, 300])
    t_conf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#214B43')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E3E8DE')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_conf)
    elements.append(Spacer(1, 14))

    # 3b. Dominant Emotion Analysis Section (Multi-Task Emotion Detection)
    if dominant_emotion and dominant_emotion.strip():
        elements.append(Paragraph("Dominant Emotion Analysis", h2_style))
        emoji_str = f"{dominant_emotion_emoji} " if dominant_emotion_emoji else ""
        prob_str = f" ({dominant_emotion_probability}%)" if dominant_emotion_probability is not None else ""

        emotion_summary_data = [
            [
                Paragraph("<b>Dominant Emotion:</b>", body_style),
                Paragraph(f"<font color='#214B43'><b>{emoji_str}{dominant_emotion}{prob_str}</b></font>", badge_style),
                Paragraph("<b>Classification:</b>", body_style),
                Paragraph("Multi-Task Multi-Label (6 Emotions)", body_style)
            ]
        ]
        t_em_sum = Table(emotion_summary_data, colWidths=[130, 150, 110, 150])
        t_em_sum.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7F8F5')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#C7F59D')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E3E8DE')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t_em_sum)
        elements.append(Spacer(1, 8))

        # Parse 6-emotion breakdown if provided as JSON string
        if emotion_probabilities:
            try:
                import json
                em_probs_dict = json.loads(emotion_probabilities) if isinstance(emotion_probabilities, str) else emotion_probabilities
                if isinstance(em_probs_dict, dict) and em_probs_dict:
                    em_table_data = [
                        [
                            Paragraph("<b>Emotion</b>", body_style),
                            Paragraph("<b>Probability</b>", body_style),
                            Paragraph("<b>Relative Level</b>", body_style)
                        ]
                    ]
                    for em_name, em_val in em_probs_dict.items():
                        val_float = float(em_val)
                        bar_len = int(val_float / 5) if val_float > 0 else 0
                        em_table_data.append([
                            Paragraph(f"{em_name}", body_style),
                            Paragraph(f"{val_float:.2f}%", body_style),
                            Paragraph("█" * bar_len if bar_len > 0 else "-", body_style)
                        ])
                    t_em_breakdown = Table(em_table_data, colWidths=[140, 100, 300])
                    t_em_breakdown.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#214B43')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E3E8DE')),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    elements.append(t_em_breakdown)
                    elements.append(Spacer(1, 14))
            except Exception:
                elements.append(Spacer(1, 10))

    # 4. Input Text
    if input_text and input_text.strip():
        elements.append(Paragraph("Analyzed Meme Text", h2_style))
        text_data = [[Paragraph(f'"{input_text.strip()}"', body_style)]]
        t_text = Table(text_data, colWidths=[540])
        t_text.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7F8F5')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E3E8DE')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(t_text)
        elements.append(Spacer(1, 14))

    # 5. Explainable AI Section (ONLY if generated)
    has_explanation = bool(explanation_summary or top_words or (gate_text_weight is not None) or heatmap_base64)
    if has_explanation:
        elements.append(Paragraph("Explainable AI Insights (Why This Prediction?)", h2_style))
        
        explain_items = []
        if explanation_summary:
            explain_items.append([Paragraph("<b>Summary:</b>", body_style), Paragraph(explanation_summary, body_style)])
        if top_words:
            explain_items.append([Paragraph("<b>Influential Words:</b>", body_style), Paragraph(top_words, body_style)])
        if gate_text_weight is not None and gate_image_weight is not None:
            explain_items.append([
                Paragraph("<b>Gated Attention Weights:</b>", body_style),
                Paragraph(f"Text Attention: {gate_text_weight}% • Image Attention: {gate_image_weight}%", body_style)
            ])

        if explain_items:
            t_explain = Table(explain_items, colWidths=[150, 390])
            t_explain.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E7F6D9')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#C7F59D')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#C7F59D')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(t_explain)
            elements.append(Spacer(1, 10))

        # Grad-CAM Heatmap Image Embed
        if heatmap_base64 and "," in heatmap_base64:
            try:
                raw_b64 = heatmap_base64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                img_io = io.BytesIO(img_bytes)
                rl_img = RLImage(img_io, width=2.4 * inch, height=2.4 * inch)
                
                img_table_data = [
                    [
                        rl_img,
                        Paragraph(
                            "<b>Grad-CAM Visual Attention Heatmap:</b><br/>"
                            "Warm regions (red/yellow) indicate image features that most strongly steered the model toward the predicted sentiment.<br/><br/>"
                            "Computed via backpropagating gradients from the final convolutional stage of ResNet-50 onto the visual feature map.",
                            body_style
                        )
                    ]
                ]
                t_img = Table(img_table_data, colWidths=[180, 360])
                t_img.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(t_img)
                elements.append(Spacer(1, 10))
            except Exception as e:
                pass

    # 6. Footer Disclaimer
    elements.append(Spacer(1, 14))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#C7D4CD'), spaceAfter=8))
    footer_text = "IndicSent-M Multimodal AI Engine • Memotion 3.0 Hinglish Meme Benchmark • Strictly for research and verification."
    elements.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor('#8D9D95'), alignment=1)))

    # Build document
    doc.build(elements)
    buffer.seek(0)

    filename = f"IndicSent-M_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
