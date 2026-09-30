import os
import json
import math
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from mineral_associations import MINERAL_TO_COMMODITY


def calculate_aoi_stats(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    width_km  = (max_lon - min_lon) * 111.32 * math.cos(math.radians(center_lat))
    height_km = (max_lat - min_lat) * 110.57
    return {
        'center_lat': round(center_lat, 5),
        'center_lon': round(center_lon, 5),
        'width_km':   round(width_km, 2),
        'height_km':  round(height_km, 2),
        'area_km2':   round(abs(width_km * height_km), 2),
    }


def generate_report(job_id, data_dir):
    summary_path = os.path.join(data_dir, 'job_summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No job_summary.json found in {data_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    metadata_path = os.path.join(data_dir, 'model_metadata.json')
    model_meta = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            model_meta = json.load(f)

    bbox = summary['bbox']
    aoi = calculate_aoi_stats(bbox)
    year_start, year_end = summary['year_start'], summary['year_end']
    results = summary['results']

    output_path = os.path.join(data_dir, f'report_{job_id}.pdf')
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=22)
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=11,
        textColor=colors.grey, alignment=TA_CENTER)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=18,
        spaceAfter=8, textColor=colors.HexColor('#1a3a5c'))
    h3 = ParagraphStyle('H3', parent=styles['Heading3'], spaceBefore=12,
        spaceAfter=6, textColor=colors.HexColor('#2c5282'))
    body = styles['BodyText']
    small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.grey)

    el = []

    # COVER
    el += [Spacer(1, 2*cm), Paragraph("GeoProspect AI", title_style),
        Paragraph("Mineral Detection Report",
            ParagraphStyle('S2', parent=styles['Heading2'], alignment=TA_CENTER,
                textColor=colors.HexColor('#1a3a5c'))),
        Spacer(1, 0.5*cm),
        Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style),
        Spacer(1, 1.5*cm)]

    cover_data = [
        ['Job ID', job_id],
        ['AOI Bounding Box', f"{bbox[0]:.4f}, {bbox[1]:.4f}  →  {bbox[2]:.4f}, {bbox[3]:.4f}"],
        ['AOI Center', f"{aoi['center_lat']}, {aoi['center_lon']}"],
        ['AOI Dimensions', f"{aoi['width_km']} km × {aoi['height_km']} km"],
        ['AOI Area', f"{aoi['area_km2']} km²"],
        ['Years Analyzed', f"{year_start} – {year_end}"],
        ['Satellite Sources', "Sentinel-2 SR, ASTER (VNIR)"],
    ]
    t = Table(cover_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1a3a5c')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8), ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    el += [t, PageBreak()]

    # EXECUTIVE SUMMARY
    all_minerals = set()
    for r in results:
        for m in r['minerals']:
            all_minerals.add(m['mineral'])

    best_mineral, best_conf = None, -1
    for r in results:
        for m in r['minerals']:
            if m['confidence'] > best_conf:
                best_conf, best_mineral = m['confidence'], m['mineral']

    total_samples = sum(r.get('n_pixels', 0) for r in results)

    el.append(Paragraph("Executive Summary", h2))
    el.append(Paragraph(
        f"This report covers a mineral detection analysis over a "
        f"{aoi['area_km2']} km² area of interest, spanning "
        f"{year_end - year_start + 1} year(s) ({year_start}–{year_end}). "
        f"A total of {total_samples:,} satellite pixel samples were analyzed. "
        f"{len(all_minerals)} distinct mineral signature(s) were identified: "
        f"{', '.join(sorted(all_minerals))}. The strongest signal detected was "
        f"<b>{best_mineral}</b> with a peak confidence of {best_conf*100:.1f}%.",
        body))
    el.append(Spacer(1, 0.3*cm))

    # METHODOLOGY
    el.append(Paragraph("Methodology", h2))
    model_clause = (
        f"a pre-trained global model built from multiple diverse training "
        f"regions worldwide" if model_meta and model_meta.get('is_global')
        else "a model trained specifically on this area of interest"
    )
    el.append(Paragraph(
        "Satellite imagery was retrieved from Google Earth Engine, combining "
        "Sentinel-2 Surface Reflectance (bands B2, B3, B4, B8, B11, B12) and "
        "ASTER visible/near-infrared data (bands B01, B02, B3N). Six derived "
        "spectral indices were computed: Iron Oxide ratio, Clay Index, NDVI, "
        "NDWI, Ferric Iron ratio, and Silica Index. Pixels were grouped by "
        "spectral similarity using HDBSCAN clustering, and each cluster's "
        "mean spectrum was matched against a reference library of 30 real "
        "mineral spectra (USGS Spectral Library, splib07) using Spectral "
        "Angle Mapper similarity. Labelled pixels were used to train/apply "
        f"{model_clause} via XGBoost multiclass classification.",
        body))
    el.append(Spacer(1, 0.3*cm))

    if model_meta:
        acc = model_meta.get('accuracy')
        mt = [
            ['Model Type', 'Global (pre-trained)' if model_meta.get('is_global') else 'Job-specific'],
            ['Mineral Classes', str(model_meta.get('num_classes', 'N/A'))],
            ['Training Samples', f"{model_meta.get('training_samples', 0):,}"],
            ['Holdout Test Samples', f"{model_meta.get('test_samples', 0):,}"],
            ['Holdout Accuracy', f"{acc*100:.1f}%" if acc is not None else 'N/A'],
            ['Model Trained On', model_meta.get('trained_at', 'N/A')],
        ]
        mtable = Table(mt, colWidths=[5*cm, 10*cm])
        mtable.setStyle(TableStyle([
            ('FONTSIZE', (0,0), (-1,-1), 9), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f7fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        el.append(mtable)
        el.append(Spacer(1, 0.3*cm))
        el.append(Paragraph(
            "<i>Holdout accuracy is measured on regions excluded from "
            "training — a realistic estimate of performance on unseen "
            "locations, not a guarantee of mineral presence.</i>", small))

    el.append(PageBreak())

    # PER-YEAR RESULTS
    el.append(Paragraph("Detailed Results by Year", h2))
    for r in results:
        el.append(Paragraph(f"Year {r['year']}", h3))
        el.append(Paragraph(f"{r.get('n_pixels', 0):,} pixel samples analyzed.", small))
        el.append(Spacer(1, 0.15*cm))

        td = [['Mineral', 'Confidence', 'Mean Prob.', 'Coverage %']]
        for m in r['minerals']:
            td.append([m['mineral'], f"{m['confidence']*100:.1f}%",
                f"{m['mean_prob']*100:.1f}%", f"{m['coverage_pct']:.1f}%"])

        rt = Table(td, colWidths=[5*cm, 3.3*cm, 3.3*cm, 3.3*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3a5c')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
            ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        el += [rt, Spacer(1, 0.5*cm)]
    el.append(PageBreak())

    # CROSS-YEAR TREND
    if len(results) > 1:
        el.append(Paragraph("Coverage Trend Across Years", h2))
        years = [r['year'] for r in results]
        trend = [['Mineral'] + [str(y) for y in years]]
        for mineral in sorted(all_minerals):
            row = [mineral]
            for r in results:
                match = next((m for m in r['minerals'] if m['mineral'] == mineral), None)
                row.append(f"{match['coverage_pct']:.1f}%" if match else "—")
            trend.append(row)
        tt = Table(trend, colWidths=[4*cm] + [11*cm/len(years)]*len(years))
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3a5c')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ]))
        el += [tt, Spacer(1, 0.5*cm), PageBreak()]

    # MINERAL PROFILES
    el.append(Paragraph("Mineral Profiles", h2))
    for mineral in sorted(all_minerals):
        assoc = MINERAL_TO_COMMODITY.get(mineral, {})
        metals, uses = assoc.get('metals', []), assoc.get('uses', [])
        el.append(Paragraph(mineral, h3))
        el.append(Paragraph(
            f"<b>Associated metals:</b> {', '.join(metals) if metals else 'None directly (non-metallic mineral)'}",
            body))
        if uses:
            el.append(Paragraph(f"<b>Common uses:</b> {', '.join(uses)}", body))
        el.append(Spacer(1, 0.25*cm))
    el.append(PageBreak())

    # FEATURE IMPORTANCE
    if model_meta and model_meta.get('feature_importance'):
        el.append(Paragraph("Model Feature Importance", h2))
        fi = [['Feature', 'Importance']]
        for item in model_meta['feature_importance']:
            fi.append([item['feature'], f"{item['importance']*100:.1f}%"])
        fit = Table(fi, colWidths=[8*cm, 4*cm])
        fit.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3a5c')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
        ]))
        el += [fit, PageBreak()]

    # DISCLAIMER
    el.append(Paragraph("Limitations & Disclaimer", h2))
    el.append(Paragraph(
        "This report is generated by an automated remote-sensing analysis "
        "pipeline and represents a screening-level, exploratory assessment "
        "only. Mineral identifications are derived from spectral similarity "
        "to laboratory reference spectra and statistical classification, not "
        "direct physical sampling. Results can be affected by vegetation "
        "cover, cloud contamination, atmospheric conditions, and surface "
        "weathering. Field verification and consultation with a qualified "
        "geologist are strongly recommended before any exploration, "
        "investment, or land-use decisions based on this report.", body))
    el += [Spacer(1, 1*cm), HRFlowable(width="100%", color=colors.HexColor('#e2e8f0')),
        Spacer(1, 0.3*cm),
        Paragraph(f"Report generated by GeoProspect AI · Job ID: {job_id} · "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", small)]

    doc.build(el)
    return output_path