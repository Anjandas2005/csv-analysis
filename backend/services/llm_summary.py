import json
import os
import requests


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"


def _build_metrics_context(metrics: dict) -> str:
    """Build a concise text representation of the analysis metrics for the LLM."""
    lines = []

    # Overview
    overview = metrics.get("overview", {})
    lines.append("=== DATASET OVERVIEW ===")
    lines.append(f"Rows: {overview.get('rows', metrics.get('rows', 'N/A'))}")
    lines.append(f"Columns: {overview.get('cols', metrics.get('cols', 'N/A'))}")
    lines.append(f"Numeric columns: {overview.get('num_cols_count', 'N/A')}")
    lines.append(f"Categorical columns: {overview.get('cat_cols_count', 'N/A')}")
    lines.append(f"Missing data ratio: {overview.get('missing_pct', metrics.get('missing_pct', 0)):.2f}%")
    lines.append(f"Total missing cells: {overview.get('total_nulls', 0)}")
    lines.append(f"Duplicate rows: {overview.get('duplicate_rows', 0)} ({overview.get('duplicate_pct', 0):.2f}%)")
    lines.append(f"Memory footprint: {overview.get('memory_mb', 0):.2f} MB")
    lines.append(f"Target column: {metrics.get('target_col', 'None')}")

    # Top missing columns
    top5 = metrics.get("top5_missing", [])
    if top5:
        lines.append("\n=== TOP 5 COLUMNS BY MISSING VALUES ===")
        for item in top5:
            lines.append(f"  {item['column']}: {item['missing_count']} missing ({item['missing_pct']:.2f}%)")

    # Categorical value counts
    diag_vc = metrics.get("diagnosis_value_counts", [])
    if diag_vc:
        lines.append(f"\n=== TARGET DISTRIBUTION ({metrics.get('target_col', 'target')}) ===")
        for item in diag_vc:
            lines.append(f"  {item['category']}: {item['count']} ({item['percentage']:.2f}%)")

    # Outlier summary (top 5 worst)
    outliers = metrics.get("outlier_proportions", [])
    if outliers:
        lines.append("\n=== TOP OUTLIER COLUMNS (IQR Method) ===")
        sorted_outliers = sorted(outliers, key=lambda x: x["outlier_pct"], reverse=True)[:5]
        for item in sorted_outliers:
            lines.append(f"  {item['column']}: {item['outlier_pct']:.2f}% outliers")

    # Feature importance
    feat_imp = metrics.get("feature_importance_ranking", [])
    if feat_imp:
        lines.append("\n=== TOP FEATURE IMPORTANCES (Random Forest) ===")
        for item in feat_imp[:10]:
            lines.append(f"  {item['feature']}: {item['importance']:.4f}")

    # ML model metrics
    lines.append("\n=== MODEL PERFORMANCE ===")
    if metrics.get("is_classification", True):
        lines.append(f"Task type: Classification")
        lines.append(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
        lines.append(f"Precision (weighted): {metrics.get('precision', 'N/A')}")
        lines.append(f"Recall (weighted): {metrics.get('recall', 'N/A')}")
        lines.append(f"F1-Score (weighted): {metrics.get('f1_score', 'N/A')}")
        if "roc_auc" in metrics:
            lines.append(f"ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        lines.append(f"Task type: Regression")
        lines.append(f"R² Score: {metrics.get('r2_score', 'N/A')}")
        lines.append(f"RMSE: {metrics.get('rmse', 'N/A')}")
        lines.append(f"MAE: {metrics.get('mae', 'N/A')}")

    # Clustering
    sil = metrics.get("silhouette_score", 0)
    lines.append(f"\n=== CLUSTERING (PCA + KMeans) ===")
    lines.append(f"Silhouette Score: {sil:.3f}")

    # Numeric summary highlights (first chunk only, to keep token count manageable)
    num_summaries = metrics.get("numeric_summaries", {})
    first_chunk = num_summaries.get("features 1–5", [])
    if first_chunk:
        lines.append("\n=== NUMERIC SUMMARY SAMPLE (Features 1-5) ===")
        for item in first_chunk:
            lines.append(
                f"  {item['feature']}: mean={item.get('mean', 0):.3f}, std={item.get('std', 0):.3f}, "
                f"skew={item.get('skewness', 0):.3f}, kurtosis={item.get('kurtosis', 0):.3f}"
            )

    return "\n".join(lines)


def generate_llm_summary(metrics: dict) -> str:
    """
    Call the Groq API to generate an executive summary of the data analysis report.
    Returns the generated summary text, or a fallback message on failure.
    """
    context = _build_metrics_context(metrics)

    if not GROQ_API_KEY:
        return _fallback_summary(metrics)

    system_prompt = (
        "You are a senior data scientist writing an executive summary of a comprehensive "
        "automated data analysis report. Your summary will be included in a professional PDF report. "
        "Write in a clear, structured, and insightful manner. Use simple plain text with basic HTML tags for bolding (e.g. <b>text</b>). DO NOT use markdown like ** or ##.\n\n"
        "Your summary MUST include these sections:\n"
        "1. <b>Executive Overview</b> — Brief dataset description and key statistics\n"
        "2. <b>Data Quality Assessment</b> — Missing values, duplicates, outliers analysis\n"
        "3. <b>Key Statistical Insights</b> — Distribution patterns, skewness, notable correlations\n"
        "4. <b>Feature Analysis</b> — Most important features and their significance\n"
        "5. <b>Model Performance Summary</b> — Classification/regression results interpretation\n"
        "6. <b>Clustering Insights</b> — PCA and silhouette score interpretation\n"
        "7. <b>Actionable Recommendations</b> — 3-5 specific, data-driven next steps\n\n"
        "Keep it concise but insightful (400-600 words). Avoid generic statements — "
        "reference specific numbers from the analysis."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the complete analysis metrics:\n\n{context}"}
        ],
        "temperature": 0.4,
        "max_tokens": 1500,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"]
        return summary.strip()
    except requests.exceptions.Timeout:
        return _fallback_summary(metrics)
    except requests.exceptions.RequestException as e:
        print(f"[LLM Summary] Groq API error: {e}")
        return _fallback_summary(metrics)
    except (KeyError, IndexError) as e:
        print(f"[LLM Summary] Unexpected response format: {e}")
        return _fallback_summary(metrics)


def _fallback_summary(metrics: dict) -> str:
    """Generate a basic non-LLM summary when the API is unavailable."""
    overview = metrics.get("overview", {})
    rows = overview.get("rows", metrics.get("rows", "N/A"))
    cols = overview.get("cols", metrics.get("cols", "N/A"))
    missing = overview.get("missing_pct", metrics.get("missing_pct", 0))

    lines = [
        "<b>Executive Summary (Auto-Generated)</b>",
        "",
        f"This dataset contains <b>{rows:,}</b> rows and <b>{cols}</b> columns "
        f"with a <b>{missing:.2f}%</b> overall missing data ratio.",
    ]

    if metrics.get("is_classification", True) and "accuracy" in metrics:
        lines.append(
            f"A Random Forest classifier achieved <b>{metrics['accuracy']:.4f}</b> accuracy "
            f"with an F1-score of <b>{metrics.get('f1_score', 0):.4f}</b>."
        )
    elif "r2_score" in metrics:
        lines.append(
            f"A Random Forest regressor achieved an R² of <b>{metrics['r2_score']:.4f}</b> "
            f"with RMSE of <b>{metrics.get('rmse', 0):.4f}</b>."
        )

    sil = metrics.get("silhouette_score", 0)
    lines.append(f"PCA-based clustering yielded a silhouette score of <b>{sil:.3f}</b>.")

    lines.append("")
    lines.append("<i>Note: LLM-powered summary unavailable. This is an auto-generated fallback.</i>")

    return "\n".join(lines)
