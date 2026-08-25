# Data Analysis Platform

A local data analysis platform that accepts CSV files, runs exploratory data analysis and machine learning diagnostics, generates a PDF report, and displays a Looker Studio dashboard URL in a Gradio interface.

## Features

- CSV upload through a Gradio web interface
- FastAPI analysis service with automatic dataset profiling
- Missing-value, duplicate, outlier, distribution, correlation, PCA, and clustering analysis
- Random Forest classification or regression when a suitable target is detected
- Downloadable PDF diagnostic report
- Optional Groq-powered executive summary with a local fallback when no API key is configured

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- The dashboard currently returns a demo Looker Studio embed URL; connect a real dashboard before using it in production

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

Start the backend in one terminal:

```powershell
python backend\main.py
```

Start the Gradio frontend in a second terminal:

```powershell
python frontend\app.py
```

Open the URL printed by the frontend, normally `http://127.0.0.1:7860`, upload a CSV file, and select **Analyze Dataset**. The backend normally starts at `http://127.0.0.1:8001`. The backend and frontend choose the next available port if their requested port is occupied; if the backend moves, set `BACKEND_BASE_URL` to the backend URL before starting the frontend.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `BACKEND_PORT` | `8001` | Preferred FastAPI port |
| `GRADIO_SERVER_PORT` | `7860` | Preferred Gradio port |
| `BACKEND_BASE_URL` | `http://localhost:8001` | Backend URL used by the frontend |
| `GROQ_API_KEY` | unset | Enables the optional LLM-generated summary |

Example PowerShell configuration:

```powershell
$env:BACKEND_BASE_URL = "http://localhost:8001"
$env:GROQ_API_KEY = "your-key"
```

Set `BACKEND_BASE_URL` to the URL printed by the backend when it uses a different port.

## API

- `POST /api/v1/analyze` accepts a multipart CSV upload under the `file` field.
- `GET /api/v1/download-report/{session_id}` downloads the generated PDF report.

## Project Layout

```text
backend/main.py                 FastAPI application and upload/report endpoints
backend/services/analysis_engine.py  EDA and ML pipeline
backend/services/llm_summary.py      Optional Groq summary and fallback summary
backend/services/pdf_generator.py     PDF report generation
frontend/app.py                 Gradio interface
requirements.txt                Python dependencies
```

## Development Notes

Uploaded CSVs, plots, and generated reports are written to `backend/uploads/`. The frontend temporarily writes downloaded reports to the project root. These runtime artifacts are ignored by Git; keep small, intentional input fixtures such as `test.csv` and `test_sample.csv` in the repository when they are useful for development. The upload directory is created automatically when the backend starts.