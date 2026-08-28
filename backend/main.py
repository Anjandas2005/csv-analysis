from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import sys
import os
import shutil
import uuid
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.analysis_engine import AnalysisEngine
from services.pdf_generator import generate_pdf_report

app = FastAPI(title="Real-Time Data Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def find_available_port(start_port: int) -> int:
    """
    Purpose: Find the first free TCP port on localhost, starting from
    `start_port`. Used at startup so the backend doesn't crash if its
    preferred port (default 8001) is already in use — it just picks the
    next open one and prints it so the frontend can be pointed at it.
    """
    for port in range(start_port, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError("No available port found for the backend server.")


def _validate_session_id(session_id: str) -> str:
    """
    Purpose: Confirm that a session_id supplied by a client is a
    well-formed UUID before it is used to build any file path. Prevents
    path traversal (Review #1) — if the string isn't a valid UUID, this
    raises a 400 immediately instead of letting it reach os.path.join.
    """
    try:
        return str(uuid.UUID(session_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session id.")


@app.post("/api/v1/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Purpose: Main upload endpoint. Accepts a CSV file from the frontend,
    saves it to disk under a fresh UUID-based session, runs the full
    EDA/ML analysis pipeline and PDF report generation on it (offloaded
    to a thread pool so it doesn't block other requests — Review #3),
    and returns the computed metrics plus URLs for the dashboard and
    the downloadable PDF report.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    session_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{session_id}.csv")
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}_report.pdf")

    with open(csv_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def _run_pipeline():
        """
        Purpose: Run the CPU-heavy part of the request (EDA, ML model
        fitting, LLM summary, PDF rendering) as a single unit of work
        that can be handed off to a worker thread via run_in_threadpool,
        keeping the async event loop free to serve other requests.
        """
        engine = AnalysisEngine(csv_path)
        metrics = engine.run_full_analysis(output_dir=UPLOAD_DIR)

        from services.llm_summary import generate_llm_summary
        metrics['llm_summary'] = generate_llm_summary(metrics)

        generate_pdf_report(metrics, pdf_path)
        return metrics

    metrics = await run_in_threadpool(_run_pipeline)

    # Simulated Looker Embed URL (Constructed via Looker SDK or Looker Studio API link)
    looker_embed_url = f"https://lookerstudio.google.com/embed/reporting/demo-dashboard?params=%7B%22session_id%22:%22{session_id}%22%7D"

    return {
        "status": "success",
        "session_id": session_id,
        "metrics": metrics,
        "looker_embed_url": looker_embed_url,
        "pdf_download_url": f"/api/v1/download-report/{session_id}"
    }

@app.get("/api/v1/download-report/{session_id}")
async def download_report(session_id: str):
    """
    Purpose: Serve the previously generated PDF report for a given
    session as a file download. Validates session_id as a real UUID
    first so a caller can never manipulate the path (Review #1), then
    returns 404 if no report exists for that session.
    """
    safe_id = _validate_session_id(session_id)
    pdf_path = os.path.join(UPLOAD_DIR, f"{safe_id}_report.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(pdf_path, media_type="application/pdf", filename="analysis_report.pdf")

if __name__ == "__main__":
    import uvicorn
    api_port = int(os.getenv("BACKEND_PORT", "8001"))
    api_port = find_available_port(api_port)
    print(f"Starting backend on http://127.0.0.1:{api_port}")
    uvicorn.run("main:app", host="127.0.0.1", port=api_port, reload=False, app_dir=BASE_DIR)