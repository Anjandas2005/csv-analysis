from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def find_available_port(start_port: int) -> int:
    for port in range(start_port, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError("No available port found for the backend server.")


def _validate_session_id(session_id: str) -> str:
    try:
        return str(uuid.UUID(session_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session id.")


@app.post("/api/v1/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    session_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{session_id}.csv")
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}_report.pdf")
    
    with open(csv_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Execute EDA & ML Pipeline
    engine = AnalysisEngine(csv_path)
    metrics = engine.run_full_analysis(output_dir=UPLOAD_DIR)
    
    # Generate LLM Summary
    from services.llm_summary import generate_llm_summary
    metrics['llm_summary'] = generate_llm_summary(metrics)
    
    # Generate Downloadable PDF
    generate_pdf_report(metrics, pdf_path)
    
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