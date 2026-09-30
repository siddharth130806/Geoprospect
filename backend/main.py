from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import run_pipeline

app = FastAPI(title="GeoProspect AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store for local development.
# Replace with Redis or Supabase for production.
JOBS = {}


class AnalyzeRequest(BaseModel):
    bbox:         list[float]   # [min_lon, min_lat, max_lon, max_lat]
    year_start:   int
    year_end:     int
    generate_map: bool = False


class PointRequest(BaseModel):
    lat:    float
    lng:    float
    job_id: str
    year:   int


from fastapi.responses import FileResponse

@app.get("/api/job/{job_id}/report")
def get_report(job_id: str):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', job_id)
    summary_path = os.path.join(data_dir, 'job_summary.json')
    if not os.path.exists(summary_path):
        return {'error': 'Job not found or not yet complete'}

    from generate_report import generate_report
    try:
        report_path = generate_report(job_id, data_dir)
    except Exception as e:
        return {'error': str(e)}

    return FileResponse(
        report_path,
        media_type='application/pdf',
        filename=f'GeoProspectAI_Report_{job_id}.pdf'
    )

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {'status': 'queued', 'message': 'Job queued'}

    # For local dev: run synchronously in a background thread
    import threading
    def run():
        try:
            JOBS[job_id] = {'status': 'running', 'message': 'Extracting features from GEE...'}
            result = run_pipeline(
                job_id=job_id,
                bbox=req.bbox,
                year_start=req.year_start,
                year_end=req.year_end,
                generate_map=req.generate_map
            )
            JOBS[job_id] = {
                'status': 'done',
                'message': f"Complete. Found minerals in {len(result)} year(s).",
                'result': {
                    'job_id': job_id,
                    'bbox': req.bbox,
                    'results': result,
                    'generate_map': req.generate_map
                }
            }
        except Exception as e:
            JOBS[job_id] = {'status': 'error', 'message': str(e)}

    threading.Thread(target=run, daemon=True).start()
    return {'job_id': job_id, 'status': 'queued'}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        return {'status': 'not_found', 'message': 'Job not found'}
    return JOBS[job_id]


@app.get("/api/health")
def health():
    return {'status': 'ok'}


@app.post("/api/predict_point")
def predict_point(req: PointRequest):
    from predict_map import predict_nearest_point
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', req.job_id)
    result = predict_nearest_point(req.lat, req.lng, req.job_id, req.year, data_dir)
    return result

@app.get("/output/{filename}")
def get_output_file(filename: str):
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    import os
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail="Data dir not found")
    
    job_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))], key=os.path.getmtime, reverse=True)
    
    for jd in job_dirs:
        filepath = os.path.join(jd, filename)
        if os.path.exists(filepath):
            return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")
