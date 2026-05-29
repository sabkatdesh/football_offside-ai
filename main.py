import os
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Lazy-import heavy ML deps so FastAPI starts fast ─────────────────────────
# These are imported inside the endpoint to avoid blocking startup.

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Offside Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the output videos so the frontend can <video src="..."> them
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
# Serve the frontend
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    """
    1. Save uploaded video to disk.
    2. Validate it (fast pre-flight).
    3. Run the full pipeline.
    4. Return verdict + output video URL + involvement log.
    """

    # ── 1. Persist upload ─────────────────────────────────────────────────────
    job_id     = uuid.uuid4().hex
    ext        = Path(video.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_output.mp4"

    with open(input_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        from validator import validate_video

        pitch_detector  = YOLO('football-pitch-detection.pt').to('cuda')
        player_detector = YOLO('football-player-detection.pt').to('cuda')

        # Run blocking validation in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        valid, reason = await loop.run_in_executor(
            None,
            validate_video,
            str(input_path),
            pitch_detector,
            player_detector,
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

    if not valid:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=reason)

    # ── 3. Run pipeline ───────────────────────────────────────────────────────
    try:
        from pipeline_3 import run_pipeline

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_pipeline,
            str(input_path),
            str(output_path),
        )
    except Exception as e:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # ── 4. Clean up input, return results ────────────────────────────────────
    input_path.unlink(missing_ok=True)

    return JSONResponse({
        "verdict":          result["verdict"],
        "output_video_url": f"/outputs/{output_path.name}",
        "involvement_log":  result["involvement_log"],
        "total_events":     len(result["involvement_log"]),
    })
