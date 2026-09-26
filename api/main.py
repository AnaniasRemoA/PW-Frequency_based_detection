import os
import sys
import shutil
import time
import uuid
import glob
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import yt_dlp

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "PwTF_weights.pth")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Add inference to path
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from inference.predict import predict_video

app = FastAPI(title="DeepGuard API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the output directory to serve the videos
app.mount("/api/video", StaticFiles(directory=OUTPUT_DIR), name="video")

# In-memory progress and job tracking store
JOBS = {}


class YouTubeInfoRequest(BaseModel):
    url: str


def fetch_youtube_info(url: str):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "YouTube Video"),
            "uploader": info.get("uploader", "Unknown Channel"),
            "duration": info.get("duration", 0),
            "thumbnail": info.get("thumbnail", ""),
            "view_count": info.get("view_count", 0),
            "url": url,
        }


def download_youtube_video(url: str, output_dir: str):
    file_id = f"yt_{uuid.uuid4().hex[:10]}"
    out_template = os.path.join(output_dir, f"{file_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "YouTube Video")
    
    expected_file = os.path.join(output_dir, f"{file_id}.mp4")
    if not os.path.exists(expected_file):
        matches = glob.glob(os.path.join(output_dir, f"{file_id}.*"))
        if matches:
            expected_file = matches[0]
        else:
            raise RuntimeError("Failed to locate downloaded YouTube video file.")
            
    return expected_file, title


@app.get("/api/status")
def get_system_status():
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (x86_64)"
    return {
        "device": "CUDA" if cuda_avail else "CPU",
        "device_name": device_name,
        "cuda_available": cuda_avail,
        "model_loaded": WEIGHTS_PATH is not None and os.path.exists(WEIGHTS_PATH),
        "threshold": 0.1500000,
        "youtube_supported": True,
        "realtime_hooks": True,
    }


@app.get("/api/samples")
def get_samples():
    return [
        {
            "id": "real",
            "name": "id1_0000_real.mp4",
            "title": "Authentic Interview (Real)",
            "description": "Original unmanipulated high-definition facial video",
            "badge": "Authentic",
            "badge_color": "green"
        },
        {
            "id": "fake",
            "name": "id1_id9_0000_fake.mp4",
            "title": "FaceSwap DeepFake (Fake)",
            "description": "FaceForensics++ AI-synthesized facial manipulation",
            "badge": "Manipulated",
            "badge_color": "red"
        }
    ]


@app.post("/api/youtube/info")
def get_yt_info(payload: YouTubeInfoRequest):
    try:
        url = payload.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        if not ("youtube.com" in url or "youtu.be" in url):
            raise HTTPException(status_code=400, detail="Please enter a valid YouTube or YouTube Shorts URL")
        
        info = fetch_youtube_info(url)
        return {"status": "success", "data": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch YouTube info: {str(e)}")


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "pending", "progress": 0.0, "step_desc": "Waiting in queue..."}
    return job


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    async def event_generator():
        last_pct = -1.0
        last_desc = ""
        # Keep stream open up to 10 minutes
        for _ in range(1200):
            job = JOBS.get(job_id, {"status": "pending", "progress": 0.0, "step_desc": "Initializing pipeline..."})
            curr_pct = job.get("progress", 0.0)
            curr_desc = job.get("step_desc", "")
            curr_status = job.get("status", "pending")
            
            if curr_pct != last_pct or curr_desc != last_desc or curr_status in ["completed", "error"]:
                last_pct = curr_pct
                last_desc = curr_desc
                data_str = json.dumps(job)
                yield f"data: {data_str}\n\n"
                
            if curr_status in ["completed", "error"]:
                break
                
            await asyncio.sleep(0.3)
            
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/analyze")
def analyze_video(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    youtube_url: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
    max_frames: int = Form(300),
    stride: int = Form(4)
):
    if job_id:
        JOBS[job_id] = {
            "status": "processing",
            "progress": 2.0,
            "step_desc": "Initializing forensic pipeline...",
            "updated_at": time.time()
        }

    source_name = None
    # Determine input video path
    if youtube_url and youtube_url.strip():
        url = youtube_url.strip()
        if not ("youtube.com" in url or "youtu.be" in url):
            if job_id:
                JOBS[job_id] = {"status": "error", "progress": 0.0, "error": "Invalid YouTube URL"}
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        try:
            if job_id:
                JOBS[job_id] = {"status": "processing", "progress": 4.0, "step_desc": "Downloading YouTube stream..."}
            print(f"Downloading YouTube video: {url}")
            file_path, yt_title = download_youtube_video(url, UPLOAD_DIR)
            source_name = yt_title
        except Exception as e:
            if job_id:
                JOBS[job_id] = {"status": "error", "progress": 0.0, "error": f"Download failed: {str(e)}"}
            raise HTTPException(status_code=400, detail=f"Failed to download YouTube video: {str(e)}")
    elif sample_id:
        if sample_id == "real":
            sample_rel = os.path.join(PROJECT_ROOT, "uploads", "id1_0000_real.mp4")
            if not os.path.exists(sample_rel):
                sample_rel = os.path.join(PROJECT_ROOT, "examples", "Sample", "id1_0000_real.mp4")
            source_name = "Authentic Interview (Real Benchmark)"
        elif sample_id == "fake":
            sample_rel = os.path.join(PROJECT_ROOT, "examples", "Sample", "id1_id9_0000_fake.mp4")
            source_name = "FaceSwap DeepFake (Fake Benchmark)"
        else:
            raise HTTPException(status_code=400, detail="Unknown sample ID")
            
        if not os.path.exists(sample_rel):
            raise HTTPException(status_code=404, detail="Sample video file not found")
        file_path = sample_rel
    elif file is not None:
        if not file.filename.endswith((".mp4", ".avi", ".mov")):
            raise HTTPException(status_code=400, detail="Invalid file type")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        source_name = file.filename
    else:
        raise HTTPException(status_code=400, detail="Either file, sample_id, or youtube_url must be provided")

    try:
        t0 = time.time()
        
        def progress_cb(val, desc):
            pct = round(val * 100, 1)
            print(f"[{pct}%] {desc}")
            if job_id:
                JOBS[job_id] = {
                    "status": "processing",
                    "progress": pct,
                    "step_desc": desc,
                    "updated_at": time.time()
                }

        res = predict_video(
            video_path=file_path,
            model_path=WEIGHTS_PATH,
            out_dir=OUTPUT_DIR,
            max_frame=max_frames,
            stride=stride,
            progress_callback=progress_cb
        )
        
        elapsed = time.time() - t0
        res["elapsed"] = elapsed
        res["source_name"] = source_name or os.path.basename(file_path)
        
        # Convert absolute path to relative URL for the frontend
        out_file_name = os.path.basename(res["out_file"])
        res["video_url"] = f"/api/video/{out_file_name}"
        
        if res.get("heatmap"):
            heatmap_name = os.path.basename(res["heatmap"])
            res["heatmap_url"] = f"/api/video/{heatmap_name}"
            
        if res.get("fft_spectrum"):
            fft_name = os.path.basename(res["fft_spectrum"])
            res["fft_url"] = f"/api/video/{fft_name}"
        
        if job_id:
            JOBS[job_id] = {
                "status": "completed",
                "progress": 100.0,
                "step_desc": "Detection complete!",
                "result": res,
                "updated_at": time.time()
            }
        
        return {"status": "success", "result": res}

    except Exception as e:
        import traceback
        traceback.print_exc()
        if job_id:
            JOBS[job_id] = {
                "status": "error",
                "progress": 0.0,
                "error": str(e),
                "updated_at": time.time()
            }
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
