import os
import sys
import shutil
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

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

@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Form(300),
    stride: int = Form(4)
):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        t0 = time.time()
        
        # We don't have async progress callbacks easily wired to FastAPI without WebSockets,
        # so we will just run the inference and return the result.
        def progress_cb(val, desc):
            print(f"[{val*100:.1f}%] {desc}")

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
        
        # Convert absolute path to relative URL for the frontend
        out_file_name = os.path.basename(res["out_file"])
        res["video_url"] = f"/api/video/{out_file_name}"
        
        if res.get("heatmap"):
            heatmap_name = os.path.basename(res["heatmap"])
            res["heatmap_url"] = f"/api/video/{heatmap_name}"
        
        return {"status": "success", "result": res}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Optionally, delete the uploaded original file after processing
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
