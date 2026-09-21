import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torch.nn.functional as F

# Ensure inference folder is in sys.path
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from model.framework import get_model
from explainability import DeepfakeGradCAM, overlay_heatmap
# The threshold used by the original SupplyWriter for fake/real labeling
# Must match exactly what SupplyWriter uses so the web label agrees with the video annotation
OPT_THRESHOLD = 0.002584857167676091

# Global model cache to avoid re-loading weights every run
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None
_CACHED_DEVICE = None


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_prediction_model(model_path="weights/PwTF_weights.pth", device=None):
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _CACHED_DEVICE

    if device is None:
        device = get_device()

    abs_model_path = os.path.abspath(model_path)

    if (
        _CACHED_MODEL is not None
        and _CACHED_MODEL_PATH == abs_model_path
        and _CACHED_DEVICE == device
    ):
        return _CACHED_MODEL

    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(
            f"Model weights file not found at: {abs_model_path}. "
            "Please place PwTF_weights.pth in the weights/ directory."
        )

    print(f"Loading model on {device} from {abs_model_path}...")
    model = get_model()
    checkpoint = torch.load(abs_model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = abs_model_path
    _CACHED_DEVICE = device
    print("Model loaded successfully!")
    return model


def _write_annotated_mp4(video_path, out_path, frames, scores, boxes, opt_thres):
    """
    Write an annotated MP4 directly using PyAV with the libx264 codec.
    This produces a highly compatible browser-playable .mp4 (H.264).
    """
    import av

    reader = cv2.VideoCapture(video_path)
    fps    = reader.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader.release()

    container = av.open(out_path, mode='w')
    stream = container.add_stream('libx264', rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '23'}

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, width / 800)          # scale label to video size
    thickness  = max(2, int(width / 300))
    box_thick  = max(3, int(width / 200))

    for image, score, box in zip(frames, scores, boxes):
        # frames are RGB from grab_all_frames — convert to BGR for OpenCV drawing
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if box is not None and score is not None:
            is_fake = score > opt_thres
            label   = "FAKE" if is_fake else "REAL"
            color   = (0, 0, 255) if is_fake else (0, 220, 90)   # BGR red / green

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

            # Draw label background pill
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 6
            lx, ly = x1, max(0, y1 - th - 2 * pad)
            cv2.rectangle(frame, (lx, ly), (lx + tw + 2 * pad, ly + th + 2 * pad), color, -1)
            cv2.putText(
                frame, label,
                (lx + pad, ly + th + pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

            # Draw confidence percentage
            pct_txt = f"{score / opt_thres * 50:.0f}%" if is_fake else f"{(1 - score / opt_thres * 50):.0f}%"
            cv2.putText(
                frame, pct_txt,
                (x1, y2 + th + pad * 2),
                font, font_scale * 0.85, color, thickness, cv2.LINE_AA,
            )

        # Convert back to RGB for PyAV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    return out_path


def predict_video(
    video_path,
    model_path="weights/PwTF_weights.pth",
    out_dir="output",
    max_frame=300,
    stride=1,
    progress_callback=None,
):
    """
    Runs frequency-based deepfake video detection on a video file.

    Returns a dict with:
        out_file       – path to the browser-playable annotated video
        label          – "FAKE" or "REAL"
        confidence     – raw model score (0–1); higher = more likely fake
        fake_prob_pct  – human-readable fake probability percentage
        total_frames   – number of frames processed
        analyzed_clips – number of 32-frame clips evaluated
        tracks         – number of face tracks found
    """
    device = get_device()
    model = load_prediction_model(model_path, device)

    crop_align_func = FasterCropAlignXRay(224)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    web_out = os.path.join(out_dir, base_name + "_detect.mp4")

    # ── Face detection ────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.05, desc="Extracting frames & detecting faces...")

    cache_file = f"{video_path}_{max_frame}.pth"
    if os.path.exists(cache_file):
        print("Loading cached face detection results...")
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
    else:
        print("Performing face detection...")
        detect_res, all_lm68, frames = detect_all(
            video_path, return_frames=True, max_size=max_frame
        )
        try:
            torch.save((detect_res, all_lm68), cache_file)
        except Exception:
            pass

    if not frames:
        raise ValueError("Could not read any frames from the provided video file.")

    # ── Landmark merge ────────────────────────────────────────────────────────
    shape = frames[0].shape[:2]
    assert len(all_lm68) == len(detect_res)

    all_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    # ── Face tracking ─────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0.20, desc="Tracking faces across frames...")

    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if not tracks:
        tuples, tracks = find_longest(detect_res)
    if not tracks:
        raise ValueError("No facial sequences could be tracked in the input video.")

    # ── Data storage ──────────────────────────────────────────────────────────
    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]

            info = (
                (box.reshape(2, 2) - top_left).reshape(-1),
                lm5 - top_left,
                lm68 - top_left,
                big_box,
            )
            x1, y1, x2, y2 = big_box
            base_key = f"{track_i}_{j}_"
            data_storage[base_key + "img"] = frames[frame_idx][y1:y2, x1:x2]
            data_storage[base_key + "ldm"] = info
            data_storage[base_key + "idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(np.int32)

    # ── Sliding-window clip generation ────────────────────────────────────────
    clips_for_video = []
    clip_size = 32
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = (post_module * (pad_length // l_post + 1))[:pad_length]
            if len(post_module) != pad_length:
                continue
            pre_module = inner_index + inner_index[1:-1][::-1]
            pre_module = (pre_module * (pad_length // len(post_module) + 1))[-pad_length:]
            if len(pre_module) != pad_length:
                continue
            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        stride_val = max(1, int(stride))
        for i in range(0, super_clip_size - clip_size + 1, stride_val):
            clips_for_video.append(
                [(super_clip_idx, t) for t in inner_index[i : i + clip_size]]
            )

    if not clips_for_video:
        raise ValueError("Could not extract suitable facial clips for temporal analysis.")

        # ── Model inference ───────────────────────────────────────────────────────
    preds = []
    frame_res = {}
    test_transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    total_clips = len(clips_for_video)
    print(f"Running inference on {total_clips} clips...")
    
    max_score = -1.0
    max_clip_tensors = None
    max_clip_images = None

    for idx, clip in enumerate(clips_for_video):
        if progress_callback and idx % 5 == 0:
            progress_callback(
                0.30 + 0.55 * (idx / total_clips),
                desc=f"Analyzing frequency domain — clip {idx+1}/{total_clips}",
            )

        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        landmarks, images = crop_align_func(landmarks, images)

        images_tensor = []
        ft_images = []
        for image in images:
            img_arr = np.array(image)
            img_pil = test_transform(Image.fromarray(img_arr))
            images_tensor.append(img_pil)
            img_filtered = cv2.medianBlur(img_arr.copy(), 5)
            ft_images.append(cv2.cvtColor(img_arr - img_filtered, cv2.COLOR_RGB2GRAY))

        ft_np = np.array(ft_images)
        ft_np = np.absolute(np.fft.fft(ft_np, axis=0)[: clip_size // 2] * (1 / clip_size))
        ft_tensor = torch.from_numpy(ft_np).to(device).unsqueeze(0)
        img_tensor = torch.stack(images_tensor, dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = F.sigmoid(model(img_tensor, ft_tensor)).squeeze(0)

        pred = float(output.item())
        
        if pred > max_score:
            max_score = pred
            max_clip_tensors = (img_tensor, ft_tensor)
            max_clip_images = images
            
        for f_id in frame_ids:
            frame_res.setdefault(f_id, []).append(pred)
        preds.append(pred)
        
    # ── Explainability ───────────────────────────────────────────────────────
    heatmap_path = None
    if max_clip_tensors is not None:
        if progress_callback:
            progress_callback(0.85, desc="Generating explainability heatmap...")
        
        try:
            gradcam = DeepfakeGradCAM(model)
            img_t, ft_t = max_clip_tensors
            heatmap = gradcam.generate_heatmap(img_t, ft_t)
            
            # Overlay on the middle frame of the most fake clip
            mid_idx = len(max_clip_images) // 2
            original_img = np.array(max_clip_images[mid_idx])
            
            overlayed_img = overlay_heatmap(original_img, heatmap)
            overlayed_img_bgr = cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR)
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            heatmap_path = os.path.join(out_dir, base_name + "_heatmap.jpg")
            cv2.imwrite(heatmap_path, overlayed_img_bgr)
            print(f"Explainability heatmap saved to {heatmap_path}")
        except Exception as e:
            print(f"Failed to generate heatmap: {e}")

    # ── Aggregate & classify ─────────────────────────────────────────────────
    mean_pred = float(np.mean(preds))

    # Use the SAME threshold as SupplyWriter so the web label matches the video annotation
    is_fake = mean_pred > OPT_THRESHOLD
    label = "FAKE" if is_fake else "REAL"

    # Build per-frame score / box arrays
    boxes, scores = [], []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            scores.append(float(np.mean(frame_res[frame_idx])))
            boxes.append(frame_boxes[frame_idx])
        else:
            scores.append(None)
            boxes.append(None)

    # ── Write annotated MP4 directly (browser-playable, no ffmpeg needed) ─────
    if progress_callback:
        progress_callback(0.88, desc="Rendering annotated output video...")

    _write_annotated_mp4(video_path, web_out, frames, scores, boxes, OPT_THRESHOLD)

    if progress_callback:
        progress_callback(1.0, desc="Detection complete!")

    # Fake probability as a 0-100 percentage for display
    # Because the raw score is very small, normalise it for UI display only
    # We clamp to [0, 100] so the bar looks sensible
    fake_prob_pct = min(100.0, mean_pred / OPT_THRESHOLD * 50) if is_fake else max(0.0, mean_pred / OPT_THRESHOLD * 50)

    return {
        "out_file": web_out,
        "heatmap": heatmap_path,
        "label": label,
        "confidence": mean_pred,
        "fake_prob_pct": fake_prob_pct,
        "is_fake": is_fake,
        "total_frames": len(frames),
        "analyzed_clips": total_clips,
        "tracks": len(tracks),
    }
