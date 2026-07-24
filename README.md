# PW-Frequency_based_detection

Pixel-Wise Temporal Frequency-Based Deepfake Video Detection using deep learning.

This repository contains the implementation of a frequency-aware deepfake video detection framework. The project detects manipulated videos by analyzing pixel-wise temporal frequency information extracted from consecutive video frames.

---

## Features

- Pixel-wise Temporal Frequency Analysis
- Deepfake Video Detection
- Video Inference Pipeline
- CPU and GPU Inference Support
- Preprocessing Utilities
- Modular PyTorch Implementation
- Easy-to-Extend Architecture

---

## Project Structure

```
PW-Frequency_based_detection/
│
├── inference/              # Video inference scripts
├── preprocessing/          # Data preprocessing pipeline
├── figures/                # Images used in documentation
├── weights/                # Model weights (download separately)
├── output/                 # Detection results
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/AnaniasRemoA/PW-Frequency_based_detection.git

cd PW-Frequency_based_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Download Model Weights

The trained model weights are **not included** in this repository due to GitHub file size limitations.

Download the model weights separately and place them inside:

```
weights/
```

Expected structure:

```
weights/
└── PwTF_weights.pth
```

---

## Running Inference

### GPU

```bash
python inference/test_on_raw_video.py
```

### CPU

```bash
python inference/test_on_raw_video_cpu.py
```

Modify the input video path and output directory inside the inference script if required.

---

## Output

The processed videos with deepfake predictions will be saved inside the configured output directory.

Example:

```
output/
```

---

## Project Workflow

```
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection & Alignment
      │
      ▼
Pixel-wise Temporal Frequency Extraction
      │
      ▼
Deep Learning Model
      │
      ▼
Deepfake Prediction
      │
      ▼
Output Video
```

---

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pillow
- tqdm
- PyYAML

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository Contents

- Deepfake inference pipeline
- Frequency-based feature extraction
- Face detection and tracking
- Preprocessing utilities
- Visualization utilities

---

## Disclaimer

This project is intended solely for research and educational purposes. The authors are not responsible for any misuse of this software.
