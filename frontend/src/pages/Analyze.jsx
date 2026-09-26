import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { 
  UploadCloud, 
  Film, 
  ShieldAlert, 
  ShieldCheck, 
  Play, 
  Link as LinkIcon, 
  Clock, 
  User, 
  CheckCircle2, 
  Sparkles, 
  FileVideo, 
  Layers, 
  Activity, 
  X, 
  Cpu, 
  Terminal, 
  Radio, 
  ScanFace,
  Maximize2
} from 'lucide-react';

const YoutubeIcon = ({ size = 16, color = "currentColor" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
    <polygon points="10 15 15 12 10 9 10 15" fill={color} />
  </svg>
);

// Helper to extract video metadata & frame snapshot in the browser
const extractVideoMetadata = (file) => {
  return new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.muted = true;
      video.playsInline = true;
      video.src = url;

      video.onloadedmetadata = () => {
        const duration = video.duration || 0;
        const width = video.videoWidth || 0;
        const height = video.videoHeight || 0;

        let resolutionLabel = `${width}x${height}`;
        if (width >= 3840 || height >= 2160) resolutionLabel = '4K UHD (2160p)';
        else if (width >= 1920 || height >= 1080) resolutionLabel = 'Full HD (1080p)';
        else if (width >= 1280 || height >= 720) resolutionLabel = 'HD (720p)';
        else if (width > 0) resolutionLabel = `SD (${height}p)`;

        video.currentTime = Math.min(0.5, duration > 1 ? 0.5 : 0.1);
      };

      video.onseeked = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = Math.min(640, video.videoWidth || 320);
          canvas.height = Math.round((canvas.width / (video.videoWidth || 16)) * (video.videoHeight || 9));
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
          URL.revokeObjectURL(url);
          resolve({
            duration: video.duration || 0,
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
            resolutionLabel: video.videoWidth ? `${video.videoWidth}x${video.videoHeight}` : 'Video Stream',
            thumbnail
          });
        } catch (e) {
          URL.revokeObjectURL(url);
          resolve({
            duration: video.duration || 0,
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
            resolutionLabel: 'Video Stream',
            thumbnail: null
          });
        }
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        resolve({ duration: 0, width: 0, height: 0, resolutionLabel: 'Video File', thumbnail: null });
      };
    } catch (e) {
      resolve({ duration: 0, width: 0, height: 0, resolutionLabel: 'Video File', thumbnail: null });
    }
  });
};


// Elapsed clock hook
const useElapsedTimer = (running) => {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) {
      setElapsed(0);
      return;
    }
    const start = performance.now();
    const interval = setInterval(() => {
      setElapsed((performance.now() - start) / 1000);
    }, 100);
    return () => clearInterval(interval);
  }, [running]);
  return elapsed;
};

// ASCII Terminal Gauge Builder
const buildGauge = (pct, total = 28) => {
  const clamped = Math.max(0, Math.min(100, pct || 0));
  const filled = Math.round((clamped / 100) * total);
  return '█'.repeat(filled) + '░'.repeat(Math.max(0, total - filled));
};

const STAGES = [
  { id: 1, min: 0, max: 20, code: '01', title: 'Spatial Face Detection & Landmark Trajectory', sub: 'RetinaFace bounding & 68-pt landmark mesh alignment' },
  { id: 2, min: 20, max: 30, code: '02', title: 'High-Pass Median Filtering (Spatial Residuals)', sub: 'Isolating high-frequency noise & synthesis boundary seams' },
  { id: 3, min: 30, max: 85, code: '03', title: '3D Spatio-Temporal Transformer (PwTF) Inference', sub: 'APNResNet frequency tokens + I3D cross-attention network' },
  { id: 4, min: 85, max: 92, code: '04', title: 'Explainability Saliency & 2D FFT Spectrum', sub: 'Grad-CAM feature attribution & 2D log magnitude mapping' },
  { id: 5, min: 92, max: 100, code: '05', title: 'Forensic Video Synthesis & Audit Compilation', sub: 'Burning frame risk bounding boxes & compiling final dossier' },
];

const Analyze = () => {
  const [inputMode, setInputMode] = useState('upload'); // 'upload', 'youtube', 'benchmark'
  
  // Upload State
  const [file, setFile] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [isExtractingMeta, setIsExtractingMeta] = useState(false);
  
  // YouTube State
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [youtubeInfo, setYoutubeInfo] = useState(null);
  const [isFetchingInfo, setIsFetchingInfo] = useState(false);
  const [youtubeError, setYoutubeError] = useState('');
  
  // Benchmark State
  const [selectedSample, setSelectedSample] = useState(null);
  const [samples, setSamples] = useState([]);
  
  // Inference Settings
  const [maxFrames, setMaxFrames] = useState(60);
  const [stride, setStride] = useState(4);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  
  // Real-time backend progress hooks & event logs
  const [liveProgress, setLiveProgress] = useState(0);
  const [liveStepDesc, setLiveStepDesc] = useState('');
  const [eventLogs, setEventLogs] = useState([]);

  const fileInputRef = useRef(null);
  const eventSourceRef = useRef(null);
  const logsEndRef = useRef(null);
  const navigate = useNavigate();
  const elapsedSecs = useElapsedTimer(isProcessing);

  useEffect(() => {
    axios.get('http://localhost:8000/api/samples')
      .then(res => setSamples(res.data))
      .catch(() => {
        setSamples([
          { id: 'real', name: 'id1_0000_real.mp4', title: 'Authentic Interview', description: 'Original HD facial video — FaceForensics++ baseline', badge: 'Authentic', badge_color: 'green' },
          { id: 'fake', name: 'id1_id9_0000_fake.mp4', title: 'FaceSwap DeepFake', description: 'AI-synthesized facial manipulation — FaceForensics++', badge: 'Manipulated', badge_color: 'red' },
        ]);
      });

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Auto-scroll event logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [eventLogs]);

  // Handle local file selection & metadata extraction
  const processSelectedFile = async (selected) => {
    if (!selected) return;
    if (!selected.name.match(/\.(mp4|avi|mov)$/i)) {
      setError('Please upload a valid video file (.mp4, .avi, .mov)');
      setFile(null);
      setFileMeta(null);
      return;
    }
    setFile(selected);
    setError('');
    setIsExtractingMeta(true);
    
    const meta = await extractVideoMetadata(selected);
    setFileMeta(meta);
    setIsExtractingMeta(false);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) processSelectedFile(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const selected = e.dataTransfer.files[0];
    if (selected) processSelectedFile(selected);
  };

  // Fetch YouTube Metadata
  const handleFetchYoutubeInfo = async (urlToFetch) => {
    const targetUrl = (urlToFetch || youtubeUrl).trim();
    if (!targetUrl) {
      setYoutubeError('Please enter a YouTube video or shorts URL.');
      return;
    }
    if (!targetUrl.includes('youtube.com') && !targetUrl.includes('youtu.be')) {
      setYoutubeError('Invalid URL. Please enter a valid youtube.com or youtu.be link.');
      return;
    }

    setIsFetchingInfo(true);
    setYoutubeError('');
    try {
      const res = await axios.post('http://localhost:8000/api/youtube/info', { url: targetUrl });
      if (res.data && res.data.data) {
        setYoutubeInfo(res.data.data);
      }
    } catch (err) {
      setYoutubeError(err.response?.data?.detail || 'Could not retrieve YouTube video info. Please verify the URL.');
      setYoutubeInfo(null);
    } finally {
      setIsFetchingInfo(false);
    }
  };

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    setError('');
  };

  // Execution with real-time backend SSE telemetry & live logs
  const handleRunAnalysis = async () => {
    if (inputMode === 'upload' && !file) {
      setError('Please select or upload a video file first.');
      return;
    }
    if (inputMode === 'youtube' && !youtubeUrl.trim()) {
      setError('Please enter a valid YouTube URL first.');
      return;
    }
    if (inputMode === 'benchmark' && !selectedSample) {
      setError('Please select a benchmark sample card below.');
      return;
    }

    setIsProcessing(true);
    setError('');
    setLiveProgress(2.0);
    setLiveStepDesc('Initializing forensic pipeline...');
    setEventLogs([
      { time: '0.0s', text: 'Initializing PwTF-FG-ViT deepfake forensic pipeline...' },
      { time: '0.2s', text: `Allocating tensor memory for ${maxFrames} frames (Stride: ${stride})` }
    ]);

    const jobId = 'job_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);

    if (eventSourceRef.current) eventSourceRef.current.close();
    
    try {
      const sse = new EventSource(`http://localhost:8000/api/jobs/${jobId}/stream`);
      eventSourceRef.current = sse;

      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.progress !== undefined) {
            setLiveProgress(Math.max(2, Math.min(100, data.progress)));
          }
          if (data.step_desc) {
            setLiveStepDesc(data.step_desc);
            setEventLogs((prev) => {
              if (prev.length > 0 && prev[prev.length - 1].text === data.step_desc) return prev;
              const nowFormatted = (performance.now() / 1000).toFixed(1) + 's';
              return [...prev, { time: nowFormatted, text: data.step_desc }];
            });
          }
          if (data.status === 'completed' || data.status === 'error') {
            sse.close();
          }
        } catch (e) {}
      };

      sse.onerror = () => {
        const pollInterval = setInterval(async () => {
          try {
            const pollRes = await axios.get(`http://localhost:8000/api/jobs/${jobId}`);
            if (pollRes.data) {
              setLiveProgress(pollRes.data.progress || 0);
              setLiveStepDesc(pollRes.data.step_desc || '');
              if (pollRes.data.status === 'completed' || pollRes.data.status === 'error') {
                clearInterval(pollInterval);
              }
            }
          } catch (e) {
            clearInterval(pollInterval);
          }
        }, 800);
      };
    } catch (e) {}

    const formData = new FormData();
    formData.append('job_id', jobId);
    if (inputMode === 'youtube') {
      formData.append('youtube_url', youtubeUrl.trim());
    } else if (inputMode === 'benchmark') {
      formData.append('sample_id', selectedSample.id);
    } else {
      formData.append('file', file);
    }
    formData.append('max_frames', maxFrames);
    formData.append('stride', stride);

    try {
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const { result } = response.data;
      const sampleMeta = inputMode === 'youtube' && youtubeInfo 
        ? { name: youtubeInfo.title, uploader: youtubeInfo.uploader, url: youtubeUrl }
        : (inputMode === 'upload' && fileMeta 
            ? { name: file.name, size: file.size, duration: fileMeta.duration, resolution: fileMeta.resolutionLabel }
            : selectedSample);

      if (eventSourceRef.current) eventSourceRef.current.close();

      try {
        sessionStorage.setItem('last_result', JSON.stringify(result));
        if (sampleMeta) sessionStorage.setItem('last_sample', JSON.stringify(sampleMeta));
      } catch (e) {}
      
      navigate('/results', { state: { result, sample: sampleMeta } });
    } catch (err) {
      if (eventSourceRef.current) eventSourceRef.current.close();
      console.error(err);
      setError(err.response?.data?.detail || 'Analysis failed. Verify the backend service is running on port 8000.');
      setIsProcessing(false);
    }
  };

  const hasValidInput = 
    (inputMode === 'upload' && file) ||
    (inputMode === 'youtube' && youtubeUrl.trim().length > 0) ||
    (inputMode === 'benchmark' && selectedSample);

  return (
    <div className="analyze-wrap">
      <div className="analyze-header">
        <div className="section-label">Forensic Analysis Ingestion</div>
        <h2>Upload & Configure</h2>
        <p>Submit local video files, YouTube streams, or benchmark targets for frequency-guided forensics.</p>
      </div>

      {/* Input Mode Selector */}
      <div className="mode-selector">
        <button 
          className={`mode-btn ${inputMode === 'upload' ? 'active' : ''}`}
          onClick={() => { setInputMode('upload'); setError(''); }}
        >
          <UploadCloud size={16} /> File Upload
        </button>
        <button 
          className={`mode-btn ${inputMode === 'youtube' ? 'active' : ''}`}
          onClick={() => { setInputMode('youtube'); setError(''); }}
        >
          <YoutubeIcon size={16} /> YouTube URL
        </button>
        <button 
          className={`mode-btn ${inputMode === 'benchmark' ? 'active' : ''}`}
          onClick={() => { setInputMode('benchmark'); setError(''); }}
        >
          <Film size={16} /> Benchmark Library
        </button>
      </div>

      {/* MODE 1: FILE UPLOAD WITH DRAG & DROP AND VISUAL PREVIEW */}
      {inputMode === 'upload' && (
        <div>
          {!file ? (
            <div
              className="upload-zone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <UploadCloud className="upload-zone-icon" size={40} strokeWidth={1.25} />
              <div className="upload-zone-title">Drag & drop video or click to browse</div>
              <div className="upload-zone-sub">MP4 · AVI · MOV &nbsp;·&nbsp; AUTOMATIC RESOLUTION & DURATION DETECTION</div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                accept=".mp4,.avi,.mov"
              />
            </div>
          ) : (
            <div className="file-preview-card">
              <div className="file-thumb-wrap">
                {fileMeta?.thumbnail ? (
                  <img src={fileMeta.thumbnail} alt="Video Preview" className="file-thumb" />
                ) : (
                  <div className="file-thumb-placeholder">
                    <FileVideo size={36} color="var(--amber)" />
                  </div>
                )}
                {fileMeta?.duration > 0 && (
                  <span className="yt-duration">
                    {Math.floor(fileMeta.duration / 60)}:{(Math.floor(fileMeta.duration % 60)).toString().padStart(2, '0')}
                  </span>
                )}
              </div>

              <div className="file-details">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div className="file-title">{file.name}</div>
                  <button 
                    className="file-remove-btn"
                    onClick={() => { setFile(null); setFileMeta(null); }}
                    title="Remove file"
                  >
                    <X size={16} />
                  </button>
                </div>

                <div className="file-badges">
                  {fileMeta?.resolutionLabel && (
                    <span className="file-badge"><Layers size={12} /> {fileMeta.resolutionLabel}</span>
                  )}
                  {fileMeta?.duration > 0 && (
                    <span className="file-badge"><Clock size={12} /> {fileMeta.duration.toFixed(1)}s timeline</span>
                  )}
                  <span className="file-badge"><FileVideo size={12} /> {(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>

                <div className="file-status-tag">
                  <CheckCircle2 size={13} color="var(--green)" /> Frame Geometry Verified · Ready for Forensic Evaluation
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* MODE 2: YOUTUBE URL */}
      {inputMode === 'youtube' && (
        <div className="youtube-panel">
          <div className="youtube-input-group">
            <div className="url-input-wrap">
              <LinkIcon size={18} className="url-icon" />
              <input
                type="text"
                className="url-input"
                placeholder="Paste YouTube Video or Shorts URL (e.g. https://www.youtube.com/watch?v=...)"
                value={youtubeUrl}
                onChange={(e) => {
                  setYoutubeUrl(e.target.value);
                  setYoutubeError('');
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleFetchYoutubeInfo();
                }}
              />
            </div>
            <button 
              className="btn-secondary"
              onClick={() => handleFetchYoutubeInfo()}
              disabled={isFetchingInfo || !youtubeUrl.trim()}
              style={{ padding: '0.8rem 1.4rem' }}
            >
              {isFetchingInfo ? 'Fetching...' : 'Fetch Info'}
            </button>
          </div>

          {youtubeError && <div className="error-bar" style={{ marginTop: '1rem' }}>{youtubeError}</div>}

          {/* YouTube Video Preview Card */}
          {youtubeInfo && (
            <div className="yt-preview-card">
              {youtubeInfo.thumbnail && (
                <div className="yt-thumb-wrap">
                  <img src={youtubeInfo.thumbnail} alt={youtubeInfo.title} className="yt-thumb" />
                  {youtubeInfo.duration > 0 && (
                    <span className="yt-duration">
                      {Math.floor(youtubeInfo.duration / 60)}:{(youtubeInfo.duration % 60).toString().padStart(2, '0')}
                    </span>
                  )}
                </div>
              )}
              <div className="yt-details">
                <div className="yt-title">{youtubeInfo.title}</div>
                <div className="yt-meta">
                  <span className="yt-meta-item"><User size={13} /> {youtubeInfo.uploader}</span>
                  {youtubeInfo.duration > 0 && (
                    <span className="yt-meta-item"><Clock size={13} /> {youtubeInfo.duration}s duration</span>
                  )}
                </div>
                <div className="yt-status-tag">
                  <CheckCircle2 size={13} color="var(--green)" /> Stream Verified · Ready for Forensic Download & Analysis
                </div>
              </div>
            </div>
          )}

          {/* Quick preset suggestions */}
          <div className="yt-presets">
            <span className="yt-preset-label">Preset Deepfake Target:</span>
            <button 
              className="yt-preset-chip"
              onClick={() => {
                const sampleUrl = 'https://www.youtube.com/watch?v=cQ54GDm1eL0';
                setYoutubeUrl(sampleUrl);
                handleFetchYoutubeInfo(sampleUrl);
              }}
            >
              <Sparkles size={12} color="var(--amber)" /> Obama Deepfake (Talking Head / Jordan Peele)
            </button>
          </div>
        </div>
      )}

      {/* MODE 3: BENCHMARK SAMPLES */}
      {inputMode === 'benchmark' && (
        <div>
          <div className="sample-cards">
            {samples.map((s) => {
              const isReal = s.id === 'real';
              const isSelected = selectedSample?.id === s.id;
              return (
                <div
                  key={s.id}
                  className={`sample-card ${isSelected ? (isReal ? 'selected-real' : 'selected-fake') : ''}`}
                  onClick={() => handleSelectSample(s)}
                >
                  <div className="sample-card-header">
                    <div className="sample-card-title">
                      {isReal
                        ? <ShieldCheck size={16} color="var(--green)" />
                        : <ShieldAlert size={16} color="var(--red)" />
                      }
                      {s.title}
                    </div>
                    <span className={`sample-badge ${isReal ? 'real' : 'fake'}`}>{s.badge}</span>
                  </div>
                  <div className="sample-card-desc">{s.description}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {error && <div className="error-bar">{error}</div>}

      {/* Settings Panel */}
      <div className="settings-section">
        <div className="settings-header">Analysis Parameters</div>

        <div className="setting-row">
          <div className="setting-label">
            <span>Max Frames</span>
            <span className="setting-val">{maxFrames} frames &nbsp;·&nbsp; ~{(maxFrames / 30).toFixed(1)}s</span>
          </div>
          <input
            type="range"
            min="15" max="300" step="15"
            value={maxFrames}
            onChange={(e) => setMaxFrames(Number(e.target.value))}
          />
          <div className="setting-hint">Lower = faster on CPU · Higher = thorough on long videos</div>
        </div>

        <div className="setting-row">
          <div className="setting-label">
            <span>Clip Sampling Stride</span>
            <span className="setting-val">{stride}</span>
          </div>
          <input
            type="range"
            min="1" max="16" step="1"
            value={stride}
            onChange={(e) => setStride(Number(e.target.value))}
          />
          <div className="setting-hint">Stride 4–8 recommended for real-time CPU evaluation</div>
        </div>
      </div>

      {/* COMPREHENSIVE FORENSIC TELEMETRY HUD & LIVE COMMAND CONSOLE */}
      {isProcessing && (
        <div className="telemetry-hud-container">
          {/* Top HUD Banner */}
          <div className="telemetry-hud-header">
            <div className="hud-radar-status">
              <span className="hud-pulse-radar" />
              <span className="hud-engine-tag">ENGINE ACTIVE // FG-ViT-PwTF // INFERENCE PIPELINE</span>
            </div>
            <div className="hud-clock-readout">
              <Clock size={13} color="var(--amber)" />
              <span>ELAPSED: {elapsedSecs.toFixed(1)}s</span>
            </div>
          </div>

          {/* Main Progress & ASCII Readout */}
          <div className="hud-score-row">
            <div className="hud-score-left">
              <div className="hud-score-header-line">
                <span className="hud-score-title">PIPELINE EXECUTION PROGRESS</span>
                <div className="hud-pct-readout">{liveProgress.toFixed(1)}%</div>
              </div>
              <div className="hud-gauge-text">{buildGauge(liveProgress)}</div>
            </div>
          </div>

          {/* Real-time Progress Bar */}
          <div className="realtime-progress-track" style={{ height: '8px', margin: '1.25rem 0' }}>
            <div 
              className="realtime-progress-fill" 
              style={{ width: `${liveProgress}%` }} 
            />
          </div>

          {/* 5-Stage Execution Matrix */}
          <div className="hud-stages-matrix">
            <div className="hud-stages-title">
              <span>FORENSIC PIPELINE EXECUTION MATRIX</span>
              <span style={{ color: 'var(--amber)' }}>STAGE {liveProgress < 20 ? '1/5' : liveProgress < 30 ? '2/5' : liveProgress < 85 ? '3/5' : liveProgress < 92 ? '4/5' : '5/5'}</span>
            </div>
            <div className="hud-stages-grid">
              {STAGES.map((stg) => {
                const isCompleted = liveProgress >= stg.max;
                const isActive = liveProgress >= stg.min && liveProgress < stg.max;
                const isQueued = liveProgress < stg.min;
                const stateClass = isCompleted ? 'stage-done' : isActive ? 'stage-active' : 'stage-queued';

                return (
                  <div key={stg.id} className={`hud-stage-item ${stateClass}`}>
                    <div className="hud-stage-top">
                      <span className="hud-stage-code">STAGE {stg.code}</span>
                      <span className={`hud-stage-badge ${stateClass}`}>
                        {isCompleted ? 'COMPLETED' : isActive ? 'PROCESSING' : 'QUEUED'}
                      </span>
                    </div>
                    <div className="hud-stage-title">{stg.title}</div>
                    <div className="hud-stage-sub">
                      {isActive ? (liveStepDesc || stg.sub) : stg.sub}
                    </div>
                    {isActive && (
                      <div className="hud-stage-minitrack">
                        <div 
                          className="hud-stage-minifill" 
                          style={{ width: `${((liveProgress - stg.min) / (stg.max - stg.min)) * 100}%` }} 
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Diagnostics Specs Matrix */}
          <div className="hud-diag-grid">
            <div className="hud-diag-cell">
              <span className="hud-diag-key">FRAMES EVALUATED</span>
              <span className="hud-diag-val">{maxFrames} FRAMES</span>
            </div>
            <div className="hud-diag-cell">
              <span className="hud-diag-key">SAMPLING STRIDE</span>
              <span className="hud-diag-val">INTERVAL {stride}</span>
            </div>
            <div className="hud-diag-cell">
              <span className="hud-diag-key">ATTENTION ENCODER</span>
              <span className="hud-diag-val">APNResNet + I3D</span>
            </div>
            <div className="hud-diag-cell">
              <span className="hud-diag-key">DECISION CUTOFF</span>
              <span className="hud-diag-val">0.1500000</span>
            </div>
          </div>

          {/* Live Terminal Stream Log */}
          <div className="hud-terminal-box">
            <div className="hud-terminal-header">
              <Terminal size={12} color="var(--amber)" />
              <span>LIVE TELEMETRY STREAM LOGS</span>
              <span className="hud-pulse-dot" />
            </div>
            <div className="hud-terminal-body">
              {eventLogs.map((log, i) => (
                <div key={i} className="hud-log-line">
                  <span className="hud-log-time">[{log.time}]</span>
                  <span className="hud-log-text">{log.text}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      <button
        className="btn-primary"
        style={{ width: '100%', marginTop: '1.5rem', padding: '1.1rem' }}
        onClick={handleRunAnalysis}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <><div className="spinner" /> Forensic Pipeline Running — {liveProgress.toFixed(0)}% Complete</>
        ) : (
          `Run Deepfake Analysis${hasValidInput ? '' : ' — select a target first'}`
        )}
      </button>
    </div>
  );
};

export default Analyze;
