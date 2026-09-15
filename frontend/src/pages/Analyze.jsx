import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { UploadCloud, Settings, Loader2 } from 'lucide-react';

const Analyze = () => {
  const [file, setFile] = useState(null);
  const [maxFrames, setMaxFrames] = useState(300);
  const [stride, setStride] = useState(4);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      if (!selected.name.match(/\.(mp4|avi|mov)$/i)) {
        setError('Please upload a valid video file (.mp4, .avi, .mov)');
        setFile(null);
        return;
      }
      setFile(selected);
      setError('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const selected = e.dataTransfer.files[0];
    if (selected) {
      if (!selected.name.match(/\.(mp4|avi|mov)$/i)) {
        setError('Please upload a valid video file (.mp4, .avi, .mov)');
        setFile(null);
        return;
      }
      setFile(selected);
      setError('');
    }
  };

  const handleRunAnalysis = async () => {
    if (!file) {
      setError('Please select a video file first.');
      return;
    }

    setIsProcessing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('max_frames', maxFrames);
    formData.append('stride', stride);

    try {
      // API call to FastAPI backend
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const { result } = response.data;
      
      // Navigate to results page with data
      navigate('/results', { state: { result } });
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred during analysis.');
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="section-title">Upload & Configuration</div>
      
      <div className="glass-card">
        <div 
          className={`upload-area ${file ? 'active' : ''}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <UploadCloud className="upload-icon" size={48} />
          {file ? (
            <>
              <div className="upload-text">{file.name}</div>
              <div className="upload-subtext">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
            </>
          ) : (
            <>
              <div className="upload-text">Drag & drop or click to upload</div>
              <div className="upload-subtext">Supports MP4, AVI, MOV</div>
            </>
          )}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            style={{ display: 'none' }} 
            accept=".mp4,.avi,.mov"
          />
        </div>

        {error && (
          <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.9rem' }}>
            {error}
          </div>
        )}

        <div className="settings-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#e2e8f0', fontWeight: '600' }}>
            <Settings size={18} /> Analysis Settings
          </div>
          
          <div className="setting-row">
            <label className="setting-label">Max Frames to Analyze: {maxFrames}</label>
            <input 
              type="range" 
              min="60" max="768" step="30" 
              value={maxFrames} 
              onChange={(e) => setMaxFrames(e.target.value)}
              className="setting-input" 
              style={{ padding: '0' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>Higher = more thorough but slower</div>
          </div>
          
          <div className="setting-row">
            <label className="setting-label">Clip Stride: {stride}</label>
            <input 
              type="range" 
              min="1" max="16" step="1" 
              value={stride} 
              onChange={(e) => setStride(e.target.value)}
              className="setting-input"
              style={{ padding: '0' }}
            />
            <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>Increase to speed up CPU inference</div>
          </div>
        </div>

        <button 
          className="btn-primary" 
          style={{ width: '100%', marginTop: '1.5rem' }}
          onClick={handleRunAnalysis}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <><div className="spinner"></div> Analyzing (this may take a few minutes)...</>
          ) : (
            'Run Deepfake Analysis'
          )}
        </button>
      </div>
    </div>
  );
};

export default Analyze;
