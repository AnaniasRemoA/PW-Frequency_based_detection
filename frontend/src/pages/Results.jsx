import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ShieldAlert, ShieldCheck, Download, ArrowLeft } from 'lucide-react';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!location.state || !location.state.result) {
      navigate('/analyze');
      return;
    }
    setResult(location.state.result);
  }, [location, navigate]);

  if (!result) return null;

  const { is_fake, confidence, fake_prob_pct, total_frames, analyzed_clips, tracks, elapsed, video_url } = result;
  
  const statusClass = is_fake ? 'fake' : 'real';
  const Icon = is_fake ? ShieldAlert : ShieldCheck;
  const verdictText = is_fake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO';
  const confidenceLabel = is_fake ? 'Fake Probability' : 'Authenticity';
  
  // Bar width logic matching your original python logic
  const barWidth = is_fake ? Math.max(2, fake_prob_pct) : Math.max(2, 100 - fake_prob_pct);

  // Format time
  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60);
  const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <button className="btn-secondary" onClick={() => navigate('/analyze')}>
          <ArrowLeft size={16} /> Analyze Another Video
        </button>
      </div>

      <div className="results-layout">
        <div>
          <div className="section-title">Detection Verdict</div>
          <div className={`glass-card verdict-card ${statusClass}`}>
            <Icon className="verdict-icon" />
            <div className="verdict-title">{verdictText}</div>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              {confidenceLabel}: <strong style={{ color: is_fake ? '#fca5a5' : '#86efac' }}>{barWidth.toFixed(1)}%</strong>
              <span style={{ opacity: 0.5, fontSize: '0.8em', marginLeft: '0.5rem' }}>· raw {confidence.toFixed(6)}</span>
            </div>
            
            <div className="confidence-bar-bg">
              <div className="confidence-bar-fill" style={{ width: `${barWidth}%` }}></div>
            </div>

            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-value">{total_frames}</div>
                <div className="stat-label">Frames</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{analyzed_clips}</div>
                <div className="stat-label">Clips</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{tracks}</div>
                <div className="stat-label">Face Tracks</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{timeStr}</div>
                <div className="stat-label">Time</div>
              </div>
            </div>
            
            <a 
              href={`http://localhost:8000${video_url}`} 
              download 
              className="btn-primary" 
              style={{ width: '100%', marginTop: '1.5rem', background: 'linear-gradient(135deg, #0369a1, #0ea5e9)' }}
            >
              <Download size={18} /> Download Annotated Video
            </a>
          </div>
        </div>

        <div>
          <div className="section-title">Annotated Output</div>
          <div className="glass-card" style={{ padding: '1rem' }}>
            <div className="video-container">
              <video src={`http://localhost:8000${video_url}`} controls autoPlay loop muted playsInline />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
