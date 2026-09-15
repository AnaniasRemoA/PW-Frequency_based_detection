import React from 'react';
import { Link } from 'react-router-dom';
import { Play, Activity, Target, Zap, Lock, ScanLine } from 'lucide-react';

const Home = () => {
  return (
    <div>
      <section className="hero">
        <h1>Detect Deepfakes<br/>with Precision</h1>
        <p>
          DeepGuard uses Pixel-Wise Temporal Frequency Analysis and 3D Transformer 
          networks to expose AI-generated facial manipulations in videos — frame by frame.
        </p>
        <Link to="/analyze" className="btn-primary" style={{ marginTop: '1rem' }}>
          <Play size={20} fill="currentColor" /> Start Detection
        </Link>
      </section>

      <section className="features-grid">
        <div className="feature-card">
          <div className="feature-icon"><Activity size={24} /></div>
          <h3 className="feature-title">Frequency Domain Analysis</h3>
          <p className="feature-desc">
            Extracts pixel-wise temporal frequency signals using FFT and median
            filtering to reveal GAN compression artifacts.
          </p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon"><ScanLine size={24} /></div>
          <h3 className="feature-title">3D Spatio-Temporal Transformer</h3>
          <p className="feature-desc">
            Combines FTCN video backbone with Spatial & Temporal Transformer Encoders
            for rich multi-scale feature fusion.
          </p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon"><Target size={24} /></div>
          <h3 className="feature-title">Per-Frame Face Tracking</h3>
          <p className="feature-desc">
            Detects, aligns, and tracks individual faces across the entire video
            using 68-point landmarks.
          </p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon"><Zap size={24} /></div>
          <h3 className="feature-title">Real-Time Performance</h3>
          <p className="feature-desc">
            Runs efficiently on GPU hardware, with configurable stride fallbacks 
            for CPU operation.
          </p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon"><Play size={24} /></div>
          <h3 className="feature-title">Annotated Output</h3>
          <p className="feature-desc">
            Produces a browser-playable MP4 with bounding boxes, labels, and 
            confidence scores per frame.
          </p>
        </div>
        
        <div className="feature-card">
          <div className="feature-icon"><Lock size={24} /></div>
          <h3 className="feature-title">Local & Private</h3>
          <p className="feature-desc">
            Everything runs on your machine. No video data is uploaded 
            to external servers. Fully private.
          </p>
        </div>
      </section>
    </div>
  );
};

export default Home;
