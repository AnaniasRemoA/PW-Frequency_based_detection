import React from 'react';
import { Link } from 'react-router-dom';

// Build bars for the terminal-style score
const buildBar = (pct, filled = '█', empty = '░', total = 20) => {
  const n = Math.round((pct / 100) * total);
  return filled.repeat(n) + empty.repeat(total - n);
};

const TICKER_ITEMS = [
  '// FG-ViT 1.0',
  '// PwTF Framework',
  '// FaceForensics++ Trained',
  '// Celeb-DF v2 Evaluated',
  '// CIFAKE Validated',
  '// GenImage Benchmarked',
  '// Threshold 0.1500000',
  '// I3D 8×8 Backbone',
  '// APNResNet Freq Attention',
  '// 68-pt Landmark Tracking',
  '// Grad-CAM Explainability',
  '// Local · Private · Open',
];

const MANIFEST = [
  {
    num: '01',
    title: 'Frequency Domain Analysis',
    desc: 'Pixel-wise temporal FFT residuals expose the periodic upsampling artifacts that GAN decoders leave in facial reconstructions — invisible to human eyes, unmistakable to a trained spectral classifier.',
  },
  {
    num: '02',
    title: '3D Spatio-Temporal Transformer',
    desc: 'A dual-branch architecture fuses spatial face features (I3D 8×8) with frequency attention tokens (APNResNet) through a Transformer Encoder — reading forgeries across both pixel space and frequency space simultaneously.',
  },
  {
    num: '03',
    title: 'Per-Frame Face Tracking',
    desc: 'RetinaFace detection followed by 68-point landmark alignment isolates every face track through the entire video timeline. Each track is analyzed independently at clip level.',
  },
  {
    num: '04',
    title: 'Grad-CAM Saliency Attribution',
    desc: 'Class activation maps trace exactly which spatial regions drove the forgery decision — blending seams, eye regions, jaw boundaries. Evidence you can read, not just a number.',
  },
  {
    num: '05',
    title: 'Annotated Forensic Output',
    desc: 'Browser-playable MP4 with per-frame bounding boxes and live forgery probability overlays. Download the complete annotated evidence video for offline review or case documentation.',
  },
  {
    num: '06',
    title: 'Local & Fully Private',
    desc: 'Every byte of inference runs on your own machine. No video is ever transmitted to an external server. Forensic evidence stays within your custody chain.',
  },
];

const Home = () => {
  return (
    <div>
      {/* Hero */}
      <section className="home-hero">
        <div className="home-hero-left">
          <div className="home-hero-eyebrow">Forensic AI · Deepfake Detection</div>
          <h1>
            Evidence-Grade<br />
            <em>Deepfake</em><br />
            Forensics
          </h1>
          <p className="home-hero-desc">
            DeepGuard applies Frequency-Guided Vision Transformers to expose AI-generated 
            facial manipulations in video — frame by frame, artifact by artifact.
          </p>
          <div className="hero-cta">
            <Link to="/analyze" className="btn-primary">
              Run Analysis
            </Link>
            <Link to="/analyze" className="cta-arrow-link">
              <span className="cta-arrow" />
              Upload a video
            </Link>
          </div>
        </div>

        <div className="home-hero-right">
          <div className="hero-meta-item">
            <div className="hero-meta-label">Framework</div>
            <div className="hero-meta-value">FG-ViT · PwTF v1.0</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Datasets & Benchmarks</div>
            <div className="hero-meta-value">FaceForensics++ · Celeb-DF · CIFAKE</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Decision Threshold</div>
            <div className="hero-meta-value">0.1500000</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Runtime</div>
            <div className="hero-meta-value">CPU · GPU CUDA</div>
          </div>
        </div>
      </section>

      {/* Ticker */}
      <div className="ticker-wrap" aria-hidden="true">
        <div className="ticker-track">
          {[...TICKER_ITEMS, ...TICKER_ITEMS].map((item, i) => (
            <span key={i} className="ticker-item">
              <span className="tick-accent">{item.slice(0, 2)}</span>
              {item.slice(2)}
            </span>
          ))}
        </div>
      </div>

      {/* Manifest */}
      <section className="manifest">
        <div className="section-label">System Capabilities</div>
        {MANIFEST.map((item) => (
          <div key={item.num} className="manifest-item">
            <div className="manifest-num">{item.num}</div>
            <div>
              <div className="manifest-title">{item.title}</div>
              <div className="manifest-desc">{item.desc}</div>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
};

export default Home;
