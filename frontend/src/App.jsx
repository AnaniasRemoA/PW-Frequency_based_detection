import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield, Activity } from 'lucide-react';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Results from './pages/Results';

const Navbar = () => {
  const location = useLocation();
  
  return (
    <nav className="navbar">
      <Link to="/" className="logo-link">
        <Shield className="logo-icon" color="#818cf8" fill="rgba(129, 140, 248, 0.2)" />
        <span className="logo-text">DeepGuard</span>
      </Link>
      
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
        <Link to="/analyze" className={`nav-link ${location.pathname === '/analyze' ? 'active' : ''}`}>Analyze</Link>
        <div className="status-badge">
          <Activity size={14} /> GPU · CUDA
        </div>
      </div>
    </nav>
  );
};

const App = () => {
  return (
    <BrowserRouter>
      <div className="app-container">
        <div className="bg-blobs">
          <div className="blob-1"></div>
          <div className="blob-2"></div>
        </div>
        
        <Navbar />
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
};

export default App;
