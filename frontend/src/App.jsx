import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield } from 'lucide-react';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import Results from './pages/Results';

const Navbar = () => {
  const location = useLocation();
  const [deviceInfo, setDeviceInfo] = React.useState('System Ready');
  
  React.useEffect(() => {
    fetch('http://localhost:8000/api/status')
      .then(res => res.json())
      .then(data => {
        if (data.cuda_available) {
          setDeviceInfo(`GPU · ${data.device_name || 'CUDA'}`);
        } else {
          setDeviceInfo('CPU · Torch');
        }
      })
      .catch(() => {
        setDeviceInfo('Offline');
      });
  }, []);
  
  return (
    <nav className="navbar">
      <Link to="/" className="logo-link">
        <Shield className="logo-icon" size={20} />
        <span className="logo-text">DeepGuard</span>
      </Link>
      
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
        <Link to="/analyze" className={`nav-link ${location.pathname === '/analyze' ? 'active' : ''}`}>Analyze</Link>
        <div className="status-badge">{deviceInfo}</div>
      </div>
    </nav>
  );
};

const App = () => {
  return (
    <BrowserRouter>
      <div className="app-container">
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
