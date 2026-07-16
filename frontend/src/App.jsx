import React, { useState, useEffect, useRef } from 'react';

const MODES = [
  { id: 'object', name: 'Object Detection', label: 'Objects' },
  { id: 'human', name: 'Human Detection', label: 'Humans' },
  { id: 'vehicle', name: 'Vehicle Detection', label: 'Vehicles' },
  { id: 'movement', name: 'Movement Analysis', label: 'Motion' },
  { id: 'emotion', name: 'Emotion Recognition', label: 'Expression' },
  { id: 'sign', name: 'Sign Language', label: 'Gesture' }
];

export default function App() {
  const [activeMode, setActiveMode] = useState('object');
  const [sourceType, setSourceType] = useState('webcam'); // 'webcam' or 'file'
  const [uploadedPath, setUploadedPath] = useState('');
  const [isLightTheme, setIsLightTheme] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  
  // Telemetry stats
  const [stats, setStats] = useState({
    fps: 0,
    object_count: 0,
    human_count: 0,
    vehicle_count: 0,
    motion_detected: false,
    current_emotion: 'Neutral',
    current_gesture: 'No Hand Detected',
    detection_log: []
  });

  // Settings states
  const [settings, setSettings] = useState({
    confidence_threshold: 0.5,
    nms_threshold: 0.4,
    motion_sensitivity: 1000,
    show_bounding_boxes: true,
    motion_display_mode: 'color'
  });

  const fileInputRef = useRef(null);

  // Fetch settings on load
  useEffect(() => {
    fetch('/api/settings')
      .then(res => res.json())
      .then(data => {
        setSettings(data);
      })
      .catch(err => console.error('Error fetching settings:', err));
  }, []);

  // Poll stats every 800ms
  useEffect(() => {
    const timer = setInterval(() => {
      fetch('/api/stats')
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(err => console.error('Error fetching stats:', err));
    }, 800);
    return () => clearInterval(timer);
  }, []);

  // Theme toggle helper
  const toggleTheme = () => {
    document.body.classList.toggle('light-theme');
    setIsLightTheme(prev => !prev);
  };

  // POST settings change
  const updateSetting = (key, value) => {
    const updatedSettings = { ...settings, [key]: value };
    setSettings(updatedSettings);

    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updatedSettings)
    }).catch(err => console.error('Error saving settings:', err));
  };

  // File upload handler
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setUploadStatus('Uploading file...');

    fetch('/api/upload', {
      method: 'POST',
      body: formData
    })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          setUploadedPath(data.file_path);
          setSourceType('file');
          setUploadStatus('Processing video...');
          setTimeout(() => setUploadStatus(''), 3000);
        } else {
          setUploadStatus(`Upload failed: ${data.error}`);
        }
      })
      .catch(err => {
        console.error(err);
        setUploadStatus('Upload error.');
      });
  };

  // Determine current stream URL
  const getStreamUrl = () => {
    const sourceStr = sourceType === 'webcam' ? 'webcam' : uploadedPath;
    // Map backend route name: backend app.py has /object/video_feed, /vehicle/video_feed, etc.
    const routeName = activeMode === 'movement' ? 'movement' : activeMode;
    return `/api/video_feed/${routeName}?source=${encodeURIComponent(sourceStr)}`;
  };

  // Get active mode's specific telemetry status values
  const getModeTelemetry = () => {
    switch (activeMode) {
      case 'object':
        return { val: stats.object_count, lbl: 'Objects' };
      case 'human':
        return { val: stats.human_count, lbl: 'Humans' };
      case 'vehicle':
        return { val: stats.vehicle_count, lbl: 'Vehicles' };
      case 'movement':
        return { val: stats.motion_detected ? 'YES' : 'NO', lbl: 'Motion Alert', isAlert: stats.motion_detected };
      case 'emotion':
        return { val: stats.current_emotion, lbl: 'Expression' };
      case 'sign':
        return { val: stats.current_gesture, lbl: 'Gesture' };
      default:
        return { val: 0, lbl: 'Count' };
    }
  };

  const currentModeInfo = MODES.find(m => m.id === activeMode);
  const telemetry = getModeTelemetry();

  return (
    <div className="app-container">
      {/* Sidebar navigation */}
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <img src="/static/assets/logo_nav.png" alt="VisioCam Logo" style={{ width: '80px', height: '50px', filter: 'drop-shadow(0 0 5px rgba(0,210,255,0.4))' }} />
          <h1>VisioCam Core</h1>
          <span>Data Divers</span>
        </div>

        <nav className="sidebar-nav">
          {MODES.map(mode => (
            <button
              key={mode.id}
              onClick={() => setActiveMode(mode.id)}
              className={`nav-item-btn ${activeMode === mode.id ? 'active' : ''}`}
            >
              <span>{mode.name}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Workspace */}
      <main className="app-content">
        <header className="app-header">
          <h2>{currentModeInfo.name} Interface</h2>
          <button onClick={toggleTheme} className="theme-btn">
            {isLightTheme ? 'Dark Mode' : 'Light Mode'}
          </button>
        </header>

        <div className="dashboard-grid">
          {/* Central Live Video Stream */}
          <div className="video-panel">
            <div className="video-header">
              Telemetry Stream - Source: {sourceType === 'webcam' ? 'Webcam' : 'Video Clip'}
            </div>
            
            <div className={`video-feed-wrapper ${settings.motion_display_mode === 'split' && activeMode === 'movement' ? 'split-feed' : ''}`}>
              <img
                src={getStreamUrl()}
                className="video-feed-image"
                alt="Live Telemetry Video Feed"
                onError={(e) => {
                  e.target.src = '/static/assets/backgrnd.png';
                }}
              />
            </div>
          </div>

          {/* Control Sidebar Widgets */}
          <div className="control-sidebar">
            {/* Source Selector */}
            <div className="sidebar-widget">
              <h4>Video Source</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                <button
                  onClick={() => setSourceType('webcam')}
                  className={`btn-toggle-switch ${sourceType === 'webcam' ? 'active' : ''}`}
                >
                  Live Webcam
                </button>
                <div style={{ borderTop: '1px dashed rgba(255,255,255,0.1)', margin: '0.2rem 0' }}></div>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Analyze pre-recorded video:</label>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="video/*"
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => fileInputRef.current.click()}
                  className={`btn-toggle-switch ${sourceType === 'file' ? 'active' : ''}`}
                >
                  {uploadedPath ? 'Change Video File' : 'Upload Video File'}
                </button>
                {uploadStatus && (
                  <span style={{ fontSize: '0.8rem', color: 'var(--accent)', textAlign: 'center' }}>
                    {uploadStatus}
                  </span>
                )}
              </div>
            </div>

            {/* Live Metrics */}
            <div className="sidebar-widget">
              <h4>Real-time Telemetry</h4>
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-val">{stats.fps}</div>
                  <div className="stat-lbl">FPS</div>
                </div>
                <div className={`stat-box ${telemetry.isAlert ? 'alert' : ''}`}>
                  <div className="stat-val" style={{ fontSize: typeof telemetry.val === 'string' && telemetry.val.length > 8 ? '1rem' : '' }}>
                    {telemetry.val}
                  </div>
                  <div className="stat-lbl">{telemetry.lbl}</div>
                </div>
              </div>
            </div>

            {/* Mode-Specific Engine Settings */}
            <div className="sidebar-widget">
              <h4>Settings</h4>
              
              {/* Show YOLO threshold sliders for object/human/vehicle modes */}
              {['object', 'human', 'vehicle'].includes(activeMode) && (
                <>
                  <div className="control-group">
                    <label>
                      Confidence Threshold: <span>{settings.confidence_threshold}</span>
                    </label>
                    <input
                      type="range"
                      className="slider-input"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={settings.confidence_threshold}
                      onChange={(e) => updateSetting('confidence_threshold', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className="control-group">
                    <label>
                      NMS Threshold: <span>{settings.nms_threshold}</span>
                    </label>
                    <input
                      type="range"
                      className="slider-input"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={settings.nms_threshold}
                      onChange={(e) => updateSetting('nms_threshold', parseFloat(e.target.value))}
                    />
                  </div>
                </>
              )}

              {/* Sensitivity slider for movement mode */}
              {activeMode === 'movement' && (
                <div className="control-group">
                  <label>
                    Sensitivity (Min Size): <span>{settings.motion_sensitivity}px</span>
                  </label>
                  <input
                    type="range"
                    className="slider-input"
                    min="200"
                    max="5000"
                    step="100"
                    value={settings.motion_sensitivity}
                    onChange={(e) => updateSetting('motion_sensitivity', parseInt(e.target.value))}
                  />
                </div>
              )}

              {/* display mode selectors for movement & sign mode */}
              {['movement', 'sign'].includes(activeMode) && (
                <div className="control-group">
                  <label>Display Mode</label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
                    <button
                      onClick={() => updateSetting('motion_display_mode', 'color')}
                      className={`btn-toggle-switch ${settings.motion_display_mode === 'color' ? 'active' : ''}`}
                    >
                      {activeMode === 'sign' ? 'Annotated RGB Feed' : 'Annotated Color Stream'}
                    </button>
                    <button
                      onClick={() => updateSetting('motion_display_mode', 'mask')}
                      className={`btn-toggle-switch ${settings.motion_display_mode === 'mask' ? 'active' : ''}`}
                    >
                      {activeMode === 'sign' ? 'Binary Skin Mask' : 'Binary Silhouette Mask'}
                    </button>
                    {activeMode === 'movement' && (
                      <button
                        onClick={() => updateSetting('motion_display_mode', 'split')}
                        className={`btn-toggle-switch ${settings.motion_display_mode === 'split' ? 'active' : ''}`}
                      >
                        Telemetry Split-Screen
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Bounding box toggle switch */}
              {['object', 'human', 'vehicle'].includes(activeMode) && (
                <div className="control-group" style={{ marginTop: '1.2rem' }}>
                  <button
                    onClick={() => updateSetting('show_bounding_boxes', !settings.show_bounding_boxes)}
                    className={`btn-toggle-switch ${settings.show_bounding_boxes ? 'active' : ''}`}
                  >
                    {settings.show_bounding_boxes ? 'Show Bounding Boxes' : 'Hide Bounding Boxes'}
                  </button>
                </div>
              )}

              {!['object', 'human', 'vehicle', 'movement', 'sign'].includes(activeMode) && (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  Configuration parameters for this cognitive engine are managed automatically.
                </p>
              )}
            </div>

            {/* Event log list */}
            <div className="sidebar-widget">
              <h4>Detection Log</h4>
              <div className="log-container">
                {stats.detection_log && stats.detection_log.length > 0 ? (
                  stats.detection_log.map((entry, idx) => (
                    <div key={idx} className="log-entry">
                      <span className="log-time">{entry.time}</span>
                      <span className="log-text">{entry.text}</span>
                    </div>
                  ))
                ) : (
                  <div className="log-entry">
                    <span className="log-time">--:--:--</span>
                    <span className="log-text">No recent events.</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
