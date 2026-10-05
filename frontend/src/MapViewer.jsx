import { useEffect, useRef, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import {
  MapContainer,
  TileLayer,
  useMapEvents,
  useMap,
  Marker,
  Popup,
  Rectangle,
  ZoomControl
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Custom Leaflet marker icons to fix issues with default icon paths in bundlers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

// Dynamic color palette — deterministic per mineral name
// so colors are consistent across page reloads
const MINERAL_PALETTE = [
  '#ffb700', '#ff4d4d', '#00fa9a', '#a78bfa',
  '#38bdf8', '#fb923c', '#34d399', '#f472b6',
  '#fbbf24', '#60a5fa', '#4ade80', '#e879f9',
];


function getMineralColor(mineralName) {
  let hash = 0;
  for (let i = 0; i < mineralName.length; i++) {
    hash = (hash * 31 + mineralName.charCodeAt(i)) >>> 0;
  }
  return MINERAL_PALETTE[hash % MINERAL_PALETTE.length];
}


// ── Click handler component ──
function SmartMapHandler({ drawMode, drawStart, onFirstClick, onSecondClick, onDrawPreview, onIdentifyClick }) {
  useMapEvents({
    click(e) {
      if (!drawMode) { onIdentifyClick(e); return; }
      if (!drawStart) {
        onFirstClick(e);
      } else {
        onSecondClick(e);
      }
    },
    mousemove(e) {
      if (drawMode && drawStart) onDrawPreview(e);
    }
  });
  return null;
}

function YearRangeSelector({ onSubmit, disabled }) {
  const [currentJobId, setCurrentJobId] = useState(null);
  const [yearStart, setYearStart] = useState(2020);
  const [yearEnd, setYearEnd]     = useState(2023);

  const years = Array.from({ length: 11 }, (_, i) => 2015 + i);

  return (
    <div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontSize: '11px', color: '#64748b', display: 'block', marginBottom: '4px' }}>
            FROM
          </label>
          <select
            value={yearStart}
            onChange={e => setYearStart(Number(e.target.value))}
            style={{
              width: '100%', padding: '8px', background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px',
              color: '#fff', fontSize: '13px'
            }}
          >
            {years.map(y => <option key={y} value={y} style={{ background: '#1e293b' }}>{y}</option>)}
          </select>
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ fontSize: '11px', color: '#64748b', display: 'block', marginBottom: '4px' }}>
            TO
          </label>
          <select
            value={yearEnd}
            onChange={e => setYearEnd(Number(e.target.value))}
            style={{
              width: '100%', padding: '8px', background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px',
              color: '#fff', fontSize: '13px'
            }}
          >
            {years.map(y => <option key={y} value={y} style={{ background: '#1e293b' }}>{y}</option>)}
          </select>
        </div>
      </div>

      <button
        onClick={() => onSubmit(yearStart, yearEnd)}
        disabled={disabled || yearEnd < yearStart}
        style={{
          width: '100%', padding: '11px',
          background: disabled ? 'rgba(255,183,0,0.2)' : '#ffb700',
          border: 'none', borderRadius: '8px',
          color: disabled ? '#ffb700' : '#0b0c10',
          fontSize: '13px', fontWeight: 700, cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        {disabled ? '⟳ Processing...' : '🚀 Run Analysis'}
      </button>
    </div>
  );
}


// ── Main component ──
function MapViewer() {

  const [drawMode, setDrawMode]       = useState(false); // true = user is drawing
  const [drawnBbox, setDrawnBbox]     = useState(null);  // [minLon, minLat, maxLon, maxLat]
  const [drawStart, setDrawStart]     = useState(null);  // {lat, lng} of first corner
  const [drawRect, setDrawRect]       = useState(null);  // live rectangle bounds while drawing
  const [jobStatus, setJobStatus]     = useState(null);  // null | 'running' | 'done' | 'error'
  const [jobMessage, setJobMessage]   = useState('');
  const [jobResult, setJobResult]     = useState(null);  // parsed job_summary.json
  const [discoveredMinerals, setDiscoveredMinerals] = useState([]);
  const [selectedYear, setSelectedYear] = useState(null);

  const [clickedPoint, setClickedPoint] = useState(null);
  const [pixelValues, setPixelValues] = useState(null);  // array of {mineral, probability, metals, uses}
  const [isIdentifying, setIsIdentifying] = useState(false);
  const [nearestDistance, setNearestDistance] = useState(null);
  const [waterInfo, setWaterInfo] = useState(null);
  const [terrainInfo, setTerrainInfo] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  const [panelPosition, setPanelPosition] = useState({ top: 20, left: 20 });
  const [panelDragStart, setPanelDragStart] = useState(null);
  const [panelDragOffset, setPanelDragOffset] = useState({ x: 0, y: 0 });
  const panelRef = useRef(null);
  
  const [manualCoordMode, setManualCoordMode] = useState(false);
  const [manualCoords, setManualCoords] = useState({ minLon: '', minLat: '', maxLon: '', maxLat: '' });

  useEffect(() => {
    if (typeof window === "undefined") return;
    setPanelPosition({ top: 20, left: window.innerWidth - 350 - 20 });
  }, []);

  useEffect(() => {
    if (!panelDragStart) return;

    const handleMouseMove = (e) => {
      const left = e.clientX - panelDragOffset.x;
      const top = e.clientY - panelDragOffset.y;
      const width = panelRef.current?.offsetWidth || 350;
      const height = panelRef.current?.offsetHeight || 300;
      const maxLeft = window.innerWidth - width - 10;
      const maxTop = window.innerHeight - height - 10;

      setPanelPosition({
        left: Math.max(10, Math.min(left, maxLeft)),
        top: Math.max(10, Math.min(top, maxTop))
      });
    };

    const handleMouseUp = () => {
      setPanelDragStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [panelDragStart, panelDragOffset]);

  const handlePanelMouseDown = (e) => {
    if (e.button !== 0) return;
    if (!panelRef.current) return;
    e.preventDefault();
    const rect = panelRef.current.getBoundingClientRect();
    setPanelDragOffset({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setPanelDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleManualCoordSubmit = useCallback(() => {
    const { minLon, minLat, maxLon, maxLat } = manualCoords;
    const coords = [minLon, minLat, maxLon, maxLat].map(c => parseFloat(c));
    
    if (coords.some(isNaN)) {
      alert('Please enter valid numbers for all coordinates');
      return;
    }
    
    if (coords[0] >= coords[2] || coords[1] >= coords[3]) {
      alert('Min coordinates must be less than max coordinates');
      return;
    }
    
    setDrawnBbox(coords);
    setManualCoordMode(false);
    setManualCoords({ minLon: '', minLat: '', maxLon: '', maxLat: '' });
  }, [manualCoords]);

  const handleFirstClick = useCallback((e) => {
    setDrawStart(e.latlng);
    setDrawRect(null);
  }, []);

  const handleSecondClick = useCallback((e) => {
    if (!drawStart) return;
    const bbox = [
      Math.min(drawStart.lng, e.latlng.lng),
      Math.min(drawStart.lat, e.latlng.lat),
      Math.max(drawStart.lng, e.latlng.lng),
      Math.max(drawStart.lat, e.latlng.lat),
    ];
    setDrawnBbox(bbox);
    setDrawMode(false);
    setDrawStart(null);
    setDrawRect(null);
  }, [drawStart]);

  const handleDrawPreview = useCallback((e) => {
    if (!drawStart) return;
    setDrawRect([
      [Math.min(drawStart.lat, e.latlng.lat), Math.min(drawStart.lng, e.latlng.lng)],
      [Math.max(drawStart.lat, e.latlng.lat), Math.max(drawStart.lng, e.latlng.lng)],
    ]);
  }, [drawStart]);

  const handleSubmitJob = useCallback(async (yearStart, yearEnd) => {
    if (!drawnBbox) return;
    setJobStatus('running');
    setJobMessage('Submitting job...');
    setJobResult(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: drawnBbox,
          year_start: yearStart,
          year_end: yearEnd,
          generate_map: false,
        }),
      });
      const { job_id } = await res.json();
      setCurrentJobId(job_id);
      pollJob(job_id);
    } catch (err) {
      setJobStatus('error');
      setJobMessage(`Submission failed: ${err.message}`);
    }
  }, [drawnBbox]);

  const pollJob = useCallback((jobId) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/job/${jobId}`);
        const { status, message, result } = await res.json();
        setJobMessage(message || status);
        if (status === 'done') {
          clearInterval(interval);
          setJobStatus('done');
          setJobResult(result);
          const minerals = result?.results?.[0]?.minerals?.map(m => m.mineral) || [];
          setDiscoveredMinerals(minerals);
          if (result?.results?.length > 0) {
            const lastYear = result.results[result.results.length - 1].year;
            setSelectedYear(lastYear);
          }
        } else if (status === 'error') {
          clearInterval(interval);
          setJobStatus('error');
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    }, 5000);
  }, []);

  const yearOptions = [2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025].map(y => ({ value: y, label: `${y}` }));
  const layerOptions = [
    { value: "composite", label: "Composite (All Minerals)" },
    ...discoveredMinerals.map(m => ({ value: m, label: m.charAt(0).toUpperCase() + m.slice(1) }))
  ];

  const handleMapClick = useCallback(async (e) => {
    const { lat, lng } = e.latlng;
    setClickedPoint({ lat, lng });

    if (!jobResult?.results?.length || !currentJobId) return;

    const bbox = jobResult.bbox || drawnBbox;
    if (bbox) {
      const [minLon, minLat, maxLon, maxLat] = bbox;
      if (lng < minLon || lng > maxLon || lat < minLat || lat > maxLat) {
        setPixelValues([]);
        setNearestDistance(null);
        setWaterInfo(null);
        setTerrainInfo(null);
        return;
      }
    }

    setIsIdentifying(true);
    try {
      const res = await fetch('/api/predict_point', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat, lng,
          job_id: currentJobId,
          year: selectedYear
        })
      });
      const data = await res.json();
      if (data.minerals) {
        setPixelValues(data.minerals);
        setNearestDistance(data.nearest_sample_distance_km);
        setWaterInfo({ present: data.water_present, ndwi: data.ndwi });
        setTerrainInfo({ mineralized: data.mineralized, type: data.terrain_type, ndvi: data.ndvi });
      }
    } catch (err) {
      console.error('Point prediction error:', err);
    } finally {
      setIsIdentifying(false);
    }
  }, [jobResult, currentJobId, selectedYear, drawnBbox]);

  // Determine dominant mineral from API response
  const getDominantAlteration = () => {
    if (!pixelValues || pixelValues.length === 0) return null;

    // If terrain is classified as unmineralized, show terrain type
    if (terrainInfo && !terrainInfo.mineralized) {
      const terrainLabels = {
        water: { name: 'Water Body', color: '#38bdf8', icon: '💧' },
        vegetation: { name: 'Vegetation / No Exposed Rock', color: '#4ade80', icon: '🌿' },
        barren_soil: { name: 'Barren Soil / No Alteration', color: '#94a3b8', icon: '🏜️' },
      };
      const info = terrainLabels[terrainInfo.type] || { name: 'Unmineralized Terrain', color: '#94a3b8', icon: '' };
      return { name: `${info.icon} ${info.name}`, color: info.color, val: 0, unmineralized: true };
    }

    const top = pixelValues[0];
    if (!top || top.probability < 10) {
      return { name: 'Low Confidence / Weak Signal', color: '#94a3b8', val: 0 };
    }
    return {
      name: `${top.mineral} alteration`,
      color: getMineralColor(top.mineral),
      val: top.probability / 100,
      label: top.mineral,
      metals: top.metals || [],
      uses: top.uses || []
    };
  };

  const dominant = getDominantAlteration();

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", background: "#0b0c10" }}>

      {/* MAP */}
      <MapContainer
        center={[-22.35, 118.65]}
        zoom={11}
        style={{ height: "100%", width: "100%", zIndex: 1 }}
        zoomControl={false}
      >
        <ZoomControl position="topright" />
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
        />
        <SmartMapHandler
          drawMode={drawMode}
          drawStart={drawStart}
          onFirstClick={handleFirstClick}
          onSecondClick={handleSecondClick}
          onDrawPreview={handleDrawPreview}
          onIdentifyClick={handleMapClick}
        />


        {/* Live drawing rectangle */}
        {drawRect && (
          <Rectangle
            bounds={drawRect}
            pathOptions={{
              color: '#ffb700', weight: 2, dashArray: '6 4',
              fillColor: '#ffb700', fillOpacity: 0.05, interactive: false
            }}
          />
        )}

        {/* Confirmed AOI rectangle */}
        {drawnBbox && (
          <Rectangle
            bounds={[
              [drawnBbox[1], drawnBbox[0]],
              [drawnBbox[3], drawnBbox[2]]
            ]}
            pathOptions={{
              color: '#ffb700', weight: 2,
              fillColor: 'transparent', fillOpacity: 0, interactive: false
            }}
          />
        )}

        {clickedPoint && (
          <Marker position={[clickedPoint.lat, clickedPoint.lng]}>
            <Popup>
              <div style={{ fontSize: "12px", color: "#e2e8f0" }}>
                Lat: {clickedPoint.lat.toFixed(5)}<br />
                Lng: {clickedPoint.lng.toFixed(5)}
              </div>
            </Popup>
          </Marker>
        )}
      </MapContainer>

      {/* ── LEFT PANEL ── */}
      <div style={{
        position: "absolute", top: "20px", left: "20px", zIndex: 10, width: "320px", pointerEvents: "none"
      }}>
        <div style={{
          display: "flex", flexDirection: "column", gap: "14px", pointerEvents: "auto",
          maxHeight: "calc(100vh - 40px)", overflowY: "auto", paddingRight: "6px"
        }}>
          {/* Header */}
          <div className="glass-panel" style={{ padding: "18px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
              <span style={{ fontSize: "22px" }}>🛰️</span>
              <h2 style={{ margin: 0, fontSize: "18px", fontWeight: 700, color: "#fff", letterSpacing: "-0.5px" }}>
                GeoProspect AI
              </h2>
            </div>
            <p style={{ margin: 0, fontSize: "12px", color: "#94a3b8", lineHeight: "1.4" }}>
              Sentinel-2 & ASTER alteration probability maps
            </p>
          </div>

          {/* Controls panel */}
          <div className="glass-panel" style={{ padding: '18px' }}>

            {/* Draw AOI button */}
            <button
              onClick={() => { setDrawMode(!drawMode); setDrawRect(null); setDrawStart(null); }}
              style={{
                width: '100%', padding: '10px', marginBottom: '12px',
                background: drawMode ? 'rgba(255,183,0,0.2)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${drawMode ? '#ffb700' : 'rgba(255,255,255,0.1)'}`,
                borderRadius: '8px', color: drawMode ? '#ffb700' : '#fff',
                fontSize: '13px', fontWeight: 600, cursor: 'pointer',
              }}
            >
              {drawMode
                ? drawStart
                  ? '🖊 Now click the opposite corner...'
                  : '🖊 Click the first corner...'
                : '🖊 Draw AOI on Map'}
            </button>

            {/* Show drawn bbox with erase button */}
            {drawnBbox && (
              <div style={{
                background: 'rgba(0,0,0,0.2)',
                padding: '12px', borderRadius: '6px', marginBottom: '12px',
                border: '1px solid rgba(255,183,0,0.2)'
              }}>
                <div style={{
                  fontSize: '11px', color: '#94a3b8', fontFamily: 'monospace',
                  marginBottom: '10px', wordBreak: 'break-all'
                }}>
                  {drawnBbox.map(v => v.toFixed(3)).join(', ')}
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => { setDrawnBbox(null); setDrawMode(false); setDrawStart(null); setDrawRect(null); }}
                    style={{
                      flex: 1, padding: '8px', background: 'rgba(255,77,77,0.15)',
                      border: '1px solid rgba(255,77,77,0.3)', borderRadius: '6px',
                      color: '#ff4d4d', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                    onMouseOver={(e) => {
                      e.target.style.background = 'rgba(255,77,77,0.25)';
                      e.target.style.borderColor = 'rgba(255,77,77,0.5)';
                    }}
                    onMouseOut={(e) => {
                      e.target.style.background = 'rgba(255,77,77,0.15)';
                      e.target.style.borderColor = 'rgba(255,77,77,0.3)';
                    }}
                  >
                    ✕ Erase
                  </button>
                  <button
                    onClick={() => { 
                      setManualCoordMode(!manualCoordMode);
                      if (!manualCoordMode && drawnBbox) {
                        setManualCoords({
                          minLon: drawnBbox[0].toString(),
                          minLat: drawnBbox[1].toString(),
                          maxLon: drawnBbox[2].toString(),
                          maxLat: drawnBbox[3].toString()
                        });
                      }
                    }}
                    style={{
                      flex: 1, padding: '8px', background: manualCoordMode ? 'rgba(255,183,0,0.2)' : 'rgba(255,255,255,0.05)',
                      border: manualCoordMode ? '1px solid rgba(255,183,0,0.3)' : '1px solid rgba(255,255,255,0.1)', borderRadius: '6px',
                      color: manualCoordMode ? '#ffb700' : '#cbd5e1', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    ✎ Edit
                  </button>
                </div>
              </div>
            )}

            {/* Manual coordinate input mode */}
            {manualCoordMode && (
              <div style={{
                background: 'rgba(255,183,0,0.05)',
                padding: '12px', borderRadius: '6px', marginBottom: '12px',
                border: '1px solid rgba(255,183,0,0.2)'
              }}>
                <label style={{ fontSize: '11px', color: '#64748b', display: 'block', marginBottom: '8px', textTransform: 'uppercase', fontWeight: 600 }}>
                  Enter Coordinates (Lon, Lat)
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
                  <div>
                    <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '3px' }}>Min Lon</label>
                    <input
                      type="number"
                      step="0.001"
                      value={manualCoords.minLon}
                      onChange={(e) => setManualCoords({ ...manualCoords, minLon: e.target.value })}
                      placeholder="e.g. 16.611"
                      style={{
                        width: '100%', padding: '6px', background: 'rgba(0,0,0,0.2)',
                        border: '1px solid rgba(255,183,0,0.2)', borderRadius: '4px',
                        color: '#fff', fontSize: '11px', fontFamily: 'monospace'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '3px' }}>Min Lat</label>
                    <input
                      type="number"
                      step="0.001"
                      value={manualCoords.minLat}
                      onChange={(e) => setManualCoords({ ...manualCoords, minLat: e.target.value })}
                      placeholder="e.g. 3.777"
                      style={{
                        width: '100%', padding: '6px', background: 'rgba(0,0,0,0.2)',
                        border: '1px solid rgba(255,183,0,0.2)', borderRadius: '4px',
                        color: '#fff', fontSize: '11px', fontFamily: 'monospace'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '3px' }}>Max Lon</label>
                    <input
                      type="number"
                      step="0.001"
                      value={manualCoords.maxLon}
                      onChange={(e) => setManualCoords({ ...manualCoords, maxLon: e.target.value })}
                      placeholder="e.g. 17.666"
                      style={{
                        width: '100%', padding: '6px', background: 'rgba(0,0,0,0.2)',
                        border: '1px solid rgba(255,183,0,0.2)', borderRadius: '4px',
                        color: '#fff', fontSize: '11px', fontFamily: 'monospace'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '10px', color: '#94a3b8', display: 'block', marginBottom: '3px' }}>Max Lat</label>
                    <input
                      type="number"
                      step="0.001"
                      value={manualCoords.maxLat}
                      onChange={(e) => setManualCoords({ ...manualCoords, maxLat: e.target.value })}
                      placeholder="e.g. 8.755"
                      style={{
                        width: '100%', padding: '6px', background: 'rgba(0,0,0,0.2)',
                        border: '1px solid rgba(255,183,0,0.2)', borderRadius: '4px',
                        color: '#fff', fontSize: '11px', fontFamily: 'monospace'
                      }}
                    />
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={handleManualCoordSubmit}
                    style={{
                      flex: 1, padding: '8px', background: 'rgba(255,183,0,0.2)',
                      border: '1px solid rgba(255,183,0,0.3)', borderRadius: '6px',
                      color: '#ffb700', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                    onMouseOver={(e) => {
                      e.target.style.background = 'rgba(255,183,0,0.3)';
                    }}
                    onMouseOut={(e) => {
                      e.target.style.background = 'rgba(255,183,0,0.2)';
                    }}
                  >
                    ✓ Set AOI
                  </button>
                  <button
                    onClick={() => { setManualCoordMode(false); }}
                    style={{
                      flex: 1, padding: '8px', background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px',
                      color: '#cbd5e1', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Year range selectors */}
            {drawnBbox && (
              <>
                <YearRangeSelector onSubmit={handleSubmitJob} disabled={jobStatus === 'running'} />
              </>
            )}

            {/* Job status */}
            {jobStatus && (
              <div style={{
                marginTop: '12px', padding: '10px', borderRadius: '8px',
                background: jobStatus === 'done'
                  ? 'rgba(0,250,154,0.1)'
                  : jobStatus === 'error'
                  ? 'rgba(255,77,77,0.1)'
                  : 'rgba(255,183,0,0.1)',
                border: `1px solid ${jobStatus === 'done' ? '#00fa9a' : jobStatus === 'error' ? '#ff4d4d' : '#ffb700'}`,
                fontSize: '12px',
                color: jobStatus === 'done' ? '#00fa9a' : jobStatus === 'error' ? '#ff4d4d' : '#ffb700',
              }}>
                {jobStatus === 'running' && <span>⟳ </span>}
                {jobStatus === 'done'    && <span>✓ </span>}
                {jobStatus === 'error'   && <span>✗ </span>}
                {jobMessage}
              </div>
            )}

            {jobStatus === 'done' && currentJobId && (
                <button
                    onClick={() => window.open(`/api/job/${currentJobId}/report`, '_blank')}
                    style={{
                        width: '100%', padding: '10px', marginTop: '10px',
                        background: 'rgba(56,189,248,0.15)', border: '1px solid #38bdf8',
                        borderRadius: '8px', color: '#38bdf8', fontSize: '13px',
                        fontWeight: 600, cursor: 'pointer'
                    }}
                >
                    📄 Download Detailed Report (PDF)
                </button>
            )}
          </div>

          {/* Dynamic mineral legend — shown after job completes */}
          {discoveredMinerals.length > 0 && (
            <div className="glass-panel" style={{ padding: '14px 18px' }}>
              <span style={{
                display: 'block', fontSize: '11px', textTransform: 'uppercase',
                letterSpacing: '1px', color: '#64748b', fontWeight: 600, marginBottom: '8px'
              }}>
                Discovered Minerals
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
                {discoveredMinerals.map(mineral => (
                  <div key={mineral} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{
                      width: '12px', height: '12px', borderRadius: '3px', flexShrink: 0,
                      background: getMineralColor(mineral)
                    }} />
                    <span style={{ textTransform: 'capitalize' }}>{mineral}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
        </div>
      </div>

      {/* ── RIGHT PANEL: Identify ── */}
      <div style={{
        position: "absolute", top: panelPosition.top, left: panelPosition.left, zIndex: 10, width: "350px", pointerEvents: "none"
      }} ref={panelRef}>
        <div style={{
          display: "flex", flexDirection: "column", gap: "14px", pointerEvents: "auto",
          maxHeight: "calc(100vh - 40px)", overflowY: "auto", paddingLeft: "6px"
        }}>
          <div className="glass-panel" style={{ padding: "20px" }}>
            <div onMouseDown={handlePanelMouseDown} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "10px", cursor: "grab", userSelect: "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "16px" }}>🔍</span>
                <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "#fff" }}>Identify Minerals</h3>
              </div>
              {isIdentifying && (
                <span className="pulse-target" style={{ fontSize: "10px", background: "rgba(255,183,0,0.15)", color: "#ffb700", padding: "2px 6px", borderRadius: "4px", fontWeight: 600 }}>
                  Querying...
                </span>
              )}
            </div>
            {!clickedPoint ? (
              <div style={{ textAlign: "center", padding: "24px 10px", color: "#64748b" }}>
                <div style={{ fontSize: "36px", marginBottom: "10px", opacity: 0.5 }}>🎯</div>
                <p style={{ margin: 0, fontSize: "13px", fontWeight: 500 }}>
                  Click on the map to inspect mineral probabilities.
                </p>
              </div>
            ) : (
              <div>
                {/* Coordinates */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(0,0,0,0.2)", padding: "8px 12px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.03)", fontFamily: "'JetBrains Mono', monospace", fontSize: "11px", color: "#cbd5e1", marginBottom: "16px" }}>
                  <div>
                    <span style={{ color: "#64748b" }}>LAT:</span> {clickedPoint.lat.toFixed(5)}<br />
                    <span style={{ color: "#64748b" }}>LNG:</span> {clickedPoint.lng.toFixed(5)}
                  </div>
                  <button
                    onClick={() => navigator.clipboard.writeText(`${clickedPoint.lat.toFixed(5)}, ${clickedPoint.lng.toFixed(5)}`)}
                    style={{ background: "rgba(255,255,255,0.05)", border: "none", borderRadius: "4px", color: "#fff", padding: "4px 8px", cursor: "pointer", fontSize: "10px" }}
                  >
                    Copy
                  </button>
                </div>
                {/* Nearest sample distance */}
                {nearestDistance != null && (
                  <div style={{ fontSize: '11px', color: nearestDistance > 5 ? '#ff4d4d' : '#64748b', marginBottom: '8px' }}>
                    {nearestDistance > 5
                      ? `⚠ Nearest sampled point: ${nearestDistance} km away — low confidence`
                      : `Nearest sampled point: ${nearestDistance} km away`}
                  </div>
                )}
                
                {/* Water Info */}
                {waterInfo && (
                  <div style={{ fontSize: '11px', color: waterInfo.present ? '#38bdf8' : '#64748b', marginBottom: '12px', fontWeight: waterInfo.present ? 600 : 400 }}>
                    {waterInfo.present ? `💧 Water Detected (NDWI: ${waterInfo.ndwi})` : `No Surface Water (NDWI: ${waterInfo.ndwi})`}
                  </div>
                )}

                {/* Dominant mineral */}
                {dominant && (
                  <div style={{
                    background: "linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.6))",
                    borderLeft: `4px solid ${dominant.color}`,
                    padding: "10px 14px", borderRadius: "0 8px 8px 0", marginBottom: "18px"
                  }}>
                    <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.5px", color: "#64748b", fontWeight: 600, marginBottom: "3px" }}>
                      Primary Signature
                    </div>
                    <div style={{ fontSize: "15px", fontWeight: 700, color: "#fff" }}>{dominant.name}</div>
                    {dominant.label && (
                      <div style={{ fontSize: "12px", color: dominant.color, fontWeight: 600, marginTop: "3px" }}>
                        Confidence: {(dominant.val * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                )}
                {/* Probability bars */}
                <div>
                  <label style={{ display: "block", fontSize: "10px", textTransform: "uppercase", letterSpacing: "1px", color: "#64748b", fontWeight: 600, marginBottom: "12px" }}>
                    Mineral Probabilities Breakdown
                  </label>
                  {terrainInfo && !terrainInfo.mineralized ? (
                    <div style={{
                      padding: '16px', background: 'rgba(255,255,255,0.03)',
                      borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '28px', marginBottom: '8px' }}>
                        {terrainInfo.type === 'water' ? '💧' : terrainInfo.type === 'vegetation' ? '🌿' : '🏜️'}
                      </div>
                      <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: 500, marginBottom: '6px' }}>
                        No mineral alteration detected
                      </div>
                      <div style={{ fontSize: '11px', color: '#475569' }}>
                        {terrainInfo.type === 'water'
                          ? `This pixel is a water body (NDWI: ${waterInfo?.ndwi})`
                          : terrainInfo.type === 'vegetation'
                            ? `Dense vegetation cover blocks mineral detection (NDVI: ${terrainInfo.ndvi})`
                            : `Barren soil with no alteration signatures detected`}
                      </div>
                    </div>
                  ) : pixelValues && pixelValues.length > 0 ? (
                    <>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                        {pixelValues.map((m) => {
                          const val = m.probability / 100;
                          const color = getMineralColor(m.mineral);
                          return (
                            <div key={m.mineral}>
                              <div style={{
                                display: 'flex', justifyContent: 'space-between',
                                alignItems: 'center', fontSize: '12px', marginBottom: '4px'
                              }}>
                                <span style={{ fontWeight: 500, color: '#fff', textTransform: 'capitalize' }}>
                                  {m.mineral}
                                </span>
                                <span style={{ fontWeight: 600, color }}>{m.probability}%</span>
                              </div>
                              <div style={{
                                height: '7px', background: 'rgba(255,255,255,0.05)',
                                borderRadius: '4px', overflow: 'hidden'
                              }}>
                                <div style={{
                                  height: '100%', width: `${m.probability}%`,
                                  background: color, borderRadius: '4px'
                                }} />
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {dominant && dominant.label && (
                        <div style={{
                          marginTop: '18px', padding: '12px 14px',
                          background: 'rgba(255,183,0,0.05)',
                          border: '1px solid rgba(255,183,0,0.15)',
                          borderRadius: '8px'
                        }}>
                          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px', color: '#64748b', fontWeight: 600, marginBottom: '8px' }}>
                            Associated Resource
                          </div>
                          {dominant.metals && dominant.metals.length > 0 && (
                            <div style={{ fontSize: '12px', color: '#fff', marginBottom: '4px' }}>
                              <span style={{ color: '#64748b' }}>Metals: </span>
                              <span style={{ color: '#ffb700', fontWeight: 600 }}>
                                {dominant.metals.join(', ')}
                              </span>
                            </div>
                          )}
                          {dominant.uses && dominant.uses.length > 0 && (
                            <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.5' }}>
                              {dominant.uses.join(', ')}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{
                      textAlign: 'center', padding: '14px', color: '#64748b',
                      fontSize: '12px', background: 'rgba(0,0,0,0.1)', borderRadius: '8px'
                    }}>
                      {!currentJobId
                        ? 'Run an analysis first to see mineral data.'
                        : pixelValues && pixelValues.length === 0
                          ? 'No pixel data at this location (outside AOI).'
                          : 'Click on the map to identify minerals.'}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}

export default MapViewer;