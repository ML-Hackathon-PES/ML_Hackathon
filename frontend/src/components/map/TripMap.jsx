import React, { useState, useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap } from 'react-leaflet'
import LayerControls from './LayerControls'

const LAYER_DEFAULTS = {
  gps_dots: { label: 'GPS Observations', enabled: true,  color: '#ef4444' },
  gps_path: { label: 'GPS Trajectory',   enabled: true,  color: '#f97316' },
  lr_route: { label: 'LR Matched Route', enabled: true,  color: '#3b82f6' },
  hmm_route:{ label: 'HMM + Viterbi',   enabled: false, color: '#a78bfa' },
}

// Recenter map when trip changes
function Recenter({ center }) {
  const map = useMap()
  useMemo(() => { if (center) map.setView(center, 14) }, [center])
  return null
}

export default function TripMap({ matchData }) {
  const [layers, setLayers] = useState(LAYER_DEFAULTS)
  const toggle = key =>
    setLayers(prev => ({ ...prev, [key]: { ...prev[key], enabled: !prev[key].enabled } }))

  const gpsPoints = matchData?.gps_points  ?? []
  const lrRoute   = matchData?.lr_route    ?? []

  const center = useMemo(() => {
    if (!gpsPoints.length) return [20, 0]
    const avgLat = gpsPoints.reduce((s, p) => s + p.lat, 0) / gpsPoints.length
    const avgLon = gpsPoints.reduce((s, p) => s + p.lon, 0) / gpsPoints.length
    return [avgLat, avgLon]
  }, [matchData?.trip_id])

  const gpsPath = gpsPoints.map(p => [p.lat, p.lon])
  const lrPath  = lrRoute.map(p  => [p.lat, p.lon])

  const defaultCenter = [20, 0]

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-gray-800">
      <MapContainer
        center={defaultCenter}
        zoom={5}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {gpsPoints.length > 0 && <Recenter center={center} />}

        {/* GPS trajectory line */}
        {layers.gps_path.enabled && gpsPath.length > 1 && (
          <Polyline positions={gpsPath} color="#f97316" weight={2} opacity={0.65} />
        )}

        {/* GPS observation dots */}
        {layers.gps_dots.enabled && gpsPoints.map((p, i) => (
          <CircleMarker
            key={`gps-${i}`}
            center={[p.lat, p.lon]}
            radius={3}
            pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.85, weight: 1 }}
          >
            <Tooltip>
              <div className="text-xs">
                <div>Obs {p.observation_index}</div>
                <div>t = {p.timestamp_s.toFixed(0)}s</div>
                <div>{p.speed_mps.toFixed(1)} m/s</div>
              </div>
            </Tooltip>
          </CircleMarker>
        ))}

        {/* LR matched route */}
        {layers.lr_route.enabled && lrPath.length > 1 && (
          <Polyline positions={lrPath} color="#3b82f6" weight={4} opacity={0.9} />
        )}
        {layers.lr_route.enabled && lrRoute.map((p, i) => (
          <CircleMarker
            key={`lr-${i}`}
            center={[p.lat, p.lon]}
            radius={2.5}
            pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 1, weight: 1 }}
          >
            <Tooltip>
              <div className="text-xs">
                <div>Arc {p.arc_id}</div>
                <div>prob {(p.predicted_prob * 100).toFixed(1)}%</div>
                {p.true_label !== '' && <div>label {p.true_label}</div>}
              </div>
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>

      <LayerControls layers={layers} onChange={toggle} />

      {!matchData && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded-xl">
          <p className="text-gray-400 text-sm">Select a trip to view the map</p>
        </div>
      )}
    </div>
  )
}
