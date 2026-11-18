import React from 'react'
import clsx from 'clsx'

export default function TripSelector({ trips, selectedId, onSelect }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Student-Test Trips</h2>
      <div className="space-y-1">
        {trips.length === 0 && (
          <p className="text-xs text-gray-500 animate-pulse">Loading trips…</p>
        )}
        {trips.map(t => (
          <button
            key={t.id}
            onClick={() => onSelect(t.id)}
            className={clsx(
              'w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors',
              selectedId === t.id
                ? 'bg-blue-600 text-white'
                : 'hover:bg-gray-800 text-gray-300'
            )}
          >
            <div className="font-mono font-semibold">{t.id}</div>
            <div className="text-xs opacity-70 mt-0.5">
              {t.observations.toLocaleString()} obs · {t.distance_km.toFixed(1)} km
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
