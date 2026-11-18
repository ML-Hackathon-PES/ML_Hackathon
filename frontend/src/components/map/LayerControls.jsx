import React from 'react'
import clsx from 'clsx'

export default function LayerControls({ layers, onChange }) {
  return (
    <div className="absolute top-3 right-3 z-[1000] bg-gray-900/95 border border-gray-700 rounded-lg p-3 shadow-lg min-w-[160px]">
      <p className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wide">Layers</p>
      <div className="space-y-2">
        {Object.entries(layers).map(([key, { label, enabled, color }]) => (
          <label key={key} className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enabled}
              onChange={() => onChange(key)}
              className="rounded"
            />
            <span className="flex items-center gap-1.5 text-xs text-gray-300">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: color }}
              />
              {label}
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}
