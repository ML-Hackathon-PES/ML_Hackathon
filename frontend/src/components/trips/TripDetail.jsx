import React from 'react'

function fmt(v, decimals = 2) {
  return v != null ? Number(v).toFixed(decimals) : '—'
}

export default function TripDetail({ matchData }) {
  if (!matchData) return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-xs text-gray-500">
      Select a trip to see details.
    </div>
  )

  const missRate = (matchData.candidate_miss_rate * 100).toFixed(2)
  const matchPct = matchData.observations_total > 0
    ? ((matchData.observations_matched / matchData.observations_total) * 100).toFixed(1)
    : '—'

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-sm">
      <h2 className="font-semibold text-gray-300 mb-3">Trip Info</h2>
      <dl className="space-y-2">
        {[
          ['Trip ID',         matchData.trip_id],
          ['Partition',       matchData.partition],
          ['GPS Observations',matchData.observations_total],
          ['LR Matched',      `${matchData.observations_matched} (${matchPct}%)`],
          ['Candidate Miss',  `${missRate}%`],
        ].map(([label, val]) => (
          <div key={label} className="flex justify-between gap-2">
            <dt className="text-gray-500 shrink-0">{label}</dt>
            <dd className="text-gray-200 font-mono text-right">{val ?? '—'}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 pt-3 border-t border-gray-800">
        <div className="flex items-center gap-2 text-xs mb-1">
          <span className="w-3 h-3 rounded-sm bg-red-500 inline-block" />
          <span className="text-gray-400">GPS observations</span>
        </div>
        <div className="flex items-center gap-2 text-xs mb-1">
          <span className="w-3 h-3 rounded-sm bg-orange-400 inline-block" />
          <span className="text-gray-400">GPS trajectory</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="w-3 h-3 rounded-sm bg-blue-500 inline-block" />
          <span className="text-gray-400">LR matched route</span>
        </div>
      </div>
    </div>
  )
}
