import React, { useEffect, useState } from 'react'
import TripSelector from '../components/trips/TripSelector'
import TripDetail   from '../components/trips/TripDetail'
import TripMap      from '../components/map/TripMap'
import { fetchTrips, fetchTripMatch } from '../api/client'

export default function TripViewer() {
  const [trips,      setTrips]      = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [matchData,  setMatchData]  = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)

  useEffect(() => {
    fetchTrips().then(data => {
      setTrips(data)
      if (data.length) setSelectedId(data[0].id)
    })
  }, [])

  useEffect(() => {
    if (!selectedId) return
    setLoading(true)
    setError(null)
    fetchTripMatch(selectedId)
      .then(setMatchData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedId])

  return (
    <div className="flex gap-4 h-[calc(100vh-7rem)]">
      {/* Left panel */}
      <div className="w-60 shrink-0 flex flex-col gap-3 overflow-y-auto">
        <TripSelector trips={trips} selectedId={selectedId} onSelect={setSelectedId} />
        <TripDetail matchData={matchData} />
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/80 rounded-xl">
            <span className="text-sm text-gray-300 animate-pulse">Loading LR predictions…</span>
          </div>
        )}
        {error && (
          <div className="absolute top-3 left-3 z-10 bg-red-900/80 text-red-200 text-xs px-3 py-2 rounded-lg">
            {error}
          </div>
        )}
        <TripMap matchData={matchData} />
      </div>
    </div>
  )
}
