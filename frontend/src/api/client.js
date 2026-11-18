/**
 * PathSnap API client
 * Backend: FastAPI running at http://localhost:8000
 * Proxied via Vite to /api
 *
 * Set MOCK_MODE = true only if the backend is not running.
 */
import axios from 'axios'
import { PLACEHOLDER_TRIPS, PLACEHOLDER_METRICS, PLACEHOLDER_TRIP_DETAIL } from '../data/placeholders'

export const MOCK_MODE = false   // ← false = use real FastAPI backend

const http = axios.create({ baseURL: '/api', timeout: 30000 })

// ── Trips ──────────────────────────────────────────────────────────────────

export async function fetchTrips() {
  if (MOCK_MODE) return PLACEHOLDER_TRIPS
  const { data } = await http.get('/trips')
  return data
}

export async function fetchTripMatch(tripId) {
  if (MOCK_MODE) return PLACEHOLDER_TRIP_DETAIL(tripId)
  const { data } = await http.get(`/trips/${tripId}`)
  return data
}

// ── Metrics ────────────────────────────────────────────────────────────────

export async function fetchModelMetrics() {
  if (MOCK_MODE) return PLACEHOLDER_METRICS
  const { data } = await http.get('/metrics')
  return data
}

// ── Status ─────────────────────────────────────────────────────────────────

export async function fetchProcessingStatus() {
  if (MOCK_MODE) return {
    candidate_generation: 'complete', feature_engineering: 'complete',
    route_alignment: 'complete', lr_training: 'complete',
    knn_training: 'running', hmm_viterbi: 'pending',
  }
  const { data } = await http.get('/status')
  return data
}
