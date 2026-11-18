/**
 * PLACEHOLDER DATA — clearly labelled mock data for development.
 * All values here are illustrative only and will be replaced by real
 * FastAPI responses when the backend is connected.
 */

// 10 student-test trip IDs from the actual dataset
export const PLACEHOLDER_TRIPS = [
  { id: '00000000', partition: 'student_test', observations: 2503, duration_s: 2502, distance_km: 20.1 },
  { id: '00000007', partition: 'student_test', observations: 1354, duration_s: 1353, distance_km: 14.2 },
  { id: '00000009', partition: 'student_test', observations: 1049, duration_s: 1048, distance_km: 9.8  },
  { id: '00000040', partition: 'student_test', observations: 2995, duration_s: 2994, distance_km: 28.4 },
  { id: '00000052', partition: 'student_test', observations: 262,  duration_s: 261,  distance_km: 5.6  },
  { id: '00000056', partition: 'student_test', observations: 3958, duration_s: 3957, distance_km: 71.7 },
  { id: '00000065', partition: 'student_test', observations: 768,  duration_s: 767,  distance_km: 11.3 },
  { id: '00000076', partition: 'student_test', observations: 1486, duration_s: 1485, distance_km: 19.6 },
  { id: '00000077', partition: 'student_test', observations: 3901, duration_s: 3900, distance_km: 55.2 },
  { id: '00000091', partition: 'student_test', observations: 874,  duration_s: 873,  distance_km: 12.9 },
]

// Generate a placeholder trip detail with fake GPS points around a central coord
export function PLACEHOLDER_TRIP_DETAIL(tripId) {
  const centers = {
    '00000000': [56.2515, 54.9635],
    '00000007': [55.7500, 37.6200],
    '00000009': [59.9300, 30.3200],
    '00000040': [55.0000, 82.9000],
    '00000052': [56.8500, 60.6100],
    '00000056': [43.1200, 131.9000],
    '00000065': [55.1600, 61.4000],
    '00000076': [50.4500, 30.5200],
    '00000077': [42.6500, -93.4200],
    '00000091': [51.7000, 36.1900],
  }
  const [lat, lon] = centers[tripId] || [55.75, 37.62]

  // Generate a simple curving path of 30 points
  const gps_points = Array.from({ length: 30 }, (_, i) => ({
    lat: lat + (i * 0.001) + (Math.sin(i * 0.5) * 0.0005),
    lon: lon + (i * 0.001) + (Math.cos(i * 0.5) * 0.0005),
    timestamp: i + 1,
    speed_mps: 8 + Math.random() * 4,
  }))

  // Reconstructed route (slightly smoother than raw GPS)
  const reconstructed_route = gps_points.map((p, i) => ({
    lat: p.lat + (Math.random() - 0.5) * 0.0002,
    lon: p.lon + (Math.random() - 0.5) * 0.0002,
    arc_id: 1000 + i,
  }))

  return {
    trip_id:            tripId,
    partition:          'student_test',
    gps_points,
    reconstructed_route,
    // [PLACEHOLDER] — real matched route from FastAPI not yet available
    matched_route_lr:   reconstructed_route,
    matched_route_hmm:  null,   // HMM+Viterbi not yet implemented
    candidate_miss_rate: 0.0,
    unlabeled_rate:      0.0,
  }
}

// Real LR metrics from actual training (from lr_baseline_cv_overall_metrics.json)
export const PLACEHOLDER_METRICS = {
  logistic_regression: {
    source: 'actual',   // ← real values from training
    cv_folds: 5,
    accuracy:    0.8446,
    precision:   0.5922,
    recall:      0.8836,
    f1:          0.7089,
    macro_f1:    0.8014,
    roc_auc:     0.9295,
    log_loss:    0.3453,
    top1_accuracy: 0.9882,
    training_time_s: 296.1,
    fold_results: [
      { fold: 1, accuracy: 0.8618, f1: 0.7324, macro_f1: 0.8196, roc_auc: 0.9419, top1: 0.9876 },
      { fold: 2, accuracy: 0.8079, f1: 0.6620, macro_f1: 0.7639, roc_auc: 0.9007, top1: 0.9936 },
      { fold: 3, accuracy: 0.8348, f1: 0.6878, macro_f1: 0.7877, roc_auc: 0.9134, top1: 0.9796 },
      { fold: 4, accuracy: 0.8668, f1: 0.7465, macro_f1: 0.8280, roc_auc: 0.9544, top1: 0.9954 },
      { fold: 5, accuracy: 0.8517, f1: 0.7158, macro_f1: 0.8077, roc_auc: 0.9372, top1: 0.9847 },
    ],
  },
  knn: {
    source: 'partial_placeholder',   // ← KNN training still running
    cv_folds: 5,
    accuracy:    null,
    precision:   null,
    recall:      null,
    f1:          null,
    macro_f1:    null,
    roc_auc:     null,
    log_loss:    null,
    top1_accuracy: null,
    training_time_s: null,
    fold_results: [
      { fold: 1, accuracy: null, f1: 0.7787, macro_f1: null, roc_auc: 0.9302, top1: 0.8910 },
    ],
    status: 'training_in_progress',
  },
  hmm_viterbi: {
    source: 'placeholder',   // ← not yet implemented
    status: 'not_implemented',
    accuracy: null, f1: null, macro_f1: null, top1_accuracy: null,
  },
}
