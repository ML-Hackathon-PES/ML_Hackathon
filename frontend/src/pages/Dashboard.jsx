import React, { useEffect, useState } from 'react'
import MetricsCard from '../components/dashboard/MetricsCard'
import ModelComparison from '../components/dashboard/ModelComparison'
import ProcessingStatus from '../components/dashboard/ProcessingStatus'
import { fetchModelMetrics, fetchProcessingStatus } from '../api/client'

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [status,  setStatus]  = useState(null)

  useEffect(() => {
    fetchModelMetrics().then(setMetrics)
    fetchProcessingStatus().then(setStatus)
  }, [])

  const lr = metrics?.logistic_regression

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-bold text-white">PathSnap Overview</h2>
        <span className="text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded-full">
          GPS Map Matching · ML Hackathon
        </span>
      </div>

      {/* KV cards — LR baseline (actual results) */}
      <div>
        <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide">
          Logistic Regression — Baseline CV Results (actual)
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricsCard title="Accuracy"   value={lr ? (lr.accuracy * 100).toFixed(1) + '%'    : null} color="blue"   badge="LR" />
          <MetricsCard title="Macro F1"   value={lr ? (lr.macro_f1 * 100).toFixed(1) + '%'    : null} color="green"  badge="LR" />
          <MetricsCard title="ROC-AUC"    value={lr ? (lr.roc_auc  * 100).toFixed(1) + '%'    : null} color="purple" badge="LR" />
          <MetricsCard title="Top-1 Acc"  value={lr ? (lr.top1_accuracy * 100).toFixed(1) + '%' : null} color="green" badge="LR"
            sub="Correct road arc selected as top candidate" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <ModelComparison metrics={metrics} />
        </div>
        <ProcessingStatus status={status} />
      </div>

      {/* Dataset summary */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Dataset Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          {[
            ['Train Trips',          '80'],
            ['Student-Test Trips',   '10'],
            ['Candidate Pairs',      '1,010,103'],
            ['Positive Rate',        '21.4%'],
            ['GPS Observations',     '227,577'],
            ['Candidate Miss Rate',  '0.60%'],
            ['Unlabeled Rows',       '59,219'],
            ['CV Folds',             '5 (trip-grouped)'],
          ].map(([label, val]) => (
            <div key={label}>
              <div className="text-gray-500 text-xs">{label}</div>
              <div className="text-white font-semibold mt-0.5">{val}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
