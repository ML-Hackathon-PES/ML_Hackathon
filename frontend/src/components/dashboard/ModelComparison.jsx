import React from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'

const METRICS = ['accuracy', 'f1', 'macro_f1', 'roc_auc', 'top1_accuracy']
const LABELS  = { accuracy: 'Accuracy', f1: 'F1', macro_f1: 'Macro F1', roc_auc: 'ROC-AUC', top1_accuracy: 'Top-1' }

function fmt(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—' }

export default function ModelComparison({ metrics }) {
  const lr  = metrics?.logistic_regression
  const knn = metrics?.knn
  const hmm = metrics?.hmm_viterbi

  const radarData = METRICS.map(m => ({
    metric: LABELS[m],
    LR:  lr?.[m]  != null ? +(lr[m]  * 100).toFixed(2) : null,
    KNN: knn?.[m] != null ? +(knn[m] * 100).toFixed(2) : null,
  }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h2 className="text-sm font-semibold text-gray-300 mb-4">Model Comparison</h2>

      {/* Radar chart */}
      <div className="h-52 mb-5">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              formatter={v => v != null ? v.toFixed(1) + '%' : '—'}
            />
            <Radar name="LR" dataKey="LR" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
            <Radar name="KNN" dataKey="KNN" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.15} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Comparison table */}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-500 border-b border-gray-800">
            <th className="text-left py-1 font-medium">Metric</th>
            <th className="text-right py-1 font-medium text-blue-400">LR</th>
            <th className="text-right py-1 font-medium text-purple-400">KNN</th>
            <th className="text-right py-1 font-medium text-gray-500">HMM+V</th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map(m => (
            <tr key={m} className="border-b border-gray-800/50">
              <td className="py-1.5 text-gray-400">{LABELS[m]}</td>
              <td className="py-1.5 text-right text-blue-300 font-mono">{fmt(lr?.[m])}</td>
              <td className="py-1.5 text-right text-purple-300 font-mono">
                {knn?.status === 'training_in_progress'
                  ? <span className="text-yellow-500">running…</span>
                  : fmt(knn?.[m])}
              </td>
              <td className="py-1.5 text-right text-gray-600 font-mono">
                {hmm?.status === 'not_implemented' ? 'pending' : fmt(hmm?.[m])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
