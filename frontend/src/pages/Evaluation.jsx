import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'
import { fetchModelMetrics } from '../api/client'

const FOLD_METRICS = ['accuracy', 'f1', 'macro_f1', 'roc_auc']
const LABELS = { accuracy: 'Accuracy', f1: 'F1', macro_f1: 'Macro F1', roc_auc: 'ROC-AUC' }

function pct(v) { return v != null ? (v * 100).toFixed(2) : '—' }

export default function Evaluation() {
  const [metrics, setMetrics] = useState(null)
  const [activeMetric, setActiveMetric] = useState('f1')

  useEffect(() => { fetchModelMetrics().then(setMetrics) }, [])

  const lrFolds = metrics?.logistic_regression?.fold_results ?? []
  const knnFolds = metrics?.knn?.fold_results ?? []

  const chartData = lrFolds.map((f, i) => ({
    fold: `Fold ${f.fold}`,
    LR:  +(f[activeMetric] * 100).toFixed(2),
    KNN: knnFolds[i] ? +(knnFolds[i][activeMetric === 'macro_f1' ? 'f1' : activeMetric] * 100).toFixed(2) : null,
  }))

  const lr  = metrics?.logistic_regression
  const knn = metrics?.knn

  const tableRows = [
    { metric: 'Accuracy',       lr: lr?.accuracy,     knn: knn?.accuracy },
    { metric: 'Precision',      lr: lr?.precision,    knn: knn?.precision },
    { metric: 'Recall',         lr: lr?.recall,       knn: knn?.recall },
    { metric: 'F1 Score',       lr: lr?.f1,           knn: knn?.f1 },
    { metric: 'Macro F1',       lr: lr?.macro_f1,     knn: knn?.macro_f1 },
    { metric: 'ROC-AUC',        lr: lr?.roc_auc,      knn: knn?.roc_auc },
    { metric: 'Log Loss',       lr: lr?.log_loss,     knn: knn?.log_loss },
    { metric: 'Top-1 Accuracy', lr: lr?.top1_accuracy, knn: knn?.top1_accuracy },
    { metric: 'Train Time (s)', lr: lr?.training_time_s, knn: knn?.training_time_s },
  ]

  return (
    <div className="space-y-5">
      <h2 className="text-lg font-bold text-white">Model Evaluation</h2>

      {/* Metric selector */}
      <div className="flex gap-2 flex-wrap">
        {FOLD_METRICS.map(m => (
          <button
            key={m}
            onClick={() => setActiveMetric(m)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              activeMetric === m
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {LABELS[m]}
          </button>
        ))}
      </div>

      {/* Per-fold bar chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          {LABELS[activeMetric]} per CV Fold
        </h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="fold" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }}
                tickFormatter={v => v + '%'} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                formatter={v => v != null ? v.toFixed(2) + '%' : '—'}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="LR"  fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="KNN" fill="#a78bfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary table */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Mean CV Results (5-fold)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800 text-left">
                <th className="py-2 font-medium">Metric</th>
                <th className="py-2 font-medium text-right text-blue-400">Logistic Regression</th>
                <th className="py-2 font-medium text-right text-purple-400">KNN</th>
                <th className="py-2 font-medium text-right text-gray-500">HMM + Viterbi</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map(({ metric, lr: lv, knn: kv }) => (
                <tr key={metric} className="border-b border-gray-800/50">
                  <td className="py-2 text-gray-300">{metric}</td>
                  <td className="py-2 text-right font-mono text-blue-300">
                    {metric.includes('Time') ? (lv ? lv.toFixed(1) + 's' : '—') : pct(lv) + (lv != null ? '%' : '')}
                  </td>
                  <td className="py-2 text-right font-mono text-purple-300">
                    {knn?.status === 'training_in_progress'
                      ? <span className="text-yellow-400 text-xs">running…</span>
                      : (metric.includes('Time') ? (kv ? kv.toFixed(1) + 's' : '—') : pct(kv) + (kv != null ? '%' : '—'))}
                  </td>
                  <td className="py-2 text-right text-gray-600 text-xs">pending</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-gray-600">
        * KNN training in progress — results will update when complete.
        HMM + Viterbi pending team implementation. All LR values are from actual training runs.
      </p>
    </div>
  )
}
