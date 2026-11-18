import React from 'react'
import { CheckCircle, Clock, Loader, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

const STEPS = [
  { key: 'candidate_generation', label: 'Candidate Generation'   },
  { key: 'feature_engineering',  label: 'Feature Engineering'    },
  { key: 'route_alignment',      label: 'Route Alignment & Labels'},
  { key: 'lr_training',          label: 'LR Training'            },
  { key: 'knn_training',         label: 'KNN Training'           },
  { key: 'hmm_viterbi',          label: 'HMM + Viterbi'          },
]

const statusConfig = {
  complete: { icon: CheckCircle, color: 'text-green-400',  label: 'Complete' },
  running:  { icon: Loader,      color: 'text-yellow-400', label: 'Running'  },
  pending:  { icon: Clock,       color: 'text-gray-500',   label: 'Pending'  },
  error:    { icon: AlertCircle, color: 'text-red-400',    label: 'Error'    },
}

export default function ProcessingStatus({ status }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h2 className="text-sm font-semibold text-gray-300 mb-4">Pipeline Status</h2>
      <div className="space-y-3">
        {STEPS.map(({ key, label }) => {
          const s = status?.[key] ?? 'pending'
          const cfg = statusConfig[s] ?? statusConfig.pending
          const Icon = cfg.icon
          return (
            <div key={key} className="flex items-center gap-3">
              <Icon
                size={16}
                className={clsx(cfg.color, s === 'running' && 'animate-spin')}
              />
              <span className="text-sm text-gray-300 flex-1">{label}</span>
              <span className={clsx('text-xs font-medium', cfg.color)}>{cfg.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
