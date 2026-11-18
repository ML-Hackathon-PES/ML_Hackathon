import React from 'react'
import clsx from 'clsx'

export default function MetricsCard({ title, value, sub, color = 'blue', badge }) {
  const colors = {
    blue:   'border-blue-500/30 bg-blue-500/5',
    green:  'border-green-500/30 bg-green-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    purple: 'border-purple-500/30 bg-purple-500/5',
    red:    'border-red-500/30 bg-red-500/5',
  }
  const textColors = {
    blue: 'text-blue-300', green: 'text-green-300',
    yellow: 'text-yellow-300', purple: 'text-purple-300', red: 'text-red-300',
  }

  return (
    <div className={clsx('border rounded-xl p-4', colors[color])}>
      <div className="flex items-start justify-between mb-1">
        <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">{title}</span>
        {badge && (
          <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full">{badge}</span>
        )}
      </div>
      <div className={clsx('text-2xl font-bold', textColors[color])}>
        {value ?? <span className="text-gray-600 text-lg">—</span>}
      </div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  )
}
