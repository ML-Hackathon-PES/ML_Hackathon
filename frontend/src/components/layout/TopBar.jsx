import React from 'react'
import { useLocation } from 'react-router-dom'

const TITLES = {
  '/dashboard':  'Dashboard',
  '/trips':      'Trip Viewer',
  '/evaluation': 'Model Evaluation',
}

export default function TopBar() {
  const { pathname } = useLocation()
  const title = TITLES[pathname] ?? 'PathSnap'

  return (
    <header className="h-14 shrink-0 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6">
      <h1 className="text-base font-semibold text-white">{title}</h1>
      <div className="flex items-center gap-3">
        <span className="text-xs bg-yellow-500/20 text-yellow-300 px-2 py-1 rounded-full font-medium">
          KNN training in progress
        </span>
        <span className="text-xs bg-green-500/20 text-green-300 px-2 py-1 rounded-full font-medium">
          LR ✓ complete
        </span>
        <span className="text-xs text-gray-500">Backend: not connected</span>
      </div>
    </header>
  )
}
