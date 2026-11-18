import React from 'react'
import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Map, BarChart2, Activity } from 'lucide-react'
import clsx from 'clsx'

const NAV = [
  { to: '/dashboard',  label: 'Dashboard',   icon: LayoutDashboard },
  { to: '/trips',      label: 'Trip Viewer',  icon: Map             },
  { to: '/evaluation', label: 'Evaluation',   icon: BarChart2       },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 py-5 border-b border-gray-800">
        <Activity className="text-blue-400" size={22} />
        <span className="font-bold text-lg tracking-tight text-white">PathSnap</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-4 space-y-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )
            }
          >
            <Icon size={17} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-500">
        PathSnap ML Hackathon
      </div>
    </aside>
  )
}
