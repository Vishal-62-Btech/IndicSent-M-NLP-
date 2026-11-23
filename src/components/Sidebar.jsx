import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Sparkles, 
  Cpu, 
  BarChart3, 
  BookOpen, 
  Smile,
  X
} from 'lucide-react';

const navItems = [
  {
    name: 'Dashboard',
    path: '/',
    icon: LayoutDashboard,
  },
  {
    name: 'Meme Analyzer',
    path: '/analyzer',
    icon: Sparkles,
  },
  {
    name: 'Model Architecture',
    path: '/architecture',
    icon: Cpu,
  },
  {
    name: 'Model Comparison',
    path: '/comparison',
    icon: BarChart3,
  },
  {
    name: 'About Project',
    path: '/about',
    icon: BookOpen,
  }
];

export default function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          onClick={onClose}
          className="fixed inset-0 bg-slate-800/40 z-40 lg:hidden"
        />
      )}

      {/* Sidebar Container */}
      <aside 
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-white border-r border-slate-200 flex flex-col transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 px-5 border-b border-slate-100 flex items-center justify-between">
          <NavLink to="/" onClick={onClose} className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center text-white shadow-sm">
              <Smile className="w-4 h-4" />
            </div>
            <div>
              <div className="font-bold text-base text-slate-800 leading-tight">
                IndicSent-M
              </div>
              <p className="text-[11px] text-slate-500 leading-tight">
                Multimodal Meme Sentiment
              </p>
            </div>
          </NavLink>

          {/* Close button for mobile */}
          <button 
            onClick={onClose} 
            className="lg:hidden p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onClose}
                className={({ isActive }) => `
                  flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors
                  ${isActive 
                    ? 'bg-purple-50 text-purple-700 font-semibold' 
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }
                `}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Live Backend Status in Sidebar Footer */}
        <div className="p-3 border-t border-slate-100 m-2 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>FastAPI Backend:</span>
            <span className="font-mono text-[10px] text-purple-700 font-semibold">:8000</span>
          </div>
          <div className="text-[11px] flex items-center justify-center gap-1.5 py-1 px-2 rounded-md bg-slate-50 border border-slate-200 text-slate-600">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="font-medium">Production Mode</span>
          </div>
        </div>
      </aside>
    </>
  );
}
