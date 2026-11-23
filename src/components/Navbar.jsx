import React, { useState } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { 
  Sparkles, 
  Menu, 
  X, 
  ArrowUpRight, 
  RefreshCw, 
  AlertCircle,
  Layers,
  BarChart3,
  BookOpen,
  Home
} from 'lucide-react';
import { useBackend } from '../context/BackendContext';

export default function Navbar() {
  const location = useLocation();
  const { isOnline, isChecking, healthData, refreshStatus } = useBackend();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { name: 'Home', path: '/', icon: Home },
    { name: 'Meme Analyzer', path: '/analyzer', icon: Sparkles },
    { name: 'Model Architecture', path: '/architecture', icon: Layers },
    { name: 'Model Comparison', path: '/comparison', icon: BarChart3 },
    { name: 'About Project', path: '/about', icon: BookOpen },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#183832] border-b border-[#2C6156] text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          
          {/* Left: Brand Logo & Title & Subtitle */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-[#C7F59D] text-[#172923] flex items-center justify-center font-bold shadow-sm group-hover:scale-105 transition-transform">
              <Sparkles className="w-5 h-5 text-[#183832]" />
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight text-white block leading-none">
                IndicSent<span className="text-[#C7F59D]">-M</span>
              </span>
              <span className="text-[11px] text-[#A3B8B0] font-medium tracking-wide">
                Hinglish Meme AI
              </span>
            </div>
          </Link>

          {/* Center: Desktop Navigation Links (Pill Style from APKDefender Inspiration) */}
          <nav className="hidden lg:flex items-center bg-[#214B43]/90 border border-[#2C6156] rounded-full px-2 py-1.5 shadow-inner">
            {navLinks.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-full text-xs font-semibold tracking-wide transition-all ${
                    isActive
                      ? 'bg-[#C7F59D] text-[#172923] shadow-sm'
                      : 'text-[#D0DDD7] hover:text-white hover:bg-white/10'
                  }`}
                >
                  {item.name}
                </NavLink>
              );
            })}
          </nav>

          {/* Right: Small API Status Badge + Launch Analyzer Button */}
          <div className="hidden sm:flex items-center gap-3">
            {/* Small API Online / Offline Badge */}
            {isOnline ? (
              <div 
                title="FastAPI server is running and ready for predictions"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#214B43] border border-[#2C6156] text-xs font-semibold text-[#C7F59D]"
              >
                <span className="w-2 h-2 rounded-full bg-[#C7F59D] animate-pulse" />
                <span>API Online</span>
                {healthData?.gpu_available && (
                  <span className="text-[10px] bg-[#183832] px-1.5 py-0.5 rounded-full text-white font-mono">
                    CUDA
                  </span>
                )}
              </div>
            ) : isChecking ? (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-950/60 border border-amber-800 text-xs font-semibold text-amber-300">
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span>Checking...</span>
              </div>
            ) : (
              <button
                onClick={refreshStatus}
                title="Backend is currently offline. Click to retry connecting."
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-rose-950/80 border border-rose-800 text-xs font-semibold text-rose-300 hover:bg-rose-900 transition-colors"
              >
                <AlertCircle className="w-3 h-3" />
                <span>API Offline (Retry)</span>
              </button>
            )}

            {/* Launch Analyzer Button */}
            <Link
              to="/analyzer"
              className="inline-flex items-center gap-1.5 bg-[#C7F59D] hover:bg-[#B5EA87] text-[#172923] text-xs font-extrabold rounded-full px-5 py-2.5 transition-all shadow-sm hover:shadow-md hover:scale-[1.02]"
            >
              <span>Launch Analyzer</span>
              <ArrowUpRight className="w-4 h-4 text-[#172923]" />
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden flex items-center gap-2">
            <Link
              to="/analyzer"
              className="inline-flex sm:hidden items-center gap-1 bg-[#C7F59D] text-[#172923] text-xs font-bold rounded-full px-3.5 py-1.5"
            >
              <span>Analyze</span>
            </Link>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-[#D0DDD7] hover:text-white hover:bg-[#214B43] transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Drawer Navigation */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-[#183832] border-b border-[#2C6156] px-4 pt-3 pb-6 space-y-2">
          <div className="flex items-center justify-between pb-3 border-b border-[#2C6156] text-xs">
            <span className="text-[#A3B8B0]">Backend Status:</span>
            {isOnline ? (
              <span className="inline-flex items-center gap-1 text-[#C7F59D] font-bold">
                <span className="w-2 h-2 rounded-full bg-[#C7F59D]" />
                API Online ({healthData?.gpu_available ? 'CUDA' : 'CPU'})
              </span>
            ) : (
              <button onClick={refreshStatus} className="text-rose-300 font-bold flex items-center gap-1">
                <span>API Offline (Retry)</span>
              </button>
            )}
          </div>

          <div className="space-y-1 pt-2">
            {navLinks.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center justify-between px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                    isActive
                      ? 'bg-[#C7F59D] text-[#172923]'
                      : 'text-[#D0DDD7] hover:bg-[#214B43] hover:text-white'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </div>
                  <ArrowUpRight className="w-4 h-4 opacity-60" />
                </NavLink>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
}
