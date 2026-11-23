import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowUpRight, Cpu, Layers, Heart } from 'lucide-react';
import { useBackend } from '../context/BackendContext';

export default function Footer() {
  const { isOnline, healthData } = useBackend();

  return (
    <footer className="bg-[#183832] text-[#D0DDD7] border-t border-[#2C6156]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        
        {/* Main Footer Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 lg:gap-12 pb-12 border-b border-[#2C6156]">
          
          {/* Brand Info */}
          <div className="md:col-span-5 space-y-4">
            <Link to="/" className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-[#C7F59D] text-[#172923] flex items-center justify-center font-bold">
                <Sparkles className="w-5 h-5 text-[#214B43]" />
              </div>
              <span className="text-xl font-bold tracking-tight text-white">
                IndicSent<span className="text-[#C7F59D]">-M</span>
              </span>
            </Link>

            <p className="text-sm text-[#A3B8B0] leading-relaxed max-w-sm">
              Multimodal deep learning framework evaluating emotional sentiment across Hinglish social media memes using text and visual features.
            </p>

            <div className="flex items-center gap-2 text-xs text-[#C7F59D]">
              <span className="w-2 h-2 rounded-full bg-[#C7F59D]" />
              <span>Memotion 3.0 Dataset &bull; 14 Models Evaluated</span>
            </div>
          </div>

          {/* Quick Navigation Links */}
          <div className="md:col-span-3 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-white">
              Navigation
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link to="/" className="hover:text-[#C7F59D] transition-colors flex items-center gap-1">
                  <span>Dashboard</span>
                </Link>
              </li>
              <li>
                <Link to="/analyzer" className="hover:text-[#C7F59D] transition-colors flex items-center gap-1">
                  <span>Meme Analyzer</span>
                </Link>
              </li>
              <li>
                <Link to="/architecture" className="hover:text-[#C7F59D] transition-colors flex items-center gap-1">
                  <span>Model Architecture</span>
                </Link>
              </li>
              <li>
                <Link to="/comparison" className="hover:text-[#C7F59D] transition-colors flex items-center gap-1">
                  <span>Model Comparison</span>
                </Link>
              </li>
              <li>
                <Link to="/about" className="hover:text-[#C7F59D] transition-colors flex items-center gap-1">
                  <span>About Project</span>
                </Link>
              </li>
            </ul>
          </div>

          {/* Technology & Hardware Info */}
          <div className="md:col-span-4 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-white">
              System Architecture
            </h4>
            <div className="space-y-2.5 text-xs text-[#A3B8B0]">
              <div className="p-3 rounded-xl bg-[#214B43] border border-[#2C6156]">
                <div className="flex items-center justify-between text-white font-semibold mb-1">
                  <span>Backend Engine</span>
                  <span className="text-[#C7F59D] font-mono text-[11px]">{isOnline ? 'Online' : 'Offline'}</span>
                </div>
                <p className="text-[11px]">
                  FastAPI on port 8000 &bull; PyTorch 2.x &bull; Hugging Face Transformers
                </p>
              </div>

              <div className="p-3 rounded-xl bg-[#214B43] border border-[#2C6156]">
                <div className="flex items-center justify-between text-white font-semibold mb-1">
                  <span>Hardware Acceleration</span>
                  <span className="text-[#C7F59D] font-mono text-[11px]">
                    {healthData?.gpu_available ? 'CUDA 12' : 'CPU'}
                  </span>
                </div>
                <p className="text-[11px]">
                  {healthData?.gpu_name || 'NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)'}
                </p>
              </div>
            </div>
          </div>

        </div>

        {/* Bottom Sub-Footer */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#7A948A]">
          <p>
            &copy; {new Date().getFullYear()} IndicSent-M. Built for academic research and multimodal NLP demonstration.
          </p>

          <div className="flex items-center gap-3">
            <span className="px-2.5 py-1 rounded-full bg-[#214B43] text-[#C7F59D] border border-[#2C6156]">
              React 18 + Vite
            </span>
            <span className="px-2.5 py-1 rounded-full bg-[#214B43] text-[#C7F59D] border border-[#2C6156]">
              FastAPI
            </span>
            <span className="px-2.5 py-1 rounded-full bg-[#214B43] text-[#C7F59D] border border-[#2C6156]">
              PyTorch
            </span>
          </div>
        </div>

      </div>
    </footer>
  );
}
