import React, { useState } from 'react';
import { 
  Cpu, 
  Image as ImageIcon, 
  FileText, 
  GitMerge, 
  ArrowDown, 
  ArrowRight,
  Smile, 
  Frown, 
  Meh,
  Sparkles,
  Layers,
  Check
} from 'lucide-react';
import { architectureComponents } from '../data/mockData';

export default function ModelArchitecture() {
  const [selectedComponentIndex, setSelectedComponentIndex] = useState(0);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12 pb-20">
      
      {/* Header Banner */}
      <div className="bg-[#214B43] rounded-3xl p-6 sm:p-8 text-white relative overflow-hidden shadow-sm">
        <div className="space-y-1.5 max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#C7F59D]/15 text-[#C7F59D] text-xs font-bold">
            <Layers className="w-3.5 h-3.5 text-[#C7F59D]" />
            <span>Neural Network Design</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Model Architecture
          </h1>
          <p className="text-xs sm:text-sm text-[#D0DDD7]">
            How IndicSent-M combines visual features from ResNet-50 and text features from MuRIL using adaptive Gated Fusion.
          </p>
        </div>
      </div>

      {/* Visual Pipeline 1: Multimodal Gated Fusion (The Main System) */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-9 space-y-8 shadow-sm">
        <div className="border-b border-[#EAEFE6] pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full">
              Simple 4-Step Process
            </span>
            <h2 className="text-xl sm:text-2xl font-extrabold text-[#172923] mt-2 tracking-tight">
              How Our AI Works
            </h2>
            <p className="text-xs sm:text-sm text-[#68776F] mt-1 leading-relaxed">
              Our AI reads meme text and images, combines the information and predicts the sentiment.
            </p>
          </div>

          <span className="text-xs px-3 py-1 rounded-full bg-[#F7F8F5] text-[#28594F] font-bold border border-[#E3E8DE] self-start sm:self-center">
            Macro-F1: 37.24%
          </span>
        </div>

        {/* Visual Pipeline Nodes */}
        <div className="max-w-4xl mx-auto space-y-5">
          
          {/* Level 1: Dual Inputs */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-2 mindful-card-hover shadow-xs">
              <div className="w-10 h-10 rounded-xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center mx-auto">
                <ImageIcon className="w-5 h-5 text-[#28594F]" />
              </div>
              <div className="text-base font-bold text-[#172923]">1. Upload Meme Image</div>
              <p className="text-xs text-[#68776F] leading-relaxed">
                Add the meme image you want to analyze.
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-2 mindful-card-hover shadow-xs">
              <div className="w-10 h-10 rounded-xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center mx-auto">
                <FileText className="w-5 h-5 text-[#28594F]" />
              </div>
              <div className="text-base font-bold text-[#172923]">2. Enter Meme Text</div>
              <p className="text-xs text-[#68776F] leading-relaxed">
                Type or paste the words written in the meme.
              </p>
            </div>
          </div>

          {/* Directional Arrows */}
          <div className="flex justify-around text-[#28594F]">
            <ArrowDown className="w-5 h-5" />
            <ArrowDown className="w-5 h-5" />
          </div>

          {/* Level 2: Image and Text Analysis */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {/* Left Card: Image Analysis */}
            <div className="p-5 rounded-2xl bg-[#214B43] text-white text-center space-y-2 shadow-sm">
              <span className="inline-block text-[11px] font-bold text-[#172923] bg-[#C7F59D] px-3 py-0.5 rounded-full">
                Image Information
              </span>
              <div className="text-lg font-extrabold text-white">
                Image Analysis
              </div>
              <p className="text-xs sm:text-sm text-[#D0DDD7] leading-relaxed">
                ResNet-50 finds useful details and patterns in the meme image.
              </p>
              <div className="text-[11px] text-[#A3B8B0] font-mono pt-1">
                Model: ResNet-50
              </div>
            </div>

            {/* Right Card: Text Analysis */}
            <div className="p-5 rounded-2xl bg-[#214B43] text-white text-center space-y-2 shadow-sm">
              <span className="inline-block text-[11px] font-bold text-[#172923] bg-[#C7F59D] px-3 py-0.5 rounded-full">
                Text Information
              </span>
              <div className="text-lg font-extrabold text-white">
                Text Analysis
              </div>
              <p className="text-xs sm:text-sm text-[#D0DDD7] leading-relaxed">
                MuRIL reads and understands the Hinglish text in the meme.
              </p>
              <div className="text-[11px] text-[#A3B8B0] font-mono pt-1">
                Model: MuRIL Transformer
              </div>
            </div>
          </div>

          {/* Converging Arrows */}
          <div className="flex justify-center text-[#28594F]">
            <ArrowDown className="w-6 h-6" />
          </div>

          {/* Level 3: Combining Text and Image */}
          <div className="p-6 rounded-3xl bg-[#E7F6D9] border border-[#C7F59D] text-center space-y-3 shadow-sm">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-[#28594F] text-[#C7F59D] text-xs font-bold">
              <GitMerge className="w-3.5 h-3.5" />
              <span>Combining Text and Image</span>
            </div>
            <h4 className="text-lg font-extrabold text-[#172923]">
              Smart Information Mixing
            </h4>
            <p className="text-xs sm:text-sm text-[#28594F] max-w-xl mx-auto leading-relaxed">
              Our AI learns how much importance to give to the text and image. It then combines both to understand the meme better.
            </p>
            
            {/* Optional Collapsible Technical Details for Formula */}
            <details className="text-xs text-[#28594F] cursor-pointer max-w-md mx-auto pt-1">
              <summary className="font-semibold text-xs text-[#28594F]/80 hover:text-[#28594F] select-none">
                Show Technical Formula (Advanced)
              </summary>
              <div className="mt-2 font-mono text-[11px] text-[#28594F] font-bold bg-white/80 py-1.5 px-3 rounded-xl border border-[#C7F59D] inline-block">
                h_fused = (g &times; h_text) + ((1 - g) &times; h_image)
              </div>
            </details>
          </div>

          {/* Arrow */}
          <div className="flex justify-center text-[#28594F]">
            <ArrowDown className="w-6 h-6" />
          </div>

          {/* Level 4: Final Sentiment Result */}
          <div className="p-6 rounded-3xl bg-white border border-[#E3E8DE] text-center space-y-3 shadow-sm">
            <div className="text-base font-extrabold text-[#172923]">
              Final Sentiment Result
            </div>
            <p className="text-xs sm:text-sm text-[#68776F]">
              After analyzing the meme, our AI predicts its sentiment.
            </p>
            <div className="flex justify-center gap-3 pt-1 text-xs">
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold">
                <Smile className="w-4 h-4 text-emerald-600" /> Positive
              </span>
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200 font-bold">
                <Meh className="w-4 h-4 text-amber-600" /> Neutral
              </span>
              <span className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-rose-50 text-rose-800 border border-rose-200 font-bold">
                <Frown className="w-4 h-4 text-rose-600" /> Negative
              </span>
            </div>
          </div>

        </div>
      </section>

      {/* Section 2: Component Deep-Dive Cards */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-9 space-y-6 shadow-sm">
        <div>
          <h3 className="text-xl font-extrabold text-[#172923]">
            Key Architecture Modules
          </h3>
          <p className="text-xs sm:text-sm text-[#68776F] mt-0.5">
            Click any component to inspect its design parameters and specifications.
          </p>
        </div>

        {/* Tab Buttons */}
        <div className="flex flex-wrap gap-2">
          {architectureComponents.map((comp, idx) => (
            <button
              key={comp.name}
              onClick={() => setSelectedComponentIndex(idx)}
              className={`px-4 py-2 rounded-full text-xs font-bold transition-all ${
                selectedComponentIndex === idx
                  ? 'bg-[#28594F] text-[#C7F59D] shadow-sm'
                  : 'bg-[#F7F8F5] text-[#68776F] hover:bg-[#EAEFE6] hover:text-[#172923]'
              }`}
            >
              {comp.name.split(' ')[0]}
            </button>
          ))}
        </div>

        {/* Selected Component Card */}
        {(() => {
          const comp = architectureComponents[selectedComponentIndex];
          return (
            <div className="p-6 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <h4 className="text-base font-extrabold text-[#172923]">
                  {comp.name}
                </h4>
                <span className="text-xs font-mono text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full font-bold self-start sm:self-auto">
                  {comp.badge}
                </span>
              </div>
              <p className="text-xs sm:text-sm text-[#68776F] leading-relaxed">
                {comp.role}
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs">
                {comp.specs.map((spec) => (
                  <div key={spec.label} className="p-3 bg-white rounded-xl border border-[#E3E8DE]">
                    <div className="text-[10px] text-[#8D9D95] font-semibold uppercase">{spec.label}</div>
                    <div className="text-xs font-bold text-[#172923] mt-0.5">{spec.value}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}
      </section>

      {/* Section 3: Layer & Shape Summary Table */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-9 space-y-4 shadow-sm">
        <h3 className="text-lg font-extrabold text-[#172923]">
          Tensor Dimensions Across Pipeline
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-[#EAEFE6] text-[#68776F] font-bold text-xs bg-[#F7F8F5]">
                <th className="py-3 px-4">Stage</th>
                <th className="py-3 px-4">Module Name</th>
                <th className="py-3 px-4">Input Tensor</th>
                <th className="py-3 px-4">Output Representation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EAEFE6] text-[#172923]">
              <tr>
                <td className="py-3.5 px-4 font-bold text-[#28594F]">1. Vision Stream</td>
                <td className="py-3.5 px-4 font-medium">ResNet-50</td>
                <td className="py-3.5 px-4 font-mono text-xs">224 × 224 × 3 RGB</td>
                <td className="py-3.5 px-4 font-mono text-xs font-bold">2,048-dim vector</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold text-[#28594F]">2. Text Stream</td>
                <td className="py-3.5 px-4 font-medium">MuRIL Transformer</td>
                <td className="py-3.5 px-4 font-mono text-xs">128 tokens</td>
                <td className="py-3.5 px-4 font-mono text-xs font-bold">768-dim vector</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold text-[#28594F]">3. Linear Projection</td>
                <td className="py-3.5 px-4 font-medium">Linear + LayerNorm + ReLU</td>
                <td className="py-3.5 px-4 font-mono text-xs">2048 (img) / 768 (txt)</td>
                <td className="py-3.5 px-4 font-mono text-xs font-bold">256-dim each</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold text-[#28594F]">4. Gated Fusion</td>
                <td className="py-3.5 px-4 font-medium">Sigmoid Attention Gate</td>
                <td className="py-3.5 px-4 font-mono text-xs">256 + 256</td>
                <td className="py-3.5 px-4 font-mono text-xs font-bold text-[#28594F]">256-dim fused</td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-bold text-[#28594F]">5. Classifier Head</td>
                <td className="py-3.5 px-4 font-medium">MLP + Softmax</td>
                <td className="py-3.5 px-4 font-mono text-xs">256-dim fused</td>
                <td className="py-3.5 px-4 font-mono text-xs font-bold text-[#28594F]">3 probabilities</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

    </div>
  );
}
