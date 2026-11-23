import React, { useState } from 'react';
import { 
  BarChart3, 
  FileText, 
  Image as ImageIcon, 
  GitMerge, 
  CheckCircle2,
  Trophy,
  Zap,
  Search,
  Filter,
  ArrowUpRight,
  Sparkles
} from 'lucide-react';
import { fullLeaderboardData } from '../data/mockData';

export default function ModelComparison() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedModality, setSelectedModality] = useState('All');

  // Filtered leaderboard
  const filteredModels = fullLeaderboardData.filter((item) => {
    const matchesSearch = item.model.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.modality.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesModality = selectedModality === 'All' || 
                            (selectedModality === 'Multimodal' && item.modality === 'Multimodal') ||
                            (selectedModality === 'Text' && item.modality.includes('Text')) ||
                            (selectedModality === 'Vision' && item.modality.includes('Vision'));
    return matchesSearch && matchesModality;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12 pb-20">
      
      {/* Header Banner */}
      <div className="bg-[#214B43] rounded-3xl p-6 sm:p-8 text-white relative overflow-hidden shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1.5 max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#C7F59D]/15 text-[#C7F59D] text-xs font-bold">
              <Trophy className="w-3.5 h-3.5 text-[#C7F59D]" />
              <span>Memotion 3.0 Benchmark</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Compare AI Models
            </h1>
            <p className="text-xs sm:text-sm text-[#D0DDD7]">
              We tested 14 AI models using 1,400 Hinglish memes.
            </p>
          </div>

          <span className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-bold text-[#172923] bg-[#C7F59D] self-start sm:self-center shadow">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>14 Models Tested</span>
          </span>
        </div>
      </div>

      {/* Triad of Top Model Highlights */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Highlight 1: Best Text + Image Model */}
        <div className="bg-[#28594F] rounded-3xl p-7 text-white space-y-4 shadow-sm relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#172923] bg-[#C7F59D] px-3 py-1 rounded-full">
                Best Text + Image Model
              </span>
              <GitMerge className="w-5 h-5 text-[#C7F59D]" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-white">
                MuRIL + ResNet-50 (Gated Fusion)
              </h3>
              <p className="text-xs text-[#D0DDD7] mt-1 leading-relaxed">
                Combines meme text and image information to predict sentiment.
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-[#377366] flex items-center justify-between text-xs">
            <span>Accuracy: <strong className="text-white font-mono">37.64%</strong></span>
            <span>Macro-F1: <strong className="text-[#C7F59D] font-mono font-extrabold text-sm">37.24%</strong></span>
          </div>
        </div>

        {/* Highlight 2: Best Overall Model */}
        <div className="bg-white rounded-3xl p-7 border border-[#E3E8DE] space-y-4 shadow-sm mindful-card-hover flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full">
                Best Overall Model
              </span>
              <FileText className="w-5 h-5 text-[#28594F]" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-[#172923]">
                TF-IDF + Logistic Regression
              </h3>
              <p className="text-xs text-[#68776F] mt-1 leading-relaxed">
                Uses words from meme text to predict sentiment. It also gives predictions very quickly (0.48 ms).
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-[#EAEFE6] flex items-center justify-between text-xs text-[#172923]">
            <span>Accuracy: <strong className="font-mono">43.79%</strong></span>
            <span>Macro-F1: <strong className="text-[#28594F] font-mono font-extrabold text-sm">43.14%</strong></span>
          </div>
        </div>

        {/* Highlight 3: Best Image Model */}
        <div className="bg-white rounded-3xl p-7 border border-[#E3E8DE] space-y-4 shadow-sm mindful-card-hover flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-[#28594F] bg-[#F7F8F5] px-3 py-1 rounded-full border border-[#E3E8DE]">
                Best Image Model
              </span>
              <ImageIcon className="w-5 h-5 text-[#28594F]" />
            </div>
            <div>
              <h3 className="text-lg font-extrabold text-[#172923]">
                ResNet-50
              </h3>
              <p className="text-xs text-[#68776F] mt-1 leading-relaxed">
                Analyzes meme images to find useful visual details.
              </p>
            </div>
          </div>

          <div className="pt-4 border-t border-[#EAEFE6] flex items-center justify-between text-xs text-[#172923]">
            <span>Accuracy: <strong className="font-mono">40.43%</strong></span>
            <span>Macro-F1: <strong className="text-[#28594F] font-mono font-extrabold text-sm">37.04%</strong></span>
          </div>
        </div>

      </section>

      {/* Complete 14-Model Leaderboard Table */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-9 space-y-6 shadow-sm">
        
        {/* Controls: Search and Filter Pills */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#EAEFE6] pb-5">
          <div className="space-y-1">
            <h3 className="text-lg font-extrabold text-[#172923] flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-[#28594F]" />
              <span>All 14 Tested Models</span>
            </h3>
            <p className="text-xs text-[#68776F]">
              Showing {filteredModels.length} of 14 models tested on 1,400 Hinglish memes
            </p>
          </div>

          {/* Search & Modality Filters */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search Box */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[#8D9D95]" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search models..."
                className="pl-8 pr-3 py-1.5 rounded-full border border-[#E3E8DE] bg-[#F7F8F5] text-xs text-[#172923] outline-none focus:border-[#28594F] w-40 sm:w-48"
              />
            </div>

            {/* Modality Filter Pills */}
            <div className="flex items-center gap-1 bg-[#F7F8F5] p-1 rounded-full border border-[#E3E8DE] text-xs">
              {['All', 'Multimodal', 'Text', 'Vision'].map((mod) => (
                <button
                  key={mod}
                  onClick={() => setSelectedModality(mod)}
                  className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                    selectedModality === mod
                      ? 'bg-[#28594F] text-[#C7F59D]'
                      : 'text-[#68776F] hover:text-[#172923]'
                  }`}
                >
                  {mod}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* The Leaderboard Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-[#EAEFE6] text-[#68776F] font-bold text-xs bg-[#F7F8F5]">
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Model Name</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Macro-F1</th>
                <th className="py-3 px-4">Weighted-F1</th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EAEFE6] text-[#172923]">
              {filteredModels.map((item, idx) => {
                const isChampion = item.model.includes('Gated Fusion');
                const isTopText = item.model.includes('Logistic Regression');
                const isTopVision = item.model === 'ResNet-50';

                return (
                  <tr 
                    key={item.model}
                    className={`transition-colors ${
                      isChampion 
                        ? 'bg-[#E7F6D9]/50 font-semibold' 
                        : isTopText
                        ? 'bg-[#F7F8F5]'
                        : 'hover:bg-[#F7F8F5]/60'
                    }`}
                  >
                    <td className="py-3.5 px-4 font-mono font-bold text-[#68776F]">
                      #{idx + 1}
                    </td>

                    <td className="py-3.5 px-4 font-bold text-[#172923] flex items-center gap-2">
                      <span>{item.model}</span>
                      {isChampion && (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#28594F] text-[#C7F59D]">
                          Best Text + Image
                        </span>
                      )}
                      {isTopText && (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#E7F6D9] text-[#28594F]">
                          Best Overall
                        </span>
                      )}
                    </td>

                    <td className="py-3.5 px-4 text-[#68776F]">
                      <span className="px-2.5 py-0.5 rounded-full bg-[#F7F8F5] border border-[#E3E8DE] text-xs">
                        {item.modality}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 font-mono font-semibold">
                      {item.accuracy}%
                    </td>

                    <td className="py-3.5 px-4 font-mono font-extrabold text-[#28594F]">
                      {item.macroF1}%
                    </td>

                    <td className="py-3.5 px-4 font-mono text-[#68776F]">
                      {item.weightedF1}%
                    </td>

                    <td className="py-3.5 px-4">
                      <span className="text-[11px] font-medium text-[#28594F] bg-[#E7F6D9] px-2 py-0.5 rounded-full">
                        {item.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

      </section>

    </div>
  );
}
