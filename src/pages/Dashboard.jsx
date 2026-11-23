import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  Sparkles, 
  ArrowRight, 
  ArrowUpRight,
  ArrowDown,
  Image as ImageIcon, 
  FileText, 
  Smile, 
  Frown, 
  Meh, 
  CheckCircle2, 
  GitMerge, 
  Cpu,
  Zap,
  BarChart3,
  Layers,
  Database,
  Code,
  AlertTriangle,
  Play,
  Info,
  X
} from 'lucide-react';
import { exampleMemes } from '../data/mockData';

export default function Dashboard() {
  const navigate = useNavigate();
  const [isHowItWorksOpen, setIsHowItWorksOpen] = useState(false);
  const [selectedMemeIdx, setSelectedMemeIdx] = useState(0);
  const activeMeme = exampleMemes[selectedMemeIdx] || exampleMemes[0];

  // Close modal on Escape key & manage body scroll lock
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setIsHowItWorksOpen(false);
      }
    };
    if (isHowItWorksOpen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isHowItWorksOpen]);

  return (
    <div className="space-y-16 sm:space-y-20 pb-20 overflow-x-hidden">
      
      {/* =========================================================================
          1. HERO SECTION (Full Viewport Coverage - Edge-to-Edge & Perfectly Centered)
          ========================================================================= */}
      <section className="bg-gradient-to-b from-[#183832] via-[#214B43] to-[#183832] text-white min-h-[calc(100vh-74px)] flex items-center px-4 sm:px-6 lg:px-8 pt-4 sm:pt-6 pb-6 sm:pb-8 relative overflow-hidden border-b border-[#2C6156]">
        
        {/* Subtle radial glows */}
        <div className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full bg-[#C7F59D]/10 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-32 -right-32 w-[500px] h-[500px] rounded-full bg-[#377366]/25 blur-3xl pointer-events-none" />

        <div className="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center relative z-10">
          
          {/* LEFT SIDE: Copy, Call to Action, and Trust Metrics */}
          <div className="lg:col-span-7 space-y-5 sm:space-y-6">
            
            {/* Small Tag Badge */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#214B43] border border-[#2C6156] text-[#C7F59D] text-xs font-bold tracking-wide shadow-sm">
              <span className="w-2 h-2 rounded-full bg-[#C7F59D] animate-pulse" />
              <span>AI MULTIMODAL SENTIMENT ANALYSIS</span>
            </div>

            {/* Giant Heading with Lime Highlight */}
            <h1 className="text-4xl sm:text-5xl lg:text-[52px] font-extrabold tracking-tight text-white leading-[1.12]">
              Understand the <br className="hidden sm:inline" />
              <span className="text-[#C7F59D]">Feeling Behind</span> <br className="hidden sm:inline" />
              Every Meme.
            </h1>

            {/* Clear Description */}
            <p className="text-base sm:text-lg text-[#D0DDD7] leading-relaxed max-w-xl font-normal">
              Memes use words, jokes and images to express emotions. IndicSent-M uses AI to analyze Hinglish meme text and images and predict whether the sentiment is Positive, Negative or Neutral.
            </p>

            {/* Prominent Action Buttons */}
            <div className="flex flex-wrap items-center gap-4 pt-1">
              <Link
                to="/analyzer"
                className="inline-flex items-center gap-2.5 px-8 py-3.5 rounded-full bg-[#C7F59D] hover:bg-[#B5EA87] text-[#172923] text-sm font-extrabold transition-all shadow-lg hover:shadow-xl hover:scale-[1.02]"
              >
                <span>Try Meme Analyzer</span>
                <ArrowRight className="w-4 h-4 text-[#172923]" />
              </Link>

              <button
                type="button"
                onClick={() => setIsHowItWorksOpen(true)}
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-[#28594F]/70 hover:bg-[#28594F] border border-[#377366] text-white text-sm font-bold transition-all backdrop-blur-sm hover:scale-[1.02]"
              >
                <Play className="w-3.5 h-3.5 fill-[#C7F59D] text-[#C7F59D]" />
                <span>How It Works</span>
              </button>
            </div>

            {/* Bottom Proof Metrics */}
            <div className="pt-5 border-t border-[#2C6156] flex flex-wrap items-center gap-6 sm:gap-10">
              {/* Stat 1 */}
              <div>
                <div className="text-2xl sm:text-3xl font-extrabold font-mono text-[#C7F59D] leading-none">
                  14
                </div>
                <div className="text-xs text-[#A3B8B0] font-semibold mt-1">
                  Models Evaluated
                </div>
              </div>

              {/* Stat 2 */}
              <div className="border-l border-[#2C6156] pl-6 sm:pl-10">
                <div className="text-2xl sm:text-3xl font-extrabold font-mono text-white leading-none">
                  Memotion 3.0
                </div>
                <div className="text-xs text-[#A3B8B0] font-semibold mt-1">
                  Benchmark (7,000 Memes)
                </div>
              </div>

              {/* Stat 3: Sentiment Classes */}
              <div className="border-l border-[#2C6156] pl-6 sm:pl-10">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" title="Positive" />
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400" title="Neutral" />
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-400" title="Negative" />
                </div>
                <div className="text-xs text-[#A3B8B0] font-semibold">
                  Hinglish Text + Image
                </div>
              </div>
            </div>

          </div>

          {/* RIGHT SIDE: Device / Scanner Mockup Card (Adjusted Upward to Guarantee Full Visibility) */}
          <div className="lg:col-span-5 relative flex justify-center lg:justify-end lg:-translate-y-5 xl:-translate-y-7">
            
            {/* Outer Device Frame Container */}
            <div className="bg-[#183832] rounded-[2rem] p-3.5 sm:p-4 border-2 border-[#2C6156] shadow-2xl w-full max-w-[430px] relative">
              
              {/* Inner Phone Screen Canvas */}
              <div className="bg-[#F7F8F5] rounded-2xl p-3.5 sm:p-4 text-[#172923] space-y-2.5 sm:space-y-3 relative overflow-hidden">
                
                {/* Device Status Bar */}
                <div className="flex items-center justify-between pb-2 border-b border-[#E3E8DE] text-xs">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-[#E7F6D9] text-[#28594F] font-bold text-[10px] sm:text-[11px]">
                    <Zap className="w-3 h-3 text-[#28594F]" />
                    <span>LIVE MULTIMODAL ANALYSIS</span>
                  </span>
                  <span className="text-[10px] sm:text-[11px] font-mono text-[#68776F] font-semibold">
                    v1.0
                  </span>
                </div>

                {/* Model Architecture Header */}
                <div className="flex items-center justify-between text-xs pt-0.5">
                  <span className="text-[#68776F] font-semibold text-[11px]">Architecture:</span>
                  <span className="font-extrabold text-[#28594F] bg-white px-2.5 py-0.5 rounded-full border border-[#E3E8DE] text-[11px] shadow-2xs">
                    MuRIL + ResNet-50 Gated Fusion
                  </span>
                </div>

                {/* Interactive Benchmark Sample Tabs */}
                <div className="flex items-center justify-between bg-white px-2.5 py-1 rounded-xl border border-[#E3E8DE] text-xs">
                  <span className="text-[10px] uppercase font-bold text-[#8D9D95] tracking-wider">
                    Sample:
                  </span>
                  <div className="flex items-center gap-1.5">
                    {exampleMemes.slice(0, 3).map((meme, idx) => (
                      <button
                        key={meme.id}
                        type="button"
                        onClick={() => setSelectedMemeIdx(idx)}
                        className={`px-2.5 py-0.5 rounded-lg text-xs font-bold transition-all ${
                          selectedMemeIdx === idx
                            ? 'bg-[#214B43] text-white shadow-xs'
                            : 'bg-[#F7F8F5] hover:bg-[#EAEFE6] text-[#68776F]'
                        }`}
                      >
                        {meme.suggestedSentiment === 'Negative' ? '😭 Neg' : meme.suggestedSentiment === 'Positive' ? '🎉 Pos' : '🥱 Neu'}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Meme Graphic Visual - Proportioned & fitted */}
                <div className="relative rounded-xl overflow-hidden bg-[#183832] border border-[#E3E8DE] h-36 sm:h-40 flex items-center justify-center p-2">
                  <img 
                    src={activeMeme.imageUrl} 
                    alt={activeMeme.title} 
                    className="w-full h-full object-contain rounded-lg"
                  />
                  
                  {/* Floating OCR Text Tag */}
                  <div className="absolute top-2.5 left-2.5 bg-[#183832]/90 backdrop-blur-sm text-white text-[9px] font-bold px-2 py-0.5 rounded-full border border-[#2C6156]">
                    Hinglish OCR Text
                  </div>
                </div>

                {/* Sample Caption Display */}
                <div className="bg-white p-2.5 rounded-xl border border-[#E3E8DE] space-y-0.5">
                  <span className="text-[9px] uppercase font-bold text-[#8D9D95] tracking-wider block">
                    Meme Punchline Caption
                  </span>
                  <p className="text-xs text-[#172923] font-medium leading-relaxed italic line-clamp-2">
                    "{activeMeme.text}"
                  </p>
                </div>

                {/* Sentiment Result Area */}
                <div className="bg-white p-3 rounded-xl border border-[#E3E8DE] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#68776F]">
                      Predicted Sentiment:
                    </span>
                    {activeMeme.suggestedSentiment === 'Negative' && (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-rose-50 text-rose-800 border border-rose-200">
                        <Frown className="w-3.5 h-3.5 text-rose-600" />
                        <span>Negative ({activeMeme.confidence}%)</span>
                      </span>
                    )}
                    {activeMeme.suggestedSentiment === 'Positive' && (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-emerald-50 text-emerald-800 border border-emerald-200">
                        <Smile className="w-3.5 h-3.5 text-emerald-600" />
                        <span>Positive ({activeMeme.confidence}%)</span>
                      </span>
                    )}
                    {activeMeme.suggestedSentiment === 'Neutral' && (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-amber-50 text-amber-800 border border-amber-200">
                        <Meh className="w-3.5 h-3.5 text-amber-600" />
                        <span>Neutral ({activeMeme.confidence}%)</span>
                      </span>
                    )}
                  </div>

                  {/* Confidence Distribution Bars */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-[#68776F] font-semibold">
                      <span>Negative: {activeMeme.scores.negative}%</span>
                      <span>Neutral: {activeMeme.scores.neutral}%</span>
                      <span>Positive: {activeMeme.scores.positive}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-[#F7F8F5] rounded-full overflow-hidden flex border border-[#E3E8DE]">
                      <div style={{ width: `${activeMeme.scores.negative}%` }} className="h-full bg-rose-500 rounded-full transition-all duration-300" />
                      <div style={{ width: `${activeMeme.scores.neutral}%` }} className="h-full bg-amber-400 rounded-full transition-all duration-300" />
                      <div style={{ width: `${activeMeme.scores.positive}%` }} className="h-full bg-[#C7F59D] rounded-full transition-all duration-300" />
                    </div>
                  </div>

                  {/* Gated Attention Weight Breakdown */}
                  <div className="pt-1.5 border-t border-[#F0F3EC] flex items-center justify-between text-[10px]">
                    <span className="text-[#8D9D95] font-semibold">Gated Attention:</span>
                    <span className="text-[#28594F] font-bold">
                      Text {activeMeme.modalityWeights?.textWeight || 65}% &bull; Image {activeMeme.modalityWeights?.imageWeight || 35}%
                    </span>
                  </div>
                </div>

                {/* Interactive Action Button to Try Live Analyzer */}
                <button
                  type="button"
                  onClick={() => navigate(`/analyzer?example=${activeMeme.id}`)}
                  className="w-full py-2.5 rounded-xl bg-[#214B43] hover:bg-[#28594F] text-white text-xs font-bold transition-colors flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <span>Test This Meme in Analyzer</span>
                  <ArrowRight className="w-3.5 h-3.5 text-[#C7F59D]" />
                </button>

              </div>

            </div>

          </div>

        </div>
      </section>

      {/* =========================================================================
          2. ANALYSIS MODES SECTION (Three Beautiful Dedicated Cards)
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#E7F6D9] text-[#28594F] text-xs font-bold mb-2">
              <span>Three Analysis Pipelines</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-[#172923] tracking-tight">
              Choose an Analysis Mode
            </h2>
            <p className="text-sm text-[#68776F] mt-1 max-w-xl">
              Each pipeline uses our top-performing machine learning model for maximum sentiment accuracy.
            </p>
          </div>

          <Link
            to="/comparison"
            className="text-xs font-bold text-[#28594F] hover:text-[#214B43] flex items-center gap-1 self-start sm:self-auto"
          >
            <span>View All 14 Benchmarks</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {/* The 3 Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Card 1: Multimodal Analysis */}
          <div 
            onClick={() => navigate('/analyzer?mode=multimodal')}
            className="bg-white rounded-3xl p-7 border border-[#E3E8DE] cursor-pointer mindful-card-hover flex flex-col justify-between space-y-6 shadow-sm group"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full">
                  Best Multimodal Pipeline
                </span>
                <div className="w-10 h-10 rounded-full bg-[#E7F6D9] text-[#28594F] group-hover:bg-[#C7F59D] group-hover:text-[#172923] flex items-center justify-center transition-colors">
                  <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-[#172923] group-hover:text-[#28594F] transition-colors">
                  Multimodal Analysis
                </h3>
                <p className="text-sm text-[#68776F] mt-2 leading-relaxed">
                  Analyze text and images together.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-[#EAEFE6] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Model Architecture:</span>
                <span className="font-bold text-[#172923]">MuRIL + ResNet-50 Gated Fusion</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Validation Macro-F1:</span>
                <span className="font-extrabold text-[#28594F] bg-[#E7F6D9] px-2.5 py-0.5 rounded-full font-mono">
                  37.24%
                </span>
              </div>
            </div>
          </div>

          {/* Card 2: Text Analysis */}
          <div 
            onClick={() => navigate('/analyzer?mode=text')}
            className="bg-white rounded-3xl p-7 border border-[#E3E8DE] cursor-pointer mindful-card-hover flex flex-col justify-between space-y-6 shadow-sm group"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#28594F] bg-[#F7F8F5] px-3 py-1 rounded-full border border-[#E3E8DE]">
                  Best Overall Baseline
                </span>
                <div className="w-10 h-10 rounded-full bg-[#F7F8F5] text-[#28594F] group-hover:bg-[#C7F59D] group-hover:text-[#172923] flex items-center justify-center transition-colors border border-[#E3E8DE]">
                  <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-[#172923] group-hover:text-[#28594F] transition-colors">
                  Text Analysis
                </h3>
                <p className="text-sm text-[#68776F] mt-2 leading-relaxed">
                  Analyze Hinglish meme text.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-[#EAEFE6] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Model Architecture:</span>
                <span className="font-bold text-[#172923]">TF-IDF + Logistic Regression</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Validation Macro-F1:</span>
                <span className="font-extrabold text-[#28594F] bg-[#E7F6D9] px-2.5 py-0.5 rounded-full font-mono">
                  43.14%
                </span>
              </div>
            </div>
          </div>

          {/* Card 3: Image Analysis */}
          <div 
            onClick={() => navigate('/analyzer?mode=image')}
            className="bg-white rounded-3xl p-7 border border-[#E3E8DE] cursor-pointer mindful-card-hover flex flex-col justify-between space-y-6 shadow-sm group"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#28594F] bg-[#F7F8F5] px-3 py-1 rounded-full border border-[#E3E8DE]">
                  Top Vision Model
                </span>
                <div className="w-10 h-10 rounded-full bg-[#F7F8F5] text-[#28594F] group-hover:bg-[#C7F59D] group-hover:text-[#172923] flex items-center justify-center transition-colors border border-[#E3E8DE]">
                  <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-[#172923] group-hover:text-[#28594F] transition-colors">
                  Image Analysis
                </h3>
                <p className="text-sm text-[#68776F] mt-2 leading-relaxed">
                  Analyze meme images.
                </p>
              </div>
            </div>

            <div className="pt-4 border-t border-[#EAEFE6] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Model Architecture:</span>
                <span className="font-bold text-[#172923]">ResNet-50</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#68776F]">Validation Macro-F1:</span>
                <span className="font-extrabold text-[#28594F] bg-[#E7F6D9] px-2.5 py-0.5 rounded-full font-mono">
                  37.04%
                </span>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* =========================================================================
          3. HOW IT WORKS (Three-Step Process with Visual Diagrams & Arrows)
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        
        <div className="text-center max-w-2xl mx-auto space-y-2">
          <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3.5 py-1 rounded-full uppercase tracking-wider">
            Simple 3-Step Process
          </span>
          <h2 className="text-3xl font-extrabold text-[#172923] tracking-tight">
            How IndicSent-M Works
          </h2>
          <p className="text-sm text-[#68776F]">
            From raw social media memes to calibrated sentiment scores in milliseconds.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
          
          {/* Step 1 */}
          <div className="bg-white rounded-3xl p-8 border border-[#E3E8DE] space-y-4 text-center mindful-card-hover relative shadow-sm">
            <div className="w-14 h-14 rounded-2xl bg-[#E7F6D9] text-[#28594F] font-extrabold text-xl flex items-center justify-center mx-auto">
              1
            </div>
            <h3 className="text-lg font-bold text-[#172923]">
              Upload a Meme or Enter Text
            </h3>
            <p className="text-sm text-[#68776F] leading-relaxed">
              Upload a meme image (JPG, PNG, WebP) or type a Hinglish text caption. You can also analyze both modalities together.
            </p>
          </div>

          {/* Step 2 */}
          <div className="bg-white rounded-3xl p-8 border border-[#E3E8DE] space-y-4 text-center mindful-card-hover relative shadow-sm">
            <div className="w-14 h-14 rounded-2xl bg-[#C7F59D] text-[#172923] font-extrabold text-xl flex items-center justify-center mx-auto">
              2
            </div>
            <h3 className="text-lg font-bold text-[#172923]">
              AI Analyzes the Selected Input
            </h3>
            <p className="text-sm text-[#68776F] leading-relaxed">
              Our neural network extracts visual patterns and text semantics, then balances them using adaptive gating.
            </p>
          </div>

          {/* Step 3 */}
          <div className="bg-white rounded-3xl p-8 border border-[#E3E8DE] space-y-4 text-center mindful-card-hover relative shadow-sm">
            <div className="w-14 h-14 rounded-2xl bg-[#E7F6D9] text-[#28594F] font-extrabold text-xl flex items-center justify-center mx-auto">
              3
            </div>
            <h3 className="text-lg font-bold text-[#172923]">
              Get the Sentiment Result
            </h3>
            <p className="text-sm text-[#68776F] leading-relaxed">
              Get an instant prediction (Positive, Negative, or Neutral) with complete confidence scores and processing speed.
            </p>
          </div>

        </div>
      </section>

      {/* =========================================================================
          4. MODEL PERFORMANCE SECTION (Clean Comparison of the Three Best Models)
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        
        <div className="bg-[#214B43] rounded-3xl p-8 sm:p-10 text-white space-y-6 shadow-md border border-[#2C6156]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs font-bold text-[#C7F59D] bg-[#183832] px-3.5 py-1 rounded-full uppercase tracking-wider border border-[#2C6156]">
                Benchmark Evaluation
              </span>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white mt-2 tracking-tight">
                Top Model Comparison
              </h2>
              <p className="text-xs sm:text-sm text-[#D0DDD7] mt-1">
                We tested 14 AI models using 1,400 Hinglish memes.
              </p>
            </div>

            <Link
              to="/comparison"
              className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-full bg-[#C7F59D] text-[#172923] text-xs font-extrabold hover:bg-[#B5EA87] transition-all self-start sm:self-center"
            >
              <span>Full 14-Model Leaderboard</span>
              <ArrowUpRight className="w-4 h-4" />
            </Link>
          </div>

          {/* Clean Performance Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead>
                <tr className="border-b border-[#2C6156] text-[#A3B8B0] font-semibold text-xs">
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Best Model</th>
                  <th className="py-3 px-4">Modality</th>
                  <th className="py-3 px-4">Accuracy</th>
                  <th className="py-3 px-4">Macro-F1</th>
                  <th className="py-3 px-4">Speed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2C6156] text-white">
                <tr className="bg-[#183832]/60">
                  <td className="py-3.5 px-4">
                    <span className="font-bold text-[#C7F59D] bg-[#214B43] px-2.5 py-0.5 rounded-full text-xs border border-[#2C6156]">
                      Best Overall Model
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-bold">TF-IDF + Logistic Regression</td>
                  <td className="py-3.5 px-4 text-[#D0DDD7]">Text Only</td>
                  <td className="py-3.5 px-4 font-mono font-bold">43.79%</td>
                  <td className="py-3.5 px-4 font-mono font-extrabold text-[#C7F59D]">43.14%</td>
                  <td className="py-3.5 px-4 font-mono text-xs text-[#A3B8B0]">0.48 ms</td>
                </tr>
                <tr>
                  <td className="py-3.5 px-4">
                    <span className="font-bold text-white bg-[#28594F] px-2.5 py-0.5 rounded-full text-xs">
                      Best Text + Image Model
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-bold">MuRIL + ResNet-50 (Gated Fusion)</td>
                  <td className="py-3.5 px-4 text-[#D0DDD7]">Text + Image</td>
                  <td className="py-3.5 px-4 font-mono font-bold">37.64%</td>
                  <td className="py-3.5 px-4 font-mono font-extrabold text-[#C7F59D]">37.24%</td>
                  <td className="py-3.5 px-4 font-mono text-xs text-[#A3B8B0]">25.51 ms</td>
                </tr>
                <tr>
                  <td className="py-3.5 px-4">
                    <span className="font-bold text-[#D0DDD7] bg-[#183832] px-2.5 py-0.5 rounded-full text-xs">
                      Best Image Model
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-bold">ResNet-50</td>
                  <td className="py-3.5 px-4 text-[#D0DDD7]">Image Only</td>
                  <td className="py-3.5 px-4 font-mono font-bold">40.43%</td>
                  <td className="py-3.5 px-4 font-mono font-bold text-white">37.04%</td>
                  <td className="py-3.5 px-4 font-mono text-xs text-[#A3B8B0]">14.16 ms</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

      </section>

      {/* =========================================================================
          5. PROJECT OVERVIEW & DATASET INTRODUCTION
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Project Overview */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-4 shadow-sm">
          <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-[#28594F]" />
          </div>
          <h3 className="text-xl font-extrabold text-[#172923]">
            What is IndicSent-M?
          </h3>
          <p className="text-sm text-[#68776F] leading-relaxed">
            Internet memes are a primary form of communication in India, blending Romanized Hindi (Hinglish) with pop culture graphics. Because memes frequently use sarcasm or opposite expressions (like a smiling face paired with an exam stress caption), reading text alone or images alone often fails. IndicSent-M joins text and image models to predict the true sentiment.
          </p>
        </div>

        {/* Memotion 3.0 Dataset */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-4 shadow-sm">
          <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
            <Database className="w-5 h-5 text-[#28594F]" />
          </div>
          <h3 className="text-xl font-extrabold text-[#172923]">
            Memotion 3.0 Dataset
          </h3>
          <p className="text-sm text-[#68776F] leading-relaxed">
            The system is trained and benchmarked on the official Memotion 3.0 dataset with 5,600 training memes and an unseen validation set of 1,400 memes. Every meme is labeled into three sentiment classes: Positive, Negative, or Neutral.
          </p>
        </div>

      </section>

      {/* =========================================================================
          6. MODEL ARCHITECTURE PREVIEW
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-8 sm:p-10 space-y-6 shadow-sm">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3.5 py-1 rounded-full uppercase tracking-wider">
                Architecture Overview
              </span>
              <h3 className="text-2xl font-extrabold text-[#172923] mt-2">
                Dual-Stream Gated Fusion
              </h3>
              <p className="text-sm text-[#68776F] mt-1">
                How text and visual representations merge inside the neural network.
              </p>
            </div>

            <Link
              to="/architecture"
              className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-full bg-[#214B43] hover:bg-[#28594F] text-white text-xs font-bold transition-all self-start sm:self-center"
            >
              <span>Explore Architecture</span>
              <ArrowRight className="w-4 h-4 text-[#C7F59D]" />
            </Link>
          </div>

          {/* Clean Flow Nodes */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-center text-xs">
            <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
              <span className="font-extrabold text-[#172923] block text-sm">1. Dual Input</span>
              <p className="text-[#68776F]">Hinglish text + 224x224 RGB image</p>
            </div>
            <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
              <span className="font-extrabold text-[#172923] block text-sm">2. Feature Encoders</span>
              <p className="text-[#68776F]">MuRIL (768-d) &amp; ResNet-50 (2048-d)</p>
            </div>
            <div className="p-4 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] space-y-1">
              <span className="font-extrabold text-[#28594F] block text-sm">3. Gated Attention</span>
              <p className="text-[#28594F] font-medium">Adaptive sigmoid weighting gate</p>
            </div>
            <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
              <span className="font-extrabold text-[#172923] block text-sm">4. Classification</span>
              <p className="text-[#68776F]">Positive, Negative, or Neutral</p>
            </div>
          </div>

        </div>
      </section>

      {/* =========================================================================
          7. TECHNOLOGY STACK RIBBON
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-3xl shadow-sm border border-[#E3E8DE] p-6 sm:p-8 space-y-4">
          <div className="text-center max-w-xl mx-auto">
            <h4 className="text-xs uppercase font-extrabold tracking-wider text-[#8D9D95]">
              Core Technologies &amp; Libraries
            </h4>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-center items-center pt-2">
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">Deep Learning</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">PyTorch 2.x</span>
            </div>
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">Transformers</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">MuRIL &amp; XLM-R</span>
            </div>
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">Vision Model</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">ResNet-50</span>
            </div>
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">API Framework</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">FastAPI</span>
            </div>
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">Acceleration</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">CUDA 12</span>
            </div>
            <div className="p-3 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
              <span className="text-[11px] text-[#8D9D95] font-semibold block uppercase">Web Client</span>
              <span className="text-sm font-extrabold text-[#172923] mt-0.5 block">React 18 + Vite</span>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================================
          8. PROJECT LIMITATIONS & FUTURE IMPROVEMENTS
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Current Limitations */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-3 shadow-sm">
          <h3 className="text-base font-extrabold text-[#172923]">
            Current Limitations
          </h3>
          <ul className="space-y-2 text-xs sm:text-sm text-[#68776F]">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 shrink-0" />
              <span>Memotion 3.0 has natural class imbalance (42% Neutral, 32% Positive, 25% Negative).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 shrink-0" />
              <span>Text is extracted via OCR, which can contain occasional spelling artifacts or watermark noise.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-2 shrink-0" />
              <span>Optimized primarily for Hindi-English code-mixing (Hinglish).</span>
            </li>
          </ul>
        </div>

        {/* Future Scope */}
        <div className="bg-[#214B43] rounded-3xl p-7 sm:p-9 text-white space-y-3 shadow-sm border border-[#2C6156]">
          <h3 className="text-base font-extrabold text-[#C7F59D]">
            Future Improvements
          </h3>
          <ul className="space-y-2 text-xs sm:text-sm text-[#D0DDD7]">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#C7F59D] mt-2 shrink-0" />
              <span>Direct in-browser OCR model (PaddleOCR / EasyOCR) for automated text extraction from image files.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#C7F59D] mt-2 shrink-0" />
              <span>Support for regional Indian language memes (Tanglish, Manglish, Benglish).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#C7F59D] mt-2 shrink-0" />
              <span>Fine-grained sarcasm and humor intensity scoring.</span>
            </li>
          </ul>
        </div>

      </section>

      {/* =========================================================================
          9. SAMPLE MEMES FOR INSTANT TESTING
          ========================================================================= */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#E7F6D9] text-[#28594F] text-xs font-bold mb-2">
              <span>Try Real Memes</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-[#172923] tracking-tight">
              Test Sample Memes Instantly
            </h2>
            <p className="text-sm text-[#68776F] mt-0.5">
              Click any meme below to load its image and caption into the Analyzer.
            </p>
          </div>

          <Link
            to="/analyzer"
            className="text-xs font-bold text-[#28594F] hover:text-[#214B43] flex items-center gap-1.5"
          >
            <span>Open Analyzer</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-5">
          {exampleMemes.map((meme) => (
            <div
              key={meme.id}
              onClick={() => navigate(`/analyzer?example=${meme.id}`)}
              className="bg-white rounded-3xl p-4 border border-[#E3E8DE] cursor-pointer mindful-card-hover space-y-3 group shadow-sm"
            >
              <div className="aspect-square rounded-2xl overflow-hidden bg-[#F7F8F5] border border-[#E3E8DE]">
                <img 
                  src={meme.imageUrl} 
                  alt={meme.title} 
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" 
                />
              </div>
              <div className="space-y-1">
                <div className="text-xs font-bold text-[#172923] truncate group-hover:text-[#28594F] transition-colors">
                  {meme.title}
                </div>
                <div className="text-[11px] text-[#28594F] font-bold flex items-center gap-1">
                  <span>Analyze &rarr;</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* =========================================================================
          10. HOW IT WORKS POPUP MODAL (Based on Screenshot Reference)
          ========================================================================= */}
      {isHowItWorksOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/70 backdrop-blur-sm transition-opacity"
          onClick={() => setIsHowItWorksOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="how-it-works-modal-title"
        >
          <div 
            className="bg-[#183832] border border-[#2C6156] rounded-3xl w-full max-w-lg max-h-[85vh] flex flex-col shadow-2xl overflow-hidden relative animate-in fade-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header (Fixed at top, never cut off) */}
            <div className="flex items-center justify-between px-6 sm:px-7 pt-6 pb-4 border-b border-[#2C6156]/70 flex-shrink-0">
              <h2 id="how-it-works-modal-title" className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                How IndicSent-M Works
              </h2>
              <div className="flex items-center gap-2.5">
                <span className="px-3 py-1 rounded-full bg-[#16332C] border border-[#2C6156] text-[#C7F59D] text-xs font-bold">
                  4 Steps
                </span>
                <button
                  type="button"
                  onClick={() => setIsHowItWorksOpen(false)}
                  aria-label="Close dialog"
                  className="w-8 h-8 rounded-full bg-[#214B43] hover:bg-[#28594F] border border-[#2C6156] text-[#A3B8B0] hover:text-white flex items-center justify-center transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Modal Scrollable Body (Internal scroll when screen height is small) */}
            <div className="p-6 sm:p-7 overflow-y-auto space-y-6 flex-1">
              
              {/* Step 1 */}
              <div className="relative flex items-start gap-4">
                <div className="absolute left-[13px] top-7 bottom-[-24px] w-[2px] bg-[#2C6156]" />
                <div className="w-7 h-7 rounded-full bg-[#214B43] border border-[#2C6156] text-white font-extrabold text-sm flex items-center justify-center flex-shrink-0 z-10 shadow-sm">
                  1
                </div>
                <div className="space-y-1 pt-0.5">
                  <h3 className="text-base font-bold text-white">
                    Upload a Meme or Enter Text
                  </h3>
                  <p className="text-sm text-[#C2D3CC] leading-relaxed">
                    Upload a meme image, enter its text, or provide both.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative flex items-start gap-4">
                <div className="absolute left-[13px] top-7 bottom-[-24px] w-[2px] bg-[#2C6156]" />
                <div className="w-7 h-7 rounded-full bg-[#214B43] border border-[#2C6156] text-white font-extrabold text-sm flex items-center justify-center flex-shrink-0 z-10 shadow-sm">
                  2
                </div>
                <div className="space-y-1 pt-0.5">
                  <h3 className="text-base font-bold text-white">
                    AI Reads the Input
                  </h3>
                  <p className="text-sm text-[#C2D3CC] leading-relaxed">
                    MuRIL understands the text and ResNet-50 extracts image features.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative flex items-start gap-4">
                <div className="absolute left-[13px] top-7 bottom-[-24px] w-[2px] bg-[#2C6156]" />
                <div className="w-7 h-7 rounded-full bg-[#214B43] border border-[#2C6156] text-white font-extrabold text-sm flex items-center justify-center flex-shrink-0 z-10 shadow-sm">
                  3
                </div>
                <div className="space-y-1 pt-0.5">
                  <h3 className="text-base font-bold text-white">
                    AI Combines the Information
                  </h3>
                  <p className="text-sm text-[#C2D3CC] leading-relaxed">
                    Gated Fusion combines text and image features for multimodal analysis.
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div className="relative flex items-start gap-4">
                <div className="w-7 h-7 rounded-full bg-[#214B43] border border-[#2C6156] text-white font-extrabold text-sm flex items-center justify-center flex-shrink-0 z-10 shadow-sm">
                  4
                </div>
                <div className="space-y-1 pt-0.5">
                  <h3 className="text-base font-bold text-white">
                    Get Your Sentiment Result
                  </h3>
                  <p className="text-sm text-[#C2D3CC] leading-relaxed">
                    View the predicted sentiment: Positive, Negative or Neutral.
                  </p>
                </div>
              </div>

            </div>

            {/* Modal Footer (Fixed at bottom, never cut off) */}
            <div className="p-6 pt-3 border-t border-[#2C6156]/70 bg-[#183832] flex-shrink-0">
              <button
                type="button"
                onClick={() => setIsHowItWorksOpen(false)}
                className="w-full py-3.5 px-6 rounded-2xl bg-[#C7F59D] hover:bg-[#B5EA87] text-[#172923] font-extrabold text-sm sm:text-base transition-all shadow-md hover:shadow-lg active:scale-[0.99] text-center"
              >
                Got it
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
