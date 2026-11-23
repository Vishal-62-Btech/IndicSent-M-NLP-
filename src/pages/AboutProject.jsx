import React from 'react';
import { 
  Sparkles, 
  HelpCircle,
  AlertTriangle, 
  Database, 
  Target, 
  Code2, 
  CheckCircle2, 
  Trophy, 
  Zap,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';

export default function AboutProject() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12 sm:space-y-14 pb-20">
      
      {/* =========================================================================
          1. HEADER BANNER: About IndicSent-M
          ========================================================================= */}
      <div className="bg-[#214B43] rounded-3xl p-6 sm:p-9 text-white relative overflow-hidden shadow-sm border border-[#2C6156]">
        <div className="space-y-2 max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#C7F59D]/15 text-[#C7F59D] text-xs font-bold">
            <Sparkles className="w-3.5 h-3.5 text-[#C7F59D]" />
            <span>AI Research Project</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            About IndicSent-M
          </h1>
          <p className="text-sm sm:text-base text-[#D0DDD7] leading-relaxed">
            IndicSent-M is an AI project that studies the feelings behind Hinglish memes using both text and images.
          </p>
        </div>

        {/* Subtle decorative background glow */}
        <div className="absolute -top-24 -right-24 w-72 h-72 rounded-full bg-[#C7F59D]/10 blur-3xl pointer-events-none" />
      </div>

      {/* =========================================================================
          2. WHAT IS INDICSENT-M & PROBLEM WE ARE SOLVING (Side-by-Side Cards)
          ========================================================================= */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Card 1: What is IndicSent-M? */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-8 space-y-5 shadow-sm mindful-card-hover flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
                <HelpCircle className="w-5 h-5 text-[#28594F]" />
              </div>
              <h2 className="text-xl font-extrabold text-[#172923]">
                What is IndicSent-M?
              </h2>
            </div>

            <ul className="space-y-3 text-xs sm:text-sm text-[#172923]">
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>An AI-based meme sentiment analysis system.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Understands Hindi-English mixed text.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Uses both meme images and text.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Predicts Positive, Negative or Neutral sentiment.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Card 2: Problem We Are Solving */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-8 space-y-5 shadow-sm mindful-card-hover flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-[#28594F]" />
              </div>
              <h2 className="text-xl font-extrabold text-[#172923]">
                Problem We Are Solving
              </h2>
            </div>

            <ul className="space-y-3 text-xs sm:text-sm text-[#172923]">
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Memes often contain jokes, sarcasm and hidden meanings.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Text alone may not explain the full meaning.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Images can change the meaning of written words.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
                <span>Our project uses text and images together to study meme sentiment.</span>
              </li>
            </ul>
          </div>
        </div>

      </section>

      {/* =========================================================================
          3. DATASET USED: Memotion 3.0
          ========================================================================= */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#EAEFE6] pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
              <Database className="w-5 h-5 text-[#28594F]" />
            </div>
            <div>
              <h2 className="text-xl font-extrabold text-[#172923]">
                Dataset Used
              </h2>
              <p className="text-xs sm:text-sm text-[#68776F]">
                Memotion 3.0 benchmark dataset focused on Hinglish memes.
              </p>
            </div>
          </div>

          <span className="text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full self-start sm:self-center">
            Focus: Hinglish Memes
          </span>
        </div>

        {/* 4 Separate Statistic Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          
          <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-1">
            <span className="text-[11px] font-bold text-[#8D9D95] uppercase tracking-wider block">
              Dataset
            </span>
            <div className="text-xl sm:text-2xl font-extrabold text-[#172923] font-mono">
              Memotion 3.0
            </div>
            <span className="text-[11px] text-[#68776F] block">
              Benchmark Dataset
            </span>
          </div>

          <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-1">
            <span className="text-[11px] font-bold text-[#8D9D95] uppercase tracking-wider block">
              Training Data
            </span>
            <div className="text-xl sm:text-2xl font-extrabold text-[#28594F] font-mono">
              5,600
            </div>
            <span className="text-[11px] text-[#68776F] block">
              Meme Samples
            </span>
          </div>

          <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-1">
            <span className="text-[11px] font-bold text-[#8D9D95] uppercase tracking-wider block">
              Validation Data
            </span>
            <div className="text-xl sm:text-2xl font-extrabold text-[#28594F] font-mono">
              1,400
            </div>
            <span className="text-[11px] text-[#68776F] block">
              Evaluation Samples
            </span>
          </div>

          <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-1">
            <span className="text-[11px] font-bold text-[#8D9D95] uppercase tracking-wider block">
              Test Data
            </span>
            <div className="text-xl sm:text-2xl font-extrabold text-[#172923] font-mono">
              1,500
            </div>
            <span className="text-[11px] text-[#68776F] block">
              Test Samples
            </span>
          </div>

        </div>
      </section>

      {/* =========================================================================
          4. MAIN OBJECTIVES & MAIN FEATURES (2 Columns)
          ========================================================================= */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Main Objectives */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-8 space-y-5 shadow-sm mindful-card-hover">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
              <Target className="w-5 h-5 text-[#28594F]" />
            </div>
            <h2 className="text-xl font-extrabold text-[#172923]">
              Main Objectives
            </h2>
          </div>

          <ul className="space-y-3 text-xs sm:text-sm text-[#172923]">
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Understand Hinglish meme text.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Find useful details in meme images.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Combine text and image information.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Predict the sentiment of memes.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Compare different AI models.</span>
            </li>
          </ul>
        </div>

        {/* Main Features */}
        <div className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-8 space-y-5 shadow-sm mindful-card-hover">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
              <Zap className="w-5 h-5 text-[#28594F]" />
            </div>
            <h2 className="text-xl font-extrabold text-[#172923]">
              Main Features
            </h2>
          </div>

          <ul className="space-y-3 text-xs sm:text-sm text-[#172923]">
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Text-only analysis.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Image-only analysis.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Text and image analysis together.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Confidence scores.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>Model comparison.</span>
            </li>
            <li className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#28594F] shrink-0 mt-0.5" />
              <span>GPU-based prediction support.</span>
            </li>
          </ul>
        </div>

      </section>

      {/* =========================================================================
          5. PROJECT RESULTS (Three Highlighted Cards)
          ========================================================================= */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-6 shadow-sm">
        <div className="border-b border-[#EAEFE6] pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
              <Trophy className="w-5 h-5 text-[#28594F]" />
            </div>
            <div>
              <h2 className="text-xl font-extrabold text-[#172923]">
                Project Results
              </h2>
              <p className="text-xs sm:text-sm text-[#68776F] mt-0.5">
                These results are based on our reported validation experiments.
              </p>
            </div>
          </div>

          <Link
            to="/comparison"
            className="text-xs font-bold text-[#28594F] hover:text-[#214B43] flex items-center gap-1 self-start sm:self-center"
          >
            <span>View All 14 Models</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {/* Three Separate Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          
          {/* Card 1: Best Overall Model */}
          <div className="p-6 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-3 mindful-card-hover flex flex-col justify-between">
            <div className="space-y-2">
              <span className="inline-block text-xs font-bold text-[#28594F] bg-[#E7F6D9] px-3 py-1 rounded-full">
                Best Overall Model
              </span>
              <h3 className="text-base font-extrabold text-[#172923]">
                TF-IDF + Logistic Regression
              </h3>
              <p className="text-xs text-[#68776F] leading-relaxed">
                Uses words from meme text to predict sentiment accurately and quickly.
              </p>
            </div>

            <div className="pt-3 border-t border-[#EAEFE6] flex items-center justify-between text-xs">
              <span className="text-[#68776F] font-semibold">Macro-F1 Score:</span>
              <span className="font-mono font-extrabold text-sm text-[#28594F] bg-white px-2.5 py-0.5 rounded-full border border-[#E3E8DE]">
                43.14%
              </span>
            </div>
          </div>

          {/* Card 2: Best Text + Image Model */}
          <div className="p-6 rounded-2xl bg-[#28594F] text-white space-y-3 shadow-sm flex flex-col justify-between">
            <div className="space-y-2">
              <span className="inline-block text-xs font-bold text-[#172923] bg-[#C7F59D] px-3 py-1 rounded-full">
                Best Text + Image Model
              </span>
              <h3 className="text-base font-extrabold text-white">
                MuRIL + ResNet-50 Gated Fusion
              </h3>
              <p className="text-xs text-[#D0DDD7] leading-relaxed">
                Combines meme text and image information to predict sentiment.
              </p>
            </div>

            <div className="pt-3 border-t border-[#377366] flex items-center justify-between text-xs">
              <span className="text-[#D0DDD7]">Macro-F1 Score:</span>
              <span className="font-mono font-extrabold text-sm text-[#C7F59D] bg-[#214B43] px-2.5 py-0.5 rounded-full border border-[#377366]">
                37.24%
              </span>
            </div>
          </div>

          {/* Card 3: Best Image Model */}
          <div className="p-6 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-3 mindful-card-hover flex flex-col justify-between">
            <div className="space-y-2">
              <span className="inline-block text-xs font-bold text-[#28594F] bg-[#F7F8F5] px-3 py-1 rounded-full border border-[#E3E8DE]">
                Best Image Model
              </span>
              <h3 className="text-base font-extrabold text-[#172923]">
                ResNet-50
              </h3>
              <p className="text-xs text-[#68776F] leading-relaxed">
                Analyzes meme images to find useful visual details.
              </p>
            </div>

            <div className="pt-3 border-t border-[#EAEFE6] flex items-center justify-between text-xs">
              <span className="text-[#68776F] font-semibold">Macro-F1 Score:</span>
              <span className="font-mono font-extrabold text-sm text-[#28594F] bg-white px-2.5 py-0.5 rounded-full border border-[#E3E8DE]">
                37.04%
              </span>
            </div>
          </div>

        </div>
      </section>

      {/* =========================================================================
          6. TECHNOLOGIES USED
          ========================================================================= */}
      <section className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-6 shadow-sm">
        <div className="flex items-center gap-3 border-b border-[#EAEFE6] pb-4">
          <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center">
            <Code2 className="w-5 h-5 text-[#28594F]" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-[#172923]">
              Technologies Used
            </h2>
            <p className="text-xs sm:text-sm text-[#68776F]">
              Tools and libraries used to build and run the project.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          
          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              Python
            </div>
            <p className="text-xs text-[#68776F]">
              Main programming language.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              PyTorch
            </div>
            <p className="text-xs text-[#68776F]">
              Used to train and run AI models.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              MuRIL
            </div>
            <p className="text-xs text-[#68776F]">
              Understands meme text.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              ResNet-50
            </div>
            <p className="text-xs text-[#68776F]">
              Analyzes meme images.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              FastAPI
            </div>
            <p className="text-xs text-[#68776F]">
              Connects the website with AI models.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              React + Vite
            </div>
            <p className="text-xs text-[#68776F]">
              Used to build the website.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1 sm:col-span-2 lg:col-span-1">
            <div className="font-extrabold text-[#172923] text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#28594F]" />
              CUDA
            </div>
            <p className="text-xs text-[#68776F]">
              Uses NVIDIA GPU for faster processing.
            </p>
          </div>

        </div>
      </section>

      {/* =========================================================================
          7. FUTURE IMPROVEMENTS
          ========================================================================= */}
      <section className="bg-[#28594F] rounded-3xl p-7 sm:p-9 text-white space-y-5 shadow-sm border border-[#377366]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-[#C7F59D] text-[#172923] flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-[#172923]" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">
              Future Improvements
            </h2>
            <p className="text-xs sm:text-sm text-[#D0DDD7]">
              Planned features to enhance IndicSent-M in future updates.
            </p>
          </div>
        </div>

        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 text-xs sm:text-sm text-[#D0DDD7] pt-1">
          <li className="flex items-start gap-2.5 bg-[#214B43] p-3.5 rounded-2xl border border-[#377366]">
            <CheckCircle2 className="w-4 h-4 text-[#C7F59D] shrink-0 mt-0.5" />
            <span>Add automatic text reading from meme images.</span>
          </li>
          <li className="flex items-start gap-2.5 bg-[#214B43] p-3.5 rounded-2xl border border-[#377366]">
            <CheckCircle2 className="w-4 h-4 text-[#C7F59D] shrink-0 mt-0.5" />
            <span>Support more Indian languages.</span>
          </li>
          <li className="flex items-start gap-2.5 bg-[#214B43] p-3.5 rounded-2xl border border-[#377366]">
            <CheckCircle2 className="w-4 h-4 text-[#C7F59D] shrink-0 mt-0.5" />
            <span>Improve sarcasm detection.</span>
          </li>
          <li className="flex items-start gap-2.5 bg-[#214B43] p-3.5 rounded-2xl border border-[#377366]">
            <CheckCircle2 className="w-4 h-4 text-[#C7F59D] shrink-0 mt-0.5" />
            <span>Add humour detection.</span>
          </li>
          <li className="flex items-start gap-2.5 bg-[#214B43] p-3.5 rounded-2xl border border-[#377366] sm:col-span-2">
            <CheckCircle2 className="w-4 h-4 text-[#C7F59D] shrink-0 mt-0.5" />
            <span>Improve prediction accuracy.</span>
          </li>
        </ul>
      </section>

    </div>
  );
}
