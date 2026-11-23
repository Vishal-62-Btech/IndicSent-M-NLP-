import React, { useState } from 'react';
import { Info, X, Sparkles, CheckCircle2 } from 'lucide-react';

export default function PrototypeNotice() {
  const [isDismissed, setIsDismissed] = useState(false);

  if (isDismissed) {
    return (
      <div className="bg-purple-50/80 border-b border-purple-100 py-1.5 px-4 text-xs flex justify-between items-center text-purple-700">
        <span className="flex items-center gap-1.5 font-medium">
          <Sparkles className="w-3.5 h-3.5 text-purple-600 animate-pulse" />
          Frontend Prototype Mode &bull; Simulated Inference &bull; Pre-Model Training Phase
        </span>
        <button 
          onClick={() => setIsDismissed(false)}
          className="text-purple-600 hover:text-purple-800 underline font-semibold text-[11px]"
        >
          Show Details
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-purple-50 via-indigo-50 to-purple-50 border-b border-purple-200/80 px-4 py-2.5 sm:px-6 transition-all">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2.5">
        <div className="flex items-start gap-2.5">
          <div className="p-1.5 bg-purple-100 text-purple-700 rounded-lg shrink-0 mt-0.5 sm:mt-0">
            <Info className="w-4 h-4" />
          </div>
          <div>
            <p className="text-xs sm:text-sm text-purple-950 font-medium">
              <span className="font-bold text-purple-900 bg-purple-200/70 px-1.5 py-0.5 rounded text-xs mr-1.5 uppercase tracking-wide">
                Interactive Prototype
              </span>
              This interface demonstrates the complete planned user workflow for <strong className="text-purple-900">IndicSent-M</strong>.
            </p>
            <p className="text-xs text-purple-700 mt-0.5">
              Sentiment predictions and performance metrics are simulated demonstration outputs. Actual ML model training (XLM-RoBERTa + ResNet-50) is scheduled post dataset verification.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 self-end sm:self-center shrink-0">
          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded-full border border-emerald-200">
            <CheckCircle2 className="w-3 h-3" /> UI Functional
          </span>
          <button
            onClick={() => setIsDismissed(true)}
            className="text-purple-500 hover:text-purple-800 p-1 rounded-md hover:bg-purple-100 transition-colors"
            title="Dismiss banner"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
