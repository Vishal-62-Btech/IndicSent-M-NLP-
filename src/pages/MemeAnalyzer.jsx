import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  UploadCloud, 
  Trash2, 
  Sparkles, 
  AlertCircle, 
  RefreshCw, 
  FileText, 
  Image as ImageIcon,
  GitMerge,
  Smile, 
  Frown, 
  Meh, 
  Check,
  Clock, 
  Cpu, 
  ArrowRight,
  ShieldCheck,
  HelpCircle,
  Download,
  Layers,
  Eye,
  SlidersHorizontal,
  X,
  Info,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { exampleMemes } from '../data/mockData';
import { useBackend } from '../context/BackendContext';
import { 
  predictText, 
  predictImage, 
  predictMultimodal,
  extractOcrText,
  suggestMemeText,
  explainPrediction,
  downloadPdfReport
} from '../services/api';
import { svgDataUrlToJpegFile } from '../utils/imageHelper';

// Verified best model for each modality
const BEST_MODELS = {
  multimodal: {
    id: 'multitask_emotion',
    name: 'MuRIL + ResNet-50 (Multi-Task Sentiment & Emotion Fusion)',
    macroF1: '35.30%',
    typeLabel: 'Multimodal (Text + Image + 6 Emotions)',
    description: 'Shared cross-modal fusion predicting 3-class sentiment and dominant emotion.'
  },
  multimodal_baseline: {
    id: 'gated_fusion',
    name: 'MuRIL + ResNet-50 (Gated Fusion Baseline)',
    macroF1: '37.24%',
    typeLabel: 'Multimodal (Text + Image)',
    description: 'Combines meme text and visual features using attention gating.'
  },
  text: {
    id: 'logistic_regression',
    name: 'TF-IDF + Logistic Regression',
    macroF1: '43.14%',
    typeLabel: 'Text Only',
    description: 'N-gram language model evaluated on Hinglish text.'
  },
  image: {
    id: 'resnet50',
    name: 'ResNet-50',
    macroF1: '37.04%',
    typeLabel: 'Image Only',
    description: 'Analyzes image features and visual patterns to predict meme sentiment.'
  }
};

export default function MemeAnalyzer() {
  const location = useLocation();
  const navigate = useNavigate();
  const { isOnline, isChecking, refreshStatus } = useBackend();

  // Mode Selection: 'multimodal' | 'text' | 'image'
  const [activeMode, setActiveMode] = useState('multimodal');

  // Form states
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [memeText, setMemeText] = useState('');
  const [activeExampleId, setActiveExampleId] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // OCR States (Manual execution only in Multimodal mode)
  const [isOcrLoading, setIsOcrLoading] = useState(false);
  const [ocrStatus, setOcrStatus] = useState('idle'); // 'idle' | 'loading' | 'success' | 'empty' | 'error'
  const [ocrConfidence, setOcrConfidence] = useState(null);
  const [ocrMessage, setOcrMessage] = useState('');
  const userHasEditedTextRef = useRef(false);

  // Meme-Based Hinglish Suggestions States
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionStatus, setSuggestionStatus] = useState('idle'); // 'idle' | 'loading' | 'success' | 'empty' | 'error'
  const [suggestionMessage, setSuggestionMessage] = useState('');
  const [pendingTextToConfirm, setPendingTextToConfirm] = useState(null); // { text, type, title }

  // Analysis / Response states
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Multi-Task Emotion Detection States
  const [multimodalSubModel, setMultimodalSubModel] = useState('multitask_emotion');
  const [showEmotionBreakdown, setShowEmotionBreakdown] = useState(false);

  // Explainable AI States
  const [isExplaining, setIsExplaining] = useState(false);
  const [explanationData, setExplanationData] = useState(null);
  const [explainError, setExplainError] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const explanationCacheRef = useRef({});

  // PDF Export State
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);
  const explanationRef = useRef(null);

  // Sync mode and example query parameters from URL
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const modeParam = params.get('mode');
    if (modeParam && ['multimodal', 'text', 'image'].includes(modeParam.toLowerCase())) {
      setActiveMode(modeParam.toLowerCase());
    }

    const exampleId = params.get('example');
    if (exampleId) {
      loadExampleById(exampleId);
    }
  }, [location.search]);

  // Load a pre-defined example meme (Loads image and text directly without auto OCR)
  const loadExampleById = async (exampleId) => {
    const example = exampleMemes.find((m) => m.id === exampleId);
    if (example) {
      setActiveExampleId(example.id);
      setImagePreviewUrl(example.imageUrl);
      setMemeText(example.text);
      userHasEditedTextRef.current = false;
      setErrorMessage('');
      setAnalysisResult(null);
      clearExplanation();
      setOcrStatus('idle');
      setSuggestions([]);
      setPendingTextToConfirm(null);

      try {
        const fileObj = await svgDataUrlToJpegFile(example.imageUrl, `${example.id}.jpg`);
        setSelectedFile(fileObj);
        // OCR is NOT triggered automatically here
      } catch (err) {
        console.warn('Could not convert SVG data URL to file:', err);
      }
    }
  };

  // =========================================================================
  // EXPLICIT ON-DEMAND OCR WORKFLOW (Multimodal mode only, triggered by button)
  // =========================================================================
  const handleExtractOcr = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload a meme image first to extract text.');
      return;
    }
    if (activeMode !== 'multimodal') return;

    setIsOcrLoading(true);
    setOcrStatus('loading');
    setOcrMessage('Reading text from image...');
    setErrorMessage('');

    try {
      const res = await extractOcrText(selectedFile);
      if (res && res.success && res.has_text) {
        setOcrConfidence(res.confidence);
        setOcrStatus('success');
        setOcrMessage('Text extracted successfully via OCR.');

        // Protect manual user text: if text already exists, prompt for confirmation instead of silent overwrite
        if (memeText.trim() && userHasEditedTextRef.current) {
          setPendingTextToConfirm({
            text: res.text,
            type: 'ocr',
            title: 'Extracted OCR Text'
          });
        } else {
          setMemeText(res.text);
          userHasEditedTextRef.current = false;
        }
      } else {
        setOcrStatus('empty');
        setOcrMessage(res?.message || 'No text found in image. You can enter text manually.');
      }
    } catch (err) {
      console.warn('OCR extraction warning:', err);
      setOcrStatus('error');
      setOcrMessage('Could not read text from image. You can enter text manually.');
    } finally {
      setIsOcrLoading(false);
    }
  };

  // =========================================================================
  // MEME-BASED HINGLISH TEXT SUGGESTIONS (Triggered by user button click)
  // =========================================================================
  const handleSuggestMemeText = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload a meme image first to generate suggestions.');
      return;
    }
    if (activeMode !== 'multimodal') return;

    setIsSuggesting(true);
    setSuggestionStatus('loading');
    setSuggestionMessage('Analyzing visual content for suggestions...');
    setErrorMessage('');

    try {
      const res = await suggestMemeText(selectedFile);
      if (res && res.success && res.suggestions && res.suggestions.length > 0) {
        setSuggestions(res.suggestions);
        setSuggestionStatus('success');
        setSuggestionMessage(res.message);
      } else {
        setSuggestionStatus('empty');
        setSuggestionMessage(res?.message || 'Could not determine visual context. Please enter text manually.');
        setSuggestions([]);
      }
    } catch (err) {
      console.warn('Meme suggestion error:', err);
      setSuggestionStatus('error');
      const errDetail = err?.message || 'Could not connect to the backend server to generate suggestions.';
      setSuggestionMessage(errDetail);
      setErrorMessage(errDetail);
      setSuggestions([]);
    } finally {
      setIsSuggesting(false);
    }
  };

  const handleSelectSuggestion = (phrase) => {
    if (memeText.trim() && userHasEditedTextRef.current) {
      setPendingTextToConfirm({
        text: phrase,
        type: 'suggestion',
        title: 'Selected Suggestion'
      });
    } else {
      setMemeText(phrase);
      userHasEditedTextRef.current = false;
      setErrorMessage('');
    }
  };

  const handleConfirmReplaceText = () => {
    if (pendingTextToConfirm) {
      setMemeText(pendingTextToConfirm.text);
      userHasEditedTextRef.current = false;
      setPendingTextToConfirm(null);
    }
  };

  const handleConfirmAppendText = () => {
    if (pendingTextToConfirm) {
      setMemeText((prev) => (prev ? `${prev} ${pendingTextToConfirm.text}` : pendingTextToConfirm.text));
      userHasEditedTextRef.current = true;
      setPendingTextToConfirm(null);
    }
  };

  const handleCancelPendingText = () => {
    setPendingTextToConfirm(null);
  };

  // =========================================================================
  // FILE HANDLING
  // =========================================================================
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = (file) => {
    setErrorMessage('');
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setErrorMessage('Unsupported file format. Please upload a JPG, PNG, or WebP image.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 10 MB limit. Please upload a smaller image.');
      return;
    }

    setSelectedFile(file);
    setActiveExampleId(null);
    setAnalysisResult(null);
    clearExplanation();

    const reader = new FileReader();
    reader.onload = () => {
      setImagePreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);

    // OCR is NOT triggered automatically on upload.
    // It runs ONLY when the user clicks "Extract Text from Image" in Multimodal mode.
    setOcrStatus('idle');
    setSuggestions([]);
    setPendingTextToConfirm(null);
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setImagePreviewUrl(null);
    setActiveExampleId(null);
    setAnalysisResult(null);
    clearExplanation();
    setOcrStatus('idle');
    setOcrMessage('');
    setSuggestions([]);
    setPendingTextToConfirm(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClearAll = () => {
    handleRemoveImage();
    setMemeText('');
    userHasEditedTextRef.current = false;
    setAnalysisResult(null);
    clearExplanation();
    setErrorMessage('');
    setSuggestions([]);
    setPendingTextToConfirm(null);
  };

  const clearExplanation = () => {
    setExplanationData(null);
    setExplainError(null);
    setShowExplanation(false);
    setIsExplaining(false);
  };

  // Mode Switcher (updates UI immediately and isolates input modalities)
  const handleModeChange = (mode) => {
    setActiveMode(mode);
    setAnalysisResult(null);
    clearExplanation();
    setErrorMessage('');
    setPendingTextToConfirm(null);
    setOcrStatus('idle');
    setSuggestions([]);

    if (mode === 'text') {
      setSelectedFile(null);
      setImagePreviewUrl(null);
    } else if (mode === 'image') {
      setMemeText('');
      userHasEditedTextRef.current = false;
    }

    const params = new URLSearchParams(location.search);
    params.set('mode', mode);
    navigate({ search: params.toString() }, { replace: true });
  };

  // =========================================================================
  // SENTIMENT PREDICTION (NORMAL INFERENCE - ZERO EXPLANATION DELAY)
  // =========================================================================
  const handleAnalyzeSentiment = async () => {
    setErrorMessage('');

    if (activeMode === 'text' && !memeText.trim()) {
      setErrorMessage('Please enter Hinglish meme text to analyze.');
      return;
    }

    if (activeMode === 'image' && !selectedFile) {
      setErrorMessage('Please upload a meme image file to analyze.');
      return;
    }

    if (activeMode === 'multimodal') {
      if (!selectedFile && !memeText.trim()) {
        setErrorMessage('Multimodal analysis requires both meme text and an image upload.');
        return;
      }
      if (!selectedFile) {
        setErrorMessage('Please upload a meme image for multimodal analysis.');
        return;
      }
      if (!memeText.trim()) {
        setErrorMessage('Multimodal analysis requires meme text. Please enter text manually, click "Extract Text from Image", or switch to Image Only mode.');
        return;
      }
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);
    clearExplanation();

    const currentModel = activeMode === 'multimodal'
      ? (multimodalSubModel === 'gated_fusion' ? BEST_MODELS.multimodal_baseline : BEST_MODELS.multimodal)
      : BEST_MODELS[activeMode];

    try {
      let response;
      if (activeMode === 'text') {
        response = await predictText(memeText.trim(), currentModel.id);
      } else if (activeMode === 'image') {
        response = await predictImage(selectedFile, currentModel.id);
      } else if (activeMode === 'multimodal') {
        response = await predictMultimodal(memeText.trim(), selectedFile, multimodalSubModel);
      }

      if (response && response.success && response.prediction) {
        setAnalysisResult(response.prediction);
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
      } else {
        throw new Error('Prediction API returned an unexpected response format.');
      }
    } catch (err) {
      console.error('Sentiment Analysis error:', err);
      setErrorMessage(err.message || 'Sentiment prediction failed. Please verify the backend connection.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // =========================================================================
  // EXPLAINABLE AI ON-DEMAND ("Why This Prediction?")
  // =========================================================================
  const handleWhyThisPrediction = async () => {
    if (!analysisResult) return;

    // If explanation is already visible, clicking toggles its view
    if (explanationData && showExplanation) {
      setShowExplanation(false);
      return;
    }

    setShowExplanation(true);

    // Cache key for current prediction
    const cacheKey = `${activeMode}_${currentModel.id}_${selectedFile?.name || 'none'}_${memeText.trim()}_${analysisResult.predicted_sentiment}`;

    if (explanationCacheRef.current[cacheKey]) {
      setExplanationData(explanationCacheRef.current[cacheKey]);
      setExplainError(null);
      setTimeout(() => {
        explanationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
      return;
    }

    setIsExplaining(true);
    setExplainError(null);

    try {
      const response = await explainPrediction({
        mode: activeMode,
        model: currentModel.id,
        text: memeText.trim() || null,
        imageFile: selectedFile || null,
        predictedSentiment: analysisResult.predicted_sentiment
      });

      if (response && response.success && response.explanation) {
        explanationCacheRef.current[cacheKey] = response.explanation;
        setExplanationData(response.explanation);
        setTimeout(() => {
          explanationRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
      } else {
        setExplainError(response?.message || 'Explanation is not available for this model configuration.');
      }
    } catch (err) {
      console.error('Explanation request error:', err);
      // Keep original prediction visible and show simple error message
      setExplainError(err.message || 'Could not generate explanation. Your original sentiment prediction remains available above.');
    } finally {
      setIsExplaining(false);
    }
  };

  // =========================================================================
  // PDF REPORT DOWNLOAD
  // =========================================================================
  const handleDownloadPdf = async () => {
    if (!analysisResult) return;

    setIsDownloadingPdf(true);
    try {
      const reportPayload = {
        model_name: analysisResult.model_name || currentModel.name,
        mode: activeMode,
        predicted_sentiment: analysisResult.predicted_sentiment,
        confidence_positive: analysisResult.confidence_scores.Positive,
        confidence_neutral: analysisResult.confidence_scores.Neutral,
        confidence_negative: analysisResult.confidence_scores.Negative,
        processing_time_ms: analysisResult.total_time_ms,
        input_text: memeText.trim() || undefined,
      };

      // Include Dominant Emotion if present
      if (analysisResult.emotion && analysisResult.emotion.is_available && analysisResult.emotion.dominant_emotion) {
        reportPayload.dominant_emotion = analysisResult.emotion.dominant_emotion;
        reportPayload.dominant_emotion_probability = analysisResult.emotion.dominant_probability;
        reportPayload.dominant_emotion_emoji = analysisResult.emotion.dominant_emoji;
        if (analysisResult.emotion.probabilities) {
          reportPayload.emotion_probabilities = JSON.stringify(analysisResult.emotion.probabilities);
        }
      }

      // Include explanation ONLY if generated by user
      if (explanationData) {
        reportPayload.explanation_summary = explanationData.summary;
        if (explanationData.gate_weights) {
          reportPayload.gate_text_weight = explanationData.gate_weights.text_weight;
          reportPayload.gate_image_weight = explanationData.gate_weights.image_weight;
        }
        if (explanationData.word_attributions && explanationData.word_attributions.length > 0) {
          reportPayload.top_words = explanationData.word_attributions.slice(0, 5).map(w => `${w.word} (${Math.round(w.score * 100)}%)`).join(', ');
        }
        if (explanationData.heatmap_image) {
          reportPayload.heatmap_base64 = explanationData.heatmap_image;
        }
      }

      await downloadPdfReport(reportPayload);
    } catch (err) {
      console.error('PDF generation error:', err);
      setErrorMessage('Failed to generate PDF report. Please verify backend connection.');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const getSentimentBadge = (sentiment) => {
    switch (sentiment) {
      case 'Positive':
        return {
          pill: 'bg-[#E7F6D9] text-[#214B43] border-[#C7F59D]',
          barColor: 'bg-[#C7F59D]',
          textColor: 'text-[#214B43]',
          icon: Smile
        };
      case 'Negative':
        return {
          pill: 'bg-rose-50 text-rose-800 border-rose-200',
          barColor: 'bg-rose-500',
          textColor: 'text-rose-800',
          icon: Frown
        };
      default:
        return {
          pill: 'bg-amber-50 text-amber-800 border-amber-200',
          barColor: 'bg-amber-500',
          textColor: 'text-amber-800',
          icon: Meh
        };
    }
  };

  const currentModel = activeMode === 'multimodal'
    ? (multimodalSubModel === 'gated_fusion' ? BEST_MODELS.multimodal_baseline : BEST_MODELS.multimodal)
    : BEST_MODELS[activeMode];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8 pb-20">
      
      {/* Header Section */}
      <div className="bg-[#214B43] rounded-3xl p-6 sm:p-8 text-white relative overflow-hidden shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#C7F59D]/15 text-[#C7F59D] text-xs font-bold">
              <Sparkles className="w-3.5 h-3.5 text-[#C7F59D]" />
              <span>Interactive Sentiment Engine</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Meme Sentiment Analyzer
            </h1>
            <p className="text-xs sm:text-sm text-[#D0DDD7] max-w-xl">
              Upload a meme graphic with automatic OCR, enter Hinglish punchlines, and inspect predictions with Explainable AI on demand.
            </p>
          </div>

          {(imagePreviewUrl || memeText) && (
            <button
              onClick={handleClearAll}
              className="text-xs px-4 py-2 rounded-full border border-[#377366] hover:bg-[#28594F] text-white flex items-center gap-1.5 transition-colors font-semibold self-start sm:self-center"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Clear All</span>
            </button>
          )}
        </div>
      </div>

      {/* Backend Offline Warning Banner */}
      {!isOnline && (
        <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-sm">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0" />
            <div>
              <span className="font-bold">FastAPI Backend is Offline: </span>
              <span>Start the server with <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-[11px]">python run.py</code> to enable live predictions and OCR.</span>
            </div>
          </div>
          <button
            onClick={refreshStatus}
            disabled={isChecking}
            className="px-4 py-1.5 rounded-full bg-amber-600 hover:bg-amber-700 text-white font-bold shrink-0 flex items-center gap-1.5 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isChecking ? 'animate-spin' : ''}`} />
            <span>Retry Connection</span>
          </button>
        </div>
      )}

      {/* Mindful-Inspired 3 Analysis Tabs */}
      <div className="flex items-center gap-2 p-1.5 bg-[#28594F] rounded-full max-w-fit shadow-inner overflow-x-auto">
        <button
          onClick={() => handleModeChange('multimodal')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-xs sm:text-sm font-bold transition-all whitespace-nowrap ${
            activeMode === 'multimodal'
              ? 'bg-[#C7F59D] text-[#172923] shadow-sm'
              : 'text-[#D0DDD7] hover:text-white hover:bg-white/10'
          }`}
        >
          <GitMerge className="w-4 h-4" />
          <span>Multimodal (Text + Image)</span>
          <span className="text-[10px] bg-[#214B43] text-[#C7F59D] px-2 py-0.5 rounded-full font-bold">Best Model</span>
        </button>

        <button
          onClick={() => handleModeChange('text')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-xs sm:text-sm font-bold transition-all whitespace-nowrap ${
            activeMode === 'text'
              ? 'bg-[#C7F59D] text-[#172923] shadow-sm'
              : 'text-[#D0DDD7] hover:text-white hover:bg-white/10'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Text Only</span>
          <span className="text-[10px] bg-[#214B43] text-[#C7F59D] px-2 py-0.5 rounded-full font-bold">Fastest</span>
        </button>

        <button
          onClick={() => handleModeChange('image')}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-full text-xs sm:text-sm font-bold transition-all whitespace-nowrap ${
            activeMode === 'image'
              ? 'bg-[#C7F59D] text-[#172923] shadow-sm'
              : 'text-[#D0DDD7] hover:text-white hover:bg-white/10'
          }`}
        >
          <ImageIcon className="w-4 h-4" />
          <span>Image Only</span>
          <span className="text-[10px] bg-[#214B43] text-[#C7F59D] px-2 py-0.5 rounded-full font-bold">Vision</span>
        </button>
      </div>

      {/* Model Information Card */}
      <div className="bg-white rounded-3xl border border-[#E3E8DE] p-5 sm:p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center shrink-0">
            <Cpu className="w-5 h-5 text-[#28594F]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-[#172923] text-sm sm:text-base">
                Classification Model:
              </span>
              <span className="text-sm font-bold text-[#28594F]">
                {currentModel.name}
              </span>
            </div>
            <p className="text-xs text-[#68776F] mt-0.5">
              {currentModel.description}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 self-start sm:self-center">
          {activeMode === 'multimodal' && (
            <div className="flex items-center gap-1 p-1 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE] text-xs">
              <button
                type="button"
                onClick={() => {
                  setMultimodalSubModel('multitask_emotion');
                  setAnalysisResult(null);
                }}
                className={`px-3 py-1.5 rounded-xl font-bold transition-all ${
                  multimodalSubModel === 'multitask_emotion'
                    ? 'bg-[#28594F] text-[#C7F59D] shadow-sm'
                    : 'text-[#68776F] hover:text-[#172923]'
                }`}
              >
                Multi-Task Emotion
              </button>
              <button
                type="button"
                onClick={() => {
                  setMultimodalSubModel('gated_fusion');
                  setAnalysisResult(null);
                }}
                className={`px-3 py-1.5 rounded-xl font-bold transition-all ${
                  multimodalSubModel === 'gated_fusion'
                    ? 'bg-[#28594F] text-[#C7F59D] shadow-sm'
                    : 'text-[#68776F] hover:text-[#172923]'
                }`}
              >
                Gated Fusion Baseline
              </button>
            </div>
          )}
          <span className="text-xs px-3 py-1.5 rounded-full bg-[#E7F6D9] text-[#28594F] font-bold border border-[#C7F59D]">
            Macro-F1: {currentModel.macroF1}
          </span>
          <span className="text-xs px-3 py-1.5 rounded-full bg-[#F7F8F5] text-[#172923] font-semibold border border-[#E3E8DE]">
            {currentModel.typeLabel}
          </span>
        </div>
      </div>

      {/* Quick Sample Presets */}
      {activeMode !== 'text' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-[#68776F]">
            <span>Try a sample Hinglish meme:</span>
            <span className="text-[#8D9D95]">Click to load sample meme into analyzer</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {exampleMemes.map((meme) => {
              const isSelected = activeExampleId === meme.id;
              return (
                <div
                  key={meme.id}
                  onClick={() => loadExampleById(meme.id)}
                  className={`p-3 rounded-2xl border cursor-pointer mindful-card-hover flex items-center gap-3 ${
                    isSelected
                      ? 'border-[#28594F] bg-[#E7F6D9] shadow-sm'
                      : 'border-[#E3E8DE] bg-white hover:border-[#28594F]'
                  }`}
                >
                  <div className="w-12 h-12 rounded-xl overflow-hidden bg-[#F7F8F5] shrink-0 border border-[#E3E8DE]">
                    <img src={meme.imageUrl} alt={meme.title} className="w-full h-full object-cover" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-bold text-[#172923] truncate leading-tight">
                      {meme.title}
                    </div>
                    <span className="text-[11px] text-[#28594F] font-bold inline-flex items-center gap-0.5 mt-0.5">
                      Load &rarr;
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Input Area Grid Layout */}
      <div className={`grid gap-6 items-start ${
        activeMode === 'multimodal' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
      }`}>
        
        {/* Upload Meme Image Card (Multimodal and Image Only modes) */}
        {activeMode !== 'text' && (
          <div className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-7 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-extrabold text-[#172923] flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-[#28594F]" />
                <span>Upload Meme Image</span>
              </h3>
              <span className="text-xs text-[#8D9D95]">JPG, PNG, WebP (Max 10MB)</span>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg, image/webp"
              onChange={handleFileChange}
              className="hidden"
            />

            {!imagePreviewUrl ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[240px] ${
                  isDragOver 
                    ? 'border-[#28594F] bg-[#E7F6D9]/50 scale-[0.99]' 
                    : 'border-[#C7D4CD] hover:border-[#28594F] hover:bg-[#F7F8F5]'
                }`}
              >
                <div className="w-14 h-14 rounded-2xl bg-[#E7F6D9] text-[#28594F] flex items-center justify-center mb-3">
                  <UploadCloud className="w-7 h-7 text-[#28594F]" />
                </div>
                <p className="text-sm font-bold text-[#172923]">
                  Drop your meme image here
                </p>
                <p className="text-xs text-[#68776F] mt-1">
                  or click to browse from your device
                </p>
                <button
                  type="button"
                  className="mt-4 px-5 py-2 rounded-full bg-[#214B43] hover:bg-[#28594F] text-white text-xs font-bold transition-all shadow-sm"
                >
                  Browse Files
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-2xl overflow-hidden border border-[#E3E8DE] bg-[#183832] flex items-center justify-center max-h-[280px]">
                  <img
                    src={imagePreviewUrl}
                    alt="Meme Preview"
                    className="max-h-[280px] w-auto object-contain mx-auto"
                  />
                  {isOcrLoading && (
                    <div className="absolute inset-0 bg-[#214B43]/80 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2 text-white p-4 text-center">
                      <RefreshCw className="w-7 h-7 text-[#C7F59D] animate-spin" />
                      <span className="text-sm font-bold text-[#C7F59D]">
                        Reading text from image...
                      </span>
                    </div>
                  )}
                  {isSuggesting && (
                    <div className="absolute inset-0 bg-[#214B43]/80 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2 text-white p-4 text-center">
                      <Sparkles className="w-7 h-7 text-[#C7F59D] animate-spin" />
                      <span className="text-sm font-bold text-[#C7F59D]">
                        Analyzing visual content...
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="truncate max-w-[220px] text-[#68776F] font-medium">
                    {selectedFile?.name || 'Selected meme graphic'}
                  </span>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-[#28594F] hover:underline font-bold text-xs"
                    >
                      Change Image
                    </button>
                    <span className="text-slate-300">&bull;</span>
                    <button
                      onClick={handleRemoveImage}
                      className="text-rose-600 hover:underline font-bold text-xs flex items-center gap-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>

                {/* Multimodal Dedicated Action Buttons */}
                {activeMode === 'multimodal' && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-2 border-t border-[#E3E8DE]">
                    <button
                      type="button"
                      onClick={handleExtractOcr}
                      disabled={isOcrLoading || isSuggesting}
                      className="px-4 py-2.5 rounded-xl bg-[#214B43] hover:bg-[#28594F] disabled:opacity-60 text-white text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-sm"
                    >
                      <FileText className={`w-3.5 h-3.5 text-[#C7F59D] ${isOcrLoading ? 'animate-spin' : ''}`} />
                      <span>{isOcrLoading ? 'Reading OCR...' : 'Extract Text from Image'}</span>
                    </button>

                    <button
                      type="button"
                      onClick={handleSuggestMemeText}
                      disabled={isSuggesting || isOcrLoading}
                      className="px-4 py-2.5 rounded-xl bg-[#E7F6D9] hover:bg-[#D4EDB8] border border-[#C7F59D] disabled:opacity-60 text-[#28594F] text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-sm"
                    >
                      <Sparkles className={`w-3.5 h-3.5 text-[#28594F] ${isSuggesting ? 'animate-spin' : ''}`} />
                      <span>{isSuggesting ? 'Analyzing...' : 'Suggest Meme Text'}</span>
                    </button>

                    {suggestions.length > 0 && (
                      <div className="col-span-1 sm:col-span-2 text-[11px] text-[#28594F] font-semibold flex items-center justify-center gap-1.5 py-1 px-3 bg-[#E7F6D9] rounded-xl border border-[#C7F59D]">
                        <Check className="w-3.5 h-3.5 text-[#28594F] shrink-0" />
                        <span>{suggestions.length} text suggestions ready in the text card below</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Enter / Edit Meme Text Card (Multimodal and Text Only modes) */}
        {activeMode !== 'image' && (
          <div className="bg-white rounded-3xl border border-[#E3E8DE] p-6 sm:p-7 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-extrabold text-[#172923] flex items-center gap-2">
                <FileText className="w-4 h-4 text-[#28594F]" />
                <span>{activeMode === 'text' ? 'Meme Text Input' : 'Editable Meme Text'}</span>
              </h3>
              <span className="text-xs text-[#8D9D95] font-mono">{memeText.length} characters</span>
            </div>

            {/* Multimodal Status Strips */}
            {activeMode === 'multimodal' && isOcrLoading && (
              <div className="p-3 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] flex items-center gap-2.5 text-xs text-[#28594F] font-semibold animate-pulse">
                <RefreshCw className="w-4 h-4 text-[#28594F] animate-spin shrink-0" />
                <span>Reading text from image via OCR...</span>
              </div>
            )}

            {activeMode === 'multimodal' && isSuggesting && (
              <div className="p-3 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] flex items-center gap-2.5 text-xs text-[#28594F] font-semibold animate-pulse">
                <Sparkles className="w-4 h-4 text-[#28594F] animate-spin shrink-0" />
                <span>Analyzing visual content for Hinglish suggestions...</span>
              </div>
            )}

            {activeMode === 'multimodal' && !isOcrLoading && ocrStatus === 'success' && (
              <div className="p-3 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] flex items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2 text-[#214B43] font-bold">
                  <Check className="w-4 h-4 text-[#28594F] shrink-0" />
                  <span>Text extracted via OCR {ocrConfidence ? `(${Math.round(ocrConfidence * 100)}% conf)` : ''}</span>
                </div>
                <button
                  type="button"
                  onClick={handleExtractOcr}
                  className="text-[#28594F] hover:text-[#172923] font-bold text-[11px] underline flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Run Again</span>
                </button>
              </div>
            )}

            {activeMode === 'multimodal' && !isOcrLoading && ocrStatus === 'empty' && (
              <div className="p-3 rounded-2xl bg-amber-50 border border-amber-200 flex items-center justify-between gap-2 text-xs text-amber-900">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>{ocrMessage}</span>
                </div>
                <button
                  type="button"
                  onClick={handleExtractOcr}
                  className="text-amber-800 font-bold text-[11px] underline"
                >
                  Retry OCR
                </button>
              </div>
            )}

            {activeMode === 'multimodal' && !isOcrLoading && ocrStatus === 'error' && (
              <div className="p-3 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-between gap-2 text-xs text-rose-900">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                  <span>{ocrMessage}</span>
                </div>
                <button
                  type="button"
                  onClick={handleExtractOcr}
                  className="text-rose-800 font-bold text-[11px] underline"
                >
                  Retry OCR
                </button>
              </div>
            )}

            {activeMode === 'multimodal' && !isSuggesting && suggestionStatus === 'error' && (
              <div className="p-3 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-between gap-2 text-xs text-rose-900">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                  <span>{suggestionMessage || 'Could not generate suggestions. Please enter text manually.'}</span>
                </div>
                <button
                  type="button"
                  onClick={handleSuggestMemeText}
                  className="text-rose-800 font-bold text-[11px] underline"
                >
                  Retry Suggestion
                </button>
              </div>
            )}

            {activeMode === 'multimodal' && !isSuggesting && suggestionStatus === 'empty' && (
              <div className="p-3 rounded-2xl bg-amber-50 border border-amber-200 flex items-center justify-between gap-2 text-xs text-amber-900">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>{suggestionMessage || 'No visual suggestions available for this image. You can enter text manually.'}</span>
                </div>
                <button
                  type="button"
                  onClick={handleSuggestMemeText}
                  className="text-amber-800 font-bold text-[11px] underline"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Overwrite Confirmation Notice for Existing Text */}
            {pendingTextToConfirm && (
              <div className="p-3.5 rounded-2xl bg-amber-50 border border-amber-200 space-y-2 text-xs text-amber-900 shadow-sm animate-fadeIn">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold">Text input already has content.</span>
                    <span className="block text-[#68776F] mt-0.5">
                      New {pendingTextToConfirm.title}: "{pendingTextToConfirm.text}"
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={handleConfirmReplaceText}
                    className="px-3 py-1.5 rounded-lg bg-amber-700 hover:bg-amber-800 text-white font-bold text-xs transition-colors"
                  >
                    Replace Text
                  </button>
                  <button
                    type="button"
                    onClick={handleConfirmAppendText}
                    className="px-3 py-1.5 rounded-lg bg-white hover:bg-amber-100 text-amber-900 border border-amber-300 font-bold text-xs transition-colors"
                  >
                    Append to End
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelPendingText}
                    className="px-2.5 py-1.5 text-slate-500 hover:text-slate-800 font-semibold text-xs"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* Editable Text Area */}
            <div className="relative">
              <textarea
                rows={5}
                value={memeText}
                onChange={(e) => {
                  setMemeText(e.target.value);
                  userHasEditedTextRef.current = true;
                  setErrorMessage('');
                  setActiveExampleId(null);
                  clearExplanation();
                }}
                placeholder={
                  activeMode === 'text'
                    ? "Type or paste Hinglish meme text here (e.g. 'Code bina error ke run ho gaya! 🔥')..."
                    : "Enter meme text manually, or click 'Extract Text from Image' / 'Suggest Meme Text'..."
                }
                className="w-full p-4 rounded-2xl border border-[#E3E8DE] focus:border-[#28594F] focus:ring-2 focus:ring-[#C7F59D]/50 text-sm text-[#172923] placeholder-[#8D9D95] outline-none resize-none font-sans bg-[#F7F8F5]"
              />
              {memeText && (
                <div className="text-[11px] text-[#68776F] mt-1 flex items-center justify-between">
                  <span>Editable text: will be sent to the model for sentiment prediction</span>
                  <button
                    type="button"
                    onClick={() => {
                      setMemeText('');
                      userHasEditedTextRef.current = true;
                      clearExplanation();
                    }}
                    className="text-rose-600 hover:underline font-semibold"
                  >
                    Clear text
                  </button>
                </div>
              )}
            </div>

            {/* Multimodal: Suggested Hinglish Text Section */}
            {activeMode === 'multimodal' && suggestions.length > 0 && (
              <div className="space-y-2.5 pt-3 border-t border-[#E3E8DE]">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-extrabold text-[#172923] flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#28594F]" />
                    <span>Meme-Based Hinglish Suggestions:</span>
                  </span>
                  <span className="text-[11px] text-[#68776F]">Click chip to select phrase</span>
                </div>
                {suggestionMessage && (
                  <div className="text-[11px] text-[#68776F] italic">
                    {suggestionMessage}
                  </div>
                )}
                <div className="flex flex-col gap-2">
                  {suggestions.map((sug, sIdx) => (
                    <button
                      key={sIdx}
                      type="button"
                      onClick={() => handleSelectSuggestion(sug.text)}
                      className="p-3 rounded-xl border border-[#E3E8DE] bg-[#F7F8F5] hover:bg-[#E7F6D9] hover:border-[#28594F] text-left transition-all group flex items-start justify-between gap-3 shadow-2xs"
                    >
                      <span className="text-xs text-[#172923] font-medium leading-relaxed group-hover:text-[#214B43]">
                        "{sug.text}"
                      </span>
                      <span className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full shrink-0 border ${
                        sug.source === 'dataset'
                          ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                          : sug.source === 'ocr_detected'
                          ? 'bg-sky-100 text-sky-800 border-sky-300'
                          : sug.source === 'ai_contextual'
                          ? 'bg-amber-100 text-amber-900 border-amber-300'
                          : 'bg-[#E7F6D9] text-[#28594F] border-[#C7F59D]'
                      }`}>
                        {sug.badge}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Text Only: Quick Sample Phrase Chips */}
            {activeMode === 'text' && (
              <div className="space-y-2 pt-1 border-t border-[#E3E8DE]">
                <span className="text-xs font-semibold text-[#68776F] block">
                  Sample Hinglish phrases:
                </span>
                <div className="flex flex-wrap gap-2">
                  {[
                    'Code bina error ke run ho gaya! 🔥',
                    'Teacher ne bola well done beta! 🎉',
                    'Monday morning struggle baki hai ⏰',
                    'Finally assignment complete ho gaya! 😍'
                  ].map((phrase) => (
                    <button
                      key={phrase}
                      type="button"
                      onClick={() => {
                        setMemeText(phrase);
                        userHasEditedTextRef.current = true;
                        setActiveExampleId(null);
                        setErrorMessage('');
                        clearExplanation();
                      }}
                      className="text-xs px-3 py-1.5 rounded-full bg-[#F7F8F5] hover:bg-[#E7F6D9] text-[#172923] hover:text-[#28594F] border border-[#E3E8DE] transition-colors font-medium"
                    >
                      "{phrase}"
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>

      {/* Error Message Card */}
      {errorMessage && (
        <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs sm:text-sm flex items-center gap-3 shadow-sm">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
          <span className="font-medium">{errorMessage}</span>
        </div>
      )}

      {/* Analyze Sentiment Button */}
      <div className="flex items-center justify-end gap-3 pt-2">
        {(imagePreviewUrl || memeText) && (
          <button
            type="button"
            onClick={handleClearAll}
            className="px-6 py-3 rounded-full border border-[#C7D4CD] hover:bg-[#EAEFE6] text-[#172923] text-sm font-bold transition-all"
          >
            Clear
          </button>
        )}

        <button
          onClick={handleAnalyzeSentiment}
          disabled={isAnalyzing}
          className={`px-8 py-3.5 rounded-full font-extrabold text-sm flex items-center justify-center gap-2.5 transition-all shadow-md ${
            isAnalyzing
              ? 'bg-[#C7D4CD] text-[#68776F] cursor-not-allowed'
              : 'bg-[#C7F59D] hover:bg-[#B5EA87] text-[#172923] hover:scale-[1.02] hover:shadow-lg'
          }`}
        >
          {isAnalyzing ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin text-[#172923]" />
              <span>Analyzing Sentiment...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 text-[#172923]" />
              <span>Analyze Sentiment</span>
            </>
          )}
        </button>
      </div>

      {/* =====================================================================
          PREDICTION RESULTS SECTION (Meme Sentiment Result, Confidence, Latency)
          ===================================================================== */}
      {analysisResult && !isAnalyzing && (
        <div 
          ref={resultsRef} 
          className="bg-white rounded-3xl border border-[#E3E8DE] p-7 sm:p-9 space-y-7 shadow-lg"
        >
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#EAEFE6] pb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-3 h-3 rounded-full bg-[#C7F59D] border-2 border-[#28594F]" />
              <h3 className="text-xl font-extrabold text-[#172923]">
                Prediction Result
              </h3>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-[#68776F]">
              <span className="px-3 py-1 rounded-full bg-[#F7F8F5] border border-[#E3E8DE]">
                <strong>Analysis Mode:</strong> {currentModel.typeLabel}
              </span>
              <span className="px-3 py-1 rounded-full bg-[#E7F6D9] text-[#28594F] font-bold border border-[#C7F59D]">
                <strong>Processing Time:</strong> {analysisResult.total_time_ms} ms
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
            
            {/* Input Preview Thumbnail */}
            <div className="md:col-span-4 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-[#68776F] block">
                Analyzed Input
              </span>
              <div className="rounded-2xl overflow-hidden border border-[#E3E8DE] bg-[#183832] aspect-square max-w-[240px] flex items-center justify-center p-2">
                {imagePreviewUrl ? (
                  <img src={imagePreviewUrl} alt="Analyzed Meme" className="max-h-[220px] w-auto object-contain rounded-xl" />
                ) : (
                  <div className="p-4 text-center text-[#A3B8B0] text-xs">
                    <FileText className="w-10 h-10 mx-auto mb-2 text-[#C7F59D]" />
                    <span className="font-semibold text-white">Text-Only Input</span>
                  </div>
                )}
              </div>
              {memeText && (
                <p className="text-xs text-[#172923] italic bg-[#F7F8F5] p-3 rounded-xl border border-[#E3E8DE] leading-relaxed">
                  "{memeText}"
                </p>
              )}
            </div>

            {/* Prediction Breakdown & Confidence Progress Bars */}
            <div className="md:col-span-8 space-y-6">
              
              {/* Highlight Card for Predicted Sentiment */}
              {(() => {
                const badge = getSentimentBadge(analysisResult.predicted_sentiment);
                const Icon = badge.icon;
                const topScore = analysisResult.confidence_scores[analysisResult.predicted_sentiment] || 0;
                return (
                  <div className="flex flex-wrap items-center justify-between gap-4 p-5 rounded-2xl bg-[#F7F8F5] border border-[#E3E8DE]">
                    <div>
                      <span className="text-xs text-[#68776F] block font-semibold">
                        Predicted Sentiment:
                      </span>
                      <div className="flex items-center gap-3 mt-1.5">
                        <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-base font-extrabold border ${badge.pill}`}>
                          <Icon className="w-5 h-5" />
                          <span>{analysisResult.predicted_sentiment}</span>
                        </span>
                        <span className="text-sm font-bold text-[#172923]">
                          ({topScore}% confidence)
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-[11px] text-[#8D9D95] block font-medium">Model:</span>
                      <span className="text-xs font-bold text-[#28594F]">
                        {analysisResult.model_name || currentModel.name}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Confidence Score Distribution */}
              <div className="space-y-4">
                <span className="text-xs font-bold uppercase tracking-wider text-[#68776F] block">
                  Confidence Score Distribution:
                </span>

                {/* Positive Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-[#172923] font-semibold">
                    <span className="flex items-center gap-1.5">
                      <Smile className="w-4 h-4 text-[#28594F]" />
                      <span>Positive</span>
                    </span>
                    <span className="font-extrabold font-mono text-[#28594F]">
                      {analysisResult.confidence_scores.Positive}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-[#EAEFE6] rounded-full overflow-hidden">
                    <div 
                      style={{ width: `${Math.min(100, analysisResult.confidence_scores.Positive)}%` }} 
                      className="h-full bg-[#C7F59D] rounded-full transition-all duration-700"
                    />
                  </div>
                </div>

                {/* Neutral Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-[#172923] font-semibold">
                    <span className="flex items-center gap-1.5">
                      <Meh className="w-4 h-4 text-amber-600" />
                      <span>Neutral</span>
                    </span>
                    <span className="font-extrabold font-mono text-amber-700">
                      {analysisResult.confidence_scores.Neutral}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-[#EAEFE6] rounded-full overflow-hidden">
                    <div 
                      style={{ width: `${Math.min(100, analysisResult.confidence_scores.Neutral)}%` }} 
                      className="h-full bg-amber-400 rounded-full transition-all duration-700"
                    />
                  </div>
                </div>

                {/* Negative Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-[#172923] font-semibold">
                    <span className="flex items-center gap-1.5">
                      <Frown className="w-4 h-4 text-rose-600" />
                      <span>Negative</span>
                    </span>
                    <span className="font-extrabold font-mono text-rose-700">
                      {analysisResult.confidence_scores.Negative}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-[#EAEFE6] rounded-full overflow-hidden">
                    <div 
                      style={{ width: `${Math.min(100, analysisResult.confidence_scores.Negative)}%` }} 
                      className="h-full bg-rose-500 rounded-full transition-all duration-700"
                    />
                  </div>
                </div>

              </div>

              {/* Emotion Detection Section (Dominant Emotion + Optional Expandable 6-Emotion Breakdown) */}
              {analysisResult.emotion?.is_available && (
                <div className="p-5 rounded-2xl bg-[#F7F8F5] border border-[#C7F59D] space-y-4 shadow-sm">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="w-11 h-11 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] flex items-center justify-center text-2xl shadow-sm">
                        {analysisResult.emotion.dominant_emoji || '😊'}
                      </div>
                      <div>
                        <span className="text-[11px] font-bold uppercase tracking-wider text-[#68776F] block">
                          Dominant Emotion
                        </span>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-lg font-extrabold text-[#214B43]">
                            {analysisResult.emotion.dominant_emotion}
                          </span>
                          <span className="px-2.5 py-0.5 rounded-full text-xs font-extrabold bg-[#E7F6D9] text-[#28594F] border border-[#C7F59D]">
                            {analysisResult.emotion.dominant_probability}%
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Optional Expandable Accordion Toggle */}
                    {analysisResult.emotion.probabilities && (
                      <button
                        type="button"
                        onClick={() => setShowEmotionBreakdown(!showEmotionBreakdown)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold text-[#28594F] bg-white border border-[#E3E8DE] hover:border-[#28594F] hover:bg-[#E7F6D9] transition-all self-start sm:self-center shadow-xs"
                      >
                        <span>{showEmotionBreakdown ? 'Hide Emotion Breakdown' : 'Detailed Emotion Analysis (6 Emotions)'}</span>
                        {showEmotionBreakdown ? (
                          <ChevronUp className="w-4 h-4 text-[#28594F]" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-[#28594F]" />
                        )}
                      </button>
                    )}
                  </div>

                  {/* Expandable 6-Emotion Breakdown */}
                  {showEmotionBreakdown && analysisResult.emotion.probabilities && (
                    <div className="pt-3 border-t border-[#EAEFE6] space-y-3">
                      <div className="flex items-center justify-between text-[11px] text-[#68776F]">
                        <span className="font-semibold">Independent Multi-Label Sigmoid Probabilities:</span>
                        <span className="text-[#8D9D95]">Multi-task emotion classification head</span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                        {Object.entries(analysisResult.emotion.probabilities).map(([emotionName, score]) => {
                          const isDominant = emotionName === analysisResult.emotion.dominant_emotion;
                          const emojiMap = {
                            Happiness: '😊',
                            Sadness: '😢',
                            Anger: '😡',
                            Fear: '😨',
                            Disgust: '🤢',
                            Surprise: '😲'
                          };
                          const emoji = emojiMap[emotionName] || '🎭';
                          return (
                            <div
                              key={emotionName}
                              className={`p-3 rounded-xl border transition-all ${
                                isDominant
                                  ? 'bg-[#E7F6D9] border-[#C7F59D] ring-1 ring-[#C7F59D]'
                                  : 'bg-white border-[#E3E8DE]'
                              }`}
                            >
                              <div className="flex justify-between items-center text-xs font-semibold text-[#172923] mb-1.5">
                                <span className="flex items-center gap-1.5">
                                  <span className="text-base">{emoji}</span>
                                  <span className={isDominant ? 'font-bold text-[#28594F]' : ''}>{emotionName}</span>
                                  {isDominant && (
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#C7F59D] text-[#172923] font-bold">
                                      Dominant
                                    </span>
                                  )}
                                </span>
                                <span className="font-extrabold font-mono text-[#28594F]">
                                  {score}%
                                </span>
                              </div>
                              <div className="w-full h-2 bg-[#EAEFE6] rounded-full overflow-hidden">
                                <div
                                  style={{ width: `${Math.min(100, score)}%` }}
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    isDominant ? 'bg-[#28594F]' : 'bg-[#C7F59D]'
                                  }`}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Timing Latency Card */}
              <div className="p-4 rounded-2xl bg-[#E7F6D9] border border-[#C7F59D] flex items-center justify-between text-xs">
                <span className="font-bold text-[#28594F] flex items-center gap-2">
                  <Clock className="w-4 h-4 text-[#28594F]" />
                  <span>Total Processing Time</span>
                </span>
                <span className="font-extrabold font-mono text-sm text-[#28594F]">
                  {analysisResult.total_time_ms} ms
                </span>
              </div>

            </div>

          </div>

          {/* =================================================================
              ACTION BUTTONS: "Why This Prediction?" and "Download PDF Report"
              ================================================================= */}
          <div className="pt-4 border-t border-[#EAEFE6] flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              {/* "Why This Prediction?" Button (On-demand Explainable AI) */}
              <button
                type="button"
                onClick={handleWhyThisPrediction}
                disabled={isExplaining}
                className="px-5 py-2.5 rounded-full bg-[#214B43] hover:bg-[#28594F] text-white text-xs sm:text-sm font-bold flex items-center gap-2 transition-all shadow-sm hover:scale-[1.01]"
              >
                {isExplaining ? (
                  <>
                    <RefreshCw className="w-4 h-4 text-[#C7F59D] animate-spin" />
                    <span>Understanding the prediction...</span>
                  </>
                ) : (
                  <>
                    <HelpCircle className="w-4 h-4 text-[#C7F59D]" />
                    <span>Why This Prediction?</span>
                    {showExplanation && explanationData ? (
                      <ChevronUp className="w-3.5 h-3.5 ml-1 text-[#C7F59D]" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5 ml-1 text-[#C7F59D]" />
                    )}
                  </>
                )}
              </button>

              {/* Download PDF Report Button */}
              <button
                type="button"
                onClick={handleDownloadPdf}
                disabled={isDownloadingPdf}
                className="px-5 py-2.5 rounded-full border border-[#28594F] hover:bg-[#E7F6D9] text-[#214B43] text-xs sm:text-sm font-bold flex items-center gap-2 transition-all shadow-sm"
              >
                {isDownloadingPdf ? (
                  <>
                    <RefreshCw className="w-4 h-4 text-[#214B43] animate-spin" />
                    <span>Preparing PDF...</span>
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4 text-[#28594F]" />
                    <span>Download PDF Report</span>
                  </>
                )}
              </button>
            </div>

            <div className="text-xs text-[#68776F]">
              {explanationData ? (
                <span className="text-[#28594F] font-semibold flex items-center gap-1">
                  <Check className="w-3.5 h-3.5 text-[#28594F]" /> Explanation available
                </span>
              ) : (
                <span>Click "Why This Prediction?" to see Grad-CAM &amp; word weights</span>
              )}
            </div>
          </div>

          {/* =================================================================
              ON-DEMAND EXPLANATION CARD (Visible ONLY when requested)
              ================================================================= */}
          {showExplanation && (
            <div ref={explanationRef} className="pt-4 border-t border-[#EAEFE6] space-y-6">
              
              {isExplaining && (
                <div className="p-8 rounded-3xl bg-[#F7F8F5] border border-[#E3E8DE] text-center space-y-3">
                  <RefreshCw className="w-8 h-8 text-[#28594F] animate-spin mx-auto" />
                  <p className="text-sm font-extrabold text-[#172923]">
                    Understanding the prediction...
                  </p>
                  <p className="text-xs text-[#68776F] max-w-md mx-auto">
                    Computing Grad-CAM visual attention heatmaps, word attributions, and dynamic gating weights.
                  </p>
                </div>
              )}

              {explainError && !isExplaining && (
                <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-600 shrink-0" />
                  <span className="font-medium">{explainError}</span>
                </div>
              )}

              {explanationData && !isExplaining && (
                <div className="bg-[#F7F8F5] rounded-3xl border border-[#C7F59D] p-6 sm:p-7 space-y-6">
                  
                  {/* Explanation Title & Summary */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E3E8DE] pb-4">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-[#28594F]" />
                      <h4 className="text-base font-extrabold text-[#172923]">
                        Explainable AI Insights
                      </h4>
                      <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-[#E7F6D9] text-[#28594F] font-bold border border-[#C7F59D]">
                        Calculated in {explanationData.execution_time_ms} ms
                      </span>
                    </div>
                    <span className="text-xs text-[#68776F]">
                      Target Explained: <strong>{explanationData.predicted_sentiment}</strong>
                    </span>
                  </div>

                  <p className="text-xs sm:text-sm text-[#172923] font-medium bg-white p-4 rounded-2xl border border-[#E3E8DE]">
                    {explanationData.summary}
                  </p>

                  {/* 1. Gated Multimodal Fusion Gate Weights */}
                  {explanationData.gate_weights && (
                    <div className="bg-white p-5 rounded-2xl border border-[#E3E8DE] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-[#172923] flex items-center gap-1.5">
                          <SlidersHorizontal className="w-4 h-4 text-[#28594F]" />
                          <span>Smart Information Mixing (Dynamic Gate Weights)</span>
                        </span>
                        <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#E7F6D9] text-[#28594F] font-extrabold">
                          Dominant: {explanationData.gate_weights.dominant_modality}
                        </span>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-xs font-mono font-bold text-[#172923]">
                          <span className="text-[#214B43]">Text Weight: {explanationData.gate_weights.text_weight}%</span>
                          <span className="text-[#28594F]">Image Weight: {explanationData.gate_weights.image_weight}%</span>
                        </div>
                        <div className="w-full h-3.5 bg-[#EAEFE6] rounded-full overflow-hidden flex">
                          <div 
                            style={{ width: `${explanationData.gate_weights.text_weight}%` }} 
                            className="bg-[#214B43] h-full transition-all duration-700" 
                            title={`Text: ${explanationData.gate_weights.text_weight}%`}
                          />
                          <div 
                            style={{ width: `${explanationData.gate_weights.image_weight}%` }} 
                            className="bg-[#C7F59D] h-full transition-all duration-700" 
                            title={`Image: ${explanationData.gate_weights.image_weight}%`}
                          />
                        </div>
                      </div>
                      <p className="text-[11px] text-[#68776F]">
                        {explanationData.gate_weights.explanation}
                      </p>
                    </div>
                  )}

                  {/* 2. Influential Text Words */}
                  {explanationData.word_attributions && explanationData.word_attributions.length > 0 && (
                    <div className="bg-white p-5 rounded-2xl border border-[#E3E8DE] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-[#172923] flex items-center gap-1.5">
                          <FileText className="w-4 h-4 text-[#28594F]" />
                          <span>Influential Meme Words (Token Gradient Attributions)</span>
                        </span>
                        <span className="text-[11px] text-[#8D9D95]">Relative Importance</span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                        {explanationData.word_attributions.map((item, idx) => (
                          <div key={idx} className="p-2.5 rounded-xl bg-[#F7F8F5] border border-[#E3E8DE] space-y-1.5">
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-extrabold text-[#172923]">
                                "{item.word}"
                              </span>
                              <span className="font-mono text-[11px] font-bold text-[#28594F]">
                                {Math.round(item.score * 100)}%
                              </span>
                            </div>
                            <div className="w-full h-2 bg-[#EAEFE6] rounded-full overflow-hidden">
                              <div
                                style={{ width: `${Math.round(item.score * 100)}%` }}
                                className="h-full bg-[#28594F] rounded-full"
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 3. Grad-CAM Image Attention Heatmap */}
                  {explanationData.heatmap_image && (
                    <div className="bg-white p-5 rounded-2xl border border-[#E3E8DE] space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-[#172923] flex items-center gap-1.5">
                          <Eye className="w-4 h-4 text-[#28594F]" />
                          <span>Important Image Areas (Grad-CAM Heatmap)</span>
                        </span>
                        <span className="text-[11px] text-[#8D9D95]">ResNet-50 Layer4</span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
                        <div className="space-y-1.5 text-center">
                          <span className="text-[11px] font-bold text-[#68776F] block">Grad-CAM Attention Heatmap</span>
                          <div className="rounded-2xl overflow-hidden border border-[#E3E8DE] bg-[#183832] aspect-square max-w-[260px] mx-auto flex items-center justify-center p-1">
                            <img 
                              src={explanationData.heatmap_image} 
                              alt="Grad-CAM Heatmap" 
                              className="max-h-[250px] w-auto object-contain rounded-xl"
                            />
                          </div>
                        </div>

                        {imagePreviewUrl && (
                          <div className="space-y-1.5 text-center">
                            <span className="text-[11px] font-bold text-[#68776F] block">Original Uploaded Image</span>
                            <div className="rounded-2xl overflow-hidden border border-[#E3E8DE] bg-[#183832] aspect-square max-w-[260px] mx-auto flex items-center justify-center p-1">
                              <img 
                                src={imagePreviewUrl} 
                                alt="Original Meme" 
                                className="max-h-[250px] w-auto object-contain rounded-xl"
                              />
                            </div>
                          </div>
                        )}
                      </div>

                      <p className="text-[11px] text-[#68776F] pt-1">
                        Warm regions (red/yellow) indicate where the vision network detected decisive visual clues for this meme.
                      </p>
                    </div>
                  )}

                </div>
              )}

            </div>
          )}

        </div>
      )}

    </div>
  );
}
