// Mock Data for IndicSent-M Frontend Prototype

// SVG data URLs for sample memes so they display beautifully without needing external images
const createMemeSvgUrl = (title, emoji, bgGradient, subtext) => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500" viewBox="0 0 500 500">
    <defs>
      <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${bgGradient[0]}" />
        <stop offset="100%" stop-color="${bgGradient[1]}" />
      </linearGradient>
    </defs>
    <rect width="500" height="500" fill="url(#grad)" rx="16"/>
    <circle cx="250" cy="220" r="110" fill="white" fill-opacity="0.18" />
    <text x="250" y="245" font-size="90" text-anchor="middle" font-family="sans-serif">${emoji}</text>
    <rect x="40" y="365" width="420" height="90" rx="12" fill="rgba(0,0,0,0.35)" />
    <text x="250" y="405" font-size="20" font-weight="bold" fill="white" text-anchor="middle" font-family="'Segoe UI', Roboto, sans-serif">${title}</text>
    <text x="250" y="435" font-size="15" fill="#f1f5f9" text-anchor="middle" font-family="'Segoe UI', Roboto, sans-serif">${subtext}</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
};

export const exampleMemes = [
  {
    id: 'meme-1',
    title: 'Exam Preparation',
    imageUrl: createMemeSvgUrl('Exam Preparation', '😭📚', ['#6366f1', '#8b5cf6'], 'Padhai chhod ke sab ho raha hai'),
    text: 'Padhai chhod ke sab ho raha hai life me, bas syllabus complete nahi ho raha 😭',
    suggestedSentiment: 'Negative',
    confidence: 81,
    scores: {
      positive: 7,
      negative: 81,
      neutral: 12
    },
    modalityWeights: {
      textWeight: 65,
      imageWeight: 35
    },
    emotion: {
      is_available: true,
      dominant_emotion: 'Sadness',
      dominant_probability: 78.5,
      dominant_emoji: '😢',
      probabilities: {
        Happiness: 4.2,
        Sadness: 78.5,
        Anger: 22.1,
        Fear: 35.8,
        Disgust: 12.0,
        Surprise: 8.4
      }
    },
    analysisNotes: 'The Hinglish text expresses exam stress and self-deprecating humor. Paired with a crying expression, the overall sentiment is classified as Negative.',
    tags: ['Student Life', 'Exams']
  },
  {
    id: 'meme-2',
    title: 'Assignment Submitted',
    imageUrl: createMemeSvgUrl('Assignment Submitted', '🎉🥳', ['#10b981', '#059669'], 'Well done beta! Shabash'),
    text: 'Jab assignment submit karne ke baad teacher bolti hai "Well done beta, presentation bahut accha tha!" 🥳',
    suggestedSentiment: 'Positive',
    confidence: 88,
    scores: {
      positive: 88,
      negative: 4,
      neutral: 8
    },
    modalityWeights: {
      textWeight: 58,
      imageWeight: 42
    },
    emotion: {
      is_available: true,
      dominant_emotion: 'Happiness',
      dominant_probability: 84.2,
      dominant_emoji: '😊',
      probabilities: {
        Happiness: 84.2,
        Sadness: 2.1,
        Anger: 1.5,
        Fear: 3.2,
        Disgust: 1.0,
        Surprise: 28.6
      }
    },
    analysisNotes: 'Clear positive wording ("Well done", "bahut accha tha") combined with celebration emojis and happy visual cues indicate a strong Positive sentiment.',
    tags: ['College Life', 'Success']
  },
  {
    id: 'meme-3',
    title: 'Monday Morning',
    imageUrl: createMemeSvgUrl('Monday Morning', '⏰🥱', ['#f59e0b', '#d97706'], 'Utho beta, struggle baki hai'),
    text: 'Monday morning alarm be like: utho beta abhi aur struggle baki hai tumhara',
    suggestedSentiment: 'Neutral',
    confidence: 64,
    scores: {
      positive: 11,
      negative: 25,
      neutral: 64
    },
    modalityWeights: {
      textWeight: 50,
      imageWeight: 50
    },
    emotion: {
      is_available: true,
      dominant_emotion: 'Sadness',
      dominant_probability: 46.8,
      dominant_emoji: '😢',
      probabilities: {
        Happiness: 12.3,
        Sadness: 46.8,
        Anger: 18.5,
        Fear: 14.2,
        Disgust: 8.7,
        Surprise: 11.0
      }
    },
    analysisNotes: 'Relatable everyday humor about daily routine. Without strong angry or happy emotional words, the sentiment resolves to Neutral.',
    tags: ['Relatable', 'Daily Routine']
  },
  {
    id: 'meme-4',
    title: 'Coding Success',
    imageUrl: createMemeSvgUrl('Coding Success', '🚀🔥', ['#8b5cf6', '#ec4899'], 'Code bina error ke run ho gaya!'),
    text: 'Bhai ne 200 line ka code likha aur pehli baar me hi bina kisi error ke successfully run ho gaya! 🔥',
    suggestedSentiment: 'Positive',
    confidence: 94,
    scores: {
      positive: 94,
      negative: 2,
      neutral: 4
    },
    modalityWeights: {
      textWeight: 52,
      imageWeight: 48
    },
    emotion: {
      is_available: true,
      dominant_emotion: 'Happiness',
      dominant_probability: 91.5,
      dominant_emoji: '😊',
      probabilities: {
        Happiness: 91.5,
        Sadness: 1.2,
        Anger: 0.8,
        Fear: 2.1,
        Disgust: 0.5,
        Surprise: 44.0
      }
    },
    analysisNotes: 'Words of excitement ("successfully run", "bina kisi error ke") combined with triumphant visual cues produce a very high Positive sentiment score.',
    tags: ['Coding', 'Achievement']
  }
];

export const sentimentTypes = [
  {
    type: 'Positive',
    label: 'Positive Sentiment',
    color: 'emerald',
    badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    description: 'Joy, appreciation, humor with uplifting tone, triumph, celebratory Indian meme tropes.',
    keywords: ['Shabash', 'Mast', 'Kamaal', 'Success', 'Op', 'Bahut Badhiya', 'Love it']
  },
  {
    type: 'Negative',
    label: 'Negative Sentiment',
    color: 'rose',
    badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
    description: 'Frustration, sarcasm, mock anger, existential crisis, Monday blues, exam pain memes.',
    keywords: ['Dard', 'Khatam', 'Rona aa gaya', 'Bekar', 'Barbaad', 'Dhokha', 'Struggle']
  },
  {
    type: 'Neutral',
    label: 'Neutral Sentiment',
    color: 'amber',
    badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
    description: 'Observational humor, factual meme statements, informational formats without intense emotional bias.',
    keywords: ['Normal', 'Aise hi', 'Sach baat', 'Observation', 'Common baat', 'Relatable']
  }
];

export const howItWorksSteps = [
  {
    step: '01',
    title: 'Meme Input & Ingestion',
    description: 'Upload an Indian-context meme image (PNG/JPG) along with transcribed or embedded Hinglish caption text.',
    icon: 'UploadCloud',
    badge: 'Dual Inputs'
  },
  {
    step: '02',
    title: 'Cross-Modal Feature Extraction',
    description: 'ResNet-50 extracts 2048-d spatial visual features; XLM-RoBERTa extracts 768-d contextual multilingual embeddings.',
    icon: 'Layers',
    badge: 'Vision + NLP'
  },
  {
    step: '03',
    title: 'Feature Fusion Mechanism',
    description: 'Cross-modal concatenation and attention gate merge textual nuances with facial gestures & visual context.',
    icon: 'GitMerge',
    badge: 'Intermediate Fusion'
  },
  {
    step: '04',
    title: 'Sentiment Classification',
    description: 'Dense MLP layers with dropout and Softmax output probability distribution across Positive, Negative, and Neutral.',
    icon: 'CheckCircle2',
    badge: 'Classification Head'
  }
];

export const architectureComponents = [
  {
    name: 'ResNet-50 (Vision Backbone)',
    type: 'Visual Modality',
    badge: '2048-d Feature Vector',
    color: 'from-blue-500 to-indigo-600',
    icon: 'Image',
    role: 'Deep Convolutional Residual Network pretrained on ImageNet. Processes 224x224 RGB meme graphics and analyzes image features and visual patterns to predict meme sentiment.',
    specs: [
      { label: 'Architecture', value: '50-layer Residual ConvNet' },
      { label: 'Input Resolution', value: '224 × 224 × 3' },
      { label: 'Output Dimension', value: '2,048 feature vector' },
      { label: 'Pretraining', value: 'ImageNet-1k weights' }
    ]
  },
  {
    name: 'XLM-RoBERTa (Language Backbone)',
    type: 'Textual Modality',
    badge: '768-d Token Embeddings',
    color: 'from-purple-500 to-violet-600',
    icon: 'FileText',
    role: 'Multilingual masked language model pretrained on 2.5TB of CommonCrawl data in 100 languages. Robustly handles Hinglish code-switching, Romanized Hindi phonetics, and informal colloquialisms.',
    specs: [
      { label: 'Architecture', value: '12-layer Transformer (base)' },
      { label: 'Max Token Length', value: '128 tokens' },
      { label: 'Output Dimension', value: '768-d [CLS] pooled representation' },
      { label: 'Language Coverage', value: '100+ languages including Hindi/Indic' }
    ]
  },
  {
    name: 'Multimodal Fusion Layer',
    type: 'Fusion Mechanism',
    badge: '2816-d Merged Latent Space',
    color: 'from-fuchsia-500 to-pink-600',
    icon: 'Combine',
    role: 'Combines visual cues (2048-d) and textual cues (768-d) through feature concatenation and cross-attention gating, solving visual-textual incongruity (e.g. happy face + sad Hinglish text).',
    specs: [
      { label: 'Fusion Strategy', value: 'Intermediate Feature Concatenation' },
      { label: 'Input Vector Size', value: '2,048 + 768 = 2,816 dimensions' },
      { label: 'Normalization', value: 'LayerNorm + Dropout (p=0.3)' },
      { label: 'Attention Mechanism', value: 'Cross-Modal Scaled Dot-Product' }
    ]
  },
  {
    name: 'Classification Head (MLP)',
    type: 'Prediction Head',
    badge: '3-Class Probability',
    color: 'from-emerald-500 to-teal-600',
    icon: 'PieChart',
    role: 'Two-stage fully connected Multi-Layer Perceptron that projects the fused 2816-d representation into a 3-way Softmax distribution representing Positive, Negative, and Neutral sentiment.',
    specs: [
      { label: 'Hidden Layers', value: 'Linear(2816 → 512) → ReLU → Linear(512 → 3)' },
      { label: 'Activation', value: 'Softmax / Cross-Entropy Loss' },
      { label: 'Target Classes', value: 'Positive, Negative, Neutral' },
      { label: 'Evaluation Target', value: 'Macro F1-Score & Accuracy' }
    ]
  }
];

export const modelComparisonData = [
  {
    id: 'text-only',
    name: 'Text-Only Transformer',
    model: 'XLM-RoBERTa (Fine-Tuned)',
    inputModality: 'Hinglish OCR Text only',
    accuracy: '42.36%',
    precision: '42.15%',
    recall: '42.36%',
    f1Score: '41.66%',
    status: 'Trained & Evaluated',
    statusBadge: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    parameters: '125 Million',
    hypothesis: 'Excels at understanding Romanized Hindi slang and colloquialisms, but blind to visual facial cues and reaction gestures.'
  },
  {
    id: 'image-only',
    name: 'Vision-Only Classifier',
    model: 'ResNet-50 (Transfer Learning)',
    inputModality: 'Meme Image Graphic only',
    accuracy: '40.43%',
    precision: '38.20%',
    recall: '40.43%',
    f1Score: '37.04%',
    status: 'Trained & Evaluated',
    statusBadge: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    parameters: '25.6 Million',
    hypothesis: 'Captures meme reaction template and character facial expression, but blind to the written Hinglish punchline.'
  },
  {
    id: 'multimodal',
    name: 'Multimodal Gated Fusion (Champion)',
    model: 'MuRIL + ResNet-50 (Gated Fusion)',
    inputModality: 'Hinglish OCR Text + Meme Graphic',
    accuracy: '37.64%',
    precision: '37.89%',
    recall: '37.64%',
    f1Score: '37.24%',
    status: 'Trained & Deployed',
    statusBadge: 'text-purple-700 bg-purple-50 border-purple-200',
    parameters: '136 Million',
    hypothesis: 'Dynamically weighs visual and textual signals per meme instance, addressing sarcastic incongruity between text and graphics.'
  }
];

export const fullLeaderboardData = [
  { model: 'TF-IDF + Logistic Regression', modality: 'Classical Text', accuracy: 43.79, macroF1: 43.14, weightedF1: 43.98, status: 'Baseline Champion' },
  { model: 'XLM-RoBERTa (Fine-Tuned)', modality: 'Transformer Text', accuracy: 42.36, macroF1: 41.66, weightedF1: 42.50, status: 'Top Text Model' },
  { model: 'TF-IDF + Random Forest', modality: 'Classical Text', accuracy: 44.00, macroF1: 40.75, weightedF1: 42.52, status: 'Evaluated' },
  { model: 'MuRIL (Fine-Tuned)', modality: 'Transformer Text', accuracy: 40.93, macroF1: 40.75, weightedF1: 41.12, status: 'Evaluated' },
  { model: 'MuRIL + ResNet-50 (Gated Fusion)', modality: 'Multimodal', accuracy: 37.64, macroF1: 37.24, weightedF1: 37.80, status: 'Champion Multimodal' },
  { model: 'TF-IDF + Multinomial Naive Bayes', modality: 'Classical Text', accuracy: 42.86, macroF1: 37.06, weightedF1: 39.73, status: 'Evaluated' },
  { model: 'ResNet-50', modality: 'Vision Only', accuracy: 40.43, macroF1: 37.04, weightedF1: 38.90, status: 'Top Vision Model' },
  { model: 'EfficientNet-B0', modality: 'Vision Only', accuracy: 37.50, macroF1: 36.54, weightedF1: 37.20, status: 'Evaluated' },
  { model: 'MobileNetV3-Large', modality: 'Vision Only', accuracy: 38.21, macroF1: 36.25, weightedF1: 37.10, status: 'Evaluated' },
  { model: 'MuRIL + EfficientNet-B0 Concat', modality: 'Multimodal', accuracy: 36.00, macroF1: 35.43, weightedF1: 36.20, status: 'Evaluated' },
  { model: 'MuRIL + ResNet-50 Concat', modality: 'Multimodal', accuracy: 37.57, macroF1: 35.14, weightedF1: 36.80, status: 'Evaluated' },
  { model: 'XLM-RoBERTa + ResNet-50 Concat', modality: 'Multimodal', accuracy: 31.43, macroF1: 30.00, weightedF1: 30.50, status: 'Evaluated' },
  { model: 'TF-IDF + Linear SVM', modality: 'Classical Text', accuracy: 43.57, macroF1: 27.97, weightedF1: 32.58, status: 'Evaluated' },
  { model: 'Majority Class Baseline', modality: 'Baseline', accuracy: 42.43, macroF1: 19.86, weightedF1: 25.28, status: 'Evaluated' }
];

export const expectedBenchmarkTrend = [
  { metric: 'Accuracy', textOnly: 62, imageOnly: 54, multimodalProposed: 78 },
  { metric: 'Precision', textOnly: 60, imageOnly: 51, multimodalProposed: 76 },
  { metric: 'Recall', textOnly: 59, imageOnly: 53, multimodalProposed: 77 },
  { metric: 'Macro F1', textOnly: 61, imageOnly: 52, multimodalProposed: 77 }
];

export const projectInfo = {
  name: 'IndicSent-M',
  fullName: 'Multimodal Sentiment Analysis of Indian-Language Memes Using Text and Images',
  tagline: 'Understanding Emotions Behind Memes',
  domain: 'Multimodal Natural Language Processing & Computer Vision',
  level: 'Undergraduate / B.Tech Major Project Demonstration',
  dataset: {
    name: 'Memotion 3.0 Dataset',
    description: 'Benchmark dataset containing thousands of annotated Indian-context memes with Hinglish/Hindi-English text, sarcasm annotations, and sentiment polarities.',
    size: '10,000+ Annotated Memes',
    classes: '3 Classes (Positive, Negative, Neutral)',
    language: 'Code-Mixed Hinglish (Hindi written in Latin script) and Devanagari Hindi'
  },
  problemStatement: 'Memes are the dominant currency of communication on Indian social media (Instagram, X, WhatsApp, Reddit). However, analyzing sentiment in Indian memes is uniquely challenging due to: (1) Hinglish code-mixing without standardized spelling, (2) cultural sarcasm and pop-culture references (Bollywood, cricket, political satire), and (3) visual-textual incongruity where image expression deliberately contradicts text meaning. Unimodal models fail to capture this dual context.',
  objectives: [
    'Develop a dual-stream multimodal deep learning architecture utilizing XLM-RoBERTa and ResNet-50.',
    'Mitigate Hinglish code-switching ambiguity using multilingual contextual subword tokenization.',
    'Implement intermediate cross-modal feature fusion to resolve visual-textual incongruity.',
    'Establish baseline comparisons demonstrating why multimodal fusion outperforms unimodal text-only and image-only models.',
    'Provide an intuitive, responsive frontend dashboard for academic demonstrations and user evaluation.'
  ],
  plannedTechStack: [
    { category: 'Frontend Prototype', tech: 'React.js, Vite, Tailwind CSS, Lucide React, Recharts' },
    { category: 'Deep Learning Framework', tech: 'PyTorch, PyTorch Lightning' },
    { category: 'NLP Backbone', tech: 'Hugging Face Transformers (xlm-roberta-base)' },
    { category: 'Vision Backbone', tech: 'Torchvision (resnet50 pretrained weights)' },
    { category: 'Future Backend API', tech: 'FastAPI with Uvicorn (Asynchronous REST API)' },
    { category: 'Model Training & Hardware', tech: 'Google Colab Pro / NVIDIA CUDA GPUs' }
  ],
  futureScope: [
    'Real-time Optical Character Recognition (OCR) using EasyOCR/PaddleOCR for automated in-image meme text extraction.',
    'Fine-grained emotion tagging: Sarcasm intensity, offensive/hate speech detection, and humor level.',
    'Extension to South Indian languages: Tanglish (Tamil-English) and Manglish (Malayalam-English) memes.',
    'Browser extension for automated real-time social media feed sentiment scoring.'
  ]
};

export const sixEmotionsMeta = [
  { name: 'Happiness', emoji: '😊', color: 'emerald', hex: '#10B981', description: 'Joy, delight, celebration, amusement' },
  { name: 'Sadness', emoji: '😢', color: 'blue', hex: '#3B82F6', description: 'Grief, sorrow, despair, disappointment' },
  { name: 'Anger', emoji: '😡', color: 'rose', hex: '#EF4444', description: 'Fury, outrage, annoyance, frustration' },
  { name: 'Fear', emoji: '😨', color: 'purple', hex: '#8B5CF6', description: 'Panic, anxiety, terror, apprehension' },
  { name: 'Disgust', emoji: '🤢', color: 'amber', hex: '#F59E0B', description: 'Repulsion, contempt, aversion' },
  { name: 'Surprise', emoji: '😲', color: 'cyan', hex: '#06B6D4', description: 'Astonishment, shock, wonder' }
];
