/**
 * IndicSent-M Frontend API Service
 * Centralized HTTP client communicating with the FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  constructor(message, status = 500, code = 'ERROR') {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

/**
 * Parses HTTP error response safely into an ApiError.
 */
async function handleResponseError(response) {
  let message = `Server returned status ${response.status} (${response.statusText})`;
  let code = 'HTTP_ERROR';

  try {
    const errorData = await response.json();
    if (errorData?.error?.message) {
      message = errorData.error.message;
      code = errorData.error.code || 'BAD_REQUEST';
    } else if (errorData?.detail) {
      if (Array.isArray(errorData.detail)) {
        message = errorData.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
      } else {
        message = String(errorData.detail);
      }
      code = 'VALIDATION_ERROR';
    }
  } catch {
    // If not valid JSON, retain default status text
  }

  throw new ApiError(message, response.status, code);
}

/**
 * Safe fetch wrapper that handles network disconnects gracefully.
 */
async function safeFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      await handleResponseError(response);
    }
    return await response.json();
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network level error (server offline, CORS blocked, etc.)
    throw new ApiError(
      `Cannot connect to IndicSent-M backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`,
      0,
      'NETWORK_ERROR'
    );
  }
}

/**
 * 1. Check Backend Health
 * GET /api/health
 */
export async function checkBackendHealth() {
  return await safeFetch('/api/health');
}

/**
 * 2. Fetch Supported Models Catalog
 * GET /api/models
 */
export async function fetchSupportedModels() {
  return await safeFetch('/api/models');
}

/**
 * 3. Text Sentiment Prediction
 * POST /api/predict/text
 */
export async function predictText(text, model = null) {
  const payload = { text };
  if (model) {
    payload.model = model;
  }

  return await safeFetch('/api/predict/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

/**
 * 4. Image Sentiment Prediction
 * POST /api/predict/image
 */
export async function predictImage(imageFile, model = null) {
  const formData = new FormData();
  formData.append('file', imageFile);
  if (model) {
    formData.append('model', model);
  }

  return await safeFetch('/api/predict/image', {
    method: 'POST',
    body: formData,
  });
}

/**
 * 5. Multimodal Sentiment Prediction
 * POST /api/predict/multimodal
 */
export async function predictMultimodal(text, imageFile, model = null) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('file', imageFile);
  if (model) {
    formData.append('model', model);
  }

  return await safeFetch('/api/predict/multimodal', {
    method: 'POST',
    body: formData,
  });
}

/**
 * 6. Extract OCR Text from Meme Image
 * POST /api/ocr/extract
 */
export async function extractOcrText(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  return await safeFetch('/api/ocr/extract', {
    method: 'POST',
    body: formData,
  });
}

/**
 * 6b. Suggest Meme Text
 * POST /api/suggestions/meme-text
 */
export async function suggestMemeText(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  return await safeFetch('/api/suggestions/meme-text', {
    method: 'POST',
    body: formData,
  });
}

/**
 * 7. On-Demand Explainable AI (Why This Prediction?)
 * POST /api/explain
 */
export async function explainPrediction({ mode, model = null, text = null, imageFile = null, predictedSentiment = null }) {
  const formData = new FormData();
  formData.append('mode', mode || 'multimodal');
  if (model) formData.append('model', model);
  if (text) formData.append('text', text);
  if (imageFile) formData.append('file', imageFile);
  if (predictedSentiment) formData.append('predicted_sentiment', predictedSentiment);

  return await safeFetch('/api/explain', {
    method: 'POST',
    body: formData,
  });
}

/**
 * 8. Download PDF Report
 * POST /api/report/pdf
 */
export async function downloadPdfReport(reportData) {
  const formData = new FormData();
  Object.entries(reportData).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      formData.append(key, value);
    }
  });

  const response = await fetch(`${API_BASE_URL}/api/report/pdf`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new ApiError(`Failed to generate PDF report (status ${response.status})`, response.status);
  }

  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = `IndicSent-M_Report_${Date.now()}.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(downloadUrl);
}

export { API_BASE_URL, ApiError };
