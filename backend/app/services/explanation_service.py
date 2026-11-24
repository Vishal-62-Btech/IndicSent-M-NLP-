"""
IndicSent-M: Explainable AI Service
===================================
Generates mathematically faithful model explanations on-demand:
1. Multimodal Gated Fusion: Dynamic Attention Gate weights, Grad-CAM visual heatmaps, and token gradient attributions.
2. Classical Models (TF-IDF + Logistic Regression): Feature coefficient contributions for the predicted class.
3. Image Models (ResNet-50): Convolutional Grad-CAM attention heatmaps.
4. Transformer Text Models (MuRIL / XLM-RoBERTa): Word-level gradient attributions.

Zero explanation computation is performed during normal inference.
"""

import time
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from app.schemas.explanation import (
    ExplanationResponse,
    ExplanationData,
    WordAttribution,
    GateWeights
)
from app.services.predictor_service import get_predictor_service

logger = logging.getLogger("indicsent.explain")

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}


class ExplanationService:
    _instance: Optional["ExplanationService"] = None

    def __init__(self):
        self.predictor_service = get_predictor_service()

    @classmethod
    def get_instance(cls) -> "ExplanationService":
        if cls._instance is None:
            cls._instance = ExplanationService()
        return cls._instance

    def explain(
        self,
        mode: str,
        model_id: Optional[str] = None,
        text: Optional[str] = None,
        image_path: Optional[Path] = None,
        target_sentiment: Optional[str] = None
    ) -> ExplanationResponse:
        """
        Executes on-demand explanation pipeline for the given input and model.
        Returns ExplanationResponse containing genuine Grad-CAM, token attributions, and gate weights.
        """
        start_time = time.perf_counter()
        mode_clean = (mode or "multimodal").lower().strip()

        try:
            if mode_clean == "multimodal":
                return self._explain_multimodal(model_id, text, image_path, target_sentiment, start_time)
            elif mode_clean in ("text", "transformer_text"):
                return self._explain_text(model_id, text, target_sentiment, start_time)
            elif mode_clean in ("image", "vision"):
                return self._explain_image(model_id, image_path, target_sentiment, start_time)
            else:
                return ExplanationResponse(
                    success=False,
                    explanation=None,
                    message=f"Explanation is not supported for mode '{mode}'."
                )
        except Exception as e:
            logger.error(f"Explanation calculation failed: {e}", exc_info=True)
            return ExplanationResponse(
                success=False,
                explanation=None,
                message=f"Could not calculate explanation: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # 1. Multimodal Explanation (Gated Fusion & Concat)
    # -------------------------------------------------------------------------
    def _explain_multimodal(
        self,
        model_id: Optional[str],
        text: Optional[str],
        image_path: Optional[Path],
        target_sentiment: Optional[str],
        start_time: float
    ) -> ExplanationResponse:
        if not text or not text.strip():
            return ExplanationResponse(
                success=False,
                explanation=None,
                message="Text is required for multimodal explanation."
            )
        if not image_path or not image_path.is_file():
            return ExplanationResponse(
                success=False,
                explanation=None,
                message="Valid image file is required for multimodal explanation."
            )

        target_model = model_id or "gated_fusion"
        predictor = self.predictor_service._get_or_create_predictor("multimodal", target_model)
        model = predictor.model
        device = predictor.device
        tokenizer = predictor.tokenizer
        transform = predictor.transform

        # Load image
        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            orig_w, orig_h = rgb_img.size

        # Tokenize text
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        img_tensor = transform(rgb_img).unsqueeze(0).to(device)

        # Enable gradients for explanation hooks
        for p in model.parameters():
            p.requires_grad = True

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []
        emb_grads: List[torch.Tensor] = []

        # Hook layer4 of vision backbone if available (ResNet-50)
        has_vision_cam = hasattr(model, "vision_features") and len(model.vision_features) > 7
        cam_hook_fwd = None
        cam_hook_bwd = None
        if has_vision_cam:
            target_layer = model.vision_features[7]
            def fwd_hook(mod, inp, out):
                activations.append(out.detach())
            def bwd_hook(mod, gin, gout):
                gradients.append(gout[0].detach())
            cam_hook_fwd = target_layer.register_forward_hook(fwd_hook)
            cam_hook_bwd = target_layer.register_full_backward_hook(bwd_hook)

        # Hook text word embeddings
        emb_hook = None
        if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings") and hasattr(model.text_model.embeddings, "word_embeddings"):
            def text_bwd_hook(mod, gin, gout):
                emb_grads.append(gout[0].detach())
            emb_hook = model.text_model.embeddings.word_embeddings.register_full_backward_hook(text_bwd_hook)

        # Model forward
        logits = model(input_ids, attention_mask, img_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_class = logits.argmax(dim=-1).item()
        if target_sentiment and target_sentiment in LABEL_TO_ID:
            target_class = LABEL_TO_ID[target_sentiment]
        else:
            target_class = pred_class

        predicted_label = LABEL_MAP.get(target_class, "Neutral")

        # Gated fusion attention weights if model is GatedMultimodalClassifier
        gate_info = None
        if hasattr(model, "gate") and hasattr(model, "proj_text") and hasattr(model, "proj_vision"):
            with torch.no_grad():
                t_raw = model.extract_text_feature(input_ids, attention_mask)
                v_raw = model.extract_vision_feature(img_tensor)
                h_t = model.proj_text(t_raw)
                h_v = model.proj_vision(v_raw)
                concat_tv = torch.cat([h_t, h_v], dim=-1)
                alpha = model.gate(concat_tv)
                t_wt = round(float(alpha.mean().item() * 100.0), 1)
                i_wt = round(100.0 - t_wt, 1)
                dominant = "Text" if t_wt >= 50.0 else "Image"
                gate_info = GateWeights(
                    text_weight=t_wt,
                    image_weight=i_wt,
                    dominant_modality=dominant,
                    explanation=f"The attention gate dynamically assigned {t_wt}% weight to text features and {i_wt}% to image features."
                )

        # Backpropagation for Grad-CAM & token saliency
        model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Remove hooks
        if cam_hook_fwd:
            cam_hook_fwd.remove()
        if cam_hook_bwd:
            cam_hook_bwd.remove()
        if emb_hook:
            emb_hook.remove()

        # Build Grad-CAM heatmap
        heatmap_base64 = None
        if activations and gradients:
            try:
                act = activations[0].cpu().numpy()[0]
                grad = gradients[0].cpu().numpy()[0]
                alpha_k = np.mean(grad, axis=(1, 2))
                cam = np.zeros(act.shape[1:], dtype=np.float32)
                for idx, w in enumerate(alpha_k):
                    cam += w * act[idx]
                cam = np.maximum(cam, 0)
                if cam.max() > 0:
                    cam = cam / cam.max()

                cam_resized = cv2.resize(cam, (orig_w, orig_h))
                heat_u8 = (cam_resized * 255).astype(np.uint8)
                color_map = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

                orig_np = np.array(rgb_img)
                overlay = cv2.addWeighted(orig_np, 0.55, color_map, 0.45, 0)
                heatmap_base64 = self._encode_pil_to_base64(Image.fromarray(overlay))
            except Exception as e:
                logger.warning(f"Grad-CAM rendering warning: {e}")

        # Build Word Attributions
        word_attributions = []
        if emb_grads:
            try:
                scores = emb_grads[0].norm(dim=-1).squeeze(0).cpu().numpy()
                token_ids = input_ids[0].cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                word_attributions = self._aggregate_tokens_to_words(tokens, scores)
            except Exception as e:
                logger.warning(f"Word attribution aggregation warning: {e}")

        # Summary text
        top_words = [w.word for w in word_attributions[:3]] if word_attributions else []
        summary_parts = [f"Model predicted {predicted_label} sentiment."]
        if top_words:
            summary_parts.append(f"Most influential words: {', '.join(top_words)}.")
        if gate_info:
            summary_parts.append(f"Gate emphasized {gate_info.dominant_modality.lower()} ({gate_info.text_weight if gate_info.dominant_modality == 'Text' else gate_info.image_weight}%).")
        summary_str = " ".join(summary_parts)

        exec_time = round((time.perf_counter() - start_time) * 1000.0, 2)

        return ExplanationResponse(
            success=True,
            explanation=ExplanationData(
                model_name=predictor.display_name,
                mode="multimodal",
                predicted_sentiment=predicted_label,
                predicted_class_id=target_class,
                word_attributions=word_attributions,
                heatmap_image=heatmap_base64,
                gate_weights=gate_info,
                summary=summary_str,
                execution_time_ms=exec_time
            ),
            message=None
        )

    # -------------------------------------------------------------------------
    # 2. Text Explanation (Transformer & Classical)
    # -------------------------------------------------------------------------
    def _explain_text(
        self,
        model_id: Optional[str],
        text: Optional[str],
        target_sentiment: Optional[str],
        start_time: float
    ) -> ExplanationResponse:
        if not text or not text.strip():
            return ExplanationResponse(
                success=False,
                explanation=None,
                message="Text is required for text explanation."
            )

        target_model = (model_id or "xlm_roberta").lower()

        # Classical Model (TF-IDF + Logistic Regression)
        if target_model in ("logistic_regression", "linear_svm", "multinomial_naive_bayes", "random_forest"):
            predictor = self.predictor_service._get_or_create_predictor("classical", target_model)
            model_pipe = predictor.model
            
            try:
                tfidf = model_pipe.named_steps.get("tfidf") or model_pipe.steps[0][1]
                clf = model_pipe.named_steps.get("classifier") or model_pipe.steps[-1][1]
            except Exception:
                tfidf = None
                clf = None

            if tfidf is None or clf is None or not hasattr(clf, "coef_"):
                return ExplanationResponse(
                    success=False,
                    explanation=None,
                    message=f"Feature attribution is unavailable for model '{predictor.display_name}'."
                )

            # Transform text
            feature_names = tfidf.get_feature_names_out()
            X_vec = tfidf.transform([text]).toarray()[0]
            non_zero_indices = np.where(X_vec > 0)[0]

            raw_preds = predictor.predict(text=text)
            pred_class = raw_preds["predicted_class"]
            target_class = LABEL_TO_ID.get(target_sentiment, pred_class)
            predicted_label = LABEL_MAP.get(target_class, "Neutral")

            coefs = clf.coef_[target_class] if len(clf.coef_.shape) > 1 else clf.coef_
            word_contribs = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                contrib = float(X_vec[idx] * coefs[idx])
                word_contribs.append((word, contrib))

            word_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            max_abs = max([abs(c[1]) for c in word_contribs]) if word_contribs else 1.0

            attributions = []
            for word, contrib in word_contribs[:10]:
                score = round(abs(contrib) / (max_abs + 1e-9), 3)
                influence = "Supports" if contrib >= 0 else "Opposes"
                attributions.append(WordAttribution(
                    word=word,
                    score=score,
                    raw_attribution=round(contrib, 4),
                    influence=influence
                ))

            exec_time = round((time.perf_counter() - start_time) * 1000.0, 2)
            top_words = [w.word for w in attributions[:3]]
            summary = f"Predicted {predicted_label}. Highest contributing words: {', '.join(top_words) if top_words else 'None'}."

            return ExplanationResponse(
                success=True,
                explanation=ExplanationData(
                    model_name=predictor.display_name,
                    mode="text",
                    predicted_sentiment=predicted_label,
                    predicted_class_id=target_class,
                    word_attributions=attributions,
                    heatmap_image=None,
                    gate_weights=None,
                    summary=summary,
                    execution_time_ms=exec_time
                ),
                message=None
            )

        # Transformer Text (MuRIL / XLM-RoBERTa)
        predictor = self.predictor_service._get_or_create_predictor("transformer_text", target_model)
        model = predictor.model
        tokenizer = predictor.tokenizer
        device = predictor.device

        for p in model.parameters():
            p.requires_grad = True

        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        emb_grads = []
        # Find embedding module
        emb_module = None
        if hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
            emb_module = model.roberta.embeddings.word_embeddings
        elif hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            emb_module = model.bert.embeddings.word_embeddings

        handle = None
        if emb_module:
            def bwd_hook(mod, gin, gout):
                emb_grads.append(gout[0].detach())
            handle = emb_module.register_full_backward_hook(bwd_hook)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_class = logits.argmax(dim=-1).item()
        target_class = LABEL_TO_ID.get(target_sentiment, pred_class)
        predicted_label = LABEL_MAP.get(target_class, "Neutral")

        model.zero_grad()
        logits[0, target_class].backward()
        if handle:
            handle.remove()

        word_attributions = []
        if emb_grads:
            scores = emb_grads[0].norm(dim=-1).squeeze(0).cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            word_attributions = self._aggregate_tokens_to_words(tokens, scores)

        exec_time = round((time.perf_counter() - start_time) * 1000.0, 2)
        top_words = [w.word for w in word_attributions[:3]] if word_attributions else []
        summary = f"Transformer predicted {predicted_label}. Key semantic tokens: {', '.join(top_words)}."

        return ExplanationResponse(
            success=True,
            explanation=ExplanationData(
                model_name=predictor.display_name,
                mode="text",
                predicted_sentiment=predicted_label,
                predicted_class_id=target_class,
                word_attributions=word_attributions,
                heatmap_image=None,
                gate_weights=None,
                summary=summary,
                execution_time_ms=exec_time
            ),
            message=None
        )

    # -------------------------------------------------------------------------
    # 3. Vision Explanation (ResNet-50 / EfficientNet / MobileNet)
    # -------------------------------------------------------------------------
    def _explain_image(
        self,
        model_id: Optional[str],
        image_path: Optional[Path],
        target_sentiment: Optional[str],
        start_time: float
    ) -> ExplanationResponse:
        if not image_path or not image_path.is_file():
            return ExplanationResponse(
                success=False,
                explanation=None,
                message="Valid image is required for vision explanation."
            )

        target_model = (model_id or "resnet50").lower()
        predictor = self.predictor_service._get_or_create_predictor("image", target_model)
        model = predictor.model
        device = predictor.device
        transform = predictor.transform

        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            orig_w, orig_h = rgb_img.size

        img_tensor = transform(rgb_img).unsqueeze(0).to(device)

        for p in model.parameters():
            p.requires_grad = True

        activations = []
        gradients = []

        # Find target feature layer
        target_layer = None
        if hasattr(model, "backbone"):
            # ResNet-50 Classifier
            if hasattr(model.backbone, "layer4"):
                target_layer = model.backbone.layer4
            elif hasattr(model.backbone, "features"):
                target_layer = model.backbone.features[-1]
        elif hasattr(model, "vision_features") and len(model.vision_features) > 7:
            target_layer = model.vision_features[7]

        if target_layer is None:
            return ExplanationResponse(
                success=False,
                explanation=None,
                message=f"Grad-CAM target layer is unavailable for model '{predictor.display_name}'."
            )

        def fwd_hook(mod, inp, out):
            activations.append(out.detach())
        def bwd_hook(mod, gin, gout):
            gradients.append(gout[0].detach())

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        logits = model(img_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_class = logits.argmax(dim=-1).item()
        target_class = LABEL_TO_ID.get(target_sentiment, pred_class)
        predicted_label = LABEL_MAP.get(target_class, "Neutral")

        model.zero_grad()
        logits[0, target_class].backward()
        h_fwd.remove()
        h_bwd.remove()

        heatmap_base64 = None
        if activations and gradients:
            act = activations[0].cpu().numpy()[0]
            grad = gradients[0].cpu().numpy()[0]
            weights = np.mean(grad, axis=(1, 2))
            cam = np.zeros(act.shape[1:], dtype=np.float32)
            for idx, w in enumerate(weights):
                cam += w * act[idx]
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_resized = cv2.resize(cam, (orig_w, orig_h))
            heat_u8 = (cam_resized * 255).astype(np.uint8)
            color_map = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

            orig_np = np.array(rgb_img)
            overlay = cv2.addWeighted(orig_np, 0.55, color_map, 0.45, 0)
            heatmap_base64 = self._encode_pil_to_base64(Image.fromarray(overlay))

        exec_time = round((time.perf_counter() - start_time) * 1000.0, 2)
        summary = f"Vision model predicted {predicted_label}. Grad-CAM heatmap highlights primary visual focus areas."

        return ExplanationResponse(
            success=True,
            explanation=ExplanationData(
                model_name=predictor.display_name,
                mode="image",
                predicted_sentiment=predicted_label,
                predicted_class_id=target_class,
                word_attributions=[],
                heatmap_image=heatmap_base64,
                gate_weights=None,
                summary=summary,
                execution_time_ms=exec_time
            ),
            message=None
        )

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------
    def _encode_pil_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def _aggregate_tokens_to_words(
        self,
        tokens: List[str],
        scores: np.ndarray
    ) -> List[WordAttribution]:
        """
        Reconstructs subword tokens (e.g. BERT '##dha' or RoBERTa 'Ġword') into clean words,
        combining their respective gradient norms.
        """
        word_scores: Dict[str, float] = {}
        current_word = ""
        current_score = 0.0

        special_tokens = {"[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]", "[UNK]"}

        for token, score in zip(tokens, scores):
            if token in special_tokens:
                continue

            # HuggingFace BERT / MuRIL subword: "##dha"
            if token.startswith("##"):
                current_word += token[2:]
                current_score += float(score)
            # HuggingFace RoBERTa subword: "Ġword"
            elif token.startswith("Ġ") or token.startswith(" "):
                if current_word:
                    word_scores[current_word] = word_scores.get(current_word, 0.0) + current_score
                clean_tok = token.lstrip("Ġ ").strip()
                current_word = clean_tok
                current_score = float(score)
            else:
                if current_word and not current_word.isalnum():
                    word_scores[current_word] = word_scores.get(current_word, 0.0) + current_score
                    current_word = token
                    current_score = float(score)
                elif current_word:
                    # Treat as continuation or separate
                    word_scores[current_word] = word_scores.get(current_word, 0.0) + current_score
                    current_word = token
                    current_score = float(score)
                else:
                    current_word = token
                    current_score = float(score)

        if current_word:
            word_scores[current_word] = word_scores.get(current_word, 0.0) + current_score

        # Filter out empty or pure punctuation
        cleaned_words = [
            (w, s) for w, s in word_scores.items()
            if w.strip() and any(c.isalnum() for c in w)
        ]

        if not cleaned_words:
            return []

        # Sort by importance
        cleaned_words.sort(key=lambda x: x[1], reverse=True)
        max_score = cleaned_words[0][1] if cleaned_words else 1.0

        results = []
        for word, s in cleaned_words[:12]:
            norm_score = round(s / (max_score + 1e-9), 3)
            results.append(WordAttribution(
                word=word,
                score=norm_score,
                raw_attribution=round(float(s), 4),
                influence="Supports"
            ))

        return results


def get_explanation_service() -> ExplanationService:
    return ExplanationService.get_instance()
