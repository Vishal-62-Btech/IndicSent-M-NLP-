# IndicSent-M: Limitations, Technical Debt & Future Directions
===============================================================

> **Critical Empirical Analysis of Model Bottlenecks and Roadmap for Research**  
> *Target Benchmark: Memotion 3.0 Multimodal Affective Analysis*

---

## 1. Architectural & Representation Limitations

### 1.1 MuRIL Embedding Anisotropy & Feature Collapse
Our representation audit revealed a foundational limitation in the multimodal text branch:
- **High Cosine Similarity:** Across semantically diverse validation captions, the mean pairwise cosine similarity of raw MuRIL embeddings was **0.9976**, and after the linear projection layer, it reached **0.9999**.
- **Dead ReLU Dimensions:** **272 out of 512** dimensions in the projected text representation remained completely inactive ($0.0$) across all validation samples.
- **Root Cause:** Due to keeping the text backbone frozen (`freeze_text = True`) during standard transfer learning to prevent catastrophic forgetting on small datasets ($N=5,600$), the pre-trained weights remain tuned for formal monolingual text rather than Romanized phonetic Hinglish slang. As a result, the text projection collapses into a narrow cone in representation space.
- **Impact:** The gating network dynamically discounts the text branch in ambiguous samples, occasionally causing the model to over-rely on visual expressions.

---

## 2. Dataset Distribution & Label Imbalance Bottlenecks

### 2.1 The Inherent Bias of Internet Meme Culture
Memotion 3.0 reflects organic social media memes in India:
- **Over-Representation of Humor & Struggle:** $64.51\%$ of annotated training samples express *Happiness* (celebratory, wholesome, funny), and $22.12\%$ express *Sadness* (self-deprecating student/workplace struggles).
- **Extreme Scarcity of Disgust and Fear:** *Disgust* represents only $1.92\%$ ($30$ training memes, $2$ validation memes), and *Fear* represents $5.12\%$ ($80$ training memes, $11$ validation memes).
- **Impact on Evaluation:** Due to severe sparsity, even a single false positive on *Disgust* severely degrades precision. Achieving high Macro-F1 on minority emotions remains fundamentally bounded by sample availability.

### 2.2 Sarcasm & Cultural Contextual Nuance
A significant proportion of Indian memes rely on transient cultural events (cricket tournament controversies, trending Bollywood dialogues, regional political events). Static text-image encoders lack external knowledge graphs or temporal context to understand why a seemingly neutral image of a bowler with the caption *"Kamaal kar diya sirji"* carries biting sarcastic scorn.

---

## 3. Optical Character Recognition (OCR) Fragility

### 3.1 Stylized Meme Typography
While `EasyOCR` achieves high accuracy on standard sans-serif text, memes frequently feature:
- Heavy dropshadows, rainbow gradients, and 3D decorative typography.
- Distorted text curved along image boundaries.
- Watermarks from meme generator applications overlaid on captions.
- Partial occlusion from stickers or emojis.

These artifacts occasionally introduce misspelled phonetic words into the text pipeline (e.g., *"bhai"* transcribed as *"bha1"*). IndicSent-M mitigates this via its frontend editable caption box, but fully automated pipelines remain vulnerable to OCR transcription noise.

---

## 4. Promising Future Research Directions

### 4.1 Parameter-Efficient Fine-Tuning (PEFT / LoRA / QLoRA)
Rather than keeping MuRIL fully frozen (which causes anisotropy) or unfreezing all 110M parameters (which causes overfitting), future iterations should implement **Low-Rank Adaptation (LoRA)**:
- Inject trainable rank decomposition matrices ($r=8$ or $r=16$) into the query ($W_q$) and value ($W_v$) projections of the top 4 MuRIL transformer layers.
- Trainable parameters are reduced by $>95\%$, while enabling the language model to adapt its geometric representation space to colloquial Hinglish without representation collapse.

### 4.2 Indic Multimodal Pre-Training (IndicCLIP / Contrastive Learning)
Existing multimodal models rely on general ImageNet weights for ResNet-50 and textual corpora for MuRIL. A major leap forward would involve contrastive pre-training (similar to OpenAI CLIP or ALIGN) directly on a large corpus of Indian social media memes ($100\text{k}+$ image-caption pairs):
$$\mathcal{L}_{CLIP} = - \frac{1}{2N} \sum_{i=1}^N \left( \log \frac{\exp(s(I_i, T_i)/\tau)}{\sum_{j} \exp(s(I_i, T_j)/\tau)} + \log \frac{\exp(s(I_i, T_i)/\tau)}{\sum_{j} \exp(s(I_j, T_i)/\tau)} \right)$$
This aligns visual facial expressions with colloquial Hinglish slang in a shared metric space before downstream sentiment fine-tuning.

### 4.3 Dense Token-to-Patch Cross-Attention
Current gated fusion operates on global pooled vectors ($h_t \in \mathbb{R}^{512}, h_v \in \mathbb{R}^{512}$). Future work should investigate dense cross-attention mechanisms:
- Each textual token attends directly to $7 \times 7$ spatial visual patches from ResNet-50.
- Enables localized incongruity detection (e.g., linking the word *"presentation"* directly to a nervous character's face in the image).

### 4.4 Active-Learning Curation for Minority Emotions
To resolve the scarcity of `Fear` and `Disgust`, an active-learning pipeline should be deployed to continuously harvest memes matching specialized affective lexicons (e.g., horror memes, cringe comedy, political revulsion) with human-in-the-loop verification.
