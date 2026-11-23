"""
IndicSent-M: Multimodal Fusion Architectures Module
Implements four multimodal sentiment classification architectures:
1. MuRIL + ResNet-50 Feature Concatenation Fusion
2. MuRIL + EfficientNet-B0 Feature Concatenation Fusion
3. XLM-RoBERTa + ResNet-50 Feature Concatenation Fusion
4. Gated Multimodal Fusion (Trainable Gating Mechanism)
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig
from pathlib import Path

# Local checkpoints directory for offline resolution
_SRC_DIR = Path(__file__).resolve().parents[1]
_LOCAL_CHECKPOINTS = _SRC_DIR.parent / "checkpoints"

def _resolve_text_backbone(backbone: str) -> str:
    """Resolves local checkpoint directory to avoid socket checks when offline."""
    if Path(backbone).is_dir():
        return str(Path(backbone).resolve())
    if "muril" in backbone.lower():
        local_muril = _LOCAL_CHECKPOINTS / "muril"
        if local_muril.is_dir():
            return str(local_muril)
    elif "xlm" in backbone.lower():
        local_xlm = _LOCAL_CHECKPOINTS / "xlm_roberta"
        if local_xlm.is_dir():
            return str(local_xlm)
    return backbone

class MultimodalConcatClassifier(nn.Module):
    """
    Modular Feature Concatenation Architecture.
    Combines text representations from MuRIL or XLM-RoBERTa with
    visual features from ResNet-50 or EfficientNet-B0.
    """
    def __init__(
        self,
        text_backbone: str = "google/muril-base-cased",
        vision_backbone: str = "resnet50",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_text: bool = True,
        freeze_vision: bool = True,
        pretrained_text: bool = True,
        pretrained_vision: bool = True
    ):
        super().__init__()
        self.text_name = text_backbone
        self.vision_name = vision_backbone

        resolved_text = _resolve_text_backbone(text_backbone)
        # 1. Text Encoder
        if pretrained_text:
            self.text_model = AutoModel.from_pretrained(resolved_text, local_files_only=True)
        else:
            config = AutoConfig.from_pretrained(resolved_text, local_files_only=True)
            self.text_model = AutoModel.from_config(config)
        text_dim = self.text_model.config.hidden_size  # 768

        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # 2. Vision Encoder
        if vision_backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained_vision else None
            base = models.resnet50(weights=weights)
            self.vision_features = nn.Sequential(*list(base.children())[:-1])
            vision_dim = 2048
        elif vision_backbone in ("efficientnet_b0", "efficientnet"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_vision else None
            base = models.efficientnet_b0(weights=weights)
            self.vision_features = nn.Sequential(base.features, base.avgpool)
            vision_dim = 1280
        else:
            raise ValueError(f"Unsupported vision backbone: {vision_backbone}")

        if freeze_vision:
            for p in self.vision_features.parameters():
                p.requires_grad = False

        # 3. Projection & Fusion Layers
        proj_dim = 512
        self.proj_text = nn.Linear(text_dim, proj_dim)
        self.proj_vision = nn.Linear(vision_dim, proj_dim)
        fused_dim = proj_dim * 2

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def extract_text_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        else:
            token_emb = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
            return torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def extract_vision_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.vision_features(pixel_values)
        return torch.flatten(feat, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        t_feat = self.extract_text_feature(input_ids, attention_mask)
        v_feat = self.extract_vision_feature(pixel_values)

        h_t = torch.relu(self.proj_text(t_feat))
        h_v = torch.relu(self.proj_vision(v_feat))

        fused = torch.cat([h_t, h_v], dim=-1)
        return self.fusion_head(fused)

class GatedMultimodalClassifier(nn.Module):
    """
    Gated Fusion Architecture:
    Dynamically weighs text and image importance for each meme sample.
    alpha = sigmoid(Linear([h_text, h_image]))
    fused = alpha * h_text + (1 - alpha) * h_image
    """
    def __init__(
        self,
        text_backbone: str = "google/muril-base-cased",
        vision_backbone: str = "resnet50",
        proj_dim: int = 512,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_text: bool = True,
        freeze_vision: bool = True,
        pretrained_text: bool = True,
        pretrained_vision: bool = True
    ):
        super().__init__()
        self.text_name = text_backbone
        self.vision_name = vision_backbone
        self.proj_dim = proj_dim

        resolved_text = _resolve_text_backbone(text_backbone)
        # 1. Text Encoder
        if pretrained_text:
            self.text_model = AutoModel.from_pretrained(resolved_text, local_files_only=True)
        else:
            config = AutoConfig.from_pretrained(resolved_text, local_files_only=True)
            self.text_model = AutoModel.from_config(config)
        text_dim = self.text_model.config.hidden_size

        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # 2. Vision Encoder
        if vision_backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained_vision else None
            base = models.resnet50(weights=weights)
            self.vision_features = nn.Sequential(*list(base.children())[:-1])
            vision_dim = 2048
        elif vision_backbone in ("efficientnet_b0", "efficientnet"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_vision else None
            base = models.efficientnet_b0(weights=weights)
            self.vision_features = nn.Sequential(base.features, base.avgpool)
            vision_dim = 1280
        else:
            raise ValueError(f"Unsupported vision backbone: {vision_backbone}")

        if freeze_vision:
            for p in self.vision_features.parameters():
                p.requires_grad = False

        # 3. Projections into common latent space
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
        self.proj_vision = nn.Sequential(
            nn.Linear(vision_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

        # 4. Gating Network
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )

        # 5. Classifier Head
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_text_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        else:
            token_emb = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
            return torch.sum(token_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def extract_vision_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.vision_features(pixel_values)
        return torch.flatten(feat, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        t_raw = self.extract_text_feature(input_ids, attention_mask)
        v_raw = self.extract_vision_feature(pixel_values)

        h_t = self.proj_text(t_raw)
        h_v = self.proj_vision(v_raw)

        # Gating vector alpha in [0, 1]^proj_dim
        concat_tv = torch.cat([h_t, h_v], dim=-1)
        alpha = self.gate(concat_tv)

        # Gated combination
        h_fused = alpha * h_t + (1.0 - alpha) * h_v
        h_fused = self.norm(h_fused)

        return self.classifier(h_fused)


SIX_EMOTIONS = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]
NUM_EMOTIONS = 6
EMOTION_EMOJIS = {
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😡",
    "Fear": "😨",
    "Disgust": "🤢",
    "Surprise": "😲"
}


class MultiTaskGatedMultimodalClassifier(GatedMultimodalClassifier):
    """
    Multi-Task Gated Multimodal Architecture:
    Shared text & image encoders and gated fusion representation (h_fused)
    branching into:
    1. Sentiment Classification Head (3 classes: Negative, Neutral, Positive) via Softmax/CrossEntropy
    2. Emotion Classification Head (6 classes: Happiness, Sadness, Anger, Fear, Disgust, Surprise) via Sigmoid/BCEWithLogits
    """
    def __init__(
        self,
        text_backbone: str = "google/muril-base-cased",
        vision_backbone: str = "resnet50",
        proj_dim: int = 512,
        num_classes: int = 3,
        num_emotions: int = 6,
        dropout_rate: float = 0.3,
        freeze_text: bool = True,
        freeze_vision: bool = True,
        pretrained_text: bool = True,
        pretrained_vision: bool = True
    ):
        super().__init__(
            text_backbone=text_backbone,
            vision_backbone=vision_backbone,
            proj_dim=proj_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_text=freeze_text,
            freeze_vision=freeze_vision,
            pretrained_text=pretrained_text,
            pretrained_vision=pretrained_vision
        )
        self.num_emotions = num_emotions
        # Dedicated multi-label emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_dict: bool = False
    ):
        t_raw = self.extract_text_feature(input_ids, attention_mask)
        v_raw = self.extract_vision_feature(pixel_values)

        h_t = self.proj_text(t_raw)
        h_v = self.proj_vision(v_raw)

        concat_tv = torch.cat([h_t, h_v], dim=-1)
        alpha = self.gate(concat_tv)

        h_fused = alpha * h_t + (1.0 - alpha) * h_v
        h_fused = self.norm(h_fused)

        sentiment_logits = self.classifier(h_fused)
        emotion_logits = self.emotion_classifier(h_fused)

        if return_dict:
            return {
                "sentiment_logits": sentiment_logits,
                "emotion_logits": emotion_logits,
                "alpha": alpha,
                "h_fused": h_fused
            }
        return sentiment_logits, emotion_logits
