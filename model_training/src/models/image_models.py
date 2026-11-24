"""
IndicSent-M: Deep Image Models Module
Implements transfer learning architectures for visual meme sentiment classification:
1. ResNet-50 (Pretrained ImageNet-1k)
2. EfficientNet-B0 (Pretrained ImageNet-1k)
3. MobileNetV3-Large (Pretrained ImageNet-1k)
4. CLIP ViT Image Encoder (openai/clip-vit-base-patch32)
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPVisionModel, CLIPVisionConfig

class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features  # 2048

        # Strip final fc layer
        self.features = nn.Sequential(*list(base.children())[:-1])

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

class EfficientNetB0Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features  # 1280

        self.features = base.features
        self.avgpool = base.avgpool

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

class MobileNetV3LargeClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        in_features = base.classifier[0].in_features  # 960

        self.features = base.features
        self.avgpool = base.avgpool

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

class CLIPVisionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
        pretrained: bool = True
    ):
        super().__init__()
        if pretrained:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = CLIPVisionConfig.from_pretrained(model_name)
            self.vision_model = CLIPVisionModel(config)
        hidden_size = self.vision_model.config.hidden_size  # 768

        if freeze_features:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=x)
        return outputs.pooler_output  # [Batch, 768]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)


class MultiTaskResNet50Classifier(nn.Module):
    """
    Multi-Task Vision Classifier based on ResNet-50.
    Simultaneously predicts:
    1. 3-class Sentiment (Negative, Neutral, Positive) via Softmax
    2. 6-class Multi-Label Emotions (Happiness, Sadness, Anger, Fear, Disgust, Surprise) via Sigmoid
    """
    def __init__(
        self,
        num_classes: int = 3,
        num_emotions: int = 6,
        dropout_rate: float = 0.3,
        freeze_features: bool = False,
        pretrained: bool = False
    ):
        super().__init__()
        base = ResNet50Classifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_features=freeze_features,
            pretrained=pretrained
        )
        self.features = base.features
        self.classifier = base.classifier
        self.num_emotions = num_emotions
        self.emotion_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_emotions)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        sentiment_logits = self.classifier(feat)
        emotion_logits = self.emotion_classifier(feat)
        return sentiment_logits, emotion_logits
