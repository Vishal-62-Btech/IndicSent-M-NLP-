"""
IndicSent-M: Deep Text Models Module
Implements fine-tunable neural architectures for:
1. MuRIL (google/muril-base-cased) - Pretrained on 17 Indic languages + English
2. XLM-RoBERTa (xlm-roberta-base) - Pretrained cross-lingual multilingual masked language model
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TransformerSentimentClassifier(nn.Module):
    """
    Modular Transformer-based 3-class sentiment classifier.
    Extracts the pooled representation or [CLS] token, passes through
    LayerNorm, Dropout, and a dense classification head.
    """
    def __init__(
        self,
        pretrained_model_name: str = "google/muril-base-cased",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = pretrained_model_name
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        
        hidden_dim = self.config.hidden_size
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output if present, else mean pool across valid token representations
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
        logits = self.classifier(pooled)
        return logits

def get_text_tokenizer(pretrained_model_name: str):
    """Returns the matching fast Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(pretrained_model_name)
