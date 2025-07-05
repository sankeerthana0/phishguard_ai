import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PhishGuardModel(nn.Module):
    def __init__(self, text_model_name, vision_model_name, num_structured_features):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        
        self.structured_mlp = nn.Sequential(
            nn.Linear(num_structured_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        fusion_dim = self.text_model.config.hidden_size + self.vision_model.config.hidden_size + 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text_inputs, vision_inputs, structured_inputs):
        text_embedding = self.text_model(**text_inputs).last_hidden_state[:, 0, :]
        vision_embedding = self.vision_model(**vision_inputs).pooler_output
        structured_embedding = self.structured_mlp(structured_inputs)
        
        fused_embedding = torch.cat((text_embedding, vision_embedding, structured_embedding), dim=1)
        return self.classifier(fused_embedding)