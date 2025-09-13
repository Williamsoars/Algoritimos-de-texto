import torch
import torch.nn as nn
from transformers import AutoModel

from .arrow_attention import ArrowAttention
from .biaffine import Biaffine
from .bispa import BiSPAModule

class SeNER(nn.Module):
    def __init__(self, plm_name: str = "microsoft/deberta-v3-large", hidden_size: int = 1024, num_labels: int = 10):
        """
        plm_name: nome do modelo pré-treinado (HuggingFace).
        hidden_size: tamanho da saída do PLM.
        num_labels: número de tipos de entidades.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(plm_name)

        # Camadas extras do artigo
        self.arrow_attention = ArrowAttention(hidden_size)
        self.biaffine = Biaffine(hidden_size, num_labels)
        self.bispa = BiSPAModule(hidden_size)

        # Classificador final
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Codificação inicial
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               return_dict=True)
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]

        # Arrow Attention
        hidden_states = self.arrow_attention(hidden_states, attention_mask)

        # Biaffine para spans
        span_repr = self.biaffine(hidden_states)

        # BiSPA
        span_repr = self.bispa(span_repr)

        # Classificação final
        logits = self.classifier(span_repr)

        return logits
