import torch.nn as nn
from transformers import LongformerModel


class LongformerClassifier(nn.Module):
    def __init__(self):
        super(LongformerClassifier, self).__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.classifier = nn.Linear(self.longformer.config.hidden_size, 2)  # 2 for binary classification

    def forward(self, longformer_input_ids, longformer_attention_mask):

        # Forward pass through Longformer
        outputs = self.longformer(
            input_ids=longformer_input_ids,
            attention_mask=longformer_attention_mask,
            global_attention_mask=None
        )

        # Classifier on [CLS] token (index 0 of last hidden state)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def test_name(self):
        return 'longformer_html_transformer'
