from transformers import DistilBertModel, DistilBertConfig
import torch.nn as nn

class TinyBertClassifier(nn.Module):
    def __init__(self):
        super(TinyBertClassifier, self).__init__()

        # Load DistilBERT with fewer hidden units to make tinybert
        smaller_config = DistilBertConfig(
            hidden_size=384,
            num_attention_heads=6,
            intermediate_size=768,
            num_hidden_layers=3,
        )
        self.bert = DistilBertModel(smaller_config)

        # Classifier adapted for smaller hidden size
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, html_input_ids, html_attention_mask):
        bert_outputs = self.bert(input_ids=html_input_ids, attention_mask=html_attention_mask)
        url_features = bert_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(url_features)
        return logits

    def test_name(self):
        return 'tiny_html_bert'