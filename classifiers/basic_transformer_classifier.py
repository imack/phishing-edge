import torch.nn as nn
from transformers import DistilBertModel

class BasicTransformerClassifier(nn.Module):
    def __init__(self):
        super(BasicTransformerClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, html_input_ids, html_attention_mask):
        html_input_ids = html_input_ids[:, 0, :]
        html_attention_mask = html_attention_mask[:,0,:]
        outputs = self.bert(input_ids=html_input_ids, attention_mask=html_attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def test_name(self):
        return 'basic_html_transformer'
