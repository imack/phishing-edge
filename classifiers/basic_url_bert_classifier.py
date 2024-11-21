import torch.nn as nn
from transformers import DistilBertModel

class BasicUrlBertClassifier(nn.Module):
    def __init__(self):
        super(BasicUrlBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, url_input_ids, url_attention_mask):
        bert_outputs = self.bert(input_ids=url_input_ids, attention_mask=url_attention_mask)
        url_features = bert_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(url_features)
        return logits

    def test_name(self):
        return 'basic_url_bert'
