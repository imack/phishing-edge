import torch.nn as nn
from transformers import DistilBertModel
from transformers import DistilBertConfig

class BasicUrlUltraSkinnyBertClassifier(nn.Module):
    def __init__(self):
        super(BasicUrlUltraSkinnyBertClassifier, self).__init__()

        # Load pretrained DistilBERT model
        pretrained_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        custom_config = DistilBertConfig(
            hidden_size=768,  # Keep same hidden size as pretrained
            num_hidden_layers=1,  # Reduce the number of transformer layers
            num_attention_heads=2,  # Reduce number of attention heads
            intermediate_size=512,  # Reduce feed-forward size
        )
        self.bert = DistilBertModel(custom_config)

        self.bert.load_state_dict(pretrained_bert.state_dict(), strict=False)

        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)


    def forward(self, url_input_ids, url_attention_mask):
        bert_outputs = self.bert(input_ids=url_input_ids, attention_mask=url_attention_mask)
        url_features = bert_outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(url_features)
        return logits

    def test_name(self):
        return 'basic_url_ultra_skinny_bert'
