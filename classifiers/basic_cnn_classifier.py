import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class PhishingClassifier(nn.Module):
    def __init__(self):
        super(PhishingClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn_fc_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Classifier combining both CNN and BERT features
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_fc_features + self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, url_input_ids, url_attention_mask, image):
        image_features = self.cnn(image)

        bert_outputs = self.bert(input_ids=url_input_ids, attention_mask=url_attention_mask)
        url_features = bert_outputs.last_hidden_state[:, 0, :]

        combined_features = torch.cat((image_features, url_features), dim=1)

        logits = self.classifier(combined_features)
        return logits

    def test_name(self):
        return 'cnn_with_url'