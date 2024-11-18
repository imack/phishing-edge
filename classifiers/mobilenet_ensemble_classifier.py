import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class MobilenetEnsembleClassifier(nn.Module):
    def __init__(self):
        super(MobilenetEnsembleClassifier, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn_features = self.cnn.features
        self.cnn_out_channels = 1280

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Classifier combining both CNN and BERT features
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_channels + self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, url_input_ids, url_attention_mask, image):
        image_features = self.cnn_features(image)
        image_features = self.adaptive_pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)

        # Extract URL features using DistilBERT
        bert_outputs = self.bert(input_ids=url_input_ids, attention_mask=url_attention_mask)
        url_features = bert_outputs.last_hidden_state[:, 0, :]

        # Combine features
        combined_features = torch.cat((image_features, url_features), dim=1)

        logits = self.classifier(combined_features)
        return logits

    def test_name(self):
        return 'mobilenet_with_url'

