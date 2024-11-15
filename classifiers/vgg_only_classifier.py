import torch.nn as nn
from torchvision import models

class VGGClassifier(nn.Module):
    def __init__(self):
        super(VGGClassifier, self).__init__()
        # Load pretrained VGG-19 model
        self.cnn = models.vgg19(pretrained=True)
        self.cnn_features = self.cnn.features
        self.cnn_avg_pool = self.cnn.avgpool
        self.cnn_classifier_input_features = self.cnn.classifier[0].in_features

        self.cnn.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_classifier_input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image):
        features = self.cnn_features(image)
        pooled_features = self.cnn_avg_pool(features)
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        logits = self.classifier(flat_features)
        return logits

    def test_name(self):
        return 'vgg19_only'
