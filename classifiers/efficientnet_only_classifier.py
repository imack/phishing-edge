import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super(EfficientNetClassifier, self).__init__()
        self.cnn = models.efficientnet_b0(pretrained=True)
        self.cnn.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image):
        features = self.cnn.features(image)
        pooled_features = self.cnn.avgpool(features)
        logits = self.classifier(pooled_features)
        return logits

    def test_name(self):
        return 'efficientnet_only'
