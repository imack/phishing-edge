import torch.nn as nn
from torchvision import models

class MobileNetClassifier(nn.Module):
    def __init__(self):
        super(MobileNetClassifier, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the pooled features
            nn.Linear(1280, 256),  # MobileNetV2 outputs 1280-d features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Final layer for binary classification
        )

    def forward(self, image):
        features = self.cnn.features(image)
        pooled_features = self.adaptive_pool(features)
        logits = self.classifier(pooled_features)
        return logits

    def test_name(self):
        return 'mobilenet_only'
