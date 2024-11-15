import torch.nn as nn
from torchvision import models

class SqueezenetClassifier(nn.Module):
    def __init__(self):
        super(SqueezenetClassifier, self).__init__()
        self.cnn = models.squeezenet1_1(pretrained=True)
        self.cnn.classifier = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image):
        features = self.cnn.features(image)
        pooled_features = self.adaptive_pool(features)
        logits = self.classifier(pooled_features)
        return logits

    def test_name(self):
        return 'squeezenet_only'