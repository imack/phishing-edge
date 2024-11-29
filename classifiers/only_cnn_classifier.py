import torch.nn as nn

from torchvision import models

class BasicCNNClassifier(nn.Module):
    def __init__(self):
        super(BasicCNNClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn_fc_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Remove the classification layer

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_fc_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image):
        image_features = self.cnn(image)

        logits = self.classifier(image_features)
        return logits

    def test_name(self):
        return 'cnn_only'
