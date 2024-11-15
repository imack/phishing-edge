import torch.nn as nn

from torchvision import models

class BigCNNClassifier(nn.Module):
    def __init__(self):
        super(BigCNNClassifier, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        self.cnn_fc_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Remove the classification layer

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_fc_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, image):
        image_features = self.cnn(image)

        logits = self.classifier(image_features)
        return logits

    def test_name(self):
        return 'big_cnn_only'
