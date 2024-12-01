import torch
import torch.nn as nn

from classifiers.basic_transformer_classifier import BasicTransformerClassifier
from classifiers.basic_url_ultra_skinny_bert_classifier import BasicUrlUltraSkinnyBertClassifier
from classifiers.only_cnn_classifier import BasicCNNClassifier

model_paths = {
    'BasicCNNClassifier': 'models/canonical/cnn_only_phishing_classifier_epoch_2.pt',
    'BasicTransformerClassifier': 'models/canonical/basic_html_transformer_phishing_classifier.pt',
    'BasicUrlUltraSkinnyBertClassifier': 'models/canonical/basic_url_ultra_skinny_bert_phishing_classifier_epoch_2.pt',
}

class EnsembleModel(nn.Module):
    def __init__(self, device):
        super(EnsembleModel, self).__init__()

        # Load pretrained submodels
        self.models = nn.ModuleDict()
        for model_name, model_path in model_paths.items():
            if model_name == 'BasicCNNClassifier':
                model = BasicCNNClassifier()
            elif model_name == 'BasicTransformerClassifier':
                model = BasicTransformerClassifier()
            elif model_name == 'BasicUrlUltraSkinnyBertClassifier':
                model = BasicUrlUltraSkinnyBertClassifier()
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Load pretrained weights
            model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False

            # Remove classifier
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()

            self.models[model_name] = model

        total_features = sum(
            model.cnn_fc_features if hasattr(model, 'cnn_fc_features')
            else model.bert.config.hidden_size
            for model in self.models.values()
        )

        # replace classifier with Linear -> relu -> dropout -> linear
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, html_input_ids, html_attention_mask, url_input_ids, url_attention_mask, image):
        outputs = []
        for name, model in self.models.items():
            if name == 'BasicCNNClassifier':
                outputs.append(model(image))
            elif name == 'BasicTransformerClassifier':
                outputs.append(model(html_input_ids, html_attention_mask))
            elif name == 'BasicUrlUltraSkinnyBertClassifier':
                outputs.append(model(url_input_ids, url_attention_mask))

        # Concatenate outputs as a stack to feed into ensemble
        combined_features = torch.cat(outputs, dim=1)
        logits = self.classifier(combined_features)
        return logits

    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found in ensemble.")

    def test_name(self):
        return 'ensemble'
