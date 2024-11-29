import torch
import torch.nn as nn
from transformers import DistilBertModel

class SimplifiedHTMLModel(nn.Module):
    def __init__(self, distilbert_model_name="distilbert-base-uncased", hidden_size=768):
        super(SimplifiedHTMLModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.classifier = nn.Linear(hidden_size, 1)  # Binary classification

    def forward(self, html_input_ids, html_attention_mask):
        batch_size, num_chunks, seq_len = html_input_ids.size()

        # Flatten chunks for processing with DistilBERT
        input_ids = html_input_ids.view(-1, seq_len)  # Shape: (batch_size * num_chunks, seq_len)
        attention_mask = html_attention_mask.view(-1, seq_len)  # Shape: (batch_size * num_chunks, seq_len)

        # Process with DistilBERT
        distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = distilbert_outputs.last_hidden_state  # Shape: (batch_size * num_chunks, seq_len, hidden_size)

        # Aggregate token embeddings within each chunk (mean pooling)
        chunk_embeddings = token_embeddings.mean(dim=1)  # Shape: (batch_size * num_chunks, hidden_size)

        # Reshape back to batch size and aggregate chunks (mean pooling across chunks)
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)  # Shape: (batch_size, num_chunks, hidden_size)
        aggregated_embedding = chunk_embeddings.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Classification head
        logits = self.classifier(aggregated_embedding)
        class_probs = torch.sigmoid(logits).squeeze(-1)

        return class_probs


    def test_name(self):
        return 'simple_html'
