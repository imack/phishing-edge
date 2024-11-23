import torch
import torch.nn as nn
from transformers import DistilBertModel
import torch.nn.utils.rnn as rnn_utils


class MultiTaskHTMLModel(nn.Module):
    def __init__(self, distilbert_model_name="distilbert-base-uncased", hidden_size=768, num_chunks=10, window_size=5,
                 top_k=5):
        super(MultiTaskHTMLModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name, output_attentions=True)
        self.chunk_aggregator = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, 1)  # Binary classification
        self.token_predictor = nn.Linear(hidden_size, 1)  # Token importance
        self.window_size = window_size
        self.top_k = top_k  # Number of tokens to retain for summarization

        # Summarization head
        self.summarizer_aggregator = nn.Linear(hidden_size, hidden_size)

    def forward(self, html_input_ids, html_attention_mask):
        """
        Process long sequences by chunking into manageable sizes for DistilBERT.
        """

        batch_size, num_chunks, seq_len = html_input_ids.size()
        max_seq_len = 512  # Maximum sequence length for DistilBERT

        # Validate sequence length
        if seq_len > max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum limit of {max_seq_len}.")

        # Flatten batch and chunk dimensions for DistilBERT
        input_ids = html_input_ids.view(-1, seq_len)  # Shape: (batch_size * num_chunks, seq_len)
        attention_mask = html_attention_mask.view(-1, seq_len)  # Shape: (batch_size * num_chunks, seq_len)

        # Forward pass through DistilBERT
        distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = distilbert_outputs.last_hidden_state  # Token-level embeddings

        print(f"Token embeddings shape: {token_embeddings.shape}")
        attention_weights = distilbert_outputs.attentions[-1]  # Last layer attention weights

        # Token importance scores
        token_scores = torch.sigmoid(self.token_predictor(token_embeddings)).squeeze(-1)

        # Apply window-based smoothing to token scores
        windowed_scores = self.window_attention(token_scores)

        # Summarization step
        summarized_embeddings = self.summarize(token_embeddings, windowed_scores)

        # Aggregate chunk-level outputs
        chunk_embeddings = token_embeddings[:, 0, :]  # [CLS]-like embeddings for each chunk
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)
        aggregated_embeddings, _ = self.chunk_aggregator(chunk_embeddings)
        global_embedding = aggregated_embeddings[:, -1, :]  # Use the last LSTM output

        # Combine global embedding with summarized output
        combined_embedding = global_embedding + summarized_embeddings

        # Classification head
        logits = self.classifier(combined_embedding)
        class_probs = torch.sigmoid(logits).squeeze(-1)

        return class_probs, windowed_scores, summarized_embeddings, attention_weights

    def window_attention(self, token_scores):
        """
        Apply a window-based aggregation over token importance scores.
        """
        window_size = self.window_size
        pad_size = window_size // 2
        padded_scores = torch.nn.functional.pad(token_scores, (pad_size, pad_size), mode='constant', value=0)
        windowed_scores = torch.zeros_like(token_scores)

        for i in range(token_scores.size(1)):  # Iterate over sequence length
            window = padded_scores[:, i:i + window_size]
            windowed_scores[:, i] = torch.mean(window, dim=1)  # Aggregate scores within the window

        return windowed_scores

    def summarize(self, token_embeddings, token_scores):
        """
        Generate a summary representation using window-based token scores.
        """
        # Attention-weighted sum of token embeddings
        token_scores = token_scores.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        weighted_embeddings = token_embeddings * token_scores  # Apply scores as weights
        summarized_output = torch.sum(weighted_embeddings, dim=1)  # Summarize across sequence length

        # Project to summarizer space
        summarized_output = self.summarizer_aggregator(summarized_output)

        return summarized_output

    def test_name(self):
        return 'html_multi_task'
