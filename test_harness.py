from dataset.phishing_dataset import PhishingDataset
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import evaluate_model, get_filtered_inputs
import torch.optim as optim
import inspect

def chunk_sequence(sequence, chunk_size=512):
    """
    Chunk a sequence into fixed-sized segments of `chunk_size`.
    Returns the chunks as a tensor of shape (num_chunks, chunk_size).
    """
    chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
    return torch.stack([torch.tensor(chunk) for chunk in chunks])

def custom_collate_fn(batch, chunk_size=512):
    all_chunks = []
    labels = []

    max_num_chunks = 0

    # Process each item in the batch
    for item in batch:
        # Chunk the input sequence
        chunks = chunk_sequence(item['html_input_ids'], chunk_size)
        max_num_chunks = max(max_num_chunks, chunks.size(0))

        # Append chunks and labels
        all_chunks.append(chunks)
        labels.append(item['label'])

    # Pad all chunks to ensure batch consistency
    padded_chunks = []
    for chunks in all_chunks:
        num_chunks = chunks.size(0)
        # Pad to `max_num_chunks` with zeros
        padded_chunks.append(
            torch.cat([chunks, torch.zeros((max_num_chunks - num_chunks, chunk_size), dtype=torch.long)], dim=0)
        )

    # Stack all padded chunks into a batch tensor
    batch_chunks = torch.stack(padded_chunks, dim=0)  # Shape: (batch_size, max_num_chunks, chunk_size)
    labels = torch.tensor(labels)                    # Shape: (batch_size)

    # Create attention masks: 1 for valid tokens, 0 for padding
    attention_masks = (batch_chunks != 0).long()     # Shape: (batch_size, max_num_chunks, chunk_size)

    return {
        'html_input_ids': batch_chunks,  # Shape: (batch_size, max_num_chunks, 512)
        'html_attention_mask': attention_masks,
        'label': labels
    }

def test_harness(model, local_dataset=None, epochs=10, batch_size=8, learning_rate=2e-5):
    required_data = inspect.signature(model.forward).parameters.keys()
    num_workers = 0
    if torch.cuda.is_available():
        num_workers = 6
    train_dataset = PhishingDataset(required_data, split='train', local_file_path=local_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    test_dataset = PhishingDataset(required_data, split='dev', local_file_path=local_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=custom_collate_fn)

    writer = SummaryWriter(f"runs/{model.test_name()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True, miniters=len(train_dataloader) // 10)
        for batch in progress_bar:
            optimizer.zero_grad()

            labels = batch['label'].to(device)
            filtered_inputs = get_filtered_inputs(batch)
            class_probs, windowed_scores, summarized_embeddings, attention_weights = model(**filtered_inputs)

            loss = criterion(class_probs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Evaluation after each epoch
        loss, precision, recall, f1, accuracy = evaluate_model(model, test_dataloader, device, criterion)
        writer.add_scalar("Loss/test", loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        print(
            f"Epoch {epoch + 1}/{epochs}, Dev Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        writer.flush()
    writer.close()

    return model