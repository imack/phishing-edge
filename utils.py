import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn as nn

def get_filtered_inputs(batch):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    images = batch['image'].to(device) if 'image' in batch else None
    url_input_ids = batch['url_input_ids'].to(device) if 'url_input_ids' in batch else None
    url_attention_mask = batch['url_attention_mask'].to(device) if 'url_attention_mask' in batch else None

    longformer_input_ids = batch['longformer_input_ids'].to(device) if 'longformer_input_ids' in batch else None
    longformer_attention_mask = batch['longformer_attention_mask'].to(device) if 'longformer_input_ids' in batch else None

    html_input_ids = batch['html_input_ids'].to(device) if 'html_input_ids' in batch else None
    html_attention_mask = batch['html_attention_mask'].to(device) if 'html_attention_mask' in batch else None

    inputs = {
        'image': images,
        'url_input_ids': url_input_ids,
        'url_attention_mask': url_attention_mask,
        'longformer_input_ids': longformer_input_ids,
        'longformer_attention_mask': longformer_attention_mask,
        'html_input_ids': html_input_ids,
        'html_attention_mask': html_attention_mask
    }

    return {k: v for k, v in inputs.items() if v is not None}

def evaluate_model(model, dataloader, device, criterion = nn.CrossEntropyLoss()):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)
            filtered_inputs = get_filtered_inputs(batch)
            outputs = model(**filtered_inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, precision, recall, f1, accuracy