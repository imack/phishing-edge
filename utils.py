import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch.nn as nn

def evaluate_model(model, dataloader, device, criterion = nn.CrossEntropyLoss()):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)

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