from phishing_dataset import PhishingDataset
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from utils import evaluate_model, get_filtered_inputs
import torch.optim as optim
import inspect
dataset_path = os.path.expanduser("~/transfer/phishing_output.h5")

def test_harness(model, epochs=10, batch_size=8, learning_rate=2e-5):
    required_data = inspect.signature(model.forward).parameters.keys()
    train_dataset = PhishingDataset(dataset_path, required_data, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = PhishingDataset(dataset_path, required_data, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    writer = SummaryWriter(f"runs/{model.test_name()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()

            labels = batch['label'].to(device)
            filtered_inputs = get_filtered_inputs(batch)
            outputs = model(**filtered_inputs)

            loss = criterion(outputs, labels)
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
            f"Epoch {epoch + 1}/{epochs}, Test Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        writer.flush()
    writer.close()

    return model