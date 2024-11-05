from torch.utils.data import Dataset
import h5py
import torch
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

class PhishingDataset(Dataset):
    def __init__(self, h5_file, split='train'):
        self.file = h5py.File(h5_file, 'r')
        self.urls = self.file[f'{split}/urls'][:]
        self.screenshots = self.file[f'{split}/screenshots'][:]
        self.html_content = self.file[f'{split}/html_content'][:]
        self.labels = self.file[f'{split}/labels'][:]
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        url = self.urls[idx].decode('utf-8')
        screenshot = self.screenshots[idx]
        html_content = self.html_content[idx].decode('utf-8')
        text = f"URL: {url} CONTENT: {html_content}"

        # tokenize the HTML content
        encoded_html_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Preprocess the screenshot
        image = self.transform(screenshot)

        # Tokenize the URL
        encoded_url_input = self.tokenizer(
            url,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'html_input_ids': encoded_html_input['input_ids'],
            'html_attention_mask': encoded_html_input['attention_mask'],
            'url_input_ids': encoded_url_input['input_ids'].squeeze(),
            'url_attention_mask': encoded_url_input['attention_mask'].squeeze(),
            'image': image,
            'label': label
        }