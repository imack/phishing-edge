from torch.utils.data import Dataset
import h5py
import torch
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

class PhishingDataset(Dataset):
    def __init__(self, h5_file, required_data, split='train'):
        self.file = h5py.File(h5_file, 'r')
        self.labels = self.file[f'{split}/labels'][:]
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if 'image' in required_data:
            self.screenshots = self.file[f'{split}/screenshots'][:]

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.urls = self.file[f'{split}/urls'][:]

        if 'html_input_ids' in required_data:
            self.html_content = self.file[f'{split}/html_content'][:]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        url = self.urls[idx].decode('utf-8')
        encoded_html_input = None
        image = None

        if  hasattr(self, 'screenshots'):
            screenshot = self.screenshots[idx]
            image = self.transform(screenshot)

        if hasattr(self, 'html_content'):
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

        # tokenize url contet
        encoded_url_input = self.tokenizer(
            url,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_dict = {
            'html_input_ids': encoded_html_input['input_ids'] if encoded_html_input is not None else None,
            'html_attention_mask': encoded_html_input['attention_mask'] if encoded_html_input is not None else None,
            'url_input_ids': encoded_url_input['input_ids'].squeeze(),
            'url_attention_mask': encoded_url_input['attention_mask'].squeeze(),
            'image': image if image is not None and image.numel() > 0 else None,
            'label': label
        }

        return {k: v for k, v in input_dict.items() if v is not None}
