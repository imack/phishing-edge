from torch.utils.data import Dataset
import h5py
import torch
import tempfile
import boto3
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

S3_PATH = 's3://phishing-edge/dataset/phishing_output.h5'

class PhishingDataset(Dataset):
    def __init__(self,required_data, split='train', local_file_path=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.required_data = required_data

        if local_file_path is None:
            print(f"Downloading data from s3://{S3_PATH}")
            # Use a temporary file to store the downloaded dataset
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                local_file_path = temp_file.name

            s3 = boto3.client('s3')
            bucket, key = self._parse_s3_path(S3_PATH)
            s3.download_file(bucket, key, local_file_path)
            print("Download Complete")

        self.file = h5py.File(local_file_path, 'r')
        self.labels = self.file[f'{split}/labels'][:]

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

    def _parse_s3_path(self, s3_path):
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1]
        return bucket, key

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

        encoded_url_input = self.tokenizer(
            url,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        needs_url = 'url_input_ids' in self.required_data

        input_dict = {
            'html_input_ids': encoded_html_input['input_ids'].squeeze() if encoded_html_input is not None else None,
            'html_attention_mask': encoded_html_input['attention_mask'].squeeze() if encoded_html_input is not None else None,
            'url_input_ids': encoded_url_input['input_ids'].squeeze() if needs_url else None,
            'url_attention_mask': encoded_url_input['attention_mask'].squeeze() if needs_url else None,
            'image': image if image is not None and image.numel() > 0 else None,
            'label': label
        }

        return {k: v for k, v in input_dict.items() if v is not None}
