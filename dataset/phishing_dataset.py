from torch.utils.data import Dataset
import h5py
import torch
import os
import boto3
import time
from transformers import DistilBertTokenizer
import torchvision.transforms as transforms

S3_PATH = 's3://phishing-edge/dataset/phishing_output_tokenized.h5'
LOCAL_CACHE_PATH = '/tmp/phishing_output_tokenized.h5'

class PhishingDataset(Dataset):
    def __init__(self, required_data, split='train', local_file_path=None):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.required_data = required_data

        if local_file_path is None:
            local_file_path = LOCAL_CACHE_PATH
            if not os.path.exists(local_file_path):
                print(f"Downloading data from {S3_PATH}")
                start_time = time.time()
                s3 = boto3.client('s3')
                bucket, key = self._parse_s3_path(S3_PATH)
                s3.download_file(bucket, key, local_file_path)
                print(f"Download Complete in {(time.time() - start_time):.2f} seconds")
            else:
                print(f"Using cached dataset at {local_file_path}")

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
            self.html_input_ids = self.file[f'{split}/html_content'][:]

        if 'html_attention_mask' in required_data:
            self.html_attention_masks = self.file[f'{split}/html_attention_masks'][:]

        if 'url_input_ids' in required_data:
            self.url_input_ids = self.file[f'{split}/url_input_ids'][:]

        if 'url_attention_mask' in required_data:
            self.url_attention_masks = self.file[f'{split}/url_attention_masks'][:]

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

        if hasattr(self, 'screenshots'):
            screenshot = self.screenshots[idx]
            image = self.transform(screenshot)

        input_dict = {
            'html_input_ids': self.html_input_ids[idx].squeeze() if 'html_input_ids' in self.required_data else None,
            'html_attention_mask': self.html_attention_masks[idx].squeeze() if 'html_input_ids' in self.required_data else None,
            'url_input_ids': self.url_input_ids[idx].squeeze() if 'url_input_ids' in self.required_data else None,
            'url_attention_mask': self.url_attention_masks[idx].squeeze() if 'url_attention_mask' in self.required_data else None,
            'image': image if image is not None and image.numel() > 0 else None,
            'label': label
        }

        return {k: v for k, v in input_dict.items() if v is not None}
