from torch.utils.data import Dataset
import h5py
import torch
import os
import boto3
import time
from transformers import DistilBertTokenizer

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
        self.images = self.file[f'{split}/images'][:]
        self.urls = self.file[f'{split}/urls'][:]

        if 'html_input_ids' in required_data:
            html_input_ids = self.file[f'{split}/html_input_ids'][:]

            html_input_ids = [torch.tensor(item, dtype=torch.long) for item in html_input_ids]
            self.html_input_ids = html_input_ids

        if 'html_attention_mask' in required_data:
            html_attention_masks = self.file[f'{split}/html_attention_masks'][:]

            html_attention_masks = [torch.tensor(item, dtype=torch.float) for item in html_attention_masks]
            self.html_attention_masks = html_attention_masks

        if 'longformer_input_ids' in required_data:
            longformer_input_ids = self.file[f'{split}/longformer_input_ids'][:]

            longformer_input_ids = [torch.tensor(item, dtype=torch.long) for item in longformer_input_ids]
            self.longformer_input_ids = longformer_input_ids

        if 'longformer_attention_mask' in required_data:
            longformer_attention_mask = self.file[f'{split}/longformer_attention_masks'][:]

            longformer_attention_mask = [torch.tensor(item, dtype=torch.float) for item in longformer_attention_mask]
            self.longformer_attention_mask = longformer_attention_mask

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
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        input_dict = {
            'html_input_ids': self.html_input_ids[idx] if 'html_input_ids' in self.required_data else None,
            'html_attention_mask': self.html_attention_masks[idx] if 'html_input_ids' in self.required_data else None,
            'longformer_input_ids': self.longformer_input_ids[idx] if 'longformer_input_ids' in self.required_data else None,
            'longformer_attention_mask': self.longformer_attention_mask[idx] if 'longformer_input_ids' in self.required_data else None,
            'url_input_ids': self.url_input_ids[idx] if 'url_input_ids' in self.required_data else None,
            'url_attention_mask': self.url_attention_masks[idx] if 'url_attention_mask' in self.required_data else None,
            'url': self.urls[idx].decode('utf-8') if 'url' in self.required_data else None,
            'image': self.images[idx] if 'image' in self.required_data else None,
            'label': label
        }

        return {k: v for k, v in input_dict.items() if v is not None}
