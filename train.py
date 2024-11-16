import argparse
import os

from classifiers.vgg_only_classifier import VGGClassifier

efs_mount_path = '/mnt/efs'
if not os.path.exists(efs_mount_path):
    os.makedirs(efs_mount_path)

import torch
import torch.utils.data
from classifiers.basic_transformer_classifier import BasicTransformerClassifier
from classifiers.squeezenet_only_classifier import SqueezenetClassifier
from test_harness import test_harness
import shutil

# Define paths where SageMaker inputs and outputs data
INPUT_DIR = "/opt/ml/input/data"
OUTPUT_DIR = "/opt/ml/model"

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")

    model = VGGClassifier().to(device)
    test_harness(model, epochs=args.epochs, learning_rate=args.lr)
    torch.save(model.state_dict(), f"models/{model.test_name()}_phishing_classifier.pt")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # at the end of training, save run data to output
    if os.path.exists("runs/"):
        shutil.move("runs/", os.path.join(OUTPUT_DIR, "runs/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument("--train_data_path", type=str, default=os.path.join(INPUT_DIR, "train"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    train(args)
