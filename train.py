import argparse
import os
current_directory = os.getcwd()
print(f"Current directory: {current_directory}")
contents = os.listdir(current_directory)
print("Contents of the current directory:")
for item in contents:
    print(item)
import torch
import torch.utils.data
from basic_transformer_classifier import BasicTransformerClassifier
from test_harness import test_harness
import shutil

# Define paths where SageMaker inputs and outputs data
INPUT_DIR = "/opt/ml/input/data"
OUTPUT_DIR = "/opt/ml/model"

dataset_path = os.path.expanduser("~/transfer/phishing_output.h5")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")

    model = BasicTransformerClassifier().to(device)
    test_harness(model, local_dataset=dataset_path, epochs=args.epochs, learning_rate=args.lr)
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
