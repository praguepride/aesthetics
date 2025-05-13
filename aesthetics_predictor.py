import os
import argparse
from datetime import datetime

import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

# -----------------------------
# Multi-Layer Perceptron Model
# -----------------------------
class AestheticMLP(pl.LightningModule):
    """
    Aesthetic score predictor using a fully connected neural network.
    """

    def __init__(self, input_size: int, x_key: str = 'emb', y_key: str = 'avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.x_key = x_key
        self.y_key = y_key
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.x_key]
        y = batch[self.y_key].reshape(-1, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.x_key]
        y = batch[self.y_key].reshape(-1, 1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# -----------------------------
# Utilities
# -----------------------------
def normalize_embedding(array: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """
    Normalize embeddings to unit length (L2 norm).
    """
    norm = np.atleast_1d(np.linalg.norm(array, order, axis))
    norm[norm == 0] = 1
    return array / np.expand_dims(norm, axis)


def predict_folder_images(folder_path: str, model, clip_model, preprocess, device: str, output_prefix: str):
    """
    Predict aesthetic scores for all valid images in a given folder.
    """
    valid_ext = ('.jpg', '.jpeg', '.png')
    results = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(valid_ext):
                image_path = os.path.join(root, filename)
                try:
                    source_folder = os.path.basename(os.path.dirname(image_path))
                    filesize = os.path.getsize(image_path)
                    pil_image = Image.open(image_path).convert("RGB")
                    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        clip_features = clip_model.encode_image(image_tensor)
                    embedding = normalize_embedding(clip_features.cpu().numpy())

                    prediction = model(torch.from_numpy(embedding).to(device).float())
                    score = prediction.item()

                    results.append({
                        "SOURCE": source_folder,
                        "FILENAME": filename,
                        "FILESIZE": filesize,
                        "SCORE": round(score, 4)
                    })
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_prefix}_{timestamp}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Saved predictions to {output_filename}")

# -----------------------------
# Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch aesthetic score prediction using CLIP and MLP.")
    parser.add_argument("--model_path", required=True, help="Path to the trained MLP state dict (.pth file)")
    parser.add_argument("--input_root", required=True, help="Root folder containing subfolders of images")
    parser.add_argument("--output_prefix", default="predictions", help="Prefix for output CSV files")
    parser.add_argument("--subfolders", nargs="+", required=True, help="List of subfolder names to process")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MLP model
    model = AestheticMLP(input_size=768)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # Run predictions
    for subfolder in args.subfolders:
        full_path = os.path.join(args.input_root, subfolder)
        print(f"Processing: {full_path}")
        predict_folder_images(full_path, model, clip_model, preprocess, device, f"{args.output_prefix}_{subfolder}")

if __name__ == "__main__":
    main()
