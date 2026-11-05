import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from cv_pipeline import (
    extract_images_and_calculate_cv_density,
    BATCH_SIZE,
    DENSITY_THRESHOLD,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    SEED,
    TEST_SIZE,
)
from cv_pipeline import extract_images_and_calculate_cv_density
X, cv_density, results_df = extract_images_and_calculate_cv_density()

# ---------------------- Pipeline 2: CNN regression ----------------------
class CrackDensityCNN(nn.Module):
    """Predict one crack-density value from one raw grayscale image."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image):
        return self.regression_head(self.features(image)).squeeze(1)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def main():
    set_seed()
    make_output_directories()

    X, cv_density, results_df = extract_images_and_calculate_cv_density()
    if len(X) < 5:
        raise ValueError("At least five embedded images are needed for this split.")

    # The precomputed CV density is the CNN's pseudo-target. Standardizing it
    # makes the regression loss numerically stable.
    density_mean = float(cv_density.mean())
    density_std = float(cv_density.std())
    if density_std == 0:
        raise ValueError("CV densities are identical; inspect CV masks/settings.")
    cv_density_standardized = (cv_density - density_mean) / density_std

    threshold = (
        float(np.median(cv_density))
        if DENSITY_THRESHOLD is None
        else float(DENSITY_THRESHOLD)
    )

    indices = np.arange(len(X))
    high_low_classes = (cv_density >= threshold).astype(int)
    class_counts = np.bincount(high_low_classes, minlength=2)
    use_stratified_split = np.min(class_counts) >= 2

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=high_low_classes if use_stratified_split else None,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[train_indices], dtype=torch.float32),
            torch.tensor(cv_density_standardized[train_indices], dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[test_indices], dtype=torch.float32),
            torch.tensor(cv_density_standardized[test_indices], dtype=torch.float32),
        ),
