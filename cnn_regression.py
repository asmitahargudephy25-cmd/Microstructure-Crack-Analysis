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
    RESULTS_PATH,
    SEED,
    TEST_SIZE,
    set_seed,
    make_output_directories,
)

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
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = select_device()
    model = CrackDensityCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    consistency_loss_function = nn.SmoothL1Loss()

    print(f"Images: {len(X)}; device: {device}")
    print(f"High/low density threshold: {threshold:.8f} µm/µm²")
    print("Training target: precomputed CV crack density.")

    best_test_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        training_loss_sum = 0.0

        for raw_images, cv_density_target in train_loader:
            raw_images = raw_images.to(device)
            cv_density_target = cv_density_target.to(device)

            cnn_predicted_density = model(raw_images)

            # The two pipelines meet here: CV density is the fixed reference,
            # and the CNN is optimized to give a close density prediction.
            cv_cnn_consistency_loss = consistency_loss_function(
                cnn_predicted_density,
                cv_density_target,
            )

            optimizer.zero_grad()
            cv_cnn_consistency_loss.backward()
            optimizer.step()

            training_loss_sum += (
                cv_cnn_consistency_loss.item() * raw_images.size(0)
            )

        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for raw_images, cv_density_target in test_loader:
                raw_images = raw_images.to(device)
                cv_density_target = cv_density_target.to(device)
                cnn_predicted_density = model(raw_images)
                test_loss = consistency_loss_function(
                    cnn_predicted_density,
                    cv_density_target,
                )
                test_loss_sum += test_loss.item() * raw_images.size(0)

        mean_training_loss = training_loss_sum / len(train_indices)
        mean_test_loss = test_loss_sum / len(test_indices)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train consistency loss: {mean_training_loss:.5f} | "
            f"test consistency loss: {mean_test_loss:.5f}"
        )

        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "density_mean": density_mean,
                "density_std": density_std,
                "density_threshold": threshold,
            }, MODEL_PATH)

    # -------------------------- Compare final outputs --------------------------
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        cnn_standardized_density = model(
            torch.tensor(X, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # Return CNN predictions to the same physical units used by OpenCV.
    cnn_density = cnn_standardized_density * density_std + density_mean
    cnn_density = np.maximum(cnn_density, 0.0)

    results_df["cnn_predicted_density_um_per_um2"] = cnn_density
    results_df["absolute_cv_cnn_difference"] = np.abs(
        results_df["cv_crack_density_um_per_um2"]
        - results_df["cnn_predicted_density_um_per_um2"]
    )
    results_df["cv_density_class"] = np.where(
        results_df["cv_crack_density_um_per_um2"] >= threshold,
        "high_crack_density",
        "low_crack_density",
    )
    results_df["cnn_density_class"] = np.where(
        results_df["cnn_predicted_density_um_per_um2"] >= threshold,
        "high_crack_density",
        "low_crack_density",
    )
    results_df["split"] = "train"
    results_df.loc[test_indices, "split"] = "test"
    results_df["high_low_agree"] = (
        results_df["cv_density_class"] == results_df["cnn_density_class"]
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nBest test consistency loss: {best_test_loss:.5f}")
    print(
        "Mean CV-CNN absolute difference: "
        f"{results_df['absolute_cv_cnn_difference'].mean():.8f} µm/µm²"
    )
    print(f"High/low class agreement: {results_df['high_low_agree'].mean():.1%}")
    print(f"Results: {RESULTS_PATH.resolve()}")
    print(f"Model: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()