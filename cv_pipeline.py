import io
from pathlib import Path

import cv2
import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize


# ---------------------------- Configuration ----------------------------
XLSX_PATH = "/Users/asmitahargude/Downloads/Cracked_microstructure_dataset.xlsx"
SHEET_NAME = "Dataset"

OUTPUT_DIR = Path("cv_cnn_crack_density_output")
MASK_DIR = OUTPUT_DIR / "cv_masks"
SKELETON_DIR = OUTPUT_DIR / "cv_skeletons"
MODEL_PATH = OUTPUT_DIR / "best_crack_density_cnn.pt"
RESULTS_PATH = OUTPUT_DIR / "cv_and_cnn_crack_density_results.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-3
TEST_SIZE = 0.20
SEED = 42

# Set these using the microscope calibration. Every image is assumed to cover
# this physical field of view. Do not use these example values unless correct.
FIELD_WIDTH_UM = 100.0
FIELD_HEIGHT_UM = 100.0

# Set a material-specific high/low density boundary if available. If None,
# the median CV density across this workbook is used as the boundary.
DENSITY_THRESHOLD = None

# Tune after inspecting the saved masks and skeletons.
MIN_COMPONENT_AREA_PX = 20


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_output_directories():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MASK_DIR.mkdir(exist_ok=True)
    SKELETON_DIR.mkdir(exist_ok=True)


# -------------------------- Pipeline 1: OpenCV --------------------------
def detect_crack_mask(gray_image):
    """Return a binary mask of dark, thin candidate crack regions."""
    gray_image = np.asarray(gray_image, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)

    mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    number_of_components, component_image, stats, _ = (
        cv2.connectedComponentsWithStats(mask)
    )
    clean_mask = np.zeros_like(mask)

    for component_id in range(1, number_of_components):
        if stats[component_id, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA_PX:
            clean_mask[component_image == component_id] = 255

    return clean_mask

def skeleton_length_pixels(skeleton):
    """Estimate total 8-connected skeleton length in pixel units."""
    skeleton = skeleton.astype(bool)

    horizontal_edges = np.sum(skeleton[:, :-1] & skeleton[:, 1:])
    vertical_edges = np.sum(skeleton[:-1, :] & skeleton[1:, :])
    diagonal_down_edges = np.sum(skeleton[:-1, :-1] & skeleton[1:, 1:])
    diagonal_up_edges = np.sum(skeleton[:-1, 1:] & skeleton[1:, :-1])

    return (
        horizontal_edges
        + vertical_edges
        + np.sqrt(2) * (diagonal_down_edges + diagonal_up_edges)
    )


def extract_images_and_calculate_cv_density():
    """Read embedded Excel images and calculate an OpenCV density per image."""
    workbook = openpyxl.load_workbook(XLSX_PATH)
    worksheet = workbook[SHEET_NAME]

    raw_images_for_cnn = []
    cv_densities = []
    records = []

    for image_number, excel_image in enumerate(worksheet._images):
        excel_row = excel_image.anchor._from.row + 1
        source_image = Image.open(io.BytesIO(excel_image._data())).convert("L")
        original_gray = np.asarray(source_image, dtype=np.uint8)
        image_height_px, image_width_px = original_gray.shape

        # Fixed OpenCV pipeline: image -> mask -> skeleton -> density.
        crack_mask = detect_crack_mask(original_gray)
        crack_skeleton = skeletonize(crack_mask > 0)
        crack_length_px = skeleton_length_pixels(crack_skeleton)

        pixel_width_um = FIELD_WIDTH_UM / image_width_px
        pixel_height_um = FIELD_HEIGHT_UM / image_height_px
        mean_pixel_size_um = (pixel_width_um + pixel_height_um) / 2
        image_area_um2 = FIELD_WIDTH_UM * FIELD_HEIGHT_UM
        crack_length_um = crack_length_px * mean_pixel_size_um
        cv_density = crack_length_um / image_area_um2

        Image.fromarray(crack_mask).save(
            MASK_DIR / f"row_{excel_row}_image_{image_number}_mask.png"
        )
        Image.fromarray((crack_skeleton * 255).astype(np.uint8)).save(
            SKELETON_DIR / f"row_{excel_row}_image_{image_number}_skeleton.png"
        )

        # CNN gets ONLY the original image; it never receives the CV mask.
        resized_image = source_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        cnn_input = np.asarray(resized_image, dtype=np.float32) / 255.0
        raw_images_for_cnn.append(cnn_input[np.newaxis, :, :])
        cv_densities.append(cv_density)

        records.append({
            "excel_row": excel_row,
            "image_number": image_number,
            "crack_length_px": crack_length_px,
            "crack_length_um": crack_length_um,
            "image_area_um2": image_area_um2,
            "cv_crack_density_um_per_um2": cv_density,
        })

    if not raw_images_for_cnn:
        raise ValueError("No embedded images were found in the worksheet.")

    return (
        np.asarray(raw_images_for_cnn, dtype=np.float32),
        np.asarray(cv_densities, dtype=np.float32),
        pd.DataFrame(records),
    )