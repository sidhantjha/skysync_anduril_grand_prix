from pathlib import Path
import random

import cv2
import numpy as np


# ============================================================
# DIRECT PATHS: CHANGE ONLY THESE
# ============================================================

IMAGE_PATH = Path(r"C:\Users\m_ara\Documents\Sidhant\anduril_gp_rl\drone-racing-dataset\data\autonomous\flight-01a-ellipse\camera_flight-01a-ellipse\00000_1691756159816137.jpg")
LABEL_PATH = Path(r"C:\Users\m_ara\Documents\Sidhant\anduril_gp_rl\drone-racing-dataset\data\autonomous\flight-01a-ellipse\label_flight-01a-ellipse\00000_1691756159816137.txt")


OUTPUT_DIR = Path("phase1_cnn/outputs")
CROP_DATASET_DIR = Path("phase1_cnn/debug_crop_dataset")

# Add some padding around the gate crop.
# 0.15 means 15% larger than the original bbox.
PADDING_RATIO = 0.15

# Final crop size for CNN.
CROP_SIZE = 224

# ============================================================
# END CONFIG
# ============================================================


def parse_tii_label(line: str) -> dict:
    values = line.strip().split()

    if len(values) < 17:
        raise ValueError(f"Expected at least 17 values, got {len(values)}")

    class_id = int(float(values[0]))

    cx = float(values[1])
    cy = float(values[2])
    w = float(values[3])
    h = float(values[4])

    return {
        "class_id": class_id,
        "bbox": (cx, cy, w, h),
    }


def box_to_pixels(box, image_width: int, image_height: int):
    """
    Converts normalized TII bbox:
        cx, cy, w, h

    into pixel box:
        x1, y1, x2, y2
    """
    cx, cy, w, h = box

    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)

    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)

    return x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, image_width: int, image_height: int):
    """
    Makes sure the box stays inside the image.
    """
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return x1, y1, x2, y2


def add_padding_to_box(x1, y1, x2, y2, image_width, image_height, padding_ratio):
    """
    Makes the crop slightly larger than the bbox.

    This helps the CNN see some context around the gate.
    """
    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y

    return clip_box(x1, y1, x2, y2, image_width, image_height)


def crop_and_resize(image, box, crop_size):
    """
    Crops image using pixel box and resizes to CNN input size.
    """
    x1, y1, x2, y2 = box

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Crop is empty. Check bbox conversion.")

    crop = cv2.resize(crop, (crop_size, crop_size))

    return crop


def compute_iou(box_a, box_b):
    """
    Computes intersection-over-union between two pixel boxes.
    Used to make sure no-gate crop does not overlap gate bbox.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def sample_no_gate_box(image_width, image_height, gate_box, crop_width, crop_height):
    """
    Samples a random crop that does not overlap much with the gate bbox.
    """
    max_attempts = 500

    crop_width = min(crop_width, image_width - 1)
    crop_height = min(crop_height, image_height - 1)

    for _ in range(max_attempts):
        x1 = random.randint(0, image_width - crop_width - 1)
        y1 = random.randint(0, image_height - crop_height - 1)

        x2 = x1 + crop_width
        y2 = y1 + crop_height

        candidate_box = (x1, y1, x2, y2)

        iou = compute_iou(candidate_box, gate_box)

        if iou < 0.05:
            return candidate_box

    raise RuntimeError("Could not find a no-gate crop with low overlap.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gate_dir = CROP_DATASET_DIR / "gate"
    no_gate_dir = CROP_DATASET_DIR / "no_gate"

    gate_dir.mkdir(parents=True, exist_ok=True)
    no_gate_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Could not find label: {LABEL_PATH}")

    image_height, image_width = image.shape[:2]

    label_lines = LABEL_PATH.read_text().strip().splitlines()

    if len(label_lines) == 0:
        raise ValueError("Label file is empty.")

    label = parse_tii_label(label_lines[0])

    raw_gate_box = box_to_pixels(
        label["bbox"],
        image_width,
        image_height,
    )

    raw_gate_box = clip_box(
        *raw_gate_box,
        image_width,
        image_height,
    )

    gate_box = add_padding_to_box(
        *raw_gate_box,
        image_width,
        image_height,
        PADDING_RATIO,
    )

    gate_crop = crop_and_resize(image, gate_box, CROP_SIZE)

    gate_width = gate_box[2] - gate_box[0]
    gate_height = gate_box[3] - gate_box[1]

    no_gate_box = sample_no_gate_box(
        image_width=image_width,
        image_height=image_height,
        gate_box=gate_box,
        crop_width=gate_width,
        crop_height=gate_height,
    )

    no_gate_crop = crop_and_resize(image, no_gate_box, CROP_SIZE)

    gate_crop_path = gate_dir / "gate_000001.jpg"
    no_gate_crop_path = no_gate_dir / "no_gate_000001.jpg"

    cv2.imwrite(str(gate_crop_path), gate_crop)
    cv2.imwrite(str(no_gate_crop_path), no_gate_crop)

    debug_image = image.copy()

    gx1, gy1, gx2, gy2 = gate_box
    nx1, ny1, nx2, ny2 = no_gate_box

    cv2.rectangle(debug_image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
    cv2.putText(
        debug_image,
        "GATE CROP",
        (gx1, max(gy1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.rectangle(debug_image, (nx1, ny1), (nx2, ny2), (0, 0, 255), 3)
    cv2.putText(
        debug_image,
        "NO GATE CROP",
        (nx1, max(ny1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    debug_path = OUTPUT_DIR / "block_1_2_crop_debug.png"
    cv2.imwrite(str(debug_path), debug_image)

    print("Saved gate crop:")
    print(gate_crop_path)
    print()

    print("Saved no-gate crop:")
    print(no_gate_crop_path)
    print()

    print("Saved debug image:")
    print(debug_path)
    print()

    print("Gate crop shape:")
    print(gate_crop.shape)
    print()

    print("No-gate crop shape:")
    print(no_gate_crop.shape)
    print()

    cv2.imshow("Gate Crop", gate_crop)
    cv2.imshow("No Gate Crop", no_gate_crop)
    cv2.imshow("Crop Debug", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()