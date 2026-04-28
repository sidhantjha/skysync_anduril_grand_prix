from pathlib import Path
import random
import cv2


# ============================================================
# CONFIG
# ============================================================

TII_ROOT = Path(
    r"C:\Users\m_ara\Documents\Sidhant\anduril_gp_rl\drone-racing-dataset"
)

OUTPUT_DATASET_DIR = Path("phase1_cnn/crop_dataset")

AUTONOMOUS_DIR = TII_ROOT / "data" / "autonomous"

SPLITS = {
    "train": [
        "flight-01a-ellipse",
        "flight-02a-ellipse",
    ],
    "val": [
        "flight-03a-ellipse",
    ],
    "test": [
        "flight-04a-ellipse",
    ],
}

# Small first dataset.
# This means max 100 gate crops and 100 no-gate crops per split.
MAX_SAMPLES_PER_CLASS_PER_SPLIT = 100

CROP_SIZE = 224
PADDING_RATIO = 0.15
RANDOM_SEED = 42

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
    cx, cy, w, h = box

    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)
    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)

    return x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, image_width: int, image_height: int):
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return x1, y1, x2, y2


def add_padding_to_box(x1, y1, x2, y2, image_width, image_height, padding_ratio):
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
    x1, y1, x2, y2 = box

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (crop_size, crop_size))

    return crop


def compute_iou(box_a, box_b):
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


def overlaps_any_gate(candidate_box, gate_boxes, max_iou=0.05):
    for gate_box in gate_boxes:
        if compute_iou(candidate_box, gate_box) > max_iou:
            return True

    return False


def sample_no_gate_box(image_width, image_height, gate_boxes, crop_width, crop_height):
    max_attempts = 500

    crop_width = min(crop_width, image_width - 1)
    crop_height = min(crop_height, image_height - 1)

    for _ in range(max_attempts):
        x1 = random.randint(0, image_width - crop_width - 1)
        y1 = random.randint(0, image_height - crop_height - 1)

        x2 = x1 + crop_width
        y2 = y1 + crop_height

        candidate_box = (x1, y1, x2, y2)

        if not overlaps_any_gate(candidate_box, gate_boxes):
            return candidate_box

    return None


def find_camera_and_label_dirs(flight_dir: Path):
    camera_dirs = list(flight_dir.glob("camera_*"))
    label_dirs = list(flight_dir.glob("label_*")) + list(flight_dir.glob("labels_*"))

    if len(camera_dirs) == 0:
        raise FileNotFoundError(f"No camera folder found in {flight_dir}")

    if len(label_dirs) == 0:
        raise FileNotFoundError(f"No label folder found in {flight_dir}")

    return camera_dirs[0], label_dirs[0]


def collect_image_label_pairs(flight_dir: Path):
    camera_dir, label_dir = find_camera_and_label_dirs(flight_dir)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(camera_dir.glob(ext))

    image_paths = sorted(image_paths)

    pairs = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists() and label_path.read_text().strip():
            pairs.append((image_path, label_path))

    return pairs


def prepare_output_dirs():
    for split in SPLITS.keys():
        gate_dir = OUTPUT_DATASET_DIR / split / "gate"
        no_gate_dir = OUTPUT_DATASET_DIR / split / "no_gate"

        gate_dir.mkdir(parents=True, exist_ok=True)
        no_gate_dir.mkdir(parents=True, exist_ok=True)


def process_split(split_name: str, flight_names: list[str]):
    gate_count = 0
    no_gate_count = 0

    max_count = MAX_SAMPLES_PER_CLASS_PER_SPLIT

    gate_output_dir = OUTPUT_DATASET_DIR / split_name / "gate"
    no_gate_output_dir = OUTPUT_DATASET_DIR / split_name / "no_gate"

    all_pairs = []

    for flight_name in flight_names:
        flight_dir = AUTONOMOUS_DIR / flight_name

        if not flight_dir.exists():
            print(f"Skipping missing flight: {flight_dir}")
            continue

        pairs = collect_image_label_pairs(flight_dir)
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)

    for image_path, label_path in all_pairs:
        if gate_count >= max_count and no_gate_count >= max_count:
            break

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        label_lines = label_path.read_text().strip().splitlines()

        labels = []
        for line in label_lines:
            try:
                labels.append(parse_tii_label(line))
            except ValueError:
                continue

        if len(labels) == 0:
            continue

        gate_boxes = []

        for label in labels:
            raw_box = box_to_pixels(label["bbox"], image_width, image_height)
            raw_box = clip_box(*raw_box, image_width, image_height)

            padded_box = add_padding_to_box(
                *raw_box,
                image_width,
                image_height,
                PADDING_RATIO,
            )

            gate_boxes.append(padded_box)

        for gate_box in gate_boxes:
            if gate_count >= max_count and no_gate_count >= max_count:
                break

            gate_width = gate_box[2] - gate_box[0]
            gate_height = gate_box[3] - gate_box[1]

            if gate_width <= 5 or gate_height <= 5:
                continue

            if gate_count < max_count:
                gate_crop = crop_and_resize(image, gate_box, CROP_SIZE)

                if gate_crop is not None:
                    gate_path = gate_output_dir / f"gate_{gate_count:05d}.jpg"
                    cv2.imwrite(str(gate_path), gate_crop)
                    gate_count += 1

            if no_gate_count < max_count:
                no_gate_box = sample_no_gate_box(
                    image_width=image_width,
                    image_height=image_height,
                    gate_boxes=gate_boxes,
                    crop_width=gate_width,
                    crop_height=gate_height,
                )

                if no_gate_box is not None:
                    no_gate_crop = crop_and_resize(image, no_gate_box, CROP_SIZE)

                    if no_gate_crop is not None:
                        no_gate_path = no_gate_output_dir / f"no_gate_{no_gate_count:05d}.jpg"
                        cv2.imwrite(str(no_gate_path), no_gate_crop)
                        no_gate_count += 1

    print(f"{split_name}:")
    print(f"  gate crops:    {gate_count}")
    print(f"  no-gate crops: {no_gate_count}")
    print()


def main():
    random.seed(RANDOM_SEED)

    prepare_output_dirs()

    for split_name, flight_names in SPLITS.items():
        process_split(split_name, flight_names)

    print("Done.")
    print("Dataset saved to:")
    print(OUTPUT_DATASET_DIR)


if __name__ == "__main__":
    main()