from pathlib import Path
import cv2
import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(r"C:\Users\m_ara\Documents\Sidhant\anduril_gp_rl\drone-racing-dataset\data\autonomous\flight-01a-ellipse\camera_flight-01a-ellipse\00000_1691756159816137.jpg")
LABEL_PATH = Path(r"C:\Users\m_ara\Documents\Sidhant\anduril_gp_rl\drone-racing-dataset\data\autonomous\flight-01a-ellipse\label_flight-01a-ellipse\00000_1691756159816137.txt")

OUTPUT_DIR = Path("phase1_cnn/outputs")
OUTPUT_NAME = "tii_label_visualization.png"
OUTPUT_PATH = Path("phase1_cnn/outputs/direct_label_visualization.png")

# ============================================================
# END CONFIG
# ============================================================

"""
    TII label format:
    0 cx cy w h tlx tly tlv trx try trv brx bry brv blx bly blv

    cx, cy, w, h - normalized bbox values.
    tl/tr/br/bl - normalized corner coordinates.
"""

def parse_tii_label(line: str) -> dict:
    values = line.strip().split()
    if (len(values) < 17):
        print("Expected complete label set!")
    class_id = int(float(values[0]))
    cx = float(values[1])
    cy = float(values[2])
    w = float(values[3])
    h = float(values[4])

    tl = (float(values[5]), float(values[6]), int(float(values[7])))
    tr = (float(values[8]), float(values[9]), int(float(values[10])))
    br = (float(values[11]), float(values[12]), int(float(values[13])))
    bl = (float(values[14]), float(values[15]), int(float(values[16])))

    return {
        "class_id": class_id,
        "bbox": (cx, cy, w, h),
        "keypoints": {
            "TL": tl,
            "TR": tr,
            "BR": br,
            "BL": bl,
        },
    }

def box_to_pixels(box, image_width, image_height):
    cx , cy , w, h = box
    x1 = int((cx-w/2) * image_width)
    x2 = int((cx+w/2) * image_width)
    y1 = int((cy-h/2) * image_height)
    y2 = int((cy+h/2) * image_height)
    return x1,y1,x2,y2

def point_to_pixels(x_norm: float, y_norm: float, image_width: int, image_height: int):
    x = int(x_norm * image_width)
    y = int(y_norm* image_height)
    return x,y

def draw_label(image, label: dict):
    output = image.copy()

    image_height, image_width = output.shape[:2]

    # Draw bounding box
    x1, y1, x2, y2 = box_to_pixels(
        label["bbox"],
        image_width,
        image_height,
    )

    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)

    cv2.putText(
        output,
        "gate bbox",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
    )

    # Draw corners
    colors = {
        "TL": (0, 255, 255),
        "TR": (0, 255, 0),
        "BR": (255, 0, 0),
        "BL": (0, 0, 255),
    }

    for name, (x_norm, y_norm, visibility) in label["keypoints"].items():
        x, y = point_to_pixels(
            x_norm,
            y_norm,
            image_width,
            image_height,
        )

        color = colors[name]

        cv2.circle(output, (x, y), 6, color, -1)

        cv2.putText(
            output,
            f"{name}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return output


def image_to_tensor(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    final_image = cv2.resize(image_rgb, (224, 224))
    tensor = torch.from_numpy(final_image).permute(2,0,1).float()
    tensor = tensor / 255.0
    return tensor

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))
    print("OpenCV image shape:")
    print(image.shape)
    print("Meaning: height x width x channels")
    print()
    label_lines = LABEL_PATH.read_text().strip().splitlines()
    label = parse_tii_label(label_lines[0])
    print("Parsed label:")
    print(label)
    print()
    tensor = image_to_tensor(image)

    print("PyTorch tensor shape:")
    print(tensor.shape)
    print("Meaning: channels x height x width")
    print()
    print("Tensor value range:")
    print(f"min = {tensor.min().item():.4f}")
    print(f"max = {tensor.max().item():.4f}")
    print()

    output = draw_label(image, label)

    cv2.imwrite(str(OUTPUT_PATH), output)

    print("Saved output to:")
    print(OUTPUT_PATH)

    cv2.imshow("Phase 1.1 Direct Label Inspector", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
