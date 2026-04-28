import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Config Values
IMG_PATH = "data/030105.png"
GATE_COLOR = "black"
GATE_WIDTH_METERS = 1.5
GATE_HEIGHT_METERS = 1.5
# Mask cleaning size
KERNEL_SIZE = 3
MIN_CONTOUR_AREA = 100.0
# fx = focal length in pixels in the x direction
# fy = focal length in pixels in the y direction
# cx = x-coordinate of the camera center in the image
# cy = y-coordinate of the camera center in the image
FX = None 
FY = None
CX = None
CY = None
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)
BLACK_V_MAX = 60
OUTPUT_DIR = "outputs"

# Load and return the image
def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
    return image

# Create gate mask
def create_gate_mask(image: np.ndarray, color:str, black_v_max:int) -> np.ndarray:
    #convert images to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([180,255,black_v_max])
    mask = cv2.inRange(hsv,lower,upper)
    return mask

# Clean mask and opening removes tiny white noise and closing fills small holes and connects broken white regions
def clean_mask(mask:np.ndarray, kernel_size:int) -> np.ndarray:
    kernel = np.ones((kernel_size,kernel_size), dtype = np.uint8)
    opened = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    #closed = cv2.morphologyEx(opened,cv2.MORPH_CLOSE,kernel)
    return opened

# find best contour
def find_best_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    mask_height, mask_width = mask.shape[:2]
    image_area = mask_height * mask_width

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Reject contours touching the image border.
        # These are usually background/ceiling/floor blobs.
        touches_border = (
            x <= 2 or
            y <= 2 or
            x + w >= mask_width - 2 or
            y + h >= mask_height - 2
        )

        if touches_border:
            continue

        # Reject massive blobs.
        if area > 0.25 * image_area:
            continue

        aspect_ratio = w / float(h)

        # Gate should be roughly square-ish, but allow perspective distortion.
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue

        candidates.append((area, contour))

    if len(candidates) == 0:
        return None

    # Pick largest remaining valid contour.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]

# contour to four corners
def contour_to_four_corners(contour:np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour, epsilon = 0.03*perimeter, closed = True)
    if len(approx) == 4:
        corners = approx.reshape(4,2).astype(np.float32)
        return corners

    return "Can't generate contour corners"

# order corners correctly
def order_corners(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)

    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]

    top_right = points[np.argmax(diffs)]
    bottom_left = points[np.argmin(diffs)]

    ordered = np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )

    return ordered

#create camera matrix
def build_camera_matrix(
    image_width: int,
    image_height: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> np.ndarray:
    """
    Builds the camera matrix.

    If no camera intrinsics are provided, uses a rough approximation.
    This is okay for the first demo but not for accurate distance estimates.
    """
    if fx is None:
        fx = 0.9 * image_width

    if fy is None:
        fy = 0.9 * image_width

    if cx is None:
        cx = image_width / 2.0

    if cy is None:
        cy = image_height / 2.0

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return camera_matrix

# build gate_object_points
def build_gate_object_points(gate_width: float, gate_height: float) -> np.ndarray:
    """
    Defines the real 3D gate corners in meters.

    Gate center is at (0, 0, 0).
    Gate lies on the Z = 0 plane.

    Order:
    top-left, top-right, bottom-right, bottom-left.
    """
    w = gate_width / 2.0
    h = gate_height / 2.0

    object_points = np.array(
        [
            [-w, h, 0.0],
            [w, h, 0.0],
            [w, -h, 0.0],
            [-w, -h, 0.0],
        ],
        dtype=np.float32,
    )

    return object_points 

def solve_gate_pose(
    image_points: np.ndarray,
    gate_width: float,
    gate_height: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Runs solvePnP to estimate gate pose.
    """
    object_points = build_gate_object_points(gate_width, gate_height)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return False, None, None

    return True, rvec, tvec

def rvec_to_euler_degrees(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Converts OpenCV rotation vector to approximate roll, pitch, yaw in degrees.
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def compute_reprojection_error(
    image_points: np.ndarray,
    gate_width: float,
    gate_height: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """
    Projects the 3D gate corners back into the image and compares
    them with the detected 2D corners.

    Lower reprojection error is better.
    """
    object_points = build_gate_object_points(gate_width, gate_height)

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected_points = projected_points.reshape(-1, 2)

    error = np.linalg.norm(projected_points - image_points, axis=1)
    mean_error = float(np.mean(error))

    return mean_error


def draw_results(
    image: np.ndarray,
    image_points: np.ndarray,
    gate_width: float,
    gate_height: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reprojection_error: float,
) -> np.ndarray:
    """
    Draws detected gate outline, corners, 3D axes, and pose text.
    """
    output = image.copy()

    corners_int = image_points.astype(np.int32)

    cv2.polylines(output, [corners_int], True, (255, 0, 255), 3)

    labels = ["TL", "TR", "BR", "BL"]

    for point, label in zip(corners_int, labels):
        x, y = int(point[0]), int(point[1])

        cv2.circle(output, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            output,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    axis_length = min(gate_width, gate_height) * 0.5

    axis_points_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )

    axis_points_2d, _ = cv2.projectPoints(
        axis_points_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    axis_points_2d = axis_points_2d.reshape(-1, 2).astype(np.int32)

    origin = tuple(axis_points_2d[0])
    x_axis = tuple(axis_points_2d[1])
    y_axis = tuple(axis_points_2d[2])
    z_axis = tuple(axis_points_2d[3])

    cv2.line(output, origin, x_axis, (0, 0, 255), 3)
    cv2.line(output, origin, y_axis, (0, 255, 0), 3)
    cv2.line(output, origin, z_axis, (255, 0, 0), 3)

    cv2.putText(output, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(output, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    x, y, z = tvec.flatten()
    roll, pitch, yaw = rvec_to_euler_degrees(rvec)

    text_lines = [
        f"x: {x:.2f} m",
        f"y: {y:.2f} m",
        f"z: {z:.2f} m",
        f"roll: {roll:.1f} deg",
        f"pitch: {pitch:.1f} deg",
        f"yaw: {yaw:.1f} deg",
        f"reproj err: {reprojection_error:.2f} px",
    ]

    for i, text in enumerate(text_lines):
        cv2.putText(
            output,
            text,
            (20, 35 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return output

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading image...")
    image = load_image(IMG_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {IMG_PATH}")

    image_height, image_width = image.shape[:2]

    def show_step(window_name: str, img: np.ndarray, save_name: str) -> None:
        save_path = output_dir / save_name
        cv2.imwrite(str(save_path), img)

        print(f"Showing: {window_name}")
        print(f"Saved: {save_path}")
        print("Press any key on the image window to continue.")
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    show_step("Step 1 - Original Image", image, "step_1_original.png")

    print("Step 2: Creating black gate mask...")
    raw_mask = create_gate_mask(
        image=image,
        color=GATE_COLOR,
        black_v_max=BLACK_V_MAX
    )

    show_step("Step 2 - Raw Black Gate Mask", raw_mask, "step_2_raw_mask.png")

    print("Step 3: Cleaning mask...")
    cleaned_mask = clean_mask(
        mask=raw_mask,
        kernel_size=KERNEL_SIZE
    )

    show_step("Step 3 - Cleaned Mask", cleaned_mask, "step_3_cleaned_mask.png")

    print("Step 3B: Inverting mask for contour detection...")

    # IMPORTANT:
    # cv2.findContours finds WHITE blobs.
    # In your current mask, the gate is black and the background is white.
    # So we invert the mask to make the gate white.
    mask_for_contours = (cleaned_mask)

    show_step(
        "Step 3B - Inverted Mask For Contours",
        mask_for_contours,
        "step_3b_inverted_mask_for_contours.png"
    )

    print("Step 4: Finding best contour...")
    contour = find_best_contour(
        mask=mask_for_contours,
        min_area=MIN_CONTOUR_AREA
    )

    if contour is None:
        print("No useful contour found.")
        print("Open step_2_raw_mask.png and step_3_cleaned_mask.png.")
        print("The gate should be white and the background should be mostly black.")
        print()
        print("For a black gate:")
        print("- If the gate is missing, increase BLACK_V_MAX.")
        print("- If too much background is white, decrease BLACK_V_MAX.")
        print("- Try BLACK_V_MAX = 50, 70, 90, 110, 130.")
        cv2.destroyAllWindows()
        return

    contour_debug = image.copy()
    cv2.drawContours(contour_debug, [contour], -1, (0, 255, 0), 3)

    show_step("Step 4 - Best Contour", contour_debug, "step_4_best_contour.png")

    print("Step 5: Converting contour to 4 corners...")
    corners = contour_to_four_corners(contour)

    corners_debug = image.copy()

    for point in corners.astype(int):
        x, y = point
        cv2.circle(corners_debug, (x, y), 7, (0, 255, 255), -1)

    cv2.polylines(
        corners_debug,
        [corners.astype(np.int32)],
        True,
        (255, 0, 255),
        3
    )

    show_step("Step 5 - Unordered Corners", corners_debug, "step_5_unordered_corners.png")

    print("Step 6: Ordering corners...")
    ordered_corners = order_corners(corners)

    ordered_debug = image.copy()
    labels = ["TL", "TR", "BR", "BL"]

    for point, label in zip(ordered_corners.astype(int), labels):
        x, y = point
        cv2.circle(ordered_debug, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(
            ordered_debug,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.polylines(
        ordered_debug,
        [ordered_corners.astype(np.int32)],
        True,
        (255, 0, 255),
        3
    )

    show_step("Step 6 - Ordered Corners", ordered_debug, "step_6_ordered_corners.png")

    print("Step 7: Building camera matrix...")
    camera_matrix = build_camera_matrix(
        image_width=image_width,
        image_height=image_height,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY
    )

    print("Camera matrix:")
    print(camera_matrix)
    print()

    print("Step 8: Running solvePnP...")
    success, rvec, tvec = solve_gate_pose(
        image_points=ordered_corners,
        gate_width=GATE_WIDTH_METERS,
        gate_height=GATE_HEIGHT_METERS,
        camera_matrix=camera_matrix,
        dist_coeffs=DIST_COEFFS
    )

    if not success or rvec is None or tvec is None:
        print("solvePnP failed.")
        cv2.destroyAllWindows()
        return

    print("Step 9: Computing reprojection error...")
    reprojection_error = compute_reprojection_error(
        image_points=ordered_corners,
        gate_width=GATE_WIDTH_METERS,
        gate_height=GATE_HEIGHT_METERS,
        camera_matrix=camera_matrix,
        dist_coeffs=DIST_COEFFS,
        rvec=rvec,
        tvec=tvec
    )

    print("Step 10: Drawing final pose result...")
    output = draw_results(
        image=image,
        image_points=ordered_corners,
        gate_width=GATE_WIDTH_METERS,
        gate_height=GATE_HEIGHT_METERS,
        camera_matrix=camera_matrix,
        dist_coeffs=DIST_COEFFS,
        rvec=rvec,
        tvec=tvec,
        reprojection_error=reprojection_error
    )

    show_step("Step 10 - Final Pose Result", output, "step_10_final_pose_result.png")

    x, y, z = tvec.flatten()
    roll, pitch, yaw = rvec_to_euler_degrees(rvec)

    print("Gate pose estimated.")
    print()
    print("Ordered image corners:")
    print(ordered_corners)
    print()
    print("Translation vector tvec:")
    print(f"x: {x:.3f} m")
    print(f"y: {y:.3f} m")
    print(f"z: {z:.3f} m")
    print()
    print("Rotation estimate:")
    print(f"roll:  {roll:.3f} deg")
    print(f"pitch: {pitch:.3f} deg")
    print(f"yaw:   {yaw:.3f} deg")
    print()
    print(f"Mean reprojection error: {reprojection_error:.3f} px")
    print()
    print(f"All step images saved inside: {output_dir}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()