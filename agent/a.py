import os
import cv2
import numpy as np

# TODO CHANGE THIS TO NON HARD CODED FOR THE CHANGE ANALYSIS
DETECTIONS = [
    # Roads
    (29, 0, 90, 37, "road"),
    (0, 37, 256, 197, "road"),
    # Buildings
    (0, 0, 9, 5, "building"),
    (102, 0, 140, 9, "building"),
    (198, 0, 240, 27, "building"),
    (0, 21, 25, 63, "building"),
    (188, 34, 233, 65, "building"),
    (36, 50, 80, 95, "building"),
    (185, 71, 227, 103, "building"),
    (192, 118, 224, 163, "building"),
    (86, 122, 130, 164, "building"),
    (139, 122, 172, 164, "building"),
    (39, 124, 73, 166, "building"),
    (162, 208, 200, 251, "building"),
    (208, 208, 248, 243, "building"),
    (0, 209, 20, 253, "building"),
    (72, 210, 110, 244, "building"),
    (253, 211, 256, 221, "building"),
    (116, 212, 152, 240, "building"),
    (27, 214, 62, 248, "building"),
]

# Color palette (BGR format for OpenCV)
COLORS = {
    "road": (0, 255, 255),  # Yellow
    "building": (0, 0, 255),  # Red
}


def annotate_image(image_path, detections, output_path="annotated_changes.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    annotated = img.copy()

    # Scale factors from 256x256 model resolution to actual image dimensions
    scale_x = w / 256.0
    scale_y = h / 256.0

    for x1, y1, x2, y2, label in detections:
        # Scale bounding box coordinates
        bx1 = int(round(x1 * scale_x))
        by1 = int(round(y1 * scale_y))
        bx2 = int(round(x2 * scale_x))
        by2 = int(round(y2 * scale_y))

        color = COLORS.get(label, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness=2)

        # Draw label tag
        tag = label.capitalize()
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tag_y1 = max(0, by1 - th - 4)
        tag_y2 = by1
        cv2.rectangle(annotated, (bx1, tag_y1), (bx1 + tw + 4, tag_y2), color, -1)
        cv2.putText(
            annotated,
            tag,
            (bx1 + 2, by1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Save individual annotated image
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to: {output_path}")
    return annotated


def create_side_by_side(
    img_a_path, img_b_path, detections, output_path="comparison.png"
):
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    if img_a is None or img_b is None:
        print(
            f"Warning: Could not create comparison because one of the images was not found."
        )
        return

    # Resize image A to match image B height if slightly different
    h_b, w_b = img_b.shape[:2]
    img_a = cv2.resize(img_a, (w_b, h_b))

    annotated_b = annotate_image(img_b_path, detections, output_path="annotated_b.png")

    # Add titles on top
    banner_height = 30
    banner_a = np.zeros((banner_height, w_b, 3), dtype=np.uint8)
    banner_b = np.zeros((banner_height, w_b, 3), dtype=np.uint8)

    cv2.putText(
        banner_a,
        "Pre-Phase (Before)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        banner_b,
        "Post-Phase + Changes",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    panel_a = np.vstack([banner_a, img_a])
    panel_b = np.vstack([banner_b, annotated_b])

    combined = np.hstack([panel_a, panel_b])
    cv2.imwrite(output_path, combined)
    print(f"Saved side-by-side comparison to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Change-Agent detections")
    parser.add_argument(
        "--imgA", default="1.png", help="Path to pre-phase image (optional)"
    )
    parser.add_argument("--imgB", default="2.png", help="Path to post-phase image")
    parser.add_argument(
        "--output", default="annotated_changes.png", help="Path to output image"
    )
    args = parser.parse_args()

    # 1. Annotate post-phase image
    if os.path.exists(args.imgB):
        annotate_image(args.imgB, DETECTIONS, output_path=args.output)
    else:
        print(f"File not found: {args.imgB}")

    # 2. If both pre-phase and post-phase images are present, create comparison
    if os.path.exists(args.imgA) and os.path.exists(args.imgB):
        create_side_by_side(args.imgA, args.imgB, DETECTIONS)
