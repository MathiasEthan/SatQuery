import os
import re
import cv2
import numpy as np

COLORS = {
    "road": (0, 255, 255),  # Yellow
    "building": (0, 0, 255),  # Red
}


def parse_detections(data):
    """
    Parses either a JSON dict like {"response": "..."} or a raw string like:
    '(area at (29, 0, 90, 37) turned to road), (area at (0, 0, 9, 5) turned to building)'

    Returns a list of tuples: (x1, y1, x2, y2, label)
    """
    if isinstance(data, dict):
        text = data.get("response", "")
    else:
        text = str(data)

    pattern = r"at\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*turned to\s*([a-zA-Z_]+)"
    matches = re.findall(pattern, text)
    return [
        (int(x1), int(y1), int(x2), int(y2), label.strip().lower())
        for x1, y1, x2, y2, label in matches
    ]


def annotate_image(
    image_path, detections_or_response, output_path="annotated_changes.png"
):
    """
    Annotates image with bounding boxes.
    detections_or_response can be:
      - a raw string / dict from the model, OR
      - an already-parsed list of (x1, y1, x2, y2, label) tuples.
    """
    if isinstance(detections_or_response, (str, dict)):
        detections = parse_detections(detections_or_response)
    else:
        detections = detections_or_response

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

    return annotated


def create_side_by_side(
    img_a_path, img_b_path, detections_or_response, output_path="comparison.png"
):
    if isinstance(detections_or_response, (str, dict)):
        detections = parse_detections(detections_or_response)
    else:
        detections = detections_or_response

    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    if img_a is None or img_b is None:
        print(
            "Warning: Could not create comparison because one of the images was not found."
        )
        return None

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
    cv2.imwrite("tmp/" + output_path, combined)
    print(f"Saved side-by-side comparison to: {output_path}")
    return combined


def process_and_visualize(
    model_output,
    img_b_path,
    img_a_path=None,
    output_path="annotated_changes.png",
):
    """
    Convenience wrapper: takes the model output (dict or string),
    annotates img_b, and optionally produces a side-by-side comparison with img_a.
    """
    annotate_image(img_b_path, model_output, output_path=output_path)
    if img_a_path and os.path.exists(img_a_path):
        create_side_by_side(
            img_a_path, img_b_path, model_output, output_path=output_path
        )
    return output_path
