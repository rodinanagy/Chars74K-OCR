import argparse
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from model import CLASSES, infer_tf, load_model

MIN_CHAR_HEIGHT = 8
MIN_CHAR_WIDTH = 4
LINE_MERGE_GAP = 0.6
SPACE_GAP_RATIO = 0.8


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_crop(crop_bgr, model, device):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = infer_tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return CLASSES[model(tensor).argmax(1).item()]


def read_image(image_path, model, device, save_vis=None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read {image_path}", file=sys.stderr)
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 127:
        gray = 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = [
        (stats[l, cv2.CC_STAT_LEFT], stats[l, cv2.CC_STAT_TOP],
         stats[l, cv2.CC_STAT_WIDTH], stats[l, cv2.CC_STAT_HEIGHT])
        for l in range(1, num_labels)
        if stats[l, cv2.CC_STAT_WIDTH] >= MIN_CHAR_WIDTH
        and stats[l, cv2.CC_STAT_HEIGHT] >= MIN_CHAR_HEIGHT
    ]

    if not boxes:
        print("No characters found — try lowering MIN_CHAR_HEIGHT / MIN_CHAR_WIDTH")
        return ""

    boxes.sort(key=lambda b: b[1])
    median_h = float(np.median([b[3] for b in boxes]))
    merge_dist = median_h * LINE_MERGE_GAP

    lines = []
    for box in boxes:
        cy = box[1] + box[3] / 2
        placed = False
        for line in lines:
            if abs(cy - np.mean([b[1] + b[3] / 2 for b in line])) < merge_dist:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    lines.sort(key=lambda ln: np.mean([b[1] for b in ln]))
    for line in lines:
        line.sort(key=lambda b: b[0])

    median_w = float(np.median([b[2] for b in boxes]))
    space_thresh = median_w * SPACE_GAP_RATIO
    vis = img.copy()
    result_lines = []

    for line in lines:
        text = ""
        prev_right = None
        for (x, y, w, h) in line:
            if prev_right is not None and (x - prev_right) > space_thresh:
                text += " "
            text += classify_crop(img[y:y+h, x:x+w], model, device)
            prev_right = x + w
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 1)
        result_lines.append(text)

    if save_vis:
        cv2.imwrite(save_vis, vis)
        print(f"Visualisation saved to {save_vis}")

    return "\n".join(result_lines)


def read_single_char(image_path, model, device):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot read {image_path}", file=sys.stderr)
        sys.exit(1)
    return classify_crop(img, model, device)


def main():
    parser = argparse.ArgumentParser(description="Chars74K OCR - read text from images")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default="chars74k.pth", help="Path to model weights (default: chars74k.pth)")
    parser.add_argument("--single-char", action="store_true", help="Treat the image as a single pre-cropped character")
    parser.add_argument("--save-vis", default=None, metavar="PATH", help="Save annotated image with bounding boxes to PATH")
    parser.add_argument("--min-char-height", type=int, default=MIN_CHAR_HEIGHT)
    parser.add_argument("--min-char-width", type=int, default=MIN_CHAR_WIDTH)
    parser.add_argument("--space-gap-ratio", type=float, default=SPACE_GAP_RATIO)
    args = parser.parse_args()

    global MIN_CHAR_HEIGHT, MIN_CHAR_WIDTH, SPACE_GAP_RATIO
    MIN_CHAR_HEIGHT = args.min_char_height
    MIN_CHAR_WIDTH = args.min_char_width
    SPACE_GAP_RATIO = args.space_gap_ratio

    device = get_device()
    model = load_model(args.model, device)
    model.eval()

    if args.single_char:
        print(read_single_char(args.image, model, device))
    else:
        text = read_image(args.image, model, device, save_vis=args.save_vis)
        print("\n=== Recognised Text ===")
        print(text)


if __name__ == "__main__":
    main()
