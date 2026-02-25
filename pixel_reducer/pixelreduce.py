import cv2
import numpy as np
from PIL import Image
from collections import Counter
import argparse
import urllib.request
import ssl
import re


# -----------------------------
# STEP 1 — Detect pixel size
# -----------------------------
def detect_pixel_size(image):
    """
    Detect the logical pixel size using autocorrelation of the gradient
    signal, sampled from all four edges of the image for robustness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    def autocorrelation(signal):
        """Normalized autocorrelation via FFT."""
        n = len(signal)
        sig = signal - np.mean(signal)
        f = np.fft.fft(sig, n=2 * n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        if acf[0] != 0:
            acf /= acf[0]
        return acf

    def first_peak(acf, min_lag=4):
        """Find first local maximum after min_lag in autocorrelation."""
        for i in range(min_lag, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]:
                return i
        return None

    quarter_h = max(h // 4, 2)
    quarter_w = max(w // 4, 2)

    estimates = []

    # Check pixel WIDTH from top strip, bottom strip, and full image
    for strip in [gray[:quarter_h, :], gray[h - quarter_h:, :], gray]:
        dx = np.abs(np.diff(strip, axis=1))
        avg_dx = np.mean(dx, axis=0)
        acf = autocorrelation(avg_dx)
        peak = first_peak(acf)
        if peak is not None:
            estimates.append(peak)

    # Check pixel HEIGHT from left strip, right strip, and full image
    for strip in [gray[:, :quarter_w], gray[:, w - quarter_w:], gray]:
        dy = np.abs(np.diff(strip, axis=0))
        avg_dy = np.mean(dy, axis=1)
        acf = autocorrelation(avg_dy)
        peak = first_peak(acf)
        if peak is not None:
            estimates.append(peak)

    if not estimates:
        raise Exception("Could not detect pixel size automatically")

    pixel_size = int(np.median(estimates))
    return pixel_size


# -----------------------------
# STEP 2 — Detect grid offset
# -----------------------------
def detect_grid_offset(image, pixel_size):
    """
    Find where the pixel grid actually starts (x_offset, y_offset)
    by looking for the alignment that maximizes edge energy at grid lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float64)
    h, w = edges.shape

    best_x_offset = 0
    best_x_score = -1
    best_y_offset = 0
    best_y_score = -1

    for offset in range(pixel_size):
        # Score vertical grid lines at this x offset
        cols = np.arange(offset, w, pixel_size)
        cols = cols[(cols > 0) & (cols < w)]
        if len(cols) > 0:
            score = np.sum(edges[:, cols])
            if score > best_x_score:
                best_x_score = score
                best_x_offset = offset

        # Score horizontal grid lines at this y offset
        rows = np.arange(offset, h, pixel_size)
        rows = rows[(rows > 0) & (rows < h)]
        if len(rows) > 0:
            score = np.sum(edges[rows, :])
            if score > best_y_score:
                best_y_score = score
                best_y_offset = offset

    return best_x_offset, best_y_offset


# -----------------------------
# STEP 3 — Center-sampled color
# -----------------------------
def center_color(block):
    """
    Sample color from the central region of a pixel block,
    avoiding edge artifacts from scaling/compression.
    """
    bh, bw = block.shape[:2]
    if bh == 0 or bw == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Use the central 50% of the block
    margin_y = max(bh // 4, 0)
    margin_x = max(bw // 4, 0)

    y1 = margin_y
    y2 = max(bh - margin_y, y1 + 1)
    x1 = margin_x
    x2 = max(bw - margin_x, x1 + 1)

    center = block[y1:y2, x1:x2]
    pixels = center.reshape(-1, 3)

    # Majority vote on the center pixels
    counts = Counter(map(tuple, pixels))
    return np.array(counts.most_common(1)[0][0], dtype=np.uint8)


# -----------------------------
# STEP 4 — Quantize to N colors
# -----------------------------
def quantize_colors(image, max_colors):
    """
    Reduce the image to at most max_colors using k-means clustering.
    Each pixel is snapped to its nearest palette color.
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, max_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(h, w, c)
    return quantized


# -----------------------------
# STEP 5 — Rebuild image
# -----------------------------
def rebuild_pixel_image(image, pixel_size, x_offset, y_offset):
    h, w, _ = image.shape

    # Calculate how many pixels fit, including partial edge blocks > 50%
    usable_w = w - x_offset
    usable_h = h - y_offset
    new_w = usable_w // pixel_size
    new_h = usable_h // pixel_size
    if (usable_w % pixel_size) > pixel_size / 2:
        new_w += 1
    if (usable_h % pixel_size) > pixel_size / 2:
        new_h += 1

    output = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            sy = y_offset + y * pixel_size
            sx = x_offset + x * pixel_size
            block = image[max(sy, 0):sy + pixel_size, max(sx, 0):sx + pixel_size]
            output[y, x] = center_color(block)

    return output


# -----------------------------
# MAIN PROGRAM
# -----------------------------
def load_image(source):
    """Load an image from a local file path or a URL."""
    if re.match(r'https?://', source):
        print(f"Downloading image from URL...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        resp = urllib.request.urlopen(source, context=ctx)
        data = np.frombuffer(resp.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(source)
    return image


def main(input_path, output_path, max_colors=None, width=None):
    image = load_image(input_path)

    if image is None:
        print("Could not load image")
        return

    if width is not None:
        pixel_size = image.shape[1] // width
        print(f"Pixel size from --width {width}: {pixel_size}px")
    else:
        print("Detecting pixel size...")
        pixel_size = detect_pixel_size(image)
        print(f"Detected pixel size: {pixel_size}px")

    print("Detecting grid offset...")
    x_offset, y_offset = detect_grid_offset(image, pixel_size)
    print(f"Grid offset: x={x_offset}, y={y_offset}")

    # Wrap offsets > 50% of pixel size to negative (include partial leading block)
    if x_offset > pixel_size / 2:
        x_offset -= pixel_size
        print(f"Adjusted grid offset: x={x_offset}, y={y_offset}")
    if y_offset > pixel_size / 2:
        y_offset -= pixel_size
        print(f"Adjusted grid offset: x={x_offset}, y={y_offset}")

    print("Rebuilding clean pixel image...")
    cleaned = rebuild_pixel_image(image, pixel_size, x_offset, y_offset)

    if max_colors is not None:
        print(f"Quantizing to {max_colors} colors...")
        cleaned = quantize_colors(cleaned, max_colors)

    # Convert BGR (OpenCV) to RGB (Pillow) before saving
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    Image.fromarray(cleaned).save(output_path)

    print(f"Saved cleaned image to: {output_path}")


# -----------------------------
# CLI Usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce pixel art to clean pixels")
    parser.add_argument("input", help="Input image path or URL")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--colors", type=int, default=None,
                        help="Max number of colors in the output palette")
    parser.add_argument("--width", type=int, default=None,
                        help="Known width in logical pixels (skips auto-detection)")
    args = parser.parse_args()
    main(args.input, args.output, max_colors=args.colors, width=args.width)
