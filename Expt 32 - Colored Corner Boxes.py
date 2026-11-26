import sys
import cv2
import numpy as np


def demo_image(h=480, w=720):
    """Fallback image so the program runs even if sample.jpg is missing."""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):                       # sky gradient
        img[y, :] = (200 - y // 4, 140 - y // 6, 60 + y // 5)
    cv2.rectangle(img, (0, int(h * 0.75)), (w, h), (40, 160, 40), -1)   # grass
    cv2.circle(img, (int(w * 0.65), int(h * 0.55)), 70, (30, 100, 30), -1)  # tree top
    cv2.rectangle(img, (int(w * 0.65) - 8, int(h * 0.55)), (int(w * 0.65) + 8, int(h * 0.75)), (20, 50, 90), -1)
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def load_image(default="sample.jpg"):
    """Usage: python "Expt N.py" [image_path]"""
    path = sys.argv[1] if len(sys.argv) > 1 else default
    img = cv2.imread(path)
    if img is None:
        print(f"'{path}' not found -> using a generated demo image instead.")
        img = demo_image()
    return img




def ask_int(prompt, default):
    """Ask the user for a number; press Enter to accept the default."""
    try:
        s = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        s = ""
    return int(s) if s else default


# Experiment 32: White image of user-defined size with Black, Blue, Green and Red boxes in the corners
def create_colored_corners(image_size):
    height, width = image_size

    # White image (3D array of ones scaled to 255)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Box size = 1/10th of the image size
    box_h, box_w = height // 10, width // 10

    # OpenCV colors are in BGR order
    image[:box_h, :box_w] = [0, 0, 0]            # Top-left     -> Black
    image[:box_h, -box_w:] = [255, 0, 0]         # Top-right    -> Blue
    image[-box_h:, :box_w] = [0, 255, 0]         # Bottom-left  -> Green
    image[-box_h:, -box_w:] = [0, 0, 255]        # Bottom-right -> Red

    cv2.imshow("Colored Corners Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


user_width = ask_int("Enter image width", 500)
user_height = ask_int("Enter image height", 400)
create_colored_corners((user_height, user_width))
