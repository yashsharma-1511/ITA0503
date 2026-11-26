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


# Experiment 35: Show a text string entered by the user on an image
def add_text_to_image(image_size, text):
    height, width = image_size
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)                                         # Red

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2                             # centered
    text_y = (height + text_size[1]) // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    cv2.imshow("Image with Text", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


user_width = ask_int("Enter image width", 500)
user_height = ask_int("Enter image height", 400)
try:
    user_text = input("Enter the text to display [opencv Rocks!]: ").strip() or "opencv Rocks!"
except EOFError:
    user_text = "opencv Rocks!"
add_text_to_image((user_height, user_width), user_text)
