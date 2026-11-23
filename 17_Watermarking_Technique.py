import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def find_image(names):
    """Look for the first existing file from `names` in the script folder, then the current folder."""
    for n in names:
        for d in (BASE, os.getcwd()):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError("Put one of these next to the script: " + ", ".join(names))


def show(title, img, max_w=900):
    """imshow that shrinks very large images so they fit on screen."""
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)


def save(name, img):
    cv2.imwrite(os.path.join(BASE, name), img)


# Experiment 17: Design and implement a watermarking technique using OpenCV
image = cv2.imread(find_image(["image.png", "input.png", "sample.png", "image.jpg", "sample.jpg", "input.jpg"]))
h, w = image.shape[:2]

# Build a watermark image (white text on black) - no external logo file needed
watermark = np.zeros((h, w, 3), dtype=np.uint8)
text = "SAVEETHA"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = max(w / 600.0, 0.6)
thick = max(int(scale * 2), 1)
(tw, th), _ = cv2.getTextSize(text, font, scale, thick)
cv2.putText(watermark, text, ((w - tw) // 2, (h + th) // 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# Blend watermark into the original (alpha blending)
alpha = 0.35
watermarked = cv2.addWeighted(image, 1.0, watermark, alpha, 0)

show("Original Image", image)
show("Watermark", watermark)
show("Watermarked Image", watermarked)
save("watermarked_image.jpg", watermarked)

cv2.waitKey(0)
cv2.destroyAllWindows()
