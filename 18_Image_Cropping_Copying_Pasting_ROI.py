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


# Experiment 18: Image cropping, copying and pasting of a Region of Interest (ROI)
image = cv2.imread(find_image(["image.png", "input.png", "sample.png", "image.jpg", "sample.jpg", "input.jpg"]))
h, w = image.shape[:2]

# ROI = central region (x1:x2, y1:y2)
x1, y1, x2, y2 = w // 4, h // 4, w // 2, h // 2
roi = image[y1:y2, x1:x2].copy()          # crop + copy

# Paste the copied ROI at another location (bottom-right area)
result = image.copy()
px, py = w - (x2 - x1) - 20, h - (y2 - y1) - 20
result[py:py + (y2 - y1), px:px + (x2 - x1)] = roi

# Mark the source and destination rectangles
cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.rectangle(result, (px, py), (px + (x2 - x1), py + (y2 - y1)), (0, 0, 255), 2)

show("Original Image", image)
show("Cropped ROI", roi)
show("ROI Copied and Pasted", result)
save("roi_cropped.jpg", roi)
save("roi_pasted.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
