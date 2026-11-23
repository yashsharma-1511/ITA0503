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


# Experiment 23: Implement the Top Hat morphological operation using OpenCV
image = cv2.imread(find_image(["image.png", "input.png", "sample.png", "image.jpg", "sample.jpg", "input.jpg"]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)
top_hat_image = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

show("Original Image", image)
show("Grayscale Image", gray)
show("Top Hat Image", top_hat_image)
save("top_hat_image.jpg", top_hat_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
