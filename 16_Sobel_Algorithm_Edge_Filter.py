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


# Experiment 16: Implement a Sobel algorithm using OpenCV to filter the input image
image = cv2.imread(find_image(["image.png", "input.png", "sample.png", "image.jpg", "sample.jpg", "input.jpg"]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # gradient in X
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # gradient in Y
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

show("Original Image", image)
show("Sobel X", sobel_x)
show("Sobel Y", sobel_y)
show("Sobel Combined", sobel_combined)

save("sobel_x.jpg", sobel_x)
save("sobel_y.jpg", sobel_y)
save("sobel_combined.jpg", sobel_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
