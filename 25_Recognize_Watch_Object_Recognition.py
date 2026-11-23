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


# Experiment 25: Recognize a watch from the given image by general object recognition using OpenCV
#
# OpenCV does not ship a watch Haar cascade. So this program:
#   1) uses "watch_cascade.xml" if you place one next to this file (Haar cascade method), otherwise
#   2) falls back to a shape-based recogniser: a watch dial is a circle, so it is found with the
#      Hough Circle Transform and marked with a bounding box and label "Watch".
image_path = find_image(["watch.jpg", "watch.png"] + ["image.png", "input.png", "sample.png", "image.jpg", "sample.jpg", "input.jpg"])
image = cv2.imread(image_path)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

found = []
cascade_path = os.path.join(BASE, "watch_cascade.xml")

if os.path.exists(cascade_path):
    print("Method: Haar Cascade")
    cascade = cv2.CascadeClassifier(cascade_path)
    for (x, y, w, h) in cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        found.append((x, y, w, h))
else:
    print("Method: Hough Circle (watch dial) - no watch_cascade.xml found")
    blur = cv2.medianBlur(gray, 5)
    H, W = gray.shape
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(H, W) // 2,
                               param1=100, param2=40,
                               minRadius=min(H, W) // 8, maxRadius=min(H, W) // 2)
    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)   # strongest circle = dial
        found.append((max(cx - r, 0), max(cy - r, 0), 2 * r, 2 * r))

for (x, y, w, h) in found:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(output, "Watch", (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

print("Watch detected:" , len(found) > 0, "| count =", len(found))

show("Original Image", image)
show("Detected Watch", output)
save("watch_detected.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
