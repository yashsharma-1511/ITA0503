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




def require_cascades():
    """Haar cascades (cv2.CascadeClassifier) exist in OpenCV 4.x but were removed in OpenCV 5."""
    if not hasattr(cv2, "CascadeClassifier"):
        print("OpenCV " + cv2.__version__ + " has no CascadeClassifier (removed in OpenCV 5).")
        print("Install OpenCV 4 instead:")
        print("    pip uninstall -y opencv-python opencv-contrib-python")
        print('    pip install "opencv-python<5"')
        sys.exit(1)


require_cascades()

# Experiment 29: Detect and locate human eyes in an image
def detect_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    total_eyes = 0
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]                 # eyes are searched only inside each face
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        total_eyes += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    print(f"Faces: {len(faces)} | Eyes detected: {total_eyes}")
    cv2.imshow("Eye Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_eyes(load_image("face_image.jpg"))       # use a photo containing a face
