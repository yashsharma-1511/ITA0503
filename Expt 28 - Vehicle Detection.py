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



import os
import urllib.request

# Experiment 28: Detect and locate vehicles in each frame of a video
# Usage: python "<file>.py" [video_path]      (default: traffic_video.mp4)
# Needs the pre-trained 'cars.xml' Haar cascade. It is downloaded automatically the first time;
# if that fails (no internet), a motion-based detector is used instead so the program still runs.
CASCADE_FILE = "cars.xml"
CARS_URL = "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "traffic_video.mp4"


def demo_frames(n=120, w=640, h=360):
    """Fallback clip: a red 'car' moving across a road."""
    for i in range(n):
        frame = np.full((h, w, 3), 90, np.uint8)
        x = int(20 + (w - 140) * i / (n - 1))
        cv2.rectangle(frame, (x, 200), (x + 100, 250), (0, 0, 255), -1)
        yield frame


def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"'{path}' not found -> using a generated demo video instead.")
        yield from demo_frames()
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def get_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        return None              # OpenCV 5 has no Haar cascades
    if not os.path.exists(CASCADE_FILE):
        try:
            print("Downloading cars.xml ...")
            urllib.request.urlretrieve(CARS_URL, CASCADE_FILE)
        except Exception as e:
            print("Could not download cars.xml:", e)
            return None
    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    if cascade.empty():
        os.remove(CASCADE_FILE)          # bad download - retry next time
        return None
    return cascade


def detect_vehicles(video_path):
    vehicle_cascade = get_cascade()
    if vehicle_cascade is None:
        print("Using motion-based vehicle detection (background subtraction) instead.")
        subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        kernel = np.ones((5, 5), np.uint8)

    for frame in read_frames(video_path):
        if vehicle_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        else:
            mask = subtractor.apply(frame)
            mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vehicles = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 800]

        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):          # press 'q' to exit
            break
    cv2.destroyAllWindows()


detect_vehicles(VIDEO_PATH)
