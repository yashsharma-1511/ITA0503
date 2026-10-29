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



# Experiment 7: Read a captured video and display it normally, in slow motion and in fast motion
# Usage: python "Expt 7.py" [video_path]      (default: video.mp4 in the same folder)
# Press 'q' at any time to quit.
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"


def demo_frames(n=120, w=640, h=360):
    """Fallback clip (a moving red ball) used when the video file is not found."""
    for i in range(n):
        frame = np.full((h, w, 3), 255, np.uint8)
        x = int(40 + (w - 80) * i / (n - 1))
        cv2.circle(frame, (x, h // 2), 30, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame {i + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
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


def get_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30


def play(title, delay_ms, step=1):
    """Show the video with a given delay between frames (step=2 skips every other frame)."""
    for i, frame in enumerate(read_frames(VIDEO_PATH)):
        if i % step:
            continue
        cv2.putText(frame, title, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Video Processing", frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            return False
    return True


fps = get_fps(VIDEO_PATH)
normal_delay = max(1, int(1000 / fps))

modes = [
    ("Normal speed", normal_delay, 1),
    ("Slow motion (0.25x)", normal_delay * 4, 1),     # longer delay between frames
    ("Fast motion (4x)", max(1, normal_delay // 2), 2),  # shorter delay + skip every other frame
]

for title, delay, step in modes:
    print("Playing:", title)
    if not play(title, delay, step):
        break

cv2.destroyAllWindows()
