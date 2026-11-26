import sys
import cv2
import numpy as np
import os

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"   # Usage: python "<file>.py" [video_path]


def demo_frames(n=120, w=640, h=360):
    """Fallback clip (text + moving red ball) used when the video file is not found."""
    for i in range(n):
        frame = np.full((h, w, 3), 255, np.uint8)
        x = int(40 + (w - 80) * i / (n - 1))
        cv2.circle(frame, (x, h // 2 + 70), 30, (0, 0, 255), -1)
        cv2.putText(frame, "OpenCV Rocks", (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
        cv2.putText(frame, f"Frame {i + 1}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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


def get_video_info(path):
    """Returns (fps, width, height); sensible defaults for the demo video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0, 640, 360
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps if fps and fps > 0 else 30.0), w, h


# Experiment 40: Extract text from a video using OCR (pytesseract)
# Needs:  pip install pytesseract   AND the Tesseract OCR program installed on your PC
#         (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki)
try:
    import pytesseract
except ImportError:
    print("pytesseract is not installed. Run:  pip install pytesseract")
    sys.exit(1)

# Common Windows install locations (ignored if Tesseract is already on PATH)
for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
          r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break


def extract_text_from_video(video_path):
    fps, _, _ = get_video_info(video_path)
    step = max(1, int(fps))                        # OCR about one frame per second (much faster)
    last_text = ""

    for i, frame in enumerate(read_frames(video_path)):
        if i % step:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            text = pytesseract.image_to_string(gray_frame).strip()
        except pytesseract.TesseractNotFoundError:
            print("Tesseract OCR program not found. Install it (see the link at the top of this file).")
            return
        if text and text != last_text:             # print only when the text changes
            print(f"[frame {i}] {text}")
            last_text = text

    cv2.destroyAllWindows()


extract_text_from_video(VIDEO_PATH)
