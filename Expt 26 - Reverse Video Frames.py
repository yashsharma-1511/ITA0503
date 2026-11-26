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


# Experiment 26: Reverse the frames of a video to create a reversed video
def reverse_video(input_video_path, output_video_path):
    fps, frame_width, frame_height = get_video_info(input_video_path)
    frames = list(read_frames(input_video_path))      # store all frames in a list
    frames.reverse()                                  # reverse the list

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")          # codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Reversed video saved as: {output_video_path}")
    return frames, fps


reversed_frames, fps = reverse_video(VIDEO_PATH, "output_reversed.mp4")

# Display the reversed video (press 'q' to stop)
for frame in reversed_frames:
    cv2.imshow("Reversed Video", frame)
    if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
