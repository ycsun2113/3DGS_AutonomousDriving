import cv2
import sys

def get_video_fps(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Retrieve FPS from metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_fps.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = get_video_fps(video_path)
    print(f"FPS: {fps}")
