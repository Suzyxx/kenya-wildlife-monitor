import subprocess
import shutil
import numpy as np
import cv2


def get_stream_url(url: str) -> str:
    if url.startswith("rtsp://") or url.endswith(".m3u8"):
        return url
    if shutil.which("yt-dlp") is None:
        print("[WARN] yt-dlp not found – using URL as-is.")
        return url
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4]/best", "-g", url],
            capture_output=True, text=True, timeout=30
        )
        resolved = result.stdout.strip().splitlines()[0]
        if resolved:
            return resolved
    except Exception as e:
        print("[WARN] yt-dlp resolution failed: " + str(e))
    return url


def open_stream(stream_url: str):
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        stream_url
    ], capture_output=True, text=True, timeout=30)

    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except Exception:
        w, h = 1280, 720
        print("[WARN] Could not probe dimensions, using " + str(w) + "x" + str(h))
    process = subprocess.Popen([
        "ffmpeg",
        "-i", stream_url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", "fps=2",
        "pipe:1"
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    return process, w, h


def read_frame(process, width, height):
    raw = process.stdout.read(width * height * 3)
    if len(raw) < width * height * 3:
        return False, None
    frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
    return True, frame.copy()


def estimate_lighting(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    if mean_lum < 40:
        return "night"
    elif mean_lum < 90:
        return "dawn/dusk"
    else:
        return "day"


def estimate_activity_level(current_frame: np.ndarray, prev_frame, blur_ksize: int = 21) -> float:
    if prev_frame is None:
        return 0.0
    g1 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (blur_ksize, blur_ksize), 0)
    g2 = cv2.GaussianBlur(g2, (blur_ksize, blur_ksize), 0)
    diff = cv2.absdiff(g1, g2)
    score = float(np.mean(diff)) / 255.0
    return round(min(score, 1.0), 4)
