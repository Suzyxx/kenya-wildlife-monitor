import cv2
import time
import argparse
import math
from datetime import datetime
from pathlib import Path

from detector import WildlifeDetector
from events import EventManager
from storage import DataStore
from utils import estimate_lighting, estimate_activity_level, get_stream_url

DEFAULT_STREAM = "https://www.youtube.com/watch?v=ydYDqZQpim8"
FRAME_INTERVAL = 10
CONFIDENCE_THRESHOLD = 0.35


def _shannon(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        p = n / total
        if p > 0:
            h -= p * math.log(p)
    return round(h, 4)


def _print_summary(ts, detections, counts, shannon, lighting, activity, evts):
    n = len(detections)
    species_str = ", ".join(f"{k}x{v}" for k, v in counts.items()) or "none"
    evt_str     = ", ".join(f"[{e.severity.upper()}] {e.name}" for e in evts) or "-"
    print(
        f"{ts.strftime('%H:%M:%S')} UTC | "
        f"Animals: {n:2d} | Species: {species_str} | "
        f"Shannon: {shannon:.2f} | Light: {lighting} | "
        f"Activity: {activity:.2f} | Events: {evt_str}"
    )


def run(stream_url: str, output_dir: str, headless: bool):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Kenya Wildlife Biodiversity Monitor")
    print(f"  Stream : {stream_url}")
    print(f"  Output : {output_path.resolve()}")
    print(f"{'='*60}\n")

    resolved_url = get_stream_url(stream_url)
    print(f"[STREAM] Resolved URL: {resolved_url[:80]}...\n")

    cap = cv2.VideoCapture(resolved_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {resolved_url}")

    detector   = WildlifeDetector(confidence=CONFIDENCE_THRESHOLD)
    events_mgr = EventManager()
    store      = DataStore(output_path)

    prev_frame   = None
    last_process = 0
    frame_count  = 0

    print("[INFO] Starting capture. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed - reconnecting in 5s...")
                time.sleep(5)
                cap = cv2.VideoCapture(resolved_url)
                continue

            frame_count += 1
            now = time.time()

            if not headless:
                display = cv2.resize(frame, (960, 540))
                cv2.putText(display, f"Frame {frame_count}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Kenya Wildlife Monitor - press Q to quit", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if now - last_process < FRAME_INTERVAL:
                prev_frame = frame.copy()
                continue

            last_process = now
            timestamp    = datetime.utcnow()

            detections     = detector.detect(frame)
            lighting       = estimate_lighting(frame)
            activity       = estimate_activity_level(frame, prev_frame)

            species_counts = {}
            for d in detections:
                species_counts[d.label] = species_counts.get(d.label, 0) + 1

            species_set   = set(species_counts.keys())
            shannon_index = _shannon(species_counts)

            triggered_events = events_mgr.evaluate(
                timestamp      = timestamp,
                detections     = detections,
                species_counts = species_counts,
                activity       = activity,
                lighting       = lighting,
            )

            store.save_observation(
                timestamp      = timestamp,
                detections     = detections,
                species_counts = species_counts,
                species_set    = species_set,
                shannon_index  = shannon_index,
                lighting       = lighting,
                activity_level = activity,
                events         = triggered_events,
            )

            _print_summary(timestamp, detections, species_counts,
                           shannon_index, lighting, activity, triggered_events)

            prev_frame = frame.copy()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        store.close()
        print(f"\n[INFO] Data saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kenya Wildlife Biodiversity Monitor")
    parser.add_argument("--stream",   default=DEFAULT_STREAM,
                        help="YouTube URL or direct stream URL")
    parser.add_argument("--output",   default="data",
                        help="Directory for CSV output (default: ./data)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window")
    args = parser.parse_args()

    run(args.stream, args.output, args.headless)