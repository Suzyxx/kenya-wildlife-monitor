import cv2
import time
import argparse
import math
from datetime import datetime
from pathlib import Path

from detector import WildlifeDetector
from events import EventManager
from storage import DataStore
from utils import estimate_lighting, estimate_activity_level, get_stream_url, open_stream, read_frame

DEFAULT_STREAM = "https://www.youtube.com/live/xXZqU5vnEug"
FRAME_INTERVAL = 10
CONFIDENCE_THRESHOLD = 0.20


def _shannon(counts):
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
    species_str = ", ".join(str(k) + "x" + str(v) for k, v in counts.items()) or "none"
    evt_str = ", ".join("[" + e.severity.upper() + "] " + e.name for e in evts) or "-"
    print(
        ts.strftime('%H:%M:%S') + " UTC | " +
        "Species: " + species_str + " | " +
        "Shannon: " + str(shannon) + " | " +
        "Lighting: " + str(lighting) + " | " +
        "Activity: " + str(activity) + " | Events: " + evt_str
    )


def run(stream_url, output_dir, headless):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Kenya Wildlife Biodiversity Monitor")
    print("  Stream : " + stream_url)
    print("  Output : " + str(output_path.resolve()))
    print("=" * 60)

    print("[STREAM] Resolving URL...")
    resolved_url = get_stream_url(stream_url)

    print("[STREAM] Opening stream...")
    process, width, height = open_stream(resolved_url)
    print("[STREAM] Stream open at " + str(width) + "x" + str(height))

    detector   = WildlifeDetector(confidence=CONFIDENCE_THRESHOLD)
    events_mgr = EventManager()
    store      = DataStore(output_path)

    prev_frame    = None
    frame_count   = 0
    last_process  = 0
    last_annotated = None
    print("[INFO] Starting capture. Press Ctrl+C to quit.")

    try:
        while True:
            ret, frame = read_frame(process, width, height)
            if not ret:
                print("[WARN] Read fail, reconnecting...")
                time.sleep(5)
                process, width, height = open_stream(resolved_url)
                continue

            frame_count += 1
            now = time.time()

            if now - last_process < FRAME_INTERVAL:
                if not headless:
                    display_frame = last_annotated if last_annotated is not None else frame
                    display = cv2.resize(display_frame, (960, 540))
                    cv2.putText(display, "Frame " + str(frame_count), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Kenya Wildlife Monitor", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                prev_frame = frame
                continue

            last_process = now
            timestamp  = datetime.utcnow()
            lighting   = estimate_lighting(frame)
            activity   = estimate_activity_level(frame, prev_frame)
            detections = detector.detect(frame)

            if not headless:
                last_annotated = detector.annotate_frame(frame, detections)
                display = cv2.resize(last_annotated, (960, 540))
                cv2.putText(display, "Frame " + str(frame_count), (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Kenya Wildlife Monitor", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            species_counts = {}
            for d in detections:
                species_counts[d.label] = species_counts.get(d.label, 0) + 1
            species_set = set(species_counts.keys())
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
        process.terminate()
        if not headless:
            cv2.destroyAllWindows()
        store.close()
        print("[INFO] Data saved to: " + str(output_path.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kenya Wildlife Biodiversity Monitor")
    parser.add_argument("--stream",   default=DEFAULT_STREAM)
    parser.add_argument("--output",   default="data")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args.stream, args.output, args.headless)
