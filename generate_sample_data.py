import csv
import json
import random
import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

SPECIES = ["elephant", "zebra", "giraffe", "bird", "cow"]

HOUR_WEIGHTS = {
    h: (
        0.9 if 6 <= h <= 8 else
        0.8 if 17 <= h <= 19 else
        0.3 if 10 <= h <= 15 else
        0.5 if 0  <= h <= 5  else
        0.6
    )
    for h in range(24)
}

CONSERVATION_STATUS = {
    "zebra":    "Endangered",
    "elephant": "Vulnerable",
    "giraffe":  "Vulnerable",
}

OBS_FIELDS = [
    "timestamp", "total_animals", "species_list", "species_counts_json",
    "shannon_index", "num_species", "lighting", "activity_level",
    "has_elephant", "has_zebra", "has_giraffe", "has_bird",
    "has_endangered", "has_vulnerable", "event_count", "event_names",
]

EVT_FIELDS = ["event", "severity", "timestamp", "message", "payload"]


def shannon(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        p = n / total
        if p > 0:
            h -= p * math.log(p)
    return round(h, 4)


def lighting_for_hour(h):
    if h < 5 or h >= 21:
        return "night"
    elif h in (5, 6, 18, 19, 20):
        return "dawn/dusk"
    return "day"


def generate_row(ts):
    eat_hour = (ts.hour + 3) % 24
    w = HOUR_WEIGHTS[eat_hour]

    present = []
    for sp in SPECIES:
        if random.random() < (0.25 if sp != "bird" else 0.4) * w:
            present.append(sp)

    counts = {}
    for sp in present:
        counts[sp] = random.randint(1, 6 if sp != "elephant" else 4)

    total    = sum(counts.values())
    sp_set   = set(counts.keys())
    h_idx    = shannon(counts)
    lighting = lighting_for_hour(eat_hour)
    activity = round(random.uniform(0.1, 0.9) * w, 4)

    has_endangered = int(bool(sp_set & {"zebra", "dog"}))
    has_vulnerable = int(bool(sp_set & {"elephant", "giraffe", "cat"}))

    evts = []
    if any(v >= 5 for v in counts.values()):
        evts.append("LARGE_HERD")
    if has_endangered:
        evts.append("RARE_SPECIES")
    if h_idx >= 1.5 and len(sp_set) >= 3:
        evts.append("HIGH_BIODIVERSITY")
    if lighting == "night" and total > 0:
        evts.append("NIGHT_ACTIVITY")
    if activity >= 0.6:
        evts.append("PEAK_ACTIVITY")

    return {
        "timestamp":           ts.isoformat(),
        "total_animals":       total,
        "species_list":        "|".join(sorted(sp_set)),
        "species_counts_json": json.dumps(counts),
        "shannon_index":       h_idx,
        "num_species":         len(sp_set),
        "lighting":            lighting,
        "activity_level":      activity,
        "has_elephant":        int("elephant" in sp_set),
        "has_zebra":           int("zebra"    in sp_set),
        "has_giraffe":         int("giraffe"  in sp_set),
        "has_bird":            int("bird"     in sp_set),
        "has_endangered":      has_endangered,
        "has_vulnerable":      has_vulnerable,
        "event_count":         len(evts),
        "event_names":         "|".join(evts),
    }, evts, ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",   type=int, default=500)
    parser.add_argument("--output", default="data")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    obs_path = out / "observations.csv"
    evt_path = out / "events.csv"

    start_ts = datetime.utcnow() - timedelta(hours=48)
    step     = timedelta(seconds=10)

    obs_rows = []
    evt_rows = []
    ts = start_ts

    for _ in range(args.rows):
        row, evts, row_ts = generate_row(ts)
        obs_rows.append(row)
        for evt in evts:
            severity_map = {
                "LARGE_HERD":        "info",
                "RARE_SPECIES":      "alert",
                "HIGH_BIODIVERSITY": "info",
                "NIGHT_ACTIVITY":    "info",
                "PEAK_ACTIVITY":     "info",
            }
            evt_rows.append({
                "event":     evt,
                "severity":  severity_map.get(evt, "info"),
                "timestamp": row_ts.isoformat(),
                "message":   f"Auto-generated event: {evt}",
                "payload":   "{}",
            })
        ts += step

    with open(obs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OBS_FIELDS)
        writer.writeheader()
        writer.writerows(obs_rows)

    with open(evt_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVT_FIELDS)
        writer.writeheader()
        writer.writerows(evt_rows)

    print(f"[SAMPLE] {args.rows} observations written → {obs_path}")
    print(f"[SAMPLE] {len(evt_rows)} events written     → {evt_path}")
    print("\nNow run: python analysis.py --data data/")


if __name__ == "__main__":
    main()