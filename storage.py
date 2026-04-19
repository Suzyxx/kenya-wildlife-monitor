import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict

from detector import Detection
from events import Event

OBS_FIELDS = [
    "timestamp", "total_animals",
    "species_list", "species_counts_json",
    "shannon_index", "num_species",
    "lighting", "activity_level",
    "has_elephant", "has_zebra", "has_giraffe", "has_bird",
    "has_endangered", "has_vulnerable",
    "event_count", "event_names",
]

EVT_FIELDS = ["event", "severity", "timestamp", "message", "payload"]

ENDANGERED_SPECIES = {"zebra", "dog"}
VULNERABLE_SPECIES = {"elephant", "giraffe", "cat"}


class DataStore:

    def __init__(self, output_dir: Path):
        self._obs_path = output_dir / "observations.csv"
        self._evt_path = output_dir / "events.csv"

        self._obs_file = open(self._obs_path, "a", newline="", encoding="utf-8")
        self._evt_file = open(self._evt_path, "a", newline="", encoding="utf-8")

        self._obs_writer = csv.DictWriter(self._obs_file, fieldnames=OBS_FIELDS)
        self._evt_writer = csv.DictWriter(self._evt_file, fieldnames=EVT_FIELDS)

        if self._obs_path.stat().st_size == 0:
            self._obs_writer.writeheader()
        if self._evt_path.stat().st_size == 0:
            self._evt_writer.writeheader()

        print(f"[STORAGE] Observations → {self._obs_path}")
        print(f"[STORAGE] Events        → {self._evt_path}")

    def save_observation(
        self,
        timestamp:      datetime,
        detections:     List[Detection],
        species_counts: Dict[str, int],
        species_set:    Set[str],
        shannon_index:  float,
        lighting:       str,
        activity_level: float,
        events:         List[Event],
    ):
        has_endangered = bool(species_set & ENDANGERED_SPECIES)
        has_vulnerable = bool(species_set & VULNERABLE_SPECIES)
        event_names    = "|".join(e.name for e in events)

        obs_row = {
            "timestamp":           timestamp.isoformat(),
            "total_animals":       len(detections),
            "species_list":        "|".join(sorted(species_set)),
            "species_counts_json": json.dumps(species_counts),
            "shannon_index":       shannon_index,
            "num_species":         len(species_set),
            "lighting":            lighting,
            "activity_level":      round(activity_level, 4),
            "has_elephant":        int("elephant" in species_set),
            "has_zebra":           int("zebra"    in species_set),
            "has_giraffe":         int("giraffe"  in species_set),
            "has_bird":            int("bird"     in species_set),
            "has_endangered":      int(has_endangered),
            "has_vulnerable":      int(has_vulnerable),
            "event_count":         len(events),
            "event_names":         event_names,
        }
        self._obs_writer.writerow(obs_row)
        self._obs_file.flush()

        for evt in events:
            self._evt_writer.writerow(evt.to_dict())
        if events:
            self._evt_file.flush()

    def close(self):
        self._obs_file.close()
        self._evt_file.close()
        print("[STORAGE] Files closed.")