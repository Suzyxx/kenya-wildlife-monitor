from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

@dataclass
class Event:
    name:      str
    severity:  str
    timestamp: datetime
    payload:   dict = field(default_factory=dict)
    message:   str = ""

    def to_dict(self) -> dict:
        return {
            "event":     self.name,
            "severity":  self.severity,
            "timestamp": self.timestamp.isoformat(),
            "message":   self.message,
            "payload":   json.dumps(self.payload),
        }

LARGE_HERD_THRESHOLD        = 5
HIGH_BIODIVERSITY_THRESHOLD = 1.5
HIGH_ACTIVITY_THRESHOLD     = 0.60
NO_ANIMAL_PATIENCE          = 6
RARE_SPECIES = {
    "Endangered": {"zebra", "dog"},
    "Vulnerable": {"elephant", "giraffe", "cat"},
}

class EventManager:

    def __init__(self):
        self._empty_streak = 0
        self._seen_species = set()
        self._last_peak_time: Optional[datetime] = None
        self._peak_cooldown = timedelta(minutes=5)

    def evaluate(
        self,
        timestamp:      datetime,
        detections:     list,
        species_counts: Dict[str, int],
        activity:       float,
        lighting:       str,
    ) -> List[Event]:
        events: List[Event] = []
        total_animals = len(detections)
        species_set   = set(species_counts.keys())

        # 1. No animals
        if total_animals == 0:
            self._empty_streak += 1
            if self._empty_streak == NO_ANIMAL_PATIENCE:
                events.append(Event(
                    name      = "NO_ANIMALS",
                    severity  = "warning",
                    timestamp = timestamp,
                    message   = f"No animals detected for {NO_ANIMAL_PATIENCE} consecutive observations.",
                    payload   = {"empty_streak": self._empty_streak},
                ))
        else:
            self._empty_streak = 0

        # 2. Large herd
        for species, count in species_counts.items():
            if count >= LARGE_HERD_THRESHOLD:
                events.append(Event(
                    name      = "LARGE_HERD",
                    severity  = "info",
                    timestamp = timestamp,
                    message   = f"Large herd of {count} {species}s detected.",
                    payload   = {"species": species, "count": count},
                ))

        # 3. Rare species
        for status, rare_set in RARE_SPECIES.items():
            for species in species_set & rare_set:
                events.append(Event(
                    name      = "RARE_SPECIES",
                    severity  = "alert",
                    timestamp = timestamp,
                    message   = f"{status} species detected: {species}.",
                    payload   = {"species": species, "iucn_status": status},
                ))

        # 4. First sighting
        new_species = species_set - self._seen_species
        for sp in new_species:
            events.append(Event(
                name      = "SPECIES_FIRST_SEEN",
                severity  = "info",
                timestamp = timestamp,
                message   = f"New species detected this session: {sp}.",
                payload   = {"species": sp},
            ))
        self._seen_species |= species_set

        # 5. Peak activity
        if activity >= HIGH_ACTIVITY_THRESHOLD:
            if (self._last_peak_time is None or
                    timestamp - self._last_peak_time > self._peak_cooldown):
                self._last_peak_time = timestamp
                events.append(Event(
                    name      = "PEAK_ACTIVITY",
                    severity  = "info",
                    timestamp = timestamp,
                    message   = f"High motion activity detected (score={activity:.2f}).",
                    payload   = {"activity_score": activity},
                ))

        # 6. Night activity
        if lighting == "night" and total_animals > 0:
            events.append(Event(
                name      = "NIGHT_ACTIVITY",
                severity  = "info",
                timestamp = timestamp,
                message   = f"Nocturnal activity: {total_animals} animal(s) detected at night.",
                payload   = {"count": total_animals, "species": list(species_set)},
            ))

        return events