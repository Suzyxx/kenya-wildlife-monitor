# kenya-wildlife-monitor

A Python application that monitors wildlife activity by reading the Bushtops YouTube 
livestream from the Maasai Mara, Kenya. It detects animals using YOLOv8, generates 
biodiversity events, and produces a data analysis report with insights for relevent stakeholders. Built as an IoT & Big Data course assignment.

---

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) (must be installed separately)
- Node.js (required by yt-dlp for YouTube bot detection)

Install ffmpeg on macOS:

```bash
brew install ffmpeg
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Suzyxx/kenya-wildlife-monitor.git
cd kenya-wildlife-monitor

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Live detection (headless)

```bash
caffeinate -i python -u main.py --headless
```

Runs continuously, reading frames from the Bushtops livestream every 10 seconds and writing detections to `data/observations.csv` and `data/events.csv`. Use `Ctrl+C` to stop. Data will be saved automatically to the related CSV files.

### Live detection (with display window)

```bash
python main.py
# or, to prevent interruptions during long runs:
caffeinate -i python -u main.py
```

Opens an OpenCV window showing the live frame with bounding boxes, animal names, and confidence levels. This is useful for verifying how the model is classifying animals in real time.

### Generate analysis report

```bash
python analysis.py --data data
```

Reads the collected CSVs and outputs:

- `data/hourly_summary.csv`
- `data/species_summary.csv`
- `data/report.html` — interactive dashboard with Chart.js charts

### Generate sample data (no live stream needed)

```bash
python generate_sample_data.py
```

Generates sample data to test that other components work correctly before connecting to the YouTube livestream.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Orchestrator — ffmpeg pipe pipeline |
| `detector.py` | YOLOv8n animal detection (CPU), COCO class mapping |
| `events.py` | Event generation (`NO_ANIMALS`, `LARGE_HERD`, `RARE_SPECIES`, etc.) |
| `storage.py` | Writes `observations.csv` and `events.csv` |
| `utils.py` | Stream URL fetching (yt-dlp), frame reading (ffmpeg pipe) |
| `analysis.py` | Pandas aggregations, KPIs, HTML report generation |
| `generate_sample_data.py` | Generates fake CSVs for offline testing |

---

## Livestream Source

**Bushtops — Maasai Mara, Kenya**

Public YouTube livestream. If the stream appears offline, check the [@Bushtops](https://www.youtube.com/@Bushtops) channel for an updated URL and replace the video ID in `utils.py`.

---

## Data

Raw collected data is available in the `data/` directory:

- `observations.csv` — per-frame detections with species flags, animal count, Shannon diversity index, and lighting estimate
- `events.csv` — biodiversity events with type, severity, and timestamp
- `hourly_summary.csv` — aggregated activity and biodiversity metrics by hour (EAT)
- `species_summary.csv` — per-species frame counts and presence rates

---

## Limitations

### Model & Classification

YOLOv8n is trained on the COCO dataset, which contains no African wildlife classes. Animals are mapped to the closest visual equivalent, leading to consistent misclassifications observed during live testing:

| Animal | Detected as |
|--------|-------------|
| Gazelle | Giraffe, horse, cow/buffalo (donkey at close range) |
| Warthog | Elephant |
| Zebra | Correctly classified with high confidence, but large groups are undercounted — the model cannot distinguish individual zebras in a dense herd |
| Wildebeest | Cow, horse, zebra, elephant, or sheep |
| Impala | Horse, elephant, giraffe |
| Waterbuck | Cow, horse |
| Buffalo | Elephant, cow |

- **Confidence threshold**: lowered from 0.35 to 0.20 to capture more fleeting or partially visible animals. This increases recall but also increases the risk of false positives, where background objects or movement are misclassified as animals.
- **Double bounding boxes**: the model occasionally places two bounding boxes on a single animal, registering it as two different species. This appears to happen when the model assigns different class labels at different confidence thresholds depending on the angle or framing of the animal.
- **RARE_SPECIES events are based on COCO class names rather than actual species** — since COCO contains no African wildlife classes, the following proxies were used:

| COCO class | Mapped to | Status |
|------------|-----------|--------|
| `zebra` | Zebra | Endangered |
| `elephant` | Elephant | Vulnerable |
| `giraffe` | Giraffe | Vulnerable |
| `cat` | Lion / Leopard | Vulnerable |
| `dog` | African Wild Dog | Endangered |

As a result, rare species counts should be interpreted as proxies, for example, a warthog misclassified as `elephant` would incorrectly trigger a vulnerable species alert. In practice, this also means the RARE_SPECIES event volume is inflated: animals such as gazelle and impala are frequently misclassified as `giraffe`, producing false vulnerable species alerts that account for much of the high event count seen in the final dashboard.

### Data Collection

- **Temporal sampling bias** — data was collected between 11:00–16:00 EAT, the peak activity window advertised by the Bushtops stream. Metrics do not represent night-time or early-morning behaviour.
- **Vision only** — the pipeline does not process audio. Animal calls are not captured.
  
---

## Future Directions

- **African wildlife detection model** — the core limitation of this project is that YOLOv8n is trained on COCO, which has no African wildlife classes. The natural next step would be to fine-tune a model on an African wildlife dataset (e.g. Snapshot Serengeti or LILA BC) to achieve accurate species-level classification. This was not feasible within the time and computational constraints of this project.

- **Audio + vision fusion** — the Bushtops livestream captures ambient sound including animal calls, bird songs. A combined pipeline that fuses a vision model with an audio classification model (e.g. YAMNet or BirdNET) would give a much more complete picture of animal activity. This will be helpful for detecting animals that are audible but not yet visible in frame, or confirming a detection with a matching species call.

- **Guest alert application** — Bushtops could extend this pipeline into a mobile app that sends real-time push notifications to guests when animals of interest appear near the camp. Guests could set preferences (e.g. alert me when elephants or lions are spotted) and be guided to the right viewing spot. This helps to enhance the safari experience without requiring permanent staff monitoring.
