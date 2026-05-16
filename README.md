# kenya-wildlife-monitor

A python application that reads the YouTube live streaming From the Bushtops Channel, detects animals using YOLOv8, generates biodiversity events, eventually produce a data analysis which can offer relevent advice to different steakholders. This project is built as an IoT & Big Data course assignment 

---

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) (must be installed separately)
- Node.js (required by yt-dlp for YouTube bot detection)

Install ffmpeg on macOS:
```bash
brew install ffmpeg

---
Installation

# 1. Clone the repository
git clone https://github.com/Suzyxx/kenya-wildlife-monitor.git
cd kenya-wildlife-monitor

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

---
Live detection (headless)

caffeinate -i python -u main.py --headless
Runs continuously, reading frames from the Bushtops livestream every 10 seconds and writing detections to data/observations.csv and data/events.csv. Using control+c to stop the data collecting. After quitting, data will be save automatically into related csv files>

Live detection (with display window)

python main.py
Opens an OpenCV window showing the live frame with bounding boxes.

Generate analysis report

python analysis.py --data data
Reads the collected CSVs and outputs:
- data/hourly_summary.csv
- data/species_summary.csv
- data/report.html — interactive dashboard with Chart.js charts
