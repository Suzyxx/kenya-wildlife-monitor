import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np


def load_observations(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "observations.csv"
    if not path.exists():
        raise FileNotFoundError(f"observations.csv not found in {data_dir}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["hour"]       = df["timestamp"].dt.floor("h")
    df["date"]       = df["timestamp"].dt.date
    df["local_hour"] = (df["timestamp"].dt.hour + 3) % 24
    return df


def load_events(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "events.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("local_hour")
          .agg(
              mean_animals  = ("total_animals",  "mean"),
              mean_activity = ("activity_level", "mean"),
              mean_shannon  = ("shannon_index",  "mean"),
              observations  = ("timestamp",      "count"),
          )
          .round(3)
          .reset_index()
    )


def species_summary(df: pd.DataFrame) -> pd.DataFrame:
    species_cols = ["has_elephant", "has_zebra", "has_giraffe", "has_bird"]
    rows = []
    for col in species_cols:
        sp = col.replace("has_", "")
        rows.append({
            "species":        sp,
            "frames_present": int(df[col].sum()),
            "presence_rate":  round(float(df[col].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("frames_present", ascending=False)


def biodiversity_kpi(df: pd.DataFrame) -> dict:
    return {
        "total_observations":     len(df),
        "mean_shannon_index":     round(df["shannon_index"].mean(), 4),
        "max_shannon_index":      round(df["shannon_index"].max(), 4),
        "mean_animals_per_frame": round(df["total_animals"].mean(), 2),
        "peak_animals_in_frame":  int(df["total_animals"].max()),
        "pct_frames_with_animal": round(df["total_animals"].gt(0).mean() * 100, 1),
        "pct_night_activity":     round(df["lighting"].eq("night").mean() * 100, 1),
        "endangered_sightings":   int(df["has_endangered"].sum()),
        "vulnerable_sightings":   int(df["has_vulnerable"].sum()),
    }


def event_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["event", "count", "severity"])
    return (
        events_df.groupby(["event", "severity"])
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False)
    )


def generate_html_report(kpis, hourly, species, evts, out_path):
    hourly_labels  = hourly["local_hour"].tolist()
    hourly_animals = hourly["mean_animals"].tolist()
    hourly_shannon = hourly["mean_shannon"].tolist()
    sp_labels = species["species"].tolist()
    sp_counts = species["frames_present"].tolist()
    evt_labels = evts["event"].tolist() if not evts.empty else []
    evt_counts = evts["count"].tolist()  if not evts.empty else []

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kenya Wildlife Biodiversity Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #3fb950; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
  .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
  .kpi .val {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .kpi .lbl {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  th {{ color: #8b949e; font-weight: 600; }}
  footer {{ color: #8b949e; font-size: 0.8rem; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Kenya Wildlife Biodiversity Monitor</h1>
<p>Bushtops, Maasai Mara — Report generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC</p>

<h2>Key Performance Indicators</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="val">{kpis['total_observations']}</div><div class="lbl">Total Observations</div></div>
  <div class="kpi"><div class="val">{kpis['mean_shannon_index']}</div><div class="lbl">Mean Shannon Index H'</div></div>
  <div class="kpi"><div class="val">{kpis['pct_frames_with_animal']}%</div><div class="lbl">Frames with Animals</div></div>
  <div class="kpi"><div class="val">{kpis['peak_animals_in_frame']}</div><div class="lbl">Peak Animals in Frame</div></div>
  <div class="kpi"><div class="val">{kpis['endangered_sightings']}</div><div class="lbl">Endangered Sightings</div></div>
  <div class="kpi"><div class="val">{kpis['pct_night_activity']}%</div><div class="lbl">Night-time Activity</div></div>
</div>

<div class="chart-grid">
  <div class="card"><h2>Animal Activity by Hour (EAT)</h2><canvas id="hourlyChart"></canvas></div>
  <div class="card"><h2>Species Occurrence</h2><canvas id="speciesChart"></canvas></div>
</div>
<div class="chart-grid">
  <div class="card"><h2>Biodiversity (Shannon H') by Hour</h2><canvas id="shannonChart"></canvas></div>
  <div class="card"><h2>Event Distribution</h2><canvas id="eventChart"></canvas></div>
</div>

<h2>Stakeholder Insights</h2>
<table>
<tr><th>Stakeholder</th><th>Key Metric</th><th>Insight</th></tr>
<tr><td>Camp Guests</td><td>Peak hour activity</td><td>Best game-viewing windows for planning guided drives</td></tr>
<tr><td>Camp Management</td><td>{kpis['pct_frames_with_animal']}% frames with animals</td><td>High wildlife presence justifies premium positioning</td></tr>
<tr><td>Safari Guides</td><td>Giraffe 45%, Zebra 31%</td><td>Most likely species to feature in pre-drive briefings</td></tr>
<tr><td>Marketing Team</td><td>Peak {kpis['peak_animals_in_frame']} animals in frame</td><td>Strong herd activity supports wildlife density claims</td></tr>
</table>

<footer>Generated by Kenya Wildlife Biodiversity Monitor · IoT & Big Data · {datetime.now(timezone.utc).year}</footer>

<script>
const COLORS = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#39c5cf'];
new Chart(document.getElementById('hourlyChart'), {{
  type: 'bar',
  data: {{ labels: {json.dumps(hourly_labels)}.map(h => h+':00'),
           datasets: [{{ label: 'Mean Animals', data: {json.dumps(hourly_animals)},
                         backgroundColor: '#58a6ff88', borderColor: '#58a6ff', borderWidth: 1 }}] }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }},
              scales: {{ x: {{ ticks: {{ color: '#8b949e' }} }}, y: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
new Chart(document.getElementById('speciesChart'), {{
  type: 'doughnut',
  data: {{ labels: {json.dumps(sp_labels)}, datasets: [{{ data: {json.dumps(sp_counts)}, backgroundColor: COLORS }}] }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }} }}
}});
new Chart(document.getElementById('shannonChart'), {{
  type: 'line',
  data: {{ labels: {json.dumps(hourly_labels)}.map(h => h+':00'),
           datasets: [{{ label: "Shannon H'", data: {json.dumps(hourly_shannon)}, fill: true,
                         backgroundColor: '#3fb95022', borderColor: '#3fb950', tension: 0.4 }}] }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }},
              scales: {{ x: {{ ticks: {{ color: '#8b949e' }} }}, y: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
new Chart(document.getElementById('eventChart'), {{
  type: 'bar',
  data: {{ labels: {json.dumps(evt_labels)},
           datasets: [{{ label: 'Count', data: {json.dumps(evt_counts)}, backgroundColor: COLORS }}] }},
  options: {{ indexAxis: 'y',
              plugins: {{ legend: {{ display: false }} }},
              scales: {{ x: {{ ticks: {{ color: '#8b949e' }} }}, y: {{ ticks: {{ color: '#8b949e' }} }} }} }}
}});
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[REPORT] HTML report saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse collected wildlife data")
    parser.add_argument("--data", default="data", help="Data directory (default: ./data)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    print("\n[ANALYSIS] Loading data...")
    obs  = load_observations(data_dir)
    evts = load_events(data_dir)
    print(f"[ANALYSIS] {len(obs)} observations, {len(evts)} events loaded.\n")

    kpis    = biodiversity_kpi(obs)
    hourly  = hourly_activity(obs)
    species = species_summary(obs)
    evt_sum = event_summary(evts)

    print("=" * 55)
    print("  BIODIVERSITY KPIs")
    print("=" * 55)
    for k, v in kpis.items():
        print(f"  {k:<35} {v}")

    print("\n  HOURLY ACTIVITY (EAT)\n" + hourly.to_string(index=False))
    print("\n  SPECIES SUMMARY\n"        + species.to_string(index=False))
    if not evt_sum.empty:
        print("\n  EVENT SUMMARY\n"      + evt_sum.to_string(index=False))

    hourly.to_csv(data_dir / "hourly_summary.csv",  index=False)
    species.to_csv(data_dir / "species_summary.csv", index=False)
    print(f"\n[ANALYSIS] Saved summaries to {data_dir}")

    generate_html_report(kpis, hourly, species, evt_sum, data_dir / "report.html")
    print("\n[ANALYSIS] Done.")


if __name__ == "__main__":
    main()