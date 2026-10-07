"""
notebooks/eda_emotion_analysis.py
-----------------------------------
Exploratory Data Analysis (EDA) for BMA Mental Health Chatbot dataset.
Run this script OR copy into a Jupyter notebook cell by cell.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Usage:
    pip install pandas matplotlib seaborn wordcloud
    python notebooks/eda_emotion_analysis.py
"""

import json
import csv
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
OUTPUT_DIR    = Path(__file__).resolve().parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

INTENTS_FILE  = DATA_DIR / "intents.json"
EMOTIONS_FILE = DATA_DIR / "emotions.csv"

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = ["#7a9e87","#b8d4c2","#c8917a","#f0ebe3","#1e2420",
           "#4a7c5f","#e8f0eb","#d4a574","#8fb4a0","#5c7a6e"]

sns.set_theme(style="whitegrid", font="DejaVu Sans")

# ═════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ═════════════════════════════════════════════════════════════════════════════
print("── Loading data ─────────────────────────────────────────")

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

intents      = intents_data["intents"]
intent_classes = []
tags           = []
pattern_counts = []
all_patterns   = []

for intent in intents:
    intent_classes.append(intent["tag"]) 
    tags.append(intent["tag"])
    pattern_counts.append(len(intent["patterns"]))
    all_patterns.extend(intent["patterns"])

emotions_rows = []
with open(EMOTIONS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        emotions_rows.append(row)

""" emotions      = [r.["emotion"]      for r in emotions_rows]
intent_class2 = [r["intent_class"] for r in emotions_rows]
severities    = [r["severity"]     for r in emotions_rows]
texts         = [r["text"]         for r in emotions_rows] """

emotions      = [r.get("emotion", "unknown") for r in emotions_rows]
intent_class2 = [r.get("intent_class", r.get("intent", r.get("tag", "unknown"))) for r in emotions_rows]
severities    = [r.get("severity", "moderate") for r in emotions_rows] # "moderate" par défaut si absent
texts         = [r.get("text", "") for r in emotions_rows]

print(f"  Intents loaded   : {len(intents)} intent definitions")
print(f"  Total patterns   : {len(all_patterns)}")
print(f"  Emotions CSV rows: {len(emotions_rows)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Intent class distribution (bar chart)
# ═════════════════════════════════════════════════════════════════════════════
class_counter = Counter(intent_classes)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(class_counter.keys(), class_counter.values(),
              color=PALETTE[:len(class_counter)], edgecolor="white", linewidth=1.2)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Number of Intent Tags per Class", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("Count")
ax.set_xlabel("Intent Class")
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.tight_layout()
path = OUTPUT_DIR / "01_intent_class_distribution.png"
plt.savefig(path, dpi=150)
print(f"  [saved] {path}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 3. Patterns per intent (horizontal bar)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, max(6, len(tags) * 0.38)))
y_pos = range(len(tags))
ax.barh(y_pos, pattern_counts, color="#7a9e87", edgecolor="white")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(tags, fontsize=9)
ax.set_xlabel("Number of training patterns")
ax.set_title("Patterns per Intent Tag", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
path = OUTPUT_DIR / "02_patterns_per_intent.png"
plt.savefig(path, dpi=150)
print(f"  [saved] {path}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 4. Emotion distribution (pie chart)
# ═════════════════════════════════════════════════════════════════════════════
emotion_counter = Counter(emotions)
labels   = list(emotion_counter.keys())
sizes    = list(emotion_counter.values())
explode  = [0.04] * len(labels)

fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts_p, autotexts = ax.pie(
    sizes, labels=labels, autopct="%1.1f%%",
    colors=PALETTE * 4, explode=explode,
    startangle=140, pctdistance=0.82,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for t in texts_p:   t.set_fontsize(9)
for t in autotexts: t.set_fontsize(8)

ax.set_title("Emotion Distribution in emotions.csv", fontsize=14, fontweight="bold", pad=16)
plt.tight_layout()
path = OUTPUT_DIR / "03_emotion_distribution.png"
plt.savefig(path, dpi=150)
print(f"  [saved] {path}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 5. Severity distribution (stacked by intent class)
# ═════════════════════════════════════════════════════════════════════════════
severity_order = ["positive", "mild", "moderate", "high", "critical"]
sev_color      = {
    "positive": "#7a9e87",
    "mild":     "#b8d4c2",
    "moderate": "#c8917a",
    "high":     "#e07055",
    "critical": "#e53935",
}

# Build matrix: class × severity
all_classes_uniq = sorted(set(intent_class2))
matrix = {cls: {s: 0 for s in severity_order} for cls in all_classes_uniq}
for cls, sev in zip(intent_class2, severities):
    matrix[cls][sev] += 1

fig, ax = plt.subplots(figsize=(11, 5))
bottoms = [0] * len(all_classes_uniq)
for sev in severity_order:
    vals = [matrix[cls][sev] for cls in all_classes_uniq]
    ax.bar(all_classes_uniq, vals, bottom=bottoms,
           label=sev.capitalize(), color=sev_color[sev], edgecolor="white")
    bottoms = [b + v for b, v in zip(bottoms, vals)]

ax.set_title("Severity Distribution per Intent Class", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Intent Class")
ax.set_ylabel("Sample Count")
ax.legend(title="Severity", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.tight_layout()
path = OUTPUT_DIR / "04_severity_by_class.png"
plt.savefig(path, dpi=150)
print(f"  [saved] {path}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 6. Text length distribution
# ═════════════════════════════════════════════════════════════════════════════
lengths = [len(t.split()) for t in all_patterns + texts]

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(lengths, bins=20, color="#7a9e87", edgecolor="white", linewidth=1)
ax.axvline(sum(lengths)/len(lengths), color=PALETTE[2], linestyle="--",
           linewidth=1.5, label=f"Mean: {sum(lengths)/len(lengths):.1f} words")
ax.set_xlabel("Text length (words)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Text Lengths", fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
path = OUTPUT_DIR / "05_text_length_distribution.png"
plt.savefig(path, dpi=150)
print(f"  [saved] {path}")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# 7. Summary report
# ═════════════════════════════════════════════════════════════════════════════
report = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BMA Dataset — EDA Summary Report
  Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTENTS.JSON
  Total intent definitions : {len(intents)}
  Total training patterns  : {len(all_patterns)}
  Intent classes           : {len(set(intent_classes))}
  Avg patterns per intent  : {sum(pattern_counts)/len(pattern_counts):.1f}
  Max patterns (one intent): {max(pattern_counts)}

EMOTIONS.CSV
  Total labeled samples    : {len(emotions_rows)}
  Unique emotions          : {len(set(emotions))}
  Unique intent classes    : {len(set(intent_class2))}
  Severity levels          : {sorted(set(severities))}

CLASS DISTRIBUTION
{chr(10).join(f'  {cls:<32} {count}' for cls, count in sorted(class_counter.items()))}

EMOTION DISTRIBUTION (top 5)
{chr(10).join(f'  {emo:<28} {cnt}' for emo, cnt in emotion_counter.most_common(5))}

PLOTS SAVED TO: {OUTPUT_DIR}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
print(report)

report_path = OUTPUT_DIR / "eda_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"[EDA] Report saved → {report_path}")
print("[EDA] Done ✅")
