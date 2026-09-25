"""Export a descriptive figure from the frozen calibration audit."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "artifacts")
args = parser.parse_args()
out = args.results
report = json.loads((out / "calibration_audit.json").read_text(encoding="utf-8"))
splits = ["calibration", "test"]
values = [report["by_split"][s]["coverage_two_sided"] * 100 for s in splits]
labels = [f"{s.capitalize()} (n={report['by_split'][s]['n']})" for s in splits]
fig, ax = plt.subplots(figsize=(7, 4.6))
fig.subplots_adjust(bottom=.22, top=.88, left=.13, right=.98)
bars = ax.bar(labels, values, color=["#3988a4", "#bc6148"], width=.5)
ax.axhline(90, color="#333333", linestyle="--", label="Nominal 90%")
ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=5)
ax.set(ylim=(0, 105), ylabel="Empirical two-sided coverage (%)",
       title="Frozen MIC predictor: calibration versus test")
ax.legend(loc="lower right")
fig.text(.025, .035, "Historical E. coli data; does not establish coverage for selected new peptides.", fontsize=8)
fig.savefig(out / "coverage.png", dpi=160)
fig.savefig(out / "coverage.svg")
plt.close(fig)
