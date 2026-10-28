"""Plot validation progress and held-out comparisons for a completed run."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path)
args = parser.parse_args()
folder = args.run or Path(json.loads((ROOT/'ppo_runs/latest.json').read_text(encoding='utf-8'))['run'])
before = json.loads((folder/'before_training.json').read_text(encoding='utf-8'))
rows = list(csv.DictReader((folder/'validation.csv').open(encoding='utf-8-sig')))
fig, ax = plt.subplots(figsize=(7,4))
ax.plot([0]+[float(r['timesteps'])/1000 for r in rows],
        [before['mean_final_error_mm']]+[float(r['mean_final_error_mm']) for r in rows],
        'o-', color='#1768AC', label='PPO validation error')
ax.axhline(10, color='#C56C19', linestyle='--', label='10 mm reference')
ax.set(xlabel='Training steps (thousands)', ylabel='Mean final error (mm)',
       title='FetchReach: fixed validation targets')
ax.set_yscale('log');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
fig.savefig(folder/'validation_curve.png',dpi=180);plt.close(fig)
tests = json.loads((folder/'test_summary.json').read_text(encoding='utf-8'))
labels = {'random':'Random','feedback':'Position feedback','ppo':'PPO'}
fig, axes = plt.subplots(1,2,figsize=(9,3.8))
names = [labels[r['method']] for r in tests]
colors = ['#999999','#3B8D5A','#1768AC']
axes[0].bar(names,[r['mean_final_error_mm'] for r in tests],color=colors)
axes[0].set_yscale('log');axes[0].set_ylabel('Mean final error (mm, log scale)')
axes[1].bar(names,[100*r['final_within_10mm_rate'] for r in tests],color=colors)
axes[1].set(ylabel='Final error below 10 mm (%)',ylim=(0,105))
fig.suptitle('Same 30 held-out target seeds');fig.tight_layout()
fig.savefig(folder/'test_comparison.png',dpi=180);plt.close(fig)
print(folder/'validation_curve.png');print(folder/'test_comparison.png')
