"""Plot the latest random-vs-feedback runs without retraining or rerunning physics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent/'results'
runs = {}
for file in sorted(root.glob('*/experiment.json')):
    data = json.loads(file.read_text(encoding='utf-8'))
    runs[data['mode']] = file.parent
if not {'random', 'reach'} <= runs.keys():
    raise SystemExit('Please run both random and reach modes first.')
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for mode, label, color in [('random', 'Random actions', '#999999'), ('reach', 'Position feedback', '#1768AC')]:
    rows = list(csv.DictReader((runs[mode]/'tracking.csv').open(encoding='utf-8-sig')))
    for i, ax in enumerate(axes):
        selected = [r for r in rows if int(r['target_id'])==i+1 and int(r['cycle'])==1]
        ax.plot([float(r['time_s']) for r in selected],
                [1000*float(r['error_m']) for r in selected], label=label, color=color, linewidth=1.8)
        ax.set(title=f'Target {i+1}', xlabel='Simulation time (s)')
        ax.grid(alpha=0.2)
for ax in axes:
    ax.axhline(10, color='#C56C19', linestyle=':', linewidth=1, label='10 mm reference')
axes[0].set_ylabel('End-effector position error (mm)')
axes[-1].legend(fontsize=8)
fig.suptitle('FetchReach: three specified targets', fontsize=13)
fig.tight_layout()
fig.savefig(root/'tracking_comparison.png', dpi=180)
plt.close(fig)
print(root/'tracking_comparison.png')
