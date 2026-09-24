import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from mc_experiment import cost, GRID, VOL, VREF, LAM_CAL, threshold_table

plt.rcParams['font.family'] = 'DejaVu Sans'

tt, reg, grid, base = threshold_table()

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), dpi=300)

# Panel A: cost curves
models = [('Eq. 7 literal (p=1, \u03bb=0)', 0.0, '-', None),
          ('Constant p=.70', 'const', '--', None),
          ('Fatigue \u03bb=0.36 (p=.70 at V_ref)', LAM_CAL, '-.', None),
          ('Fatigue \u03bb=0.70', 0.70, (0, (3, 1, 1, 1)), None),
          ('Fatigue \u03bb=1.00', 1.0, (0, (5, 2)), None),
          ('Fatigue \u03bb=1.50', 1.5, (0, (1, 1)), None),
          ('Fatigue \u03bb=2.00', 2.0, (0, (5, 1, 1, 1, 1, 1)), None)]
grays = [0.05, 0.20, 0.32, 0.42, 0.52, 0.65, 0.75]

ax = axes[0]
for (name, lam, ls, _), gy in zip(models, grays):
    c = cost(GRID, lam)
    i = c.argmin()
    ax.plot(GRID, c, linestyle=ls, color=str(gy), linewidth=1.9, label=name)
    ax.plot(GRID[i], c[i], 'o', color=str(gy), markersize=6, markeredgecolor='black', markeredgewidth=0.8, zorder=5)

ax.axvline(0.70, color='black', linewidth=0.9, linestyle=':', alpha=0.6)
ax.text(0.705, ax.get_ylim()[1]*0.02 + 46000, "conventional\ndefault (0.70)", fontsize=7.6, va='bottom')
noMon = tt.loc[tt.model.str.contains('literal'), 'cost_opt'].values  # not used directly
base_line = base
ax.axhline(base_line, color='black', linewidth=1.0, linestyle=(0,(1,3)))
ax.text(0.02, base_line*1.01, "no monitoring", fontsize=8)

ax.set_xlabel("Alert threshold \u03c4")
ax.set_ylabel("Modelled quarterly cost (USD)")
ax.set_title("A. Cost curves under increasing response fatigue\n(dots = cost-minimising threshold)", fontsize=10.8)
ax.legend(fontsize=7.6, loc='upper center', frameon=False, ncol=1, bbox_to_anchor=(0.32, 1.0))
ax.set_xlim(0, 1.0)

# Panel B: tau*, saving, and regret vs lambda (fine grid)
lam_grid = np.linspace(0.0, 2.2, 45)
tau_opt = []
saving = []
regret = []
tau_eq7 = GRID[cost(GRID, 0.0).argmin()]
for lam in lam_grid:
    c = cost(GRID, lam)
    i = c.argmin()
    tau_opt.append(GRID[i])
    saving.append(100 * (1 - c[i] / base))
    opt = c[i]
    regret.append(100 * (float(cost(tau_eq7, lam)) / opt - 1) if opt > 0 else np.nan)

ax2 = axes[1]
ax2.plot(lam_grid, tau_opt, color='0.05', linewidth=2.1, label='Optimal threshold \u03c4* (left axis)')
ax2.set_xlabel("Fatigue intensity \u03bb   (p = exp(\u2212\u03bbV/V_ref))")
ax2.set_ylabel("Optimal alert threshold \u03c4*")
ax2.axvline(LAM_CAL, color='black', linewidth=0.9, linestyle=':', alpha=0.6)
ax2.text(LAM_CAL + 0.03, ax2.get_ylim()[0], "p=0.70 at V_ref", fontsize=7.6, rotation=90, va='bottom')

ax3 = ax2.twinx()
ax3.plot(lam_grid, saving, color='0.45', linewidth=1.9, linestyle='--', label='Saving vs. no monitoring (right axis)')
ax3.plot(lam_grid, regret, color='0.45', linewidth=1.6, linestyle=(0,(1,1)), label='Regret of Eq.7-tuned \u03c4 (right axis)')
ax3.set_ylabel("Percent (%)")

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7.8, loc='upper left', frameon=False)
ax2.set_title("B. Fatigue raises the optimal threshold\nand erodes the cost saving", fontsize=10.8)

plt.tight_layout()
plt.savefig(str(ROOT / 'figures' / 'fig3_threshold_fatigue.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("saved")
