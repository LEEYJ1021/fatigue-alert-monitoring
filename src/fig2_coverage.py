from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_csv(ROOT / 'results' / 'mc_estimator_performance.csv')

estimators = ["DiD", "Path a (pooled)", "Indirect a*b (pooled)", "Indirect a*b (FE)",
              "Direct c' (pooled)", "Direct c' (FE)", "IV 2SLS", "IV naive OLS"]
scenarios = ["S0 baseline (p=.70)", "S1 true p=.50", "S2 true p=.90",
             "S3 pre-trend d=-.002", "S4 pre-trend d=-.005", "S5 fatigue k=.15",
             "S6 fatigue k=.30", "S7 unobs. confounder", "S8 IV exclusion violated",
             "S9 drift decay .1 (hypothesis)"]
scen_labels = ["S0\nbaseline", "S1\ntrue p=.50", "S2\ntrue p=.90", "S3\npre-trend\n-.002",
               "S4\npre-trend\n-.005", "S5\nfatigue\nk=.15", "S6\nfatigue\nk=.30",
               "S7\nconfound", "S8\nIV excl.\nviolated", "S9\ndecay .1\n(hyp.)"]
est_labels = ["DiD", "Path a\n(pooled)", "Indirect a\u00d7b\n(pooled)", "Indirect a\u00d7b\n(FE)",
              "Direct c\u2032\n(pooled)", "Direct c\u2032\n(FE)", "IV 2SLS", "IV naive\nOLS"]

groups = {"valid": [0,1,2], "pretrend": [3,4], "fatigue": [5,6], "confound": [6], "excl": [7], "decay": [8]}
group_bounds = [0,3,5,7,8,9,10]  # column boundaries for shading groups: valid(0-2), pretrend(3-4), fatigue(5-6), confound(7), IVexcl(8), decay(9)

cov = np.full((len(estimators), len(scenarios)), np.nan)
bias = np.full((len(estimators), len(scenarios)), np.nan)
na_mask = np.zeros_like(cov, dtype=bool)

for i, e in enumerate(estimators):
    for j, s in enumerate(scenarios):
        row = df[(df.scenario == s) & (df.estimator == e)]
        if len(row):
            cov[i, j] = row.coverage.values[0] * 100
            rb = row.rel_bias_pct.values[0]
            bias[i, j] = rb if not pd.isna(rb) else np.nan
            if pd.isna(rb):
                na_mask[i, j] = True

# Category: 0 = nominal (93-97), 1 = moderate breakdown (hatched hollow, hollow triangle marks 5-93 loosely), 2 = severe (<10% or >150% abs rel bias-ish)
# We define purely on coverage per pre-registered gate: nominal 93-97%; below 90% flagged; below 20% or above ~80% deviation = severe
cat = np.zeros_like(cov)
for i in range(cov.shape[0]):
    for j in range(cov.shape[1]):
        c = cov[i, j]
        if np.isnan(c):
            cat[i, j] = -1
            continue
        if 93 <= c <= 97:
            cat[i, j] = 0   # nominal
        elif c >= 90:
            cat[i, j] = 0.5  # borderline nominal (MC-noise band)
        elif c >= 50:
            cat[i, j] = 1   # moderate breakdown
        else:
            cat[i, j] = 2   # severe breakdown

color_map = {0: '0.92', 0.5: '0.78', 1: '0.60', 2: '0.30', -1: '1.0'}
hatch_map = {0: '', 0.5: '', 1: '//', 2: 'xx', -1: ''}

fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=300)

nrow, ncol = cov.shape
for i in range(nrow):
    for j in range(ncol):
        c = cat[i, j]
        rect = plt.Rectangle((j, nrow - i - 1), 1, 1, facecolor=color_map[c],
                              edgecolor='white', linewidth=1.6, hatch=hatch_map[c])
        rect.set_hatch(hatch_map[c])
        ax.add_patch(rect)
        covval = cov[i, j]
        if not np.isnan(covval):
            bval = bias[i, j]
            if np.isnan(bval):
                label = f"{covval:.0f}%"
            else:
                label = f"{covval:.0f}%\n({bval:+.0f}%)"
            ax.text(j + 0.5, nrow - i - 1 + 0.5, label, ha='center', va='center',
                    fontsize=8.4, color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.82))

# group separators (vertical thick lines)
for b in group_bounds[1:-1]:
    ax.plot([b, b], [0, nrow], color='black', linewidth=2.2)

ax.set_xlim(0, ncol)
ax.set_ylim(0, nrow)
ax.set_xticks(np.arange(ncol) + 0.5)
ax.set_xticklabels(scen_labels, fontsize=9)
ax.set_yticks(np.arange(nrow) + 0.5)
ax.set_yticklabels(est_labels[::-1], fontsize=9.3)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# group headers
group_defs = [(0,3,"Valid design"), (3,5,"Pre-trend\nviolation"), (5,7,"Alert\nfatigue"),
              (7,8,"Unobserved\nconfounder"), (8,9,"IV exclusion\nviolated"), (9,10,"Drift decay\n(hypothesis)")]
for (a,b,lab) in group_defs:
    ax.text((a+b)/2, nrow + 0.35, lab, ha='center', va='bottom', fontsize=8.8, fontweight='bold')

legend_patches = [
    mpatches.Patch(facecolor='0.92', edgecolor='black', label='Nominal (93\u201397% coverage)'),
    mpatches.Patch(facecolor='0.78', edgecolor='black', label='Borderline (90\u201393%)'),
    mpatches.Patch(facecolor='0.55', edgecolor='black', hatch='///', label='Moderate breakdown (50\u201390%)'),
    mpatches.Patch(facecolor='0.18', edgecolor='black', hatch='xxx', label='Severe breakdown (<50%)'),
]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.09),
          ncol=4, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(str(ROOT / 'figures' / 'fig2_coverage_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("saved")
