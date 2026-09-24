from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5

fig, ax = plt.subplots(figsize=(10.4, 8.0), dpi=300)
ax.set_xlim(0, 11.5)
ax.set_ylim(-1.6, 8.6)
ax.axis('off')

boxes = {
    'tau':   (1.6, 7.0, "Alert\nthreshold (\u03c4)"),
    'vol':   (5.2, 7.0, "Alert\nvolume (V)"),
    'load':  (8.8, 7.0, "Reviewer\nworkload"),
    'resp':  (8.8, 3.9, "Response\nprobability (p)"),
    'drift': (5.2, 0.9, "Unresolved\ndrift (D)"),
    'perf':  (1.6, 3.9, "Coding\nperformance (Y)"),
    'cost':  (1.6, 0.9, "Downstream\ncost"),
}

box_w, box_h = 2.15, 1.15
patches = {}
for key, (x, y, label) in boxes.items():
    fc = '0.94'
    bp = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                         boxstyle="round,pad=0.06,rounding_size=0.08",
                         linewidth=1.4, edgecolor='black', facecolor=fc, zorder=3)
    ax.add_patch(bp)
    ax.text(x, y, label, ha='center', va='center', fontsize=10.2, fontweight='bold', zorder=4)
    patches[key] = (x, y)

def arrow(p1, p2, style='solid', label=None, label_pos=0.5, sign='+', rad=0.0, lab_off=(0,0)):
    x1, y1 = boxes[p1][0], boxes[p1][1]
    x2, y2 = boxes[p2][0], boxes[p2][1]
    ls = '-' if style == 'solid' else (0, (5, 3))
    fa = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                          arrowstyle='-|>', mutation_scale=16, linewidth=1.6,
                          linestyle=ls, color='black', shrinkA=42, shrinkB=42, zorder=2)
    ax.add_patch(fa)
    if label:
        mx, my = (x1+x2)/2 + lab_off[0], (y1+y2)/2 + lab_off[1]
        ax.text(mx, my, f"{sign} {label}", ha='center', va='center', fontsize=8.8,
                style='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.88))

# Modelled arrows (solid) -- exactly the mechanisms in Eqs. (1)-(5)/(7)
arrow('tau', 'vol', label='sets', sign='', lab_off=(0, 0.32))
arrow('vol', 'load', label='raises', sign='', lab_off=(0, 0.32))
arrow('load', 'resp', label='fatigue erosion', sign='\u2212', lab_off=(1.05, 0))
arrow('resp', 'drift', label='lowers remediation', sign='\u2212', rad=0.15, lab_off=(1.15,-0.25))
arrow('vol', 'drift', label='raises exposure', sign='+', rad=-0.12, lab_off=(-0.65,0))
arrow('drift', 'perf', label='erodes (\u03b3)', sign='\u2212', lab_off=(0,-0.34))
arrow('perf', 'cost', label='detection correction', sign='\u2212', lab_off=(-1.05,0))
arrow('drift', 'cost', label='raises', sign='+', rad=-0.15, lab_off=(0,0.3))

# Hypothesised / not modelled (dashed)
arrow('drift', 'vol', style='dashed', label='drift begets more alerts (not modelled)', sign='', rad=0.32, lab_off=(1.65,0.75))

# Loop labels
ax.text(6.55, 5.35, "R1", fontsize=12, fontweight='bold', ha='center')
ax.text(3.3, 2.35, "B1", fontsize=12, fontweight='bold', ha='center')

# Estimator tags (panel B content folded in as annotations), placed clear of boxes
ax.text(5.2, 8.15, "DiD, mediation, and IV each read out a different edge of this loop",
        ha='center', fontsize=10, style='italic')

ax.text(1.6, -0.95, "IV: instruments the\n\u03c4\u2192V\u2192drift edge via\nlagged complexity", ha='center',
        fontsize=8.3, color='0.3')
ax.text(8.8, -0.95, "Cost-threshold model:\noptimises \u03c4 over the\nfull loop (Prop. 3)", ha='center',
        fontsize=8.3, color='0.3')
ax.text(5.2, -0.95, "Mediation (a, b, c\u2032):\ndecomposes the\ndrift\u2192performance path", ha='center',
        fontsize=8.3, color='0.3')

# Legend
leg_solid = plt.Line2D([0], [0], color='black', lw=1.6, ls='-', label='Modelled relationship (Eqs. 1\u20135, 7)')
leg_dash = plt.Line2D([0], [0], color='black', lw=1.6, ls=(0,(5,3)), label='Hypothesised / not modelled')
ax.legend(handles=[leg_solid, leg_dash], loc='lower center', bbox_to_anchor=(0.5, -0.22),
          ncol=2, frameon=False, fontsize=9.5)

ax.set_title("Fig. 1. The alert-monitoring system as a feedback loop.\n"
             "R1 = fatigue-erosion (reinforcing loop); B1 = detection-correction (balancing loop).",
             fontsize=12, pad=14, loc='center')

plt.tight_layout()
plt.savefig(str(ROOT / 'figures' / 'fig1_causal_loop.png'), dpi=300, bbox_inches='tight', facecolor='white')
print("saved")
