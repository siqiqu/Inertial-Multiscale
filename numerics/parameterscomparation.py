"""
Figure 1: parametercomparation.py

"""

import os
import sys, time, csv, itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from PIL import Image, ImageOps
import pandas as pd
from function import soft_shrinkage, psnr, isnr

os.makedirs("results", exist_ok=True)


#  SWITCH — True: run grid search and save CSV  |  False: load existing CSV
RUN_GRID_SEARCH = True


# ─── Image ─────
SIZE, PERCENTAGE, sigma = 256, 0.2, 50
img = Image.open("image.jpeg").resize((SIZE, SIZE))
X_ref = np.asarray(ImageOps.grayscale(img)).astype("float32")
np.random.seed(42)
mask = np.random.binomial(1, 1 - PERCENTAGE, (SIZE, SIZE))
R = lambda X: X * mask
X_corrupt = R(X_ref)
psnr_c = psnr(X_ref, X_corrupt)
print(f"Corrupted PSNR baseline: {psnr_c:.2f} dB")


def D_op(X): return R(R(X) - X_corrupt)


def B_op(X): return np.maximum(X - 255, 0) + np.minimum(X, 0)



#  ─── GRID SEARCH ───
CSV_PATH = "results/grid_search_results.csv"

if RUN_GRID_SEARCH:
    print("\n[Grid Search] Starting …")

    GS_SIZE = 64
    img_gs   = Image.open("image.jpeg").resize((GS_SIZE, GS_SIZE))
    X_ref_gs = np.asarray(ImageOps.grayscale(img_gs)).astype("float32")
    np.random.seed(42)
    mask_gs   = np.random.binomial(1, 1 - PERCENTAGE, (GS_SIZE, GS_SIZE))
    X_corr_gs = X_ref_gs * mask_gs

    def D_op_gs(X): return mask_gs * (mask_gs * X - X_corr_gs)
    def B_op_gs(X): return np.maximum(X - 255, 0) + np.minimum(X, 0)

    GS_ITER  = 500
    k_gs     = np.arange(GS_ITER)
    GS_SIGMA = 50

    def run_one_gs(beta_mult, beta_exp, eps_mult, eps_exp):
        beta_iter = beta_mult * (1 + k_gs) ** beta_exp
        eps_iter  = eps_mult  / (1 + k_gs) ** eps_exp

        # FBT pure parameter search, no inertial
        X = X_corr_gs.copy()
        try:
            for n in range(GS_ITER - 1):
                VY     = D_op_gs(X) + beta_iter[n] * B_op_gs(X) + eps_iter[n] * X
                lm     = 0.4 / beta_iter[n]
                X      = soft_shrinkage(X - lm * VY, lm * GS_SIGMA)
            pi = psnr(X_ref_gs, X)
        except Exception:
            pi = -999.

        return pi

    beta_mults = [0.1, 1.0, 10.0]
    beta_exps  = [0.6, 0.7, 0.8, 0.9, 1.0]
    eps_mults  = [0.1, 1.0, 10.0]
    eps_exps   = [0.6, 0.7, 0.8, 0.9, 1.0]
    combos = [(bm, be, em, ee)
              for bm, be, em, ee in itertools.product(beta_mults, beta_exps, eps_mults, eps_exps)
              if be > ee]          # beta_exp bigger than eps_exp
    N = len(combos)
    print(f"  {N} combinations × {GS_ITER} iterations")

    gs_rows = []
    t0 = time.time()
    for idx, (bm, be, em, ee) in enumerate(combos):
        p = run_one_gs(bm, be, em, ee)
        gs_rows.append((bm, be, em, ee, p))
        if (idx + 1) % 25 == 0 or idx + 1 == N:
            best = max(gs_rows, key=lambda r: r[4])
            print(f"  [{idx+1:3d}/{N}]  {time.time()-t0:.0f}s  "
                  f"best PSNR={best[4]:.3f} dB "
                  f"(bm={best[0]}, be={best[1]}, em={best[2]}, ee={best[3]})")
            sys.stdout.flush()

    gs_rows.sort(key=lambda r: r[4], reverse=True)
    print(f"\n  Done in {time.time()-t0:.1f}s")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "beta_mult", "beta_exp", "eps_mult", "eps_exp", "PSNR"])
        for rank, (bm, be, em, ee, p) in enumerate(gs_rows):
            w.writerow(["coarse", bm, be, em, ee, f"{p:.4f}"])
    print(f"  Saved: {CSV_PATH}\n")

else:
    print(f"\n[Grid Search] Skipped — loading {CSV_PATH}\n")



df = pd.read_csv("results/grid_search_results.csv")
coarse_df = df[df['phase'] == 'coarse'].copy()

# Fix multipliers
FIXED_B_MULT = 0.1
FIXED_E_MULT = 0.1


coarse_fixed = coarse_df[
    (coarse_df['beta_mult'] == FIXED_B_MULT) &
    (coarse_df['eps_mult'] == FIXED_E_MULT)
    ].copy()

# best exponent pair under fixed multipliers
winner_row = coarse_fixed.dropna(subset=['PSNR']).sort_values('PSNR', ascending=False).iloc[0]
BEST_B_EXP = winner_row['beta_exp']
BEST_E_EXP = winner_row['eps_exp']
print(f"Winner (fixed mults): β_exp={BEST_B_EXP}, ε_exp={BEST_E_EXP}  "
      f"PSNR={winner_row['PSNR']:.2f} dB")


pair_best = defaultdict(lambda: float('-inf'))
for _, row in coarse_fixed.dropna(subset=['PSNR']).iterrows():
    key = (row['beta_exp'], row['eps_exp'])
    if row['PSNR'] > pair_best[key]:
        pair_best[key] = row['PSNR']

all_b = sorted(coarse_df['beta_exp'].unique())
all_e = sorted(coarse_df['eps_exp'].unique())
grid = np.full((len(all_b), len(all_e)), np.nan)
for (b, e), p in pair_best.items():
    if b in all_b and e in all_e:
        grid[all_b.index(b), all_e.index(e)] = p

cmap = LinearSegmentedColormap.from_list(
    "psnr", ["#d73027", "#f46d43", "#fdae61", "#ffffbf", "#a6d96a", "#1a9850"], N=256
)

fig1, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig1.patch.set_facecolor("white")


ax = axes[0]
ax.set_facecolor("#f8f8f8")
valid_vals = grid[~np.isnan(grid)]
im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower',
               vmin=np.percentile(valid_vals, 10) if len(valid_vals) else 0,
               vmax=np.nanmax(grid) if len(valid_vals) else 1)

ax.set_xticks(range(len(all_e)))
ax.set_xticklabels([f"{e:.1f}" for e in all_e], color='black', fontsize=10)
ax.set_yticks(range(len(all_b)))
ax.set_yticklabels([f"{b:.1f}" for b in all_b], color='black', fontsize=10)
ax.set_xlabel(r"$\varepsilon$ exponent", color='black', fontsize=12, labelpad=8)
ax.set_ylabel(r"$\beta$ exponent", color='black', fontsize=12, labelpad=8)
ax.set_title(
    fr"PSNR (dB) per Exponent Pair"
    "\n"
    fr"($C_{{\beta}}={FIXED_B_MULT}$,  $C_{{\varepsilon}}={FIXED_E_MULT}$)",
    color='black', fontsize=12, pad=12)
ax.tick_params(colors='black')
for spine in ax.spines.values():
    spine.set_edgecolor('#cccccc')

mean_val = np.nanmean(grid)
for i in range(len(all_b)):
    for j in range(len(all_e)):
        v = grid[i, j]
        if not np.isnan(v):
            color = 'white' if v > mean_val else 'black'
            ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold')
        else:
            ax.text(j, i, "—", ha='center', va='center', fontsize=9, color='#aaaaaa')


if BEST_B_EXP in all_b and BEST_E_EXP in all_e:
    bi = all_b.index(BEST_B_EXP)
    ei = all_e.index(BEST_E_EXP)
    ax.add_patch(plt.Rectangle((ei - 0.5, bi - 0.5), 1, 1,
                               fill=False, edgecolor='black',
                               linewidth=2.5, zorder=5))

cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("PSNR (dB)", color='black', fontsize=10)
cb.ax.yaxis.set_tick_params(color='black')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')


ax2 = axes[1]
ax2.set_facecolor("#f8f8f8")

top_n = coarse_fixed.dropna(subset=['PSNR']).sort_values('PSNR', ascending=False).head(10)
labels = [fr"$C_{{\beta}}$={r.beta_mult},  $e_{{\beta}}$={r.beta_exp},  $C_{{\varepsilon}}$={r.eps_mult},  $e_{{\varepsilon}}$={r.eps_exp}"
          for _, r in top_n.iterrows()]
values = top_n['PSNR'].values
palette = ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b",
           "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026", "#762a83"]

bars = ax2.barh(range(len(top_n)), values, color=palette[:len(top_n)],
                height=0.65, zorder=3, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(top_n)))
ax2.set_yticklabels(labels, color='black', fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("PSNR (dB)", color='black', fontsize=12)
ax2.set_title(r"Top Exponent Pairs",
              color='black', fontsize=12, pad=12)
ax2.tick_params(axis='x', colors='black')
ax2.tick_params(axis='y', colors='black')
ax2.grid(axis='x', color='#dddddd', linestyle='--', alpha=0.8, zorder=0)
for spine in ax2.spines.values():
    spine.set_edgecolor('#cccccc')

for bar, val in zip(bars, values):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}", va='center', color='black', fontsize=9.5, fontweight='bold')

xl = ax2.get_xlim()
ax2.set_xlim(xl[0], xl[1] + 2.0)
ax2.axvline(x=psnr_c, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.9, zorder=2)
ax2.text(psnr_c + 0.1, len(top_n) - 0.6,
         f"Corrupted\n{psnr_c:.1f} dB", color='#c0392b', fontsize=8.5)

fig1.suptitle("IFBT Parameter Grid Search Results", color='black',
              fontsize=14, fontweight='bold', y=1.01)
fig1.tight_layout()
fig1.savefig("results/figure1_grid_search.png", dpi=150, bbox_inches='tight',
             facecolor='white')
print("Figure 1 saved: results/figure1_grid_search.png")
print("\nDone")
plt.show()