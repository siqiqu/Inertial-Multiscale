import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import exp
from function import soft_shrinkage, psnr, isnr

os.makedirs("results", exist_ok=True)

SIZE = 256
PERCENTAGE = 0.2

img = Image.open("image.jpeg").resize((SIZE, SIZE))
X_ref = np.asarray(ImageOps.grayscale(img)).astype("float32")
np.random.seed(42)
mask = np.random.binomial(1, 1 - PERCENTAGE, (SIZE, SIZE))
R = lambda X: X * mask
X_corrupt = R(X_ref)

def D_operator(X):
    return R(R(X) - X_corrupt)

def B_operator(X):
    return np.maximum(X - 255, 0) + np.minimum(X, 0)

def proj_box(X):
    return np.clip(X, 0, 255)


# IFBT Algorithm
def IFBT_system(X0, max_iterations, beta_iter, eps_iter, alpha_iter,
                sigma, D=None, B=None, proj=None,
                use_inertial=True, record_every=100):
    X_prev = X0.copy()
    X = X0.copy()

    psnr_curve = []
    isnr_curve = []
    record_iters = []

    label = "IFBT (Inertial)" if use_inertial else "FBT  (No Inertial)"

    for n in range(max_iterations - 1):
        if use_inertial:
            Y = X + alpha_iter[n] * (X - X_prev)
        else:
            Y = X

        VY = D_operator(Y) + beta_iter[n] * B(Y) + eps_iter[n] * Y

        L_B = 1.0
        lm_n = 0.4 / (L_B * beta_iter[n])
        Z = Y - lm_n * VY
        X_next = soft_shrinkage(Z, lm_n * sigma)

        X_prev = X
        X = X_next

        if (n + 1) % record_every == 0:
            p = psnr(X_ref, X)
            i = isnr(X_ref, X_corrupt, X)
            psnr_curve.append(p)
            isnr_curve.append(i)
            record_iters.append(n + 1)

        if (n + 1) % 1000 == 0:
            print(f"  [{label}] iter {n+1}/{max_iterations-1}  PSNR={psnr_curve[-1]:.2f} dB")

    return X, psnr_curve, isnr_curve, record_iters


max_iterations = 2000
K_max = max_iterations
k = np.arange(K_max)

beta_iter  = 0.1 * (1 + k) ** 0.8
eps_iter   = 0.1 / (1 + k) ** 0.7

# Nesterov
t_k = (1 + np.sqrt(1 + 4 * (k + 1) ** 2)) / 2
t_k1 = (1 + np.sqrt(1 + 4 * (k + 2) ** 2)) / 2
alpha_iter = (t_k - 1) / t_k1
alpha_iter = np.clip(alpha_iter, 0, 9 / 30)

sigma = 50
RECORD_EVERY = 50


print("\nRunning FBT (no inertial)...")
X_no_inertial, psnr_no, isnr_no, iters = IFBT_system(
    X0=X_corrupt, max_iterations=max_iterations,
    beta_iter=beta_iter, eps_iter=eps_iter, alpha_iter=alpha_iter,
    sigma=sigma, D=D_operator, B=B_operator, proj=proj_box,
    use_inertial=False, record_every=RECORD_EVERY
)

print("\nRunning IFBT (with inertial)...")
X_inertial, psnr_in, isnr_in, _ = IFBT_system(
    X0=X_corrupt, max_iterations=max_iterations,
    beta_iter=beta_iter, eps_iter=eps_iter, alpha_iter=alpha_iter,
    sigma=sigma, D=D_operator, B=B_operator, proj=proj_box,
    use_inertial=True, record_every=RECORD_EVERY
)


psnr_corrupt  = psnr(X_ref, X_corrupt)
psnr_no_final = psnr(X_ref, X_no_inertial)
psnr_in_final = psnr(X_ref, X_inertial)
isnr_no_final = isnr(X_ref, X_corrupt, X_no_inertial)
isnr_in_final = isnr(X_ref, X_corrupt, X_inertial)

print("\n" + "=" * 55)
print(f"Corrupted image      PSNR: {psnr_corrupt:.2f} dB")
print(f"FBT  (no inertial)   PSNR: {psnr_no_final:.2f} dB  ISNR: {isnr_no_final:.2f} dB")
print(f"IFBT (with inertial) PSNR: {psnr_in_final:.2f} dB  ISNR: {isnr_in_final:.2f} dB")
print(f"Gain from inertial:  {psnr_in_final - psnr_no_final:+.2f} dB")
print("=" * 55)


target = psnr_no_final
ifbt_reach = next((iters[i] for i, p in enumerate(psnr_in) if p >= target), None)
print(f"\nIFBT reaches FBT final level ({target:.2f} dB) at iteration {ifbt_reach}")
print(f"FBT  reaches {target:.2f} dB at iteration {max_iterations}")
if ifbt_reach:
    saved = max_iterations - ifbt_reach
    print(f"Inertial saves ~{saved} iterations ({saved/max_iterations*100:.1f}%)")


# Figure 1: 4-image comparison
fig1, axs = plt.subplots(1, 4, figsize=(18, 5))

axs[0].imshow(np.clip(X_ref, 0, 255), cmap=cm.Greys_r, vmin=0, vmax=255)
axs[0].set_title("Original Image", fontsize=12)
axs[0].axis("off")

axs[1].imshow(np.clip(X_corrupt, 0, 255), cmap=cm.Greys_r, vmin=0, vmax=255)
axs[1].set_title(f"Corrupted Image\nPSNR: {psnr_corrupt:.2f} dB", fontsize=12)
axs[1].axis("off")

axs[2].imshow(np.clip(X_no_inertial, 0, 255), cmap=cm.Greys_r, vmin=0, vmax=255)
axs[2].set_title(f"FBT (No Inertial)\nPSNR: {psnr_no_final:.2f} dB  ISNR: {isnr_no_final:.2f} dB", fontsize=12)
axs[2].axis("off")

axs[3].imshow(np.clip(X_inertial, 0, 255), cmap=cm.Greys_r, vmin=0, vmax=255)
axs[3].set_title(f"IFBT (With Inertial)\nPSNR: {psnr_in_final:.2f} dB  ISNR: {isnr_in_final:.2f} dB", fontsize=12)
axs[3].axis("off")

plt.suptitle(f"Inertial vs Non-Inertial",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("results/comparison_4images_decoupling.png", dpi=150, bbox_inches='tight')
plt.show()


# Figure 2: Convergence curves
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(iters, psnr_no, color='steelblue', linewidth=2,
         linestyle='--', label=f'FBT   final={psnr_no_final:.2f} dB')
ax1.plot(iters, psnr_in, color='tomato', linewidth=2,
         label=f'IFBT   final={psnr_in_final:.2f} dB')

if ifbt_reach:
    reach_psnr = next(p for i, p in zip(iters, psnr_in) if i == ifbt_reach)
    ax1.axvline(x=ifbt_reach, color='tomato', linestyle=':', alpha=0.6)
    ax1.axhline(y=target, color='steelblue', linestyle=':', alpha=0.6)
    ax1.annotate(f'IFBT reaches\nFBT final at iter {ifbt_reach}',
                 xy=(ifbt_reach, reach_psnr),
                 xytext=(ifbt_reach + 500, reach_psnr - 1.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9, color='gray')

ax1.set_xlabel("Iteration", fontsize=12)
ax1.set_ylabel("PSNR (dB)", fontsize=12)
ax1.set_title("Convergence: PSNR vs Iteration", fontsize=13)
ax1.legend(loc='lower right',fontsize=10)
ax1.grid(True, alpha=0.3)

cutoff = 500 // RECORD_EVERY
ax2.plot(iters[:cutoff], psnr_no[:cutoff], color='steelblue', linewidth=2,
         linestyle='--', label='FBT')
ax2.plot(iters[:cutoff], psnr_in[:cutoff], color='tomato', linewidth=2,
         label='IFBT')
ax2.fill_between(iters[:cutoff], psnr_no[:cutoff], psnr_in[:cutoff],
                 alpha=0.15, color='tomato', label='Inertial advantage region')
ax2.set_xlabel("Iteration", fontsize=12)
ax2.set_ylabel("PSNR (dB)", fontsize=12)
ax2.set_title("Early Convergence (first 500 iterations)", fontsize=13)
ax2.legend(loc='lower right',fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("IFBT vs FBT — Convergence Speed Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/convergence_curves_decoupling.png", dpi=150, bbox_inches='tight')
plt.show()