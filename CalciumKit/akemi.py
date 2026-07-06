import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ----------------------------
# Loaders (MATLAB v7.3 / HDF5)
# ----------------------------
def load_green_stat_iscell(green_mat_path):
    with h5py.File(green_mat_path, "r") as f:
        iscell = np.array(f["iscell"])
        good = iscell[0, :].astype(bool) if iscell.shape[0] == 2 else iscell[:, 0].astype(bool)

        stat_refs = f["stat"]
        stat = []
        for i in range(stat_refs.shape[0]):
            g = f[stat_refs[i, 0]]
            stat.append({
                "ypix": np.array(g["ypix"]).astype(int).squeeze(),
                "xpix": np.array(g["xpix"]).astype(int).squeeze(),
                "lam":  np.array(g["lam"]).astype(float).squeeze(),
            })
    return stat, good


def load_red_refImg(red_mat_path):
    """Load Suite2p ops['refImg'] (recommended for visualization) from v7.3 .mat."""
    with h5py.File(red_mat_path, "r") as f:
        ops = f["ops"]
        if "refImg" not in ops:
            raise KeyError("ops['refImg'] not found. Available keys: " + ", ".join(list(ops.keys())))
        red = np.array(ops["refImg"]).astype(np.float32)
    return red


# ----------------------------
# Engram tagging (no filter)
# ----------------------------
def get_tagged_cells_no_filter(stat, red, thresh_intensity, thresh_correlation):
    r_vals = np.array([np.mean(red[s["ypix"], s["xpix"]]) for s in stat])
    c_vals = np.array([np.corrcoef(red[s["ypix"], s["xpix"]], s["lam"])[0, 1] for s in stat])
    return (r_vals > thresh_intensity) & (c_vals > thresh_correlation), r_vals, c_vals


# ----------------------------
# Plotting: original, smoothed, circles on engram cells
# ----------------------------
def plot_refimg_smooth_and_engram_circles(
    red_ref,
    stat,
    is_eng,
    good=None,
    sigma=10,
    circle_radius=10,
    vmin_vmax_percentiles=(1, 99),
):
    """
    Plots:
      1) Original refImg
      2) Gaussian-smoothed refImg
      3) Original refImg with circles centered at ROI centroids for engram cells
    """

    red_ref = np.asarray(red_ref)
    Ly, Lx = red_ref.shape

    is_eng = np.asarray(is_eng).astype(bool)
    if good is None:
        good = np.ones_like(is_eng, dtype=bool)
    else:
        good = np.asarray(good).astype(bool)

    eng_idx = np.where(is_eng & good)[0]

    red_smooth = gaussian_filter(red_ref, sigma=sigma)

    def _imshow(ax, img, title):
        vmin, vmax = np.percentile(img, vmin_vmax_percentiles)
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _imshow(axes[0], red_ref, "refImg (original)")
    _imshow(axes[1], red_smooth, f"refImg (gaussian σ={sigma})")

    _imshow(axes[2], red_ref, f"refImg + engram circles (n={len(eng_idx)})")

    # draw circles at ROI centroids for engram cells
    theta = np.linspace(0, 2*np.pi, 200)
    for i in eng_idx:
        cy = float(np.mean(stat[i]["ypix"]))
        cx = float(np.mean(stat[i]["xpix"]))

        # circle param eqn
        ys = cy + circle_radius * np.sin(theta)
        xs = cx + circle_radius * np.cos(theta)

        # keep points in bounds (optional safety)
        keep = (ys >= 0) & (ys < Ly) & (xs >= 0) & (xs < Lx)
        axes[2].plot(xs[keep], ys[keep], linewidth=1.5)

    fig.tight_layout()
    return fig, axes


# ----------------------------
# Example usage
# ----------------------------
green_mat = "/Users/suthardr/Desktop/akemi/green/Fall_cellsOnly.mat"
red_mat   = "/Users/suthardr/Desktop/akemi/red/Fall_cellsOnly.mat"

stat, good = load_green_stat_iscell(green_mat)
red_ref = load_red_refImg(red_mat)

# Pick thresholds (example placeholders)
# Tip: start by printing r_vals/c_vals histograms to choose these.
is_eng, r_vals, c_vals = get_tagged_cells_no_filter(
    stat, red_ref,
    thresh_intensity=np.percentile([np.mean(red_ref[s["ypix"], s["xpix"]]) for s in stat], 90),  # top 10% ROI means
    thresh_correlation=0.2
)

plot_refimg_smooth_and_engram_circles(
    red_ref,
    stat,
    is_eng,
    good=good,
    sigma=10,
    circle_radius=10
)
plt.show()
