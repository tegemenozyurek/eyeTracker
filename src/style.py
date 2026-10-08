"""One chart style for every script (same palette as emotionDetecter's charts)."""
import matplotlib.pyplot as plt

COLORS = {
    "train": "#2a78d6", "val": "#eb6834", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
    "surface": "#fcfcfb", "bad": "#c0392b",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11, "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["muted"], "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.edgecolor": COLORS["grid"], "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
    })


def finish(ax, grid="y"):
    """Light grid, no box, no tick marks."""
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(axis="both", length=0)
