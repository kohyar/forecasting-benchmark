"""Shared look for the paper's figures.

The four family hues are slots 1, 2, 3 and 7 of the reference categorical
palette, chosen because that combination clears all-pairs colour-vision
separation (worst CVD dE 9.2, worst normal-vision dE 16.3) - scatter plots put
every pair on screen at once, so adjacent-pair validation is not enough. Each
family also carries its own marker, which is what survives a greyscale print.
"""
from pathlib import Path

FAMILIES = ("baseline", "local", "global", "foundation")

_STYLES = {
    "baseline": {"color": "#2a78d6", "marker": "o"},
    "local": {"color": "#eb6834", "marker": "s"},
    "global": {"color": "#1baf7a", "marker": "^"},
    "foundation": {"color": "#4a3aa7", "marker": "D"},
}

_FALLBACK = {"color": "#52514e", "marker": "X"}

PRINT_DPI = 300


GRID = "#d9d8d4"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def family_style(family: str) -> dict:
    return dict(_STYLES.get(family, _FALLBACK))


def apply_paper_style() -> None:
    """Journal defaults: serif text near the body font, recessive rules, no
    top/right spines competing with the data.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def save_figure(fig, path, dpi: int = PRINT_DPI) -> list[Path]:
    """Write the vector copy LaTeX embeds and the raster copy everything else
    wants. `path` carries no extension.
    """
    stem = Path(path)
    written = []
    for suffix in (".pdf", ".png"):
        target = stem.with_suffix(suffix)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
    return written
