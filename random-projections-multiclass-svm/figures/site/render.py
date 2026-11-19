"""The course book's copies of the five random-projection figures the chapter shows, for its day and night pages.

Each figure is drawn by study/figures.py's own function from the saved results that its main() reads:
    fig7_original_pipeline_100_draws     fig_replication    study/results/raw_replication.csv.gz
    fig1_accuracy_vs_k                   fig_accuracy_vs_k  study/results/accuracy_summary.csv
    fig2_paired_difference_vs_gaussian   fig_paired         study/results/paired_differences.csv
    fig3_distortion_variance_vs_theory   fig_variance       study/results/variance_check.csv
    fig4_distortion_tails_vs_bound       fig_tails          study/results/distortion_summary.csv
fig_replication jitters its points with its own generator, seeded inside the function, so every run draws the
same jitter as the committed figure.

The book's inks replace the study's colors, one per matrix family in every figure. Eq. 4, the assignment's sparse
matrix that the chapter follows, takes the chapter's ink. The other families avoid the blue the chapter's heat map
uses for "above the Gaussian"; the only red-orange family in the difference figure is the one-per-column matrix,
which is below the Gaussian in every cell. The drawing and the checks are in tools/site_figures.py.

Usage, from the repository's root or this project's folder:
    python random-projections-multiclass-svm/figures/site/render.py [--check]
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the study's modules leaves nothing beside them
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "study"))
sys.path.insert(0, str(PROJECT.parent / "tools"))
import site_figures  # noqa: E402  (selects matplotlib's Agg backend before pyplot loads)
import figures as F  # noqa: E402  (study/figures.py)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

RESULTS = "study/results"
ACCURACY = f"{RESULTS}/accuracy_summary.csv"
PAIRED = f"{RESULTS}/paired_differences.csv"
VARIANCE = f"{RESULTS}/variance_check.csv"
DISTORTION = f"{RESULTS}/distortion_summary.csv"
REPLICATION = f"{RESULTS}/raw_replication.csv.gz"

# The study's colors (figures.py's COLORS, INK, INK2, and the gray bands) and the book's ink for each
INKS = {
    F.COLORS["achlioptas3"]: "teal",     # Eq. 4: the chapter's ink
    F.COLORS["gaussian"]: "crimson",     # the dense reference
    F.COLORS["rademacher"]: "violet",    # Eq. 3
    F.COLORS["verysparse"]: "gold",      # very sparse, s = sqrt(d)
    F.COLORS["onepercol"]: "rust",       # one nonzero per column, s = d
    F.INK: "ink",                        # PCA, the Gaussian's exact tail law, medians, and the axis labels
    "black": "ink",                      # titles and legend text
    F.INK2: "ink2",                      # all features, the majority class, the zero line, theory, and the bound
    "#d7d6d1": "band",                   # the 95% interval of all features
    "#ecebe7": "band",                   # the band of plus or minus two points
    "#d9d8d3": "band",                   # that band's legend key
    "white": "paper",                    # the open diamond for the report's draw
}
# Near-black INK draws marks and axis labels and black draws titles and legend text: one ink serves both. The book's
# band is that ink at a low opacity, so the gray bands share its color but are filled regions behind the data; the
# legend key for the two-point margin is the band it names.
MERGE_OK = [(F.INK, "black", "#d7d6d1", "#ecebe7", "#d9d8d3")]


SAVED_DPI = 200  # figures.py's save() writes every one of these figures at 200 dpi


def at_saved_dpi(fn, *args):
    """Run a study function with its figures made at the resolution the study saves them at. Matplotlib lays a
    figure out again for each resolution and picks how many ticks an axis gets from the result, so at its default
    100 dpi Breast Cancer's accuracy axis in Figure 1 loses the ticks at 65, 75, 85, and 95 that the committed PNG
    shows."""
    with plt.rc_context({"figure.dpi": SAVED_DPI}):
        fn(*args)


def study_ticks(spec):
    """A `before` hook that keeps each axis's ticks where the study's own run puts them. The book's type changes
    the axes' lengths a little, and the SVG is laid out at 72 dpi, either of which can change the number of ticks
    matplotlib picks. Axes with ticks the study set are left as they are."""
    ticks = []

    def before(fig, theme):
        if not ticks:
            plain, _ = site_figures.draw(spec)
            plain.canvas.draw()
            ticks.extend((ax.get_xticks(), ax.get_yticks()) for ax in plain.axes)
            plt.close(plain)
        for ax, (xt, yt) in zip(fig.axes, ticks):
            for axis, t in ((ax.xaxis, xt), (ax.yaxis, yt)):
                if not isinstance(axis.get_major_locator(), FixedLocator):
                    axis.set_major_locator(FixedLocator(t))
    return before


def figures():
    acc = pd.read_csv(PROJECT / ACCURACY)
    pdf = pd.read_csv(PROJECT / PAIRED)
    vc = pd.read_csv(PROJECT / VARIANCE)
    ds_sum = pd.read_csv(PROJECT / DISTORTION)
    rep = pd.read_csv(PROJECT / REPLICATION)
    common = dict(inks=INKS, merge_ok=MERGE_OK)
    specs = [
        dict(name="fig7_original_pipeline_100_draws", committed="study/figures/fig7_original_pipeline_100_draws.png",
             source="study/figures.py: fig_replication", data=[REPLICATION],
             draw=lambda: at_saved_dpi(F.fig_replication, rep), **common),
        dict(name="fig1_accuracy_vs_k", committed="study/figures/fig1_accuracy_vs_k.png",
             source="study/figures.py: fig_accuracy_vs_k", data=[ACCURACY],
             draw=lambda: at_saved_dpi(F.fig_accuracy_vs_k, acc), **common),
        dict(name="fig2_paired_difference_vs_gaussian",
             committed="study/figures/fig2_paired_difference_vs_gaussian.png",
             source="study/figures.py: fig_paired", data=[PAIRED],
             draw=lambda: at_saved_dpi(F.fig_paired, pdf), **common),
        dict(name="fig3_distortion_variance_vs_theory",
             committed="study/figures/fig3_distortion_variance_vs_theory.png",
             source="study/figures.py: fig_variance", data=[VARIANCE],
             draw=lambda: at_saved_dpi(F.fig_variance, vc), **common),
        dict(name="fig4_distortion_tails_vs_bound", committed="study/figures/fig4_distortion_tails_vs_bound.png",
             source="study/figures.py: fig_tails", data=[DISTORTION],
             draw=lambda: at_saved_dpi(F.fig_tails, ds_sum), **common),
    ]
    for spec in specs:
        spec["before"] = study_ticks(spec)
    specs[0]["after"] = denser_by_day
    return specs


def denser_by_day(fig, theme):
    """fig7's draws are drawn at 55% opacity, so overlapping draws show; on the day page that leaves the book's inks
    under 3:1 against the mount, so they are drawn at 85% there. Opacity only: the points are the study's."""
    if theme == "light":
        for ax in fig.axes:
            for c in ax.collections:
                if c.get_alpha() == 0.55:
                    c.set_alpha(0.85)


if __name__ == "__main__":
    ok = site_figures.render(PROJECT, figures(), check="--check" in sys.argv)
    sys.exit(0 if ok else 1)
