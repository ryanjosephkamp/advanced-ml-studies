"""The course book's copies of this study's six chapter figures, for its day and night pages.

Each figure is drawn by study/figures.py's own function from the saved results in results/study (and, for the
permutation null, the March 2026 comparison table and the latest corrected-baseline run's comparison table):

    fig7_imf_example        fig_imf_example        imf_example.csv
    fig6_null               fig_null               split_null.csv and the two comparison tables
    fig1_heldout            fig_dots               summary.csv
    fig3_emd_vs_bandpower   fig_emd_vs_bandpower   paired_tests.csv
    fig5_leak               fig_leak               split_summary.csv and summary.csv
    fig8_clustering         fig_clustering         clustering.csv

The arguments come from make_all itself: it runs once with its drawing functions replaced by recorders, so every
file is read and filtered exactly as the study reads it, and nothing is drawn or written. The study's rcParams
(_style) stay in force for the whole run, as they are when make_all saves: matplotlib picks how many automatic ticks
an axis gets from the tick-label size in force when the figure is drawn. _style sets sizes and colors only, so the
book's type stands. In the book's copies, an axis with automatic ticks keeps the ticks of the study's own run,
which the book's type would otherwise change by changing the axes' width.

The study pairs its three colors with three marker shapes in one fixed order in every figure (blue circles,
orange squares, aqua diamonds), and the book keeps those pairs: crimson circles, gold squares, teal diamonds.
Band power, the circle, is the feature set the chapter quotes for every held-out result and measures each EMD
feature set against, so it takes the chapter's ink; the same order gives the chapter's ink to the IMFs in the
one-window example, as the chapter's IMF demo draws them. The drawing and the checks are in tools/site_figures.py.

Usage, from the repository's root or this project's folder:
    python signal-analysis-emd/figures/site/render.py [--check]
"""
import glob
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True  # leave no __pycache__ beside the study's code or the shared tools
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent / "tools"))
import site_figures  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import AutoLocator, FixedLocator, MaxNLocator  # noqa: E402

_spec = importlib.util.spec_from_file_location("emd_study_figures", PROJECT / "study/figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

RESULTS = "results/study"
MARCH = f"{RESULTS}/march_2026_comparison_table.csv"
DRAWERS = ("fig_dots", "fig_emd_vs_bandpower", "fig_spectra", "fig_leak", "fig_null", "fig_imf_example",
           "fig_clustering")
# Where a drawing function would save if the book's capture of savefig were ever bypassed: a folder that does not
# exist, so such a run fails instead of replacing a committed PNG
NOWHERE = Path(tempfile.gettempdir()) / "emd-book-figures-never-written"


def on_paper(*roles):
    """See-through inks laid on the paper one over another, as one opaque color per theme."""
    out = {}
    for theme, T in site_figures.INKS.items():
        rgb = mcolors.to_rgb(T["paper"])
        for role in roles:
            *ink, a = mcolors.to_rgba(T[role])
            rgb = [a * c + (1 - a) * q for c, q in zip(ink, rgb)]
        out[theme] = mcolors.to_hex(rgb)
    return out


# The study's colors (figures.py's constants and literals) and the book's ink for each
INKS = {
    F.BLUE: "crimson",    # circles: band power, the 70/30 split of windows, agreement with classes, the IMFs,
                          # and the corrected baseline
    F.ORANGE: "gold",     # squares: IMF features, held-out subjects, agreement with subjects, the residue
    F.AQUA: "teal",       # diamonds: Hilbert band power, agreement with EMD row count
    F.INK: "ink",         # text and the one window's signal
    F.INK2: "ink2",       # chance and zero lines, their labels, p-values, and footnotes
    F.GRAY: "ink2",       # the March 2026 run's marks
    F.GRID: "band",       # the line joining a feature set's two results
    # The permutation null's bars stand on its shaded middle 95%, whose edge is drawn in its own fill. Both are
    # opaque, as the study's grays are, since see-through ink would darken only the bars inside the span and
    # outline the span; and the bars are faint laid over the band, so that, as in the study, they stand out from
    # the span as clearly as the span does from the paper.
    "#d7d6d1": on_paper("band", "faint"),   # the null's bars
    "#f2f1ee": on_paper("band"),            # the null's middle 95%
    "white": "paper",     # open markers' fill and the gaps between the null's bars
}
# Different marks that share an ink. "band" is the text's ink, see-through, and the check compares inks without
# their opacity, so it counts the band as the text's ink.
MERGE_OK = {
    "fig5_leak": [(F.INK, F.GRID)],     # text; the band joining a feature set's two results
    "fig6_null": [(F.GRAY, F.INK2)],    # the March 2026 run's tick marks; the footnote
}


def make_all_calls():
    """{figure name: (drawing function, its arguments)} as make_all passes them, recorded without drawing."""
    calls, real = {}, {n: getattr(F, n) for n in DRAWERS}

    def recorder(n):
        def record(*args, **kw):
            bound = inspect.signature(real[n]).bind(*args, **kw)
            name = Path(bound.arguments["out"]).stem
            bound.arguments["out"] = str(NOWHERE / f"{name}.png")
            calls[name] = (real[n], bound.args, bound.kwargs)
        return record

    try:
        for n in DRAWERS:
            setattr(F, n, recorder(n))
        F.make_all(str(PROJECT / RESULTS), str(PROJECT))
    finally:
        for n, f in real.items():
            setattr(F, n, f)
    return calls


def baseline_table(calls):
    """The corrected-baseline table fig_null reads when make_all names no run: the latest run but March's."""
    assert calls["fig6_null"][1][2] is None, "make_all named a baseline run; list its table here instead"
    runs = sorted(r for r in glob.glob(str(PROJECT / "results/run_*")) if not r.endswith("run_2026-03-09_15-21-15"))
    return str(Path(runs[-1], "comparative_analysis/comparison_table.csv").relative_to(PROJECT))


def drawer(fn, args, kw):
    def draw():
        F._style()  # as make_all sets them before drawing
        fn(*args, **kw)
    return draw


def same_ticks(spec):
    """A `before` hook: every axis with automatic ticks keeps the ticks the study's own run gives it."""
    ticks = []

    def before(fig, theme):
        if not ticks:
            plain, _ = site_figures.draw(spec)
            plain.canvas.draw()
            ticks.extend((ax.get_xticks(), ax.get_yticks()) for ax in plain.axes)
            plt.close(plain)
        for ax, (xt, yt) in zip(fig.axes, ticks):
            for axis, t in ((ax.xaxis, xt), (ax.yaxis, yt)):
                if isinstance(axis.get_major_locator(), (AutoLocator, MaxNLocator)):
                    axis.set_major_locator(FixedLocator(t))
    return before


def figures():
    F._style()  # the study's rcParams, in force whenever a figure is drawn or saved
    calls = make_all_calls()
    data = {  # in the chapter's order
        "fig7_imf_example": [f"{RESULTS}/imf_example.csv"],
        "fig6_null": [f"{RESULTS}/split_null.csv", MARCH, baseline_table(calls)],
        "fig1_heldout": [f"{RESULTS}/summary.csv"],
        "fig3_emd_vs_bandpower": [f"{RESULTS}/paired_tests.csv"],
        "fig5_leak": [f"{RESULTS}/split_summary.csv", f"{RESULTS}/summary.csv"],
        "fig8_clustering": [f"{RESULTS}/clustering.csv"],
    }
    specs = []
    for name, files in data.items():
        fn, args, kw = calls[name]
        spec = dict(name=name, committed=f"{RESULTS}/figures/{name}.png",
                    source=f"study/figures.py: {fn.__name__}, with the arguments make_all passes", data=files,
                    draw=drawer(fn, args, kw), inks=INKS, merge_ok=MERGE_OK.get(name, []))
        spec["before"] = same_ticks(spec)
        specs.append(spec)
    return specs


if __name__ == "__main__":
    ok = site_figures.render(PROJECT, figures(), check="--check" in sys.argv)
    sys.exit(0 if ok else 1)
