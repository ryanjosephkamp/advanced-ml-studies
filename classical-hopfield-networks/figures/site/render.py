"""The course book's copies of this study's five chapter figures, for its day and night pages.

Each figure is drawn by study/analyze.py's own function (fig1 to fig5) from the arrays its main() computes: the
recall curves and capacities from the saved synthetic runs (study/results/synthetic_recall.csv.gz, with the pattern
overlaps from synthetic_patterns.csv), and the character results from characters_subsets.csv and
characters_full.csv. The study's summary functions also write the summary tables; here they write them into a
temporary folder, and each table is checked against the committed one, byte for byte, before any figure is drawn.

The book's inks replace the study's colors. The chapter's ink goes to what the chapter follows in each figure: the
Hebbian rule among the rules, the shared-prototype family among the synthetic families, the character subsets, and
the pseudoinverse rule's spread of results. The ten overlap levels of the recall curves take an ordered ramp
through the book's blue, violet, crimson, rust, and gold, as the study's ramp runs from blue to yellow: deep to light
on the day page and dim to bright on the night page. The drawing and the checks are in tools/site_figures.py.

Usage, from the repository's root or this project's folder:
    python classical-hopfield-networks/figures/site/render.py [--check]
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import colormaps, colors as mcolors

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "study"))
sys.path.insert(0, str(PROJECT.parent / "tools"))
import analyze as S  # noqa: E402
import site_figures  # noqa: E402

RESULTS = "study/results"
INPUTS = ["synthetic_recall.csv.gz", "synthetic_fixed.csv.gz", "synthetic_patterns.csv", "characters_full.csv",
          "characters_subsets.csv"]
SUMMARIES = ["summary_synthetic_curves.csv", "summary_capacity.csv", "summary_characters_full.csv",
             "summary_character_subsets.csv", "summary_character_subsets_p50.csv"]


def results():
    """main()'s recall curves, capacities, pattern overlaps, and character results, computed by the study's own
    functions from the saved runs. Their summary tables go to a temporary folder and must match the committed ones."""
    saved = S.RES
    with tempfile.TemporaryDirectory(prefix="hopfield-site-") as tmp:
        tmp = Path(tmp)
        for f in INPUTS:  # the runs are read in place; only the summaries are written, and only here
            os.symlink(saved / f, tmp / f)
        S.RES = tmp
        try:
            E, OV, F, PO, PD = S.load_synthetic()
            curve, fixed, cap = S.summarize_synthetic(E, OV, F, PO, PD)
            dist, rec = S.summarize_characters()
            for f in SUMMARIES:
                assert (tmp / f).read_bytes() == (saved / f).read_bytes(), f"{f} differs from the committed table"
        finally:
            S.RES = saved
    return curve, cap, PO, dist, rec


# ---- the overlap ramp of fig1 -------------------------------------------------------------------------------------

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def to_oklab(c):
    rgb = np.array(mcolors.to_rgb(c))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return _M2 @ np.cbrt(_M1 @ lin)


def from_oklab(lab):
    """The sRGB hex color for an OKLab color, with its chroma reduced until it fits in sRGB."""
    for shrink in np.linspace(1, 0, 41):
        lin = np.linalg.solve(_M1, np.linalg.solve(_M2, [lab[0], lab[1] * shrink, lab[2] * shrink]) ** 3)
        if lin.min() >= -1e-4 and lin.max() <= 1 + 1e-4:
            break
    lin = np.clip(lin, 0, 1)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return mcolors.to_hex(np.clip(rgb, 0, 1))


LEVELS = len(S.PROTO_LEVELS)
# As in the study's plasma ramp, the levels run from blue (overlap 0) to gold (0.9) on both pages, so a level keeps
# its hue from day to night; the lightness runs deep to light by day and dim to bright by night
RAMP_HUES = {"light": ("blue", "violet", "crimson", "rust", "gold"),
             "dark": ("blue", "violet", "crimson", "rust", "gold")}
RAMP_LIGHTNESS = {"light": (0.28, 0.62), "dark": (0.56, 0.92)}  # the palest day level stays at 3:1 on the mount  # OKLab lightness of the first and last level


def ramp(theme):
    """One color per overlap level, in order. The lightness steps evenly (deep to light on the day page, dim to
    bright on the night page), so the order reads from lightness alone; the hue moves through the book's inks, so
    neighboring levels also differ in hue."""
    hues = [to_oklab(site_figures.INKS[theme][k])[1:] for k in RAMP_HUES[theme]]
    out = []
    for k, L in enumerate(np.linspace(*RAMP_LIGHTNESS[theme], LEVELS)):
        t = k / (LEVELS - 1) * (len(hues) - 1)
        i = min(int(t), len(hues) - 2)
        ab = (1 - (t - i)) * hues[i] + (t - i) * hues[i + 1]
        out.append(from_oklab([L, *ab]))
    return out


# fig1's colors: plasma(0.85 t) for level k at t = k / 9 (analyze.fig1's cmap), each taking the ramp's color k
PLASMA = [mcolors.to_hex(colormaps["plasma"](0.85 * k / (LEVELS - 1))) for k in range(LEVELS)]
RAMP = {"light": ramp("light"), "dark": ramp("dark")}
FIG1_INKS = {c: {"light": RAMP["light"][k], "dark": RAMP["dark"][k]} for k, c in enumerate(PLASMA)}
FIG1_INKS["black"] = "ink"                                    # titles, labels, and the legend

# ---- fig2: one color per rule (analyze.COLOR) --------------------------------------------------------------------

HEBBIAN = "#000001"  # the Hebbian rule's black, told apart from the black text (see hebbian_apart)
FIG2_INKS = {
    HEBBIAN: "rust",                 # Hebbian: the rule the chapter follows
    S.COLOR["hebb01"]: "ink2",       # Hebbian with the 0/1 update: gray in the study too
    S.COLOR["storkey"]: "gold",
    S.COLOR["pinv"]: "blue",
    S.COLOR["cov"]: "teal",
    S.COLOR["modern16"]: "violet",
    S.COLOR["modern64"]: "ink",
    S.COLOR["nn"]: "crimson",
    "#F2F2F2": "band",               # the band where capacities above 40 are drawn
    "black": "ink",                  # titles, labels, and the legend's words
}
# The modern network at beta = 64 is drawn as a line with markers, black is left only on the words, and the band is
# a see-through shaded region (the "band" ink is the page's ink at low opacity): different marks, so one ink serves
# all three.
FIG2_MERGE_OK = [("black", S.COLOR["modern64"], "#F2F2F2")]


def hebbian_apart(fig, theme):
    """Give the Hebbian rule's lines, bands, and legend key (the only black marks in fig2) a stand-in color, so they
    take the chapter's ink while the black words take the page's ink. Changes colors only."""
    black = mcolors.to_rgb("black")
    lines = [ln for ax in fig.axes for ln in ax.lines] + [h for lg in fig.legends for h in lg.legend_handles]
    for ln in lines:
        if mcolors.to_rgb(ln.get_color()) == black:
            ln.set_color(HEBBIAN)
    for ax in fig.axes:
        for band in ax.collections:
            fc = band.get_facecolor()
            if len(fc) and tuple(fc[0][:3]) == black:
                band.set_facecolor(mcolors.to_rgba(HEBBIAN, fc[0][3]))


# ---- fig3 to fig5: the study's blue (#0072B2) and vermilion (#D55E00) series --------------------------------------

FIG3_INKS = {
    "#0072B2": "rust",               # patterns that share a prototype: the study's main family
    "#D55E00": "teal",               # independent bits at densities 0.65 and 0.80
    "black": "ink",
}
FIG4_INKS = {
    "#0072B2": "rust",               # the character subsets: what the chapter is about
    "#D55E00": "teal",               # synthetic patterns at density 0.80, as in fig3
    "#555555": "ink2",               # synthetic patterns at overlaps 0.4 and 0.5
    "#999999": "ink2",               # their bands
    "black": "ink",
}
# #999999 is only the bands drawn under the #555555 lines (at 18% opacity): a band and its own line share an ink.
FIG4_MERGE_OK = [("#555555", "#999999")]
def see_through(role, alpha):
    """A book ink at partial opacity on each page ({"light": a, "dark": b}, or one value for both)."""
    a = alpha if isinstance(alpha, dict) else {"light": alpha, "dark": alpha}
    return {t: mcolors.to_hex(mcolors.to_rgba(site_figures.INKS[t][role], a[t]), keep_alpha=True)
            for t in ("light", "dark")}


FIG5_INKS = {
    # the share of draws at each count: the spread the chapter is about. By night the bars are see-through so the
    # chalk of the mean's dashed line and the label over them stay legible; by day they are nearly solid, at 3:1 or
    # more against the mount
    "#0072B2": see_through("rust", {"light": 0.9, "dark": 0.65}),
    "#D55E00": "ink",                # the mean (dashed)
    "#333333": "ink2",               # the arrow to the original report
    "black": "ink",
}
# The mean is a dashed vertical line and black is only the words: different marks, so one ink serves both.
FIG5_MERGE_OK = [("black", "#D55E00")]


def figures():
    curve, cap, PO, dist, rec = results()
    synthetic = f"{RESULTS}/synthetic_recall.csv.gz"
    checked = ("curves and capacities recomputed with load_synthetic and summarize_synthetic, checked against "
               "summary_synthetic_curves.csv and summary_capacity.csv")
    characters = ("character results recomputed with summarize_characters, checked against "
                  "summary_character_subsets.csv and summary_characters_full.csv")
    return [
        dict(name="fig1_recall_vs_load", committed="study/figures/fig1_recall_vs_load.png",
             source=f"study/analyze.py: fig1 at 10% noise ({checked})", data=[synthetic],
             draw=lambda: S.fig1(curve, 0, S.FIG / "fig1_recall_vs_load.png"), inks=FIG1_INKS),
        dict(name="fig2_capacity_vs_overlap", committed="study/figures/fig2_capacity_vs_overlap.png",
             source=f"study/analyze.py: fig2 ({checked})", data=[synthetic],
             draw=lambda: S.fig2(cap, S.FIG / "fig2_capacity_vs_overlap.png"), inks=FIG2_INKS,
             merge_ok=FIG2_MERGE_OK, before=hebbian_apart),
        dict(name="fig3_density_vs_structure", committed="study/figures/fig3_density_vs_structure.png",
             source=f"study/analyze.py: fig3 ({checked})", data=[synthetic, f"{RESULTS}/synthetic_patterns.csv"],
             draw=lambda: S.fig3(cap, PO, S.FIG / "fig3_density_vs_structure.png"), inks=FIG3_INKS),
        dict(name="fig4_characters_vs_synthetic", committed="study/figures/fig4_characters_vs_synthetic.png",
             source=f"study/analyze.py: fig4 ({checked}; {characters})",
             data=[synthetic, f"{RESULTS}/characters_subsets.csv"],
             draw=lambda: S.fig4(curve, rec, S.FIG / "fig4_characters_vs_synthetic.png"), inks=FIG4_INKS,
             merge_ok=FIG4_MERGE_OK),
        dict(name="fig5_pseudoinverse_order_spread", committed="study/figures/fig5_pseudoinverse_order_spread.png",
             source=f"study/analyze.py: fig5 ({characters})", data=[f"{RESULTS}/characters_full.csv"],
             draw=lambda: S.fig5(dist, S.FIG / "fig5_pseudoinverse_order_spread.png"), inks=FIG5_INKS,
             merge_ok=FIG5_MERGE_OK),
    ]


if __name__ == "__main__":
    ok = site_figures.render(PROJECT, figures(), check="--check" in sys.argv)
    sys.exit(0 if ok else 1)
