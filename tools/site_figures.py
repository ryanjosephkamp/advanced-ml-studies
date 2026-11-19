"""The course book's copies of a study's figures: drawn by the study's own plotting code, in the book's inks.

Each study's figures/site/render.py lists its figures: the study function that draws each one, the saved results it
reads, and which of the book's inks takes the place of each color the function uses. For every figure, this module

1. runs the function as the study does and checks that the result matches the PNG the study committed, every pixel
   in every channel, so the function and the data named are the ones that drew it;
2. runs it again for the book's day and night pages, with the book's type, and recolors what it drew: the
   background, the frame, and every ink, through the study's list (a color missing from the list stops the run);
3. checks that each restyled figure plots exactly what the plain run plots, before and after it is saved: every line,
   marker, band, bar, arrow, and label at the same coordinates, in the same coordinate system, with the same
   visibility, masks, and clipping, and the same axis ranges, scales, and major and minor ticks; that it holds no
   artist the check cannot compare; and that colors the study kept apart stay apart;
4. writes <name>.light.svg and <name>.dark.svg (WebP at twice the page's width when an SVG would pass 300 KB) and
   figures/site/manifest.json, which records each check and each file's SHA-256, only when every check passes.

The type is the book's Computer Modern (the woff2 files in docs/assets/themes/monograph/fonts, under the SIL Open
Font License), with DejaVu for the few signs those files leave out. Needs matplotlib (the files here were drawn with
3.10.7; another version can draw the plain run differently, which the first check reports), numpy, pillow, and
fonttools with brotli (pip install "fonttools[woff]").
"""
import hashlib
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors, font_manager  # noqa: E402
from matplotlib.collections import Collection, LineCollection, PathCollection, QuadMesh  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# the book's fonts: the built site's copy in this repository, or the folder BOOK_FONTS names
FONTS = Path(os.environ.get("BOOK_FONTS") or REPO / "docs/assets/themes/monograph/fonts")

# The book's inks (docs/assets/themes/monograph/theme.css and the chapters' own inks in the course file). "paper" is
# the plate's mount, so an open marker filled with it reads as hollow; "band" and "faint" are see-through ink.
INKS = {
    "light": {"ink": "#131a2b", "ink2": "#4a5468", "faint": "#131a2b2e", "band": "#131a2b1c", "paper": "#fcfcf9",
              "blue": "#2645a8", "rust": "#a8561c", "teal": "#12756a", "violet": "#5b3fa0", "crimson": "#a3283a",
              "gold": "#7d6200"},
    "dark": {"ink": "#e8ebe2", "ink2": "#a9b6ad", "faint": "#e8ebe233", "band": "#e8ebe221", "paper": "#1f2d28",
             "blue": "#8db4ff", "rust": "#ffbf8a", "teal": "#86e3d3", "violet": "#c9b8ff", "crimson": "#ff9aa6",
             "gold": "#f0d47a"},
}
SERIF = "CMU Concrete"  # the book's text face; CMU Serif and DejaVu Serif fill in what its cut leaves out
GRID_ALPHA = {"light": 0.13, "dark": 0.16}
SVG_LIMIT = 300_000
RASTER_WIDTH = 2400  # px: twice the widest track a figure gets on the page (72rem at 16 px is 1,152 px)


def load_fonts():
    """Register the book's Computer Modern cuts with matplotlib (converted from the site's woff2 files)."""
    from fontTools.ttLib import TTFont
    names = ("cm-concrete", "cm-roman", "cm-italic", "cm-bold")
    key = hashlib.sha256(b"".join((FONTS / f"{n}.woff2").read_bytes() for n in names)).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / f"book-fonts-{key}"  # one converted copy, reused by every run
    tmp.mkdir(exist_ok=True)
    for name in names:
        if not (tmp / f"{name}.ttf").exists():
            f = TTFont(FONTS / f"{name}.woff2")
            f.flavor = None
            f.save(tmp / f"{name}.ttf.part")
            os.replace(tmp / f"{name}.ttf.part", tmp / f"{name}.ttf")
        font_manager.fontManager.addfont(str(tmp / f"{name}.ttf"))


def site_rc():
    return {"font.family": [SERIF, "CMU Serif", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.unicode_minus": True,
            "svg.fonttype": "path", "svg.hashsalt": "course-book", "axes.formatter.use_mathtext": False}


@contextmanager
def captured():
    """Figures a study function saves, kept in memory instead: {file name: (figure, savefig keywords)}."""
    figs = {}
    save, close = Figure.savefig, plt.close

    def keep(self, fname, *a, **k):
        figs[Path(str(fname)).name] = (self, k)

    Figure.savefig = keep
    plt.close = lambda *a, **k: None
    try:
        yield figs
    finally:
        Figure.savefig, plt.close = save, close


def draw(spec, rc=None):
    """Run the study's function for one figure; return (figure, the keywords it saved with)."""
    with plt.rc_context(rc or {}), captured() as figs:
        spec["draw"]()
    hits = [v for k, v in figs.items() if Path(k).stem == spec["name"]]
    assert hits, f"{spec['name']}: the study function saved {sorted(figs)}, not this figure"
    for k, (f, _) in figs.items():
        if Path(k).stem != spec["name"]:
            plt.close(f)
    return hits[0]


# ---- what a figure plots, without its inks or type --------------------------------------------------------------

def _arr(a):
    """An array as plain floats, with a masked entry as NaN and the mask kept beside it."""
    if np.ma.isMaskedArray(a):
        return (np.ma.filled(np.asarray(a, dtype=float), np.nan), np.ma.getmaskarray(a).copy())
    return np.asarray(a, dtype=float)


def _where(a, ax):
    """Which coordinates an artist is placed in (data, axes, figure, or a blend), not where those land on the page."""
    t = a.get_transform()
    if ax is None:
        return "figure"
    names = []
    for name, base in (("data", ax.transData), ("axes", ax.transAxes), ("x data", ax.get_xaxis_transform()),
                       ("y data", ax.get_yaxis_transform()), ("figure", a.figure.transFigure)):
        if t == base or t.contains_branch(base):
            names.append(name)
    return " ".join(names) or type(t).__name__


def _style(x):
    """An arrow or connection style as its kind and its settings (not its address in memory)."""
    return (type(x).__name__, tuple(sorted((k, v) for k, v in vars(x).items() if isinstance(v, (int, float, str, bool)))))


class Unsupported(Exception):
    pass


def plotted(fig, redraw=True):
    """Everything a figure plots, in order, as (what, values) pairs: the check compares these, never the inks. With
    redraw=False, it reads the figure as the last drawing left it (after a save, the saved file's drawing)."""
    if redraw:
        fig.canvas.draw()
    out = []

    def text(t, what):
        if t.get_text():
            out.append((what, (t.get_text(), tuple(np.round(t.get_position(), 12)), t.get_ha(), t.get_va(),
                               t.get_rotation(), t.get_visible())))
        arrow = getattr(t, "arrow_patch", None)
        if arrow is not None:  # an annotation: where it points, and its arrow's shape
            out.append((f"{what} arrow", (tuple(np.round(t.xy, 12)), str(t.xycoords), str(getattr(t, "_textcoords", "")),
                                          _style(arrow.get_arrowstyle()), _style(arrow.get_connectionstyle()),
                                          arrow.get_linewidth(), arrow.get_visible())))

    def artist(a, where, ax):
        kind = type(a).__name__
        common = (a.get_visible(), _where(a, ax), a.get_clip_on(), a.get_zorder())
        if isinstance(a, Line2D):
            out.append((f"{where} {kind}", (_arr(a.get_xydata()), a.get_marker(), a.get_linestyle(),
                                            a.get_markersize(), a.get_linewidth(), a.get_drawstyle(), common)))
        elif isinstance(a, (QuadMesh, AxesImage)) or (isinstance(a, Collection) and a.get_array() is not None):
            n = a.norm
            out.append((f"{where} {kind}", (_arr(a.get_array()), type(n).__name__, n.vmin, n.vmax, common,
                                            tuple(a.get_extent()) if isinstance(a, AxesImage) else
                                            _arr(a.get_coordinates()) if isinstance(a, QuadMesh) else
                                            _arr(a.get_offsets()))))
        elif isinstance(a, LineCollection):
            out.append((f"{where} {kind}", ([_arr(s) for s in a.get_segments()], _arr(a.get_linewidths()),
                                            str(a.get_linestyles()), common)))
        elif isinstance(a, PathCollection):
            out.append((f"{where} {kind}", (_arr(a.get_offsets()), _arr(a.get_sizes()),
                                            [_arr(p.vertices) for p in a.get_paths()], _arr(a.get_linewidths()),
                                            _where_offsets(a, ax), common)))
        elif isinstance(a, Collection):
            out.append((f"{where} {kind}", ([_arr(p.vertices) for p in a.get_paths()], _arr(a.get_linewidths()),
                                            common)))
        elif isinstance(a, Rectangle):
            out.append((f"{where} {kind}", (a.get_xy(), a.get_width(), a.get_height(), a.get_linewidth(),
                                            a.get_hatch(), common)))
        elif isinstance(a, Patch):
            p = a.get_path()
            out.append((f"{where} {kind}", (_arr(a.get_patch_transform().transform(p.vertices)), a.get_linewidth(),
                                            a.get_hatch(), common)))
        elif isinstance(a, Text):
            text(a, f"{where} text")
        else:
            raise Unsupported(f"{where}: a {kind}, which the check cannot compare")

    def words(t, what):  # a title or label: where it sits depends on the type around it, so only its words count
        if t.get_text():
            out.append((what, (t.get_text(), t.get_ha(), t.get_visible())))

    def key(h, what):  # a legend key: its mark, not where the legend's layout put it
        if isinstance(h, Line2D):
            out.append((what, (h.get_marker(), h.get_linestyle(), h.get_markersize(), h.get_linewidth())))
        elif isinstance(h, Patch):
            out.append((what, (type(h).__name__, h.get_hatch(), h.get_linewidth())))
        elif isinstance(h, Collection):
            out.append((what, (type(h).__name__, _arr(h.get_sizes()) if hasattr(h, "get_sizes") else None)))
        else:
            raise Unsupported(f"{what}: a {type(h).__name__}, which the check cannot compare")

    def legend(lg, w):
        out.append((f"{w} legend", (lg.get_visible(), len(lg.get_texts()))))
        if lg.get_title().get_text():
            words(lg.get_title(), f"{w} legend title")
        for t in lg.get_texts():
            words(t, f"{w} legend")
        for h in lg.legend_handles:
            key(h, f"{w} legend key")

    def ticks(axis):
        return (_arr(axis.get_majorticklocs()), tuple(t.get_text() for t in axis.get_majorticklabels()),
                _arr(axis.get_minorticklocs()), tuple(t.get_text() for t in axis.get_minorticklabels()))

    for i, ax in enumerate(fig.axes):
        w = f"axes {i}"
        out.append((f"{w} limits", (ax.get_xlim(), ax.get_ylim(), ax.get_xscale(), ax.get_yscale(),
                                    ax.xaxis_inverted(), ax.yaxis_inverted(), ax.get_visible(), ax.axison)))
        out.append((f"{w} x ticks", ticks(ax.xaxis)))
        out.append((f"{w} y ticks", ticks(ax.yaxis)))
        for t, what in ((ax.title, "title"), (ax._left_title, "left title"), (ax._right_title, "right title"),
                        (ax.xaxis.label, "x label"), (ax.yaxis.label, "y label")):
            words(t, f"{w} {what}")
        for a in [*ax.lines, *ax.collections, *ax.patches, *ax.images, *ax.texts, *ax.artists]:
            artist(a, w, ax)
        if ax.get_legend():
            legend(ax.get_legend(), w)
    for t in fig.texts:
        text(t, "figure text")
    for a in [*fig.lines, *fig.patches, *fig.images, *fig.artists]:
        artist(a, "figure", None)
    for lg in fig.legends:
        legend(lg, "figure")
    return out


def _where_offsets(a, ax):
    """The coordinates a scatter's points are placed in."""
    t = a.get_offset_transform()
    if ax is None:
        return "figure"
    return "data" if (t == ax.transData or t.contains_branch(ax.transData)) else type(t).__name__


def _same(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and (a.dtype == bool or b.dtype == bool):
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def compare(plain, site):
    """None when the two lists match item for item; otherwise the first difference, in words."""
    if len(plain) != len(site):
        return f"{len(plain)} plotted items in the study's figure, {len(site)} in the book's"
    for (w1, v1), (w2, v2) in zip(plain, site):
        if w1 != w2 or not _same(v1, v2):
            return f"{w1} differs"
    return None


def fingerprint(items):
    h = hashlib.sha256()

    def feed(v):
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(np.asarray(v, float)).tobytes())
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            for x in v:
                feed(x)
            h.update(b"]")
        else:
            h.update(repr(v).encode())

    for w, v in items:
        h.update(w.encode())
        feed(v)
    return h.hexdigest()[:16]


# ---- recoloring -------------------------------------------------------------------------------------------------

def _key(c):
    return mcolors.to_hex(mcolors.to_rgba(c), keep_alpha=False).lower()


class Inks:
    """Maps each color a study function uses to the book's ink for one theme, keeping the color's own opacity."""

    def __init__(self, table, theme, extra=None):
        self.table = {_key(k): v for k, v in table.items()}
        self.theme, self.extra, self.used = theme, extra, {}

    def __call__(self, c, where="", force_alpha=None):
        rgba = mcolors.to_rgba(c)
        if rgba[3] == 0:
            return rgba
        k = _key(rgba)
        role = self.table.get(k)
        if role is None and self.extra:
            role = self.extra(rgba)
        if role is None:
            raise KeyError(f"no book ink for {k} ({where}); add it to the figure's ink list")
        if isinstance(role, dict):  # {"light": color, "dark": color}
            role = role[self.theme]
        new = mcolors.to_rgba(INKS[self.theme][role] if role in INKS[self.theme] else role)
        a = new[3] * rgba[3] if force_alpha is None else force_alpha
        self.used.setdefault(k, set()).add(mcolors.to_hex(new[:3]))
        return (*new[:3], a)

    def many(self, arr, where=""):
        arr = np.asarray(arr)
        if arr.size == 0:
            return arr
        return np.array([self(c, where) for c in arr])


def restyle(fig, inks, theme, boxed=()):
    """Recolor one figure for a theme: no background, a light frame without a box, the book's type, and every
    color through the ink table. Changes only how things are drawn, never where or what."""
    T = INKS[theme]
    fig.canvas.draw()  # so every tick and its label exists
    fig.patch.set_alpha(0)
    skip = {id(fig.patch)}
    mapped = []  # colormapped artists: the figure's own `after` gives each a book colormap
    for i, ax in enumerate(fig.axes):
        skip.add(id(ax.patch))
        ax.patch.set_alpha(0)
        for side, sp in ax.spines.items():
            skip.add(id(sp))
            if i in boxed:
                sp.set_color(T["ink2"]); sp.set_linewidth(0.8)
            elif side in ("top", "right"):
                sp.set_visible(False)
            else:
                sp.set_color(T["ink2"]); sp.set_linewidth(0.8)
        for axis in (ax.xaxis, ax.yaxis):
            skip.update(id(o) for o in axis.findobj())
            axis.label.set_color(inks(axis.label.get_color(), "axis label"))
            axis.label.set_fontfamily([SERIF, "CMU Serif", "DejaVu Serif"])
        ax.tick_params(which="both", color=T["ink2"], labelcolor=T["ink2"], grid_color=T["ink"],
                       grid_alpha=GRID_ALPHA[theme])
        if ax.get_legend():
            skip.add(id(ax.get_legend().get_frame()))
            ax.get_legend().get_frame().set_visible(False)
    for lg in fig.legends:
        skip.add(id(lg.get_frame()))
        lg.get_frame().set_visible(False)
    for a in fig.findobj():
        if id(a) in skip:
            continue
        skip.add(id(a))  # an artist can be reached twice (a legend key, say); recolor it once
        where = f"{type(a).__name__} {getattr(a, 'get_label', lambda: '')()}"
        if isinstance(a, Line2D):  # read every color first: a marker's "auto" colors follow the line's
            c, fc, ec, alt = (a.get_color(), a.get_markerfacecolor(), a.get_markeredgecolor(),
                              a.get_markerfacecoloralt())
            line = a.get_linestyle() not in ("None", "none", "", " ") and a.get_linewidth() > 0
            mark = a.get_marker() not in (None, "None", "none", "", " ")
            if line:  # a color that is never drawn keeps its value
                a.set_color(inks(c, where))
            if mark:
                a.set_markerfacecolor(inks(fc, where))
                a.set_markeredgecolor(inks(ec, where))
                if alt not in (None, "none"):
                    a.set_markerfacecoloralt(inks(alt, where))
        elif isinstance(a, (AxesImage, QuadMesh)) or (isinstance(a, Collection) and a.get_array() is not None):
            mapped.append(a)
            if isinstance(a, Collection) and not isinstance(a, QuadMesh):
                ec = a.get_edgecolor()
                if len(ec):
                    a.set_edgecolor(inks.many(ec, where))
        elif isinstance(a, Collection):
            fc, ec = a.get_facecolor(), a.get_edgecolor()
            if len(fc):
                a.set_facecolor(inks.many(fc, where))
            if len(ec):
                a.set_edgecolor(inks.many(ec, where))
        elif isinstance(a, Patch):
            a.set_facecolor(inks(a.get_facecolor(), where))
            a.set_edgecolor(inks(a.get_edgecolor(), where))
        elif isinstance(a, Text):
            a.set_color(inks(a.get_color(), where))
            if a.get_fontfamily() and a.get_fontfamily()[0] != SERIF:
                a.set_fontfamily([SERIF, "CMU Serif", "DejaVu Serif"])
    for ax in fig.axes:  # an annotation's arrow is not among the figure's children
        for t in ax.texts:
            arrow = getattr(t, "arrow_patch", None)
            if arrow is not None and id(arrow) not in skip:
                skip.add(id(arrow))
                fc, ec = arrow.get_facecolor(), arrow.get_edgecolor()
                arrow.set_facecolor(inks(fc, "annotation arrow"))
                arrow.set_edgecolor(inks(ec, "annotation arrow"))
    return mapped


def distinct(inks):
    """Colors the study kept apart that the book's inks would merge, if any."""
    seen = {}
    for old, new in inks.used.items():
        for n in new:
            seen.setdefault(n, set()).add(old)
    return {n: sorted(o) for n, o in seen.items() if len(o) > 1}


# ---- the committed figure ---------------------------------------------------------------------------------------

def pixels_against(fig, kw, committed):
    """How many pixels of the plain run differ from the committed PNG in any channel, at all: (differing, total,
    same size)."""
    from PIL import Image
    buf = io.BytesIO()
    k = dict(kw)
    k.pop("format", None)
    fig.savefig(buf, format="png", **k)
    a = np.asarray(Image.open(buf).convert("RGBA"), dtype=np.int16)
    b = np.asarray(Image.open(committed).convert("RGBA"), dtype=np.int16)
    if a.shape != b.shape:
        return None, int(b.shape[0] * b.shape[1]), False
    diff = (a != b).any(axis=2)
    return int(diff.sum()), int(diff.size), True


# ---- writing ----------------------------------------------------------------------------------------------------

def encode(fig, kw):
    """The bytes of a figure's file and their extension: SVG, or WebP when the SVG would be too large."""
    pad = kw.get("pad_inches", 0.1)
    buf = io.BytesIO()
    with plt.rc_context({"svg.hashsalt": "course-book", "svg.fonttype": "path"}):  # the same bytes every run
        fig.savefig(buf, format="svg", transparent=True, bbox_inches="tight", pad_inches=pad,
                    metadata={"Date": None, "Creator": None})
    data = buf.getvalue()
    if len(data) <= SVG_LIMIT:
        return data, ".svg"
    from PIL import Image
    buf = io.BytesIO()
    w_in = fig.get_size_inches()[0]
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=pad,
                dpi=RASTER_WIDTH / w_in)
    out = io.BytesIO()
    Image.open(buf).save(out, "WEBP", quality=90, method=6)
    return out.getvalue(), ".webp"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(project, figures, check=False):
    """Draw and check every figure in `figures` for one study. The files and the manifest are written only when
    every check passes (the manifest then says so); a failed run leaves the last good files as they were. With
    check=True, it writes nothing and fails when a file on disk differs from what a fresh render writes."""
    load_fonts()
    out_dir = project / "figures/site"
    manifest, failed, files = [], [], {}
    for spec in figures:
        name = spec["name"]
        fails_before = len(failed)
        fig, kw = draw(spec)
        diff, total, same_size = pixels_against(fig, kw, project / spec["committed"])
        if isinstance(kw.get("dpi"), (int, float)):
            fig.set_dpi(kw["dpi"])  # lay it out at the resolution the study saved it at, as the committed PNG is
        try:
            plain = plotted(fig)
        except Unsupported as e:
            failed.append(f"{name}: {e}"); plt.close(fig); continue
        ticks = [(ax.xaxis.get_majorticklocs(), ax.yaxis.get_majorticklocs()) for ax in fig.axes]
        plt.close(fig)
        entry = {"name": name, "source": spec["source"], "data": {d: sha(project / d) for d in spec["data"]},
                 "committed": spec["committed"], "committed_sha256": sha(project / spec["committed"]),
                 "plain_run_vs_committed": {"same_size": same_size, "pixels_differing": diff, "pixels": total},
                 "plotted_items": len(plain), "plotted_fingerprint": fingerprint(plain), "files": {}}
        if not (same_size and diff == 0):
            failed.append(f"{name}: the plain run does not match {spec['committed']} "
                          f"({diff if diff is not None else 'size differs'} of {total} pixels differ)")
        for theme in ("light", "dark"):
            inks = Inks(spec["inks"], theme, spec.get("extra"))
            fig, kw = draw(spec, rc=site_rc())
            if spec.get("before"):
                spec["before"](fig, theme)
            mapped = restyle(fig, inks, theme, boxed=spec.get("boxed", ()))
            if spec.get("after"):
                spec["after"](fig, theme)
            for a in mapped:  # no study figure here maps colors through a colormap; one that does needs its own check
                failed.append(f"{name} ({theme}): a colormapped {type(a).__name__}, which this engine does not restyle")
            # the study's ticks: the book's type and the SVG's 72-dpi layout could otherwise change how many
            # automatic ticks an axis gets
            for ax, (xt, yt) in zip(fig.axes, ticks):
                for axis, t in ((ax.xaxis, xt), (ax.yaxis, yt)):
                    if not isinstance(axis.get_major_locator(), FixedLocator):
                        axis.set_major_locator(FixedLocator(t))
            try:
                problem = compare(plain, plotted(fig))
                data, ext = encode(fig, kw)
                saved = compare(plain, plotted(fig, redraw=False))  # the figure as the file was drawn
            except Unsupported as e:
                problem, saved, data, ext = str(e), None, b"", ".svg"
            if problem or saved:
                failed.append(f"{name} ({theme}): {problem or 'as saved, ' + saved}")
            allowed = [{_key(c) for c in group} for group in spec.get("merge_ok", [])]
            merged = {n: o for n, o in distinct(inks).items() if not any(set(o) <= g for g in allowed)}
            if merged:
                failed.append(f"{name} ({theme}): colors the study keeps apart share an ink: {merged}")
            plt.close(fig)
            fname = f"{name}.{theme}{ext}"
            files[fname] = data
            entry["files"][theme] = {"file": fname, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                     "plotted_items_match": not (problem or saved)}
        entry["passed"] = len(failed) == fails_before
        manifest.append(entry)
        print(f"{'ok  ' if entry['passed'] else 'FAIL'} {name}: {entry['plotted_items']} plotted items; plain run "
              f"vs committed PNG: {diff} of {total} pixels differ; files {entry['files'].get('light', {}).get('file')}, "
              f"{entry['files'].get('dark', {}).get('file')}")
    doc = {"about": "The course book's copies of this study's figures. Each is drawn by the study's own function "
                    "from the saved data listed, in the book's inks; the checks are described in "
                    "tools/site_figures.py.",
           "passed": not failed, "figures": manifest}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    mpath = out_dir / "manifest.json"
    if check:
        for fname, data in files.items():
            if not (out_dir / fname).exists() or (out_dir / fname).read_bytes() != data:
                failed.append(f"{fname} is not what a fresh render writes")
        if not mpath.exists() or mpath.read_text() != text:
            failed.append("manifest.json is not what a fresh render writes")
    elif not failed:
        out_dir.mkdir(parents=True, exist_ok=True)
        for spec in figures:
            for ext in (".svg", ".webp"):
                for theme in ("light", "dark"):
                    f = out_dir / f"{spec['name']}.{theme}{ext}"
                    if f.name not in files:
                        f.unlink(missing_ok=True)
        for fname, data in files.items():
            (out_dir / fname).write_bytes(data)
        mpath.write_text(text)
    for f in failed:
        print("FAIL", f, file=sys.stderr)
    if failed and not check:
        print("nothing written: a check failed", file=sys.stderr)
    return not failed
