// The book's page script: the day and night printing, the page turn's direction, and lifting a plate off the
// page to look closer (pinch, scroll, or drag to zoom and pan; Escape or the close button to put it back).
(function () {
  var root = document.documentElement;
  // The page turn (the theme's stylesheet) runs forward, or back when the page you came from comes later in the
  // book. A page's place is its folder's place in the book's order (the title page first); a page outside the book
  // has none, and turns forward.
  var TURN = "course-book-from";
  var place = function (url) {
    var link = document.querySelector('link[rel="index"]'), meta = document.querySelector('meta[name="book-order"]');
    if (!link || !meta || !url) return -1;
    var base = new URL(link.href).pathname.replace(/index\.html$/, ""), path = new URL(url, location.href).pathname;
    if (path.indexOf(base) !== 0) return -1;
    var rest = path.slice(base.length).replace(/index\.html$/, "").replace(/\/$/, "");
    return rest === "" ? 0 : meta.content.split(" ").indexOf(rest);
  };
  window.addEventListener("pagereveal", function (e) {
    if (!e.viewTransition || !e.viewTransition.types) return;
    var from = null;
    try { from = navigation.activation.from.url; } catch (x) {}
    if (!from) { try { from = sessionStorage.getItem(TURN); } catch (x) {} }
    var a = place(from), b = place(location.href);
    if (a >= 0 && b >= 0 && b < a) e.viewTransition.types.add("back");
  });
  var leaving = function () { try { sessionStorage.setItem(TURN, location.href); } catch (x) {} };
  window.addEventListener("pageswap", leaving);
  window.addEventListener("pagehide", leaving);
  var KEY = "course-book-theme";
  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function store(v) { try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch (e) {} }
  function apply(v) { if (v) root.setAttribute("data-theme", v); else root.removeAttribute("data-theme"); }
  apply(stored());
  document.addEventListener("DOMContentLoaded", function () {
    var mode = document.querySelector(".mode");
    if (mode) {
      var label = function () {
        var v = stored() || "auto";
        mode.textContent = v === "dark" ? "☾" : v === "light" ? "☀" : "◐";
        mode.setAttribute("aria-label", "Printing: " + (v === "auto" ? "follows the system" : v === "dark" ? "night" : "day") + ". Change it.");
      };
      label();
      mode.addEventListener("click", function () {
        var v = stored() || "auto";
        var next = v === "auto" ? "light" : v === "light" ? "dark" : null;
        store(next); apply(next); label();
      });
    }

    // Lifting a plate
    var box = document.querySelector(".lifted");
    if (!box) return;
    var sheet = box.querySelector(".sheet"), img = sheet.querySelector("img"), title = box.querySelector(".title");
    var foot = box.querySelector(".lift-cap"), close = box.querySelector(".close"), opener = null;
    var s = 1, x = 0, y = 0, pts = new Map(), last = null;
    function set() { sheet.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")"; }
    function reset() { s = 1; x = 0; y = 0; set(); }
    function open(fig) {
      // the print the page shows: the day or the night copy, whichever the printing displays
      var imgs = Array.prototype.slice.call(fig.querySelectorAll("img"));
      var src = imgs.filter(function (i) { return getComputedStyle(i).display !== "none"; })[0] || imgs[0];
      img.src = src.currentSrc || src.src; img.alt = src.alt;
      title.textContent = fig.querySelector("figcaption b").textContent;
      foot.textContent = fig.querySelector("figcaption .cap").textContent;
      opener = fig.querySelector("button.lift");
      box.hidden = false; reset(); close.focus();
      document.body.style.overflow = "hidden";
    }
    function shut() { box.hidden = true; document.body.style.overflow = ""; if (opener) opener.focus(); }
    document.querySelectorAll(".plate > button.lift").forEach(function (b) {
      b.addEventListener("click", function () { open(b.closest(".plate")); });
    });
    close.addEventListener("click", shut);
    box.addEventListener("keydown", function (e) { if (e.key === "Escape") shut(); if (e.key === "Tab") { e.preventDefault(); close.focus(); } });
    box.querySelector(".stage").addEventListener("click", function (e) { if (e.target === e.currentTarget && s === 1) shut(); });
    box.addEventListener("wheel", function (e) {
      e.preventDefault();
      var k = Math.exp(-e.deltaY * 0.0015), r = sheet.getBoundingClientRect();
      var ns = Math.min(6, Math.max(1, s * k)), f = ns / s;
      x = e.clientX - (e.clientX - r.left) * f - (r.left - x); y = e.clientY - (e.clientY - r.top) * f - (r.top - y); s = ns;
      if (s === 1) { x = 0; y = 0; }
      set();
    }, { passive: false });
    sheet.addEventListener("pointerdown", function (e) { sheet.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); last = null; });
    sheet.addEventListener("pointermove", function (e) {
      if (!pts.has(e.pointerId)) return;
      var prev = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 1) { if (s > 1) { x += e.clientX - prev[0]; y += e.clientY - prev[1]; set(); } return; }
      var p = Array.from(pts.values()), d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
      var cx = (p[0][0] + p[1][0]) / 2, cy = (p[0][1] + p[1][1]) / 2;
      if (last) {
        var ns = Math.min(6, Math.max(1, s * d / last.d)), f = ns / s, r = sheet.getBoundingClientRect();
        x = cx - (cx - r.left) * f - (r.left - x) + (cx - last.cx); y = cy - (cy - r.top) * f - (r.top - y) + (cy - last.cy); s = ns;
        if (s === 1) { x = 0; y = 0; }
        set();
      }
      last = { d: d, cx: cx, cy: cy };
    });
    var up = function (e) { pts.delete(e.pointerId); last = null; };
    sheet.addEventListener("pointerup", up); sheet.addEventListener("pointercancel", up);
    sheet.addEventListener("dblclick", reset);
  });
})();
