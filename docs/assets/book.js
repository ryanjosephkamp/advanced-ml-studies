// The book's page script: the day and night printing, the page turn's direction, the thumb index's flags (and,
// on a touch screen, scrubbing through them), and lifting a plate off the page to look closer (pinch, scroll, or
// drag to zoom and pan; Escape or the close button to put it back).
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
  // a classic scrollbar's width (overlay scrollbars take none), so the theme keeps plates and the wide track clear
  // of the thumb index measured from the page's real width, not 100vw
  var bar = function () { root.style.setProperty("--sb", Math.max(0, innerWidth - root.clientWidth) + "px"); };
  document.addEventListener("DOMContentLoaded", bar);
  window.addEventListener("resize", bar);
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

    // The thumb index: a flag names the chapter under the pointer or the keyboard's focus
    var edge = document.querySelector("nav.edge");
    if (edge) {
      var flag = document.createElement("span");
      flag.className = "edge-flag"; flag.setAttribute("aria-hidden", "true"); flag.hidden = true;
      edge.appendChild(flag);
      // the flag's words: the chapter's numeral in the tab's own ink (whichever ink the printing paints the tab in),
      // then its short name
      var word = function (a) {
        var parts = (a.dataset.flag || "").split(" · ");
        flag.textContent = "";
        flag.style.setProperty("--flag-ink", getComputedStyle(a).getPropertyValue("--tab").trim() || "currentColor");
        if (parts.length > 1) {
          var no = document.createElement("span");
          no.className = "flag-no"; no.textContent = parts[0];
          var name = document.createElement("span");
          name.className = "flag-name"; name.textContent = parts.slice(1).join(" · ");
          flag.appendChild(no); flag.appendChild(name);
        } else flag.textContent = parts[0];
      };
      var show = function (a) {
        if (!a || !a.dataset.flag) return;
        var r = a.getBoundingClientRect();
        word(a); flag.hidden = false;
        flag.style.top = (r.top + r.height / 2) + "px";
      };
      // the pointer's tab wins while it is there; when it leaves, the focused tab's flag comes back. The pointer can
      // rest on the flag itself, and Escape puts the flag away until the next tab. A thumb's scrub owns the flag.
      var hovered = null, focused = null, dismissed = null, scrub = null;
      var update = function () {
        if (scrub) return;
        var a = hovered || focused;
        if (a && a !== dismissed) show(a); else flag.hidden = true;
      };
      var aim = function (a) { if (a !== dismissed) dismissed = null; return a; };
      edge.addEventListener("mouseover", function (e) { if (flag.contains(e.target)) return; hovered = aim(e.target.closest("a")); update(); });
      edge.addEventListener("mouseleave", function () { hovered = null; update(); });
      // a new keyboard focus wins over a pointer parked on another tab
      edge.addEventListener("focusin", function (e) { focused = aim(e.target.closest("a")); hovered = null; update(); });
      edge.addEventListener("focusout", function () { focused = null; update(); });

      // A thumb on the edge: press a tab to see its name beside the thumb, slide along the edge to run through the
      // tabs, and let go to turn to the one named. Slide away from the edge first to stay on this page. The strip's
      // empty rail is not a target (theme.css), so a press there does nothing.
      var tabs = Array.prototype.slice.call(edge.querySelectorAll("a")), swallowed = 0;
      // touch laptops count: the strip is a touch strip wherever a finger can reach it (theme.css asks the same)
      var touchy = function () { return matchMedia("(any-pointer: coarse)").matches; };
      var under = function (y) {
        for (var i = 0; i < tabs.length; i++) if (y < tabs[i].getBoundingClientRect().bottom) return tabs[i];
        return tabs[tabs.length - 1];
      };
      var pick = function (a, y) {
        if (a !== scrub.on) {
          // a tick under the thumb at each new tab, once the page has had a tap (before that the browser refuses it)
          var ready = navigator.vibrate && (!navigator.userActivation || navigator.userActivation.hasBeenActive);
          if (scrub.on) { scrub.on.classList.remove("is-on"); if (ready) { try { navigator.vibrate(3); } catch (x) {} } }
          scrub.on = a; a.classList.add("is-on");
        }
        if (scrub.off) { flag.textContent = "Let go to stay here"; flag.style.removeProperty("--flag-ink"); } else word(a);
        flag.hidden = false;
        flag.style.top = Math.max(84, Math.min(innerHeight - 40, y - 56)) + "px";
      };
      var end = function (go) {
        if (!scrub) return;
        var a = scrub.on, off = scrub.off;
        scrub = null;
        // the browser may follow a handled gesture with a click on whatever link is under the finger: that one click
        // is the gesture's, so it is dropped (a mouse or keyboard click later is not)
        swallowed = Date.now();
        if (a) a.classList.remove("is-on");
        flag.classList.remove("is-scrub", "is-off");
        flag.hidden = true;
        if (go && a && !off && a.getAttribute("aria-current") !== "page") location.href = a.href;
        else update();
      };
      var mine = function (e) { return scrub && e.pointerId === scrub.id; };
      edge.addEventListener("pointerdown", function (e) {
        // a finger or pen on a tab, where a finger can reach the strip (it doesn't scroll the page there); a mouse,
        // or any pointer elsewhere, gets a plain link
        var a = e.target.closest && e.target.closest("a");
        if (!a || e.pointerType === "mouse" || !e.isPrimary || scrub || !touchy()) return;
        e.preventDefault();
        // capture on the tab, so the rest of the gesture comes here wherever the thumb goes; if the browser
        // refuses, the window's listeners below still see the release
        try { a.setPointerCapture(e.pointerId); } catch (x) {}
        scrub = { id: e.pointerId, on: null, off: false };
        flag.classList.add("is-scrub");
        pick(under(e.clientY), e.clientY);
      });
      var track = function (e) {
        // "away" is measured from the strip's own right edge (a classic scrollbar sits beyond it)
        scrub.off = e.clientX < edge.getBoundingClientRect().right - 120;
        flag.classList.toggle("is-off", scrub.off);
        pick(under(e.clientY), e.clientY);
      };
      window.addEventListener("pointermove", function (e) { if (mine(e)) track(e); }, true);
      // the thumb's own release decides: where it lifts, not where it last moved
      window.addEventListener("pointerup", function (e) { if (!mine(e)) return; track(e); end(true); }, true);
      window.addEventListener("pointercancel", function (e) { if (mine(e)) end(false); }, true);
      // capture lost before the release (the release ends the scrub first, so this is only an interruption)
      edge.addEventListener("lostpointercapture", function (e) { if (mine(e)) end(false); });
      window.addEventListener("blur", function () { end(false); });
      // Escape puts the flag away: it cancels a scrub, or hides a hover or focus flag until another tab
      document.addEventListener("keydown", function (e) {
        if (e.key !== "Escape") return;
        if (scrub) end(false);
        else if (!flag.hidden) { dismissed = hovered || focused; flag.hidden = true; }
      });
      // a long press on a tab is the scrub's, not the browser's link menu; a mouse or keyboard menu still opens
      edge.addEventListener("contextmenu", function (e) { if (scrub || Date.now() - swallowed < 800) e.preventDefault(); });
      edge.addEventListener("click", function (e) { if (Date.now() - swallowed < 800) { swallowed = 0; e.preventDefault(); } }, true);
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
