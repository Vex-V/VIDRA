// VIDRA docs: theme, menu, highlighting, copy buttons, the page's contents in
// the sidebar, and search. Everything works from file://, so the search index
// is a script (search-index.js) rather than a JSON fetch.

(function () {
  "use strict";

  // ------------------------------------------------------------ theme
  function stored(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private window */ }
  }
  var theme = stored("vidra-docs-theme");
  if (theme === "light" || theme === "dark") {
    document.documentElement.setAttribute("data-theme", theme);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var toggle = document.getElementById("theme");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var dark = document.documentElement.getAttribute("data-theme") !== "light";
        var next = dark ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        store("vidra-docs-theme", next);
      });
    }

    var menu = document.getElementById("menu");
    if (menu) {
      menu.addEventListener("click", function () {
        document.body.classList.toggle("nav-open");
      });
    }
    document.querySelectorAll(".side a").forEach(function (a) {
      a.addEventListener("click", function () { document.body.classList.remove("nav-open"); });
    });

    highlight();
    copyButtons();
    anchors();
    contents();
    search();
  });

  // ------------------------------------------------------ highlighting
  var KEYWORDS = new Set(("False None True and as assert async await break class continue " +
    "def del elif else except finally for from global if import in is lambda nonlocal not " +
    "or pass raise return try while with yield").split(" "));

  function escape(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function python(source) {
    var pattern = /("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfRBF]{0,2}"(?:[^"\\\n]|\\.)*"|[rbfRBF]{0,2}'(?:[^'\\\n]|\\.)*'|#[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b)/g;
    var out = "", last = 0, match;
    while ((match = pattern.exec(source)) !== null) {
      out += escape(source.slice(last, match.index));
      var token = match[0], cls = null;
      if (token[0] === "#") cls = "c";
      else if (/^[rbfRBF]{0,2}["']/.test(token)) cls = "s";
      else if (/^\d/.test(token)) cls = "n";
      else if (KEYWORDS.has(token)) cls = "k";
      else if (source[pattern.lastIndex] === "(") cls = "f";
      out += cls ? '<span class="' + cls + '">' + escape(token) + "</span>" : escape(token);
      last = pattern.lastIndex;
    }
    return out + escape(source.slice(last));
  }

  function highlight() {
    document.querySelectorAll("pre code.python").forEach(function (code) {
      code.innerHTML = python(code.textContent);
    });
  }

  // ------------------------------------------------------------- copying
  function copyButtons() {
    document.querySelectorAll("pre").forEach(function (pre) {
      var button = document.createElement("button");
      button.className = "copy";
      button.type = "button";
      button.textContent = "Copy";
      button.addEventListener("click", function () {
        var text = pre.querySelector("code") ? pre.querySelector("code").textContent : pre.textContent;
        var done = function () {
          button.textContent = "Copied";
          setTimeout(function () { button.textContent = "Copy"; }, 1200);
        };
        if (navigator.clipboard && window.isSecureContext) {
          navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
        } else { fallback(text); done(); }
      });
      pre.appendChild(button);
    });
  }

  function fallback(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing to do */ }
    document.body.removeChild(area);
  }

  // ----------------------------------------------------- heading anchors
  function anchors() {
    document.querySelectorAll("main h2[id], main h3[id]").forEach(function (h) {
      var a = document.createElement("a");
      a.className = "anchor";
      a.href = "#" + h.id;
      a.textContent = "#";
      a.setAttribute("aria-label", "Link to this section");
      h.appendChild(a);
    });
  }

  // -------------------------------------------- this page, in the sidebar
  function contents() {
    var here = document.querySelector(".side a.here");
    var headings = Array.prototype.slice.call(document.querySelectorAll("main h2[id]"));
    if (!here || headings.length < 2) return;
    var toc = document.createElement("div");
    toc.className = "toc";
    var links = headings.map(function (h) {
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.firstChild ? h.firstChild.textContent : h.textContent;
      toc.appendChild(a);
      return a;
    });
    here.parentNode.insertBefore(toc, here.nextSibling);

    function mark() {
      var current = 0;
      headings.forEach(function (h, i) {
        if (h.getBoundingClientRect().top < 120) current = i;
      });
      links.forEach(function (a, i) { a.classList.toggle("on", i === current); });
    }
    window.addEventListener("scroll", mark, { passive: true });
    mark();
  }

  // -------------------------------------------------------------- search
  function search() {
    var input = document.getElementById("q");
    var box = document.getElementById("results");
    var index = window.VIDRA_DOCS_INDEX || [];
    if (!input || !box) return;
    // Index urls are relative to docs/; this page says how to get there.
    var meta = document.querySelector('meta[name="docs-root"]');
    var root = meta ? meta.getAttribute("content") : "";
    var selected = -1;

    function terms(q) {
      return q.toLowerCase().split(/[^a-z0-9_.:]+/).filter(function (t) { return t.length > 1; });
    }

    function score(entry, words) {
      var title = entry.title.toLowerCase(), text = entry.text.toLowerCase(), total = 0;
      for (var i = 0; i < words.length; i++) {
        var w = words[i], inTitle = title.indexOf(w) !== -1, at = text.indexOf(w);
        if (!inTitle && at === -1) return 0;
        total += (inTitle ? 10 : 0) + (at !== -1 ? 1 + Math.min(4, text.split(w).length - 1) : 0);
      }
      return total;
    }

    function snippet(text, words) {
      var lower = text.toLowerCase(), at = lower.indexOf(words[0]);
      var start = Math.max(0, at - 50), piece = text.slice(start, start + 150);
      var html = escape(piece);
      words.forEach(function (w) {
        html = html.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
      });
      return (start > 0 ? "&hellip;" : "") + html + "&hellip;";
    }

    function render() {
      var words = terms(input.value);
      selected = -1;
      if (!words.length) { box.classList.remove("open"); box.innerHTML = ""; return; }
      var hits = index.map(function (e) { return { e: e, s: score(e, words) }; })
        .filter(function (h) { return h.s > 0; })
        .sort(function (a, b) { return b.s - a.s; })
        .slice(0, 12);
      box.innerHTML = hits.length ? hits.map(function (h) {
        return '<a href="' + root + h.e.url + '"><div class="where">' + escape(h.e.page) + "</div><b>" +
          escape(h.e.title) + '</b><div class="snip">' + snippet(h.e.text, words) + "</div></a>";
      }).join("") : '<div class="empty">Nothing matches.</div>';
      box.classList.add("open");
    }

    input.addEventListener("input", render);
    input.addEventListener("keydown", function (event) {
      var links = box.querySelectorAll("a");
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!links.length) return;
        selected = (selected + (event.key === "ArrowDown" ? 1 : -1) + links.length) % links.length;
        links.forEach(function (a, i) { a.classList.toggle("active", i === selected); });
      } else if (event.key === "Enter" && links.length) {
        window.location.href = links[Math.max(0, selected)].getAttribute("href");
      } else if (event.key === "Escape") {
        box.classList.remove("open");
        input.blur();
      }
    });
    document.addEventListener("click", function (event) {
      if (!box.contains(event.target) && event.target !== input) box.classList.remove("open");
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "/" && document.activeElement !== input) {
        event.preventDefault();
        input.focus();
      }
    });
  }
})();
