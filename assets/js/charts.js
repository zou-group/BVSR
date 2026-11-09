/* BVSR project page: figures.
   All charts are hand-built SVG/HTML so they inherit the page's CSS variables
   (and therefore the light / dark theme) with no chart library. */
(function () {
  'use strict';

  var D = window.BVSR_DATA;
  var NS = 'http://www.w3.org/2000/svg';
  var $ = function (s, r) { return (r || document).querySelector(s); };

  function S(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function H(tag, cls, html, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  }
  function lin(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) * (r1 - r0) / (d1 - d0); }; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function linePath(pts, sx, sy) {
    return pts.map(function (p, i) { return (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1); }).join('');
  }
  /* piecewise-linear interpolation; returns null outside the sampled range */
  function interp(pts, x, tx) {
    tx = tx || function (v) { return v; };
    if (x < pts[0][0] || x > pts[pts.length - 1][0]) return null;
    for (var i = 1; i < pts.length; i++) {
      if (x <= pts[i][0]) {
        var a = pts[i - 1], b = pts[i];
        var f = (tx(x) - tx(a[0])) / ((tx(b[0]) - tx(a[0])) || 1);
        return a[1] + f * (b[1] - a[1]);
      }
    }
    return pts[pts.length - 1][1];
  }
  function setFill(input) {
    var p = (input.value - input.min) / (input.max - input.min) * 100;
    input.style.setProperty('--fill', p + '%');
  }
  /* pointer x in viewBox units */
  function vbX(svg, evt) {
    var r = svg.getBoundingClientRect();
    return (evt.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
  }

  var tip = $('#tooltip');
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.classList.add('on');
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = clamp(x + 14, 8, window.innerWidth - w - 8);
    var top = y - h - 12; if (top < 8) top = y + 16;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function hideTip() { tip.classList.remove('on'); }

  /* ---------------------------------------------------------------- hero dots */
  (function hero() {
    var svg = $('#hero-dots'); if (!svg) return;
    var W = 1200, Hh = 520; svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh); svg.setAttribute('preserveAspectRatio', 'xMidYMid slice');
    var seed = 7; function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    var cols = ['var(--bv)', 'var(--sr)', 'var(--base)', 'var(--bv)', 'var(--base)'];
    for (var i = 0; i < 150; i++) {
      var x = rnd() * W, y = rnd() * Hh;
      var dx = Math.abs(x - W / 2) / (W / 2), dy = Math.abs(y - Hh / 2) / (Hh / 2);
      var edge = Math.max(dx * 1.05, dy * 0.75);           /* keep the centre (title) clear */
      if (edge < 0.62) continue;
      S('circle', { cx: x.toFixed(0), cy: y.toFixed(0), r: (2 + rnd() * 3.5).toFixed(1), fill: cols[i % cols.length], opacity: (0.1 + (edge - 0.62) * 0.9).toFixed(2) }, svg);
    }
  })();

  /* ---------------------------------------------------------------- creativity calculator */
  (function calc() {
    if (!$('#calc-p')) return;
    var ids = ['p', 'u', 'v'];
    var presets = [
      { n: 'Creative discovery', p: 0.05, u: 0.9, v: 0.15 },
      { n: 'Textbook baseline', p: 0.85, u: 0.7, v: 0.9 },
      { n: 'Rare, but known to work', p: 0.05, u: 0.8, v: 0.92 },
      { n: 'Wild idea that fails', p: 0.03, u: 0.04, v: 0.1 }
    ];
    var chipBox = $('#calc-presets');
    var chips = presets.map(function (ps) {
      var b = H('button', 'chip', ps.n, chipBox); b.type = 'button'; b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', function () {
        ids.forEach(function (k) { $('#calc-' + k).value = ps[k]; });
        update(); chips.forEach(function (c) { c.setAttribute('aria-pressed', c === b ? 'true' : 'false'); });
      });
      return b;
    });
    function update() {
      var p = +$('#calc-p').value, u = +$('#calc-u').value, v = +$('#calc-v').value;
      var o = 1 - p, n = 1 - v, c = o * u * n;
      ids.forEach(function (k) { var el = $('#calc-' + k); $('#calc-' + k + '-out').textContent = (+el.value).toFixed(2); setFill(el); });
      [['o', o], ['u', u], ['n', n], ['c', c]].forEach(function (t) {
        $('#calc-b-' + t[0]).style.width = (t[1] * 100) + '%';
        $('#calc-n-' + t[0]).textContent = t[1].toFixed(2);
      });
      var msg;
      var terms = [[o, 'the model proposes it all the time, so it is not original'], [u, 'it does not work well when tested'], [n, 'existing evidence already said it would work, so there is no surprise']];
      terms.sort(function (a, b) { return a[0] - b[0]; });
      if (c >= 0.5) msg = '<b>Creative.</b> Rarely proposed, effective when tested, and not something prior knowledge would have predicted.';
      else if (c >= 0.25) msg = '<b>Moderately creative.</b> The weakest factor: ' + terms[0][1] + '.';
      else msg = '<b>Not creative.</b> One weak factor is enough to sink the product: ' + terms[0][1] + '.';
      $('#calc-verdict').innerHTML = msg;
    }
    ids.forEach(function (k) {
      $('#calc-' + k).addEventListener('input', function () { chips.forEach(function (c) { c.setAttribute('aria-pressed', 'false'); }); update(); });
    });
    ids.forEach(function (k) { $('#calc-' + k).value = presets[0][k]; });
    chips[0].setAttribute('aria-pressed', 'true');
    update();
  })();

  /* ---------------------------------------------------------------- theory illustration: shared-signal generation */
  (function toy() {
    var svg = $('#toy-svg'); if (!svg) return;
    var K = 16, N = 112, W = 640, Hh = 262, base = 222, colW = W / K, pitch = 9.6, r = 4.1, perRow = 3;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
    var a = [], i;
    for (i = 0; i < K; i++) a.push(1 - i / (K - 1));                      /* task-association score a_i */
    var util = [0.42, 0.50, 0.38, 0.47, 0.55, 0.44, 0.62, 0.52, 0.58, 0.71, 0.49, 0.93, 0.66, 0.84, 0.57, 1.00];
    var hi = util.map(function (u) { return u >= 0.84; });
    for (i = 0; i < K; i++) {
      S('rect', { x: (i * colW + 3).toFixed(1), y: 8, width: (colW - 6).toFixed(1), height: base - 4, rx: 6, 'class': 'toy-col' + (hi[i] ? ' hi' : '') }, svg);
      if (hi[i]) S('use', { href: '#star', x: (i * colW + colW / 2 - 5).toFixed(1), y: 14, fill: 'var(--gold)' }, svg);
    }
    S('text', { x: 3, y: Hh - 5, 'class': 'ax-label' }, svg).textContent = '← most task-associated ideas';
    S('text', { x: W - 3, y: Hh - 5, 'class': 'ax-label', 'text-anchor': 'end' }, svg).textContent = 'least task-associated ideas →';
    var dots = [];
    for (i = 0; i < N; i++) dots.push(S('circle', { r: r, 'class': 'toy-dot' }, svg));

    var seed = 20261108, eps = [];
    function rnd() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; var t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
    function sample() {
      eps = [];
      for (var t = 0; t < N; t++) { var row = []; for (var j = 0; j < K; j++) row.push(-Math.log(-Math.log(rnd() || 1e-9))); eps.push(row); }
    }
    var slider = $('#toy-lambda');
    function update() {
      var lam = +slider.value, counts = new Array(K).fill(0), cls = [], t, j;
      for (t = 0; t < N; t++) {
        var best = 0, bv = -Infinity;
        for (j = 0; j < K; j++) { var s = lam * a[j] + eps[t][j]; if (s > bv) { bv = s; best = j; } }
        cls.push(best); counts[best]++;
      }
      var maxRows = Math.ceil(Math.max.apply(null, counts) / perRow);
      var vp = Math.min(pitch, (base - 14) / maxRows);
      var seen = new Array(K).fill(0);
      for (t = 0; t < N; t++) {
        j = cls[t]; var k = seen[j]++;
        var x = j * colW + colW / 2 + ((k % perRow) - 1) * pitch;
        var y = base - r - 2 - Math.floor(k / perRow) * vp;
        dots[t].style.transform = 'translate(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px)';
        dots[t].setAttribute('class', 'toy-dot' + (hi[j] ? ' star' : ''));
      }
      var orig = 0, distinct = 0, bestU = 0;
      for (j = 0; j < K; j++) if (counts[j]) { distinct++; orig += counts[j] * (1 - Math.log(counts[j]) / Math.log(N)); bestU = Math.max(bestU, util[j]); }
      $('#toy-orig').textContent = (orig / N).toFixed(2);
      $('#toy-distinct').innerHTML = distinct + ' <small>of ' + K + '</small>';
      $('#toy-best').textContent = bestU.toFixed(2);
      $('#toy-out').textContent = 'λ = ' + lam.toFixed(1);
      setFill(slider);
    }
    slider.addEventListener('input', update);
    $('#toy-resample').addEventListener('click', function () { sample(); update(); });
    sample(); update();
  })();

  /* ---------------------------------------------------------------- tasks grid */
  (function tasks() {
    var box = $('#tasks'); if (!box) return;
    var doms = [
      ['Biology', [['Spatially variable genes', 'rank genes by spatial structure in a tissue section'], ['Label projection', 'transfer cell-type labels to new nuclei'], ['Perturbation prediction', 'predict gene-expression response to drugs'], ['Cell–cell communication', 'infer which cell types interact']]],
      ['Machine learning', [['Anomaly detection', 'unsupervised, four tabular datasets'], ['Probability calibration', 'post-hoc, four fixed classifiers'], ['Scaling law · LR and batch size', 'predict final loss across training setups'], ['Scaling law · U-shape', 'model the double-descent regime']]],
      ['Astrodynamics', [['Cassini', 'Earth-to-Saturn gravity-assist trajectory'], ['Voyager 2', 'Earth-to-Neptune gravity-assist trajectory']]],
      ['Neuroscience', [['ZAPBench · horizon 16', 'forecast whole-brain activity of 71,721 zebrafish neurons'], ['ZAPBench · horizon 32', 'the same, twice as far ahead']]]
    ];
    doms.forEach(function (d) {
      var el = H('div', 'dom', '<h4>' + d[0] + '<span>' + d[1].length + '</span></h4>', box);
      d[1].forEach(function (t) { H('div', 'task', '<b>' + t[0] + '</b><span>' + t[1] + '</span>', el); });
    });
  })();

  /* ---------------------------------------------------------------- Fig. 2: information dial */
  (function dial() {
    var box = $('#dial-panels'); if (!box) return;
    var LS = 1.2222;                                           /* width of the linear part, in decades */
    function T(b) { return Math.abs(b) <= 20 ? b / 20 * LS : (b < 0 ? -1 : 1) * (LS + Math.log10(Math.abs(b) / 20)); }
    function Tinv(t) { return t <= LS ? t / LS * 20 : 20 * Math.pow(10, t - LS); }
    var BMAX = 143.08, W = 380, Hh = 196, m = { l: 34, r: 8, t: 8, b: 24 };
    var sx = lin(T(-3), T(178), m.l, W - m.r);
    var cfg = [
      { k: 'creativity', name: 'Top creativity', desc: 'how often proposals reach the top 1% by creativity', dom: [-0.7, 10.4], fmt: function (v) { return v.toFixed(1) + '×'; }, tick: function (v) { return v + '×'; } },
      { k: 'utility', name: 'Top utility', desc: 'how often proposals reach the top 1% by utility', dom: [-0.9, 13.9], fmt: function (v) { return v.toFixed(1) + '×'; }, tick: function (v) { return v + '×'; } },
      { k: 'originality', name: 'Originality', desc: 'mean over all proposals', dom: [0.33, 1.06], fmt: function (v) { return v.toFixed(2); }, tick: function (v) { return v.toFixed(1); } },
      { k: 'nonobviousness', name: 'Non-obviousness', desc: 'mean over all proposals', dom: [-0.08, 1.04], fmt: function (v) { return v.toFixed(2); }, tick: function (v) { return v.toFixed(1); } }
    ];
    var panels = cfg.map(function (c, idx) {
      var d = D.main[c.k];
      var el = H('div', 'panel', '<div class="p-top"><div><div class="p-name">' + c.name + '</div><div class="p-desc">' + c.desc + '</div></div><div class="p-val"><b></b><span class="delta"></span></div></div>', box);
      var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, role: 'img', 'aria-label': c.name + ' versus task information' }, el);
      var sy = lin(c.dom[0], c.dom[1], Hh - m.b, m.t);
      d.yticks.forEach(function (v) {
        S('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), 'class': 'gridline' }, svg);
        var g = S('g', { 'class': 'ax' }, svg);
        S('text', { x: m.l - 6, y: sy(v) + 3.5, 'text-anchor': 'end' }, g).textContent = c.tick(v);
      });
      var gx = S('g', { 'class': 'ax' }, svg);
      S('line', { x1: m.l, x2: W - m.r, y1: Hh - m.b, y2: Hh - m.b }, gx);
      [0, 5, 10, 20, 100].forEach(function (v) {
        S('line', { x1: sx(T(v)), x2: sx(T(v)), y1: Hh - m.b, y2: Hh - m.b + 4 }, gx);
        S('text', { x: sx(T(v)), y: Hh - m.b + 15, 'text-anchor': 'middle' }, gx).textContent = v;
      });
      var id = 'dclip' + idx, clip = S('clipPath', { id: id }, S('defs', {}, svg));
      var clipRect = S('rect', { x: 0, y: 0, width: 0, height: Hh }, clip);
      var sorted = d.dots.slice().sort(function (p, q) { return p[0] - q[0]; });
      var gd = S('g', {}, svg);
      var circles = sorted.map(function (p) {
        return S('circle', { cx: sx(T(p[0])).toFixed(1), cy: sy(clamp(p[1], c.dom[0], c.dom[1])).toFixed(1), r: 2.2, 'class': 'd-dot' }, gd);
      });
      var trendD = linePath(d.trend.map(function (p) { return [T(p[0]), p[1]]; }), sx, sy);
      S('path', { d: trendD, 'class': 'd-trend ghost' }, svg);
      var rev = S('g', { 'clip-path': 'url(#' + id + ')' }, svg);
      S('path', { d: linePath(d.band.map(function (p) { return [T(p[0]), p[1]]; }), sx, sy) + 'Z', 'class': 'd-band', opacity: 0.85 }, rev);
      S('path', { d: trendD, 'class': 'd-trend' }, rev);
      var cursor = S('line', { y1: m.t, y2: Hh - m.b, 'class': 'd-cursor' }, svg);
      var knob = S('circle', { r: 4.6, 'class': 'd-knob' }, svg);
      var v0 = d.trend[0][1];
      return { c: c, d: d, el: el, svg: svg, sy: sy, xs: sorted.map(function (p) { return p[0]; }), circles: circles, clipRect: clipRect, cursor: cursor, knob: knob, v0: v0, n: 0,
        val: el.querySelector('.p-val b'), delta: el.querySelector('.delta') };
    });

    var tiers = [
      { max: 5.6, label: 'no task context', ctx: '' },
      { max: 16.7, label: 'example context · 11.2 bits', ctx: 'Context: the samples sit on a 2-D lattice, and the goal is to rank features by how much of their variation is organized across the lattice rather than noise.' },
      { max: 40, label: 'example context · 22.2 bits', ctx: 'Context: this is spatially-variable-gene detection in spatial transcriptomics - ranking genes by whether their expression varies in a spatially structured way across a tissue section.' },
      { max: 1e9, label: 'full task description', ctx: 'Task: find spatially variable genes. Data: DBiT-seq mouse eye, 2,500 spots, 1,050 genes. Score: Kendall correlation within variant groups. Reference baseline: SPARK-X (0.853). …' }
    ];
    var ask = 'Find ONE real paper from a scientific field whose single core mechanism is most promising to transplant into a different algorithm.';
    var slider = $('#dial'), bitsEl = $('#dial-bits'), ctxEl = $('#dial-ctx'), tierEl = $('#dial-tier'), lastTier = -1;
    var knots = [0, 11.24, 27.15, 57.07, 143.08], ticks = $('#dial-ticks');
    knots.forEach(function (b) { var s = H('span', '', Math.round(b) + '', ticks); s.style.left = (T(b) / T(BMAX) * 100) + '%'; });

    function setBits(b) {
      b = clamp(b, 0, BMAX);
      bitsEl.textContent = b < 20 ? b.toFixed(1).replace(/\.0$/, '') : Math.round(b);
      var ti = 0; while (b >= tiers[ti].max) ti++;
      if (ti !== lastTier) {
        lastTier = ti; tierEl.textContent = tiers[ti].label;
        ctxEl.innerHTML = (tiers[ti].ctx ? '<span class="ctx">' + tiers[ti].ctx + '</span> ' : '<span class="dim">[ no context ]</span> ') + '<span class="dim">' + ask + '</span>';
      }
      var x = sx(T(b));
      panels.forEach(function (p) {
        var v = interp(p.d.trend, b, T);
        p.cursor.setAttribute('x1', x); p.cursor.setAttribute('x2', x);
        p.knob.setAttribute('cx', x); p.knob.setAttribute('cy', p.sy(v));
        p.clipRect.setAttribute('width', x + 0.5);
        p.val.textContent = p.c.fmt(v);
        var ch = (v / p.v0 - 1) * 100;
        if (b < 0.05) { p.delta.textContent = 'at 0 bits'; p.delta.className = 'delta'; }
        else { p.delta.textContent = (ch > 0 ? '+' : '−') + Math.abs(ch).toFixed(0) + '% vs 0 bits'; p.delta.className = 'delta' + (ch < 0 ? ' neg' : ''); }
        var n = 0, lim = b + 3.5; while (n < p.xs.length && p.xs[n] <= lim) n++;
        var lo = Math.min(n, p.n), hi = Math.max(n, p.n);
        for (var i = lo; i < hi; i++) p.circles[i].setAttribute('class', i < n ? 'd-dot on' : 'd-dot');
        p.n = n;
      });
    }
    function fromSlider() { setFill(slider); setBits(Tinv(slider.value / 1000 * T(BMAX))); }
    function toSlider(b) { slider.value = Math.round(T(clamp(b, 0, BMAX)) / T(BMAX) * 1000); fromSlider(); }
    slider.addEventListener('input', function () { stop(); fromSlider(); });

    panels.forEach(function (p) {
      var drag = false;
      function at(e) { stop(); toSlider(Tinv(clamp(sx.inv(vbX(p.svg, e)), 0, T(BMAX)))); }
      p.svg.style.cursor = 'ew-resize';
      p.svg.addEventListener('pointerdown', function (e) { drag = true; p.svg.setPointerCapture(e.pointerId); at(e); });
      p.svg.addEventListener('pointermove', function (e) { if (drag) at(e); });
      p.svg.addEventListener('pointerup', function () { drag = false; });
      p.svg.addEventListener('pointercancel', function () { drag = false; });
    });
    sx.inv = lin(m.l, W - m.r, T(-3), T(178));

    var raf = null, playBtn = $('#dial-play');
    function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; playBtn.querySelector('span').textContent = 'Play'; } }
    function play() {
      var t0 = null, dur = 7000;
      playBtn.querySelector('span').textContent = 'Pause';
      function step(ts) {
        if (t0 === null) t0 = ts;
        var f = Math.min(1, (ts - t0) / dur);
        slider.value = Math.round(f * 1000); fromSlider();
        if (f < 1) raf = requestAnimationFrame(step); else { raf = null; playBtn.querySelector('span').textContent = 'Replay'; }
      }
      raf = requestAnimationFrame(step);
    }
    playBtn.addEventListener('click', function () { if (raf) stop(); else play(); });
    window.BVSR_dial = { set: toSlider, play: play };
    toSlider(0);
  })();

  /* ---------------------------------------------------------------- Table 1 as a bar table */
  (function cmp() {
    var tb = $('#cmp'); if (!tb) return;
    var cols = ['Creativity<small>top 1%</small>', 'Utility<small>top 1%</small>', 'Originality<small>mean</small>', 'Non-obviousness<small>mean</small>'];
    var rows = [
      ['BV', 'limited information', 'top', [[0.530, 0.043], [0.571, 0.041], [0.955, 0.002], [0.713, 0.012]]],
      ['BV', 'full information (control)', 'ctl', [[0.462, 0.045], [0.524, 0.043], [0.926, 0.004], [0.609, 0.015]]],
      ['Analogical Reasoning', '', '', [[0.429, 0.049], [0.508, 0.050], [0.914, 0.007], [0.492, 0.037]]],
      ['Cross-Domain', '', '', [[0.369, 0.052], [0.531, 0.045], [0.871, 0.012], [0.263, 0.020]]],
      ['No-Domain', '', '', [[0.117, 0.025], [0.530, 0.050], [0.598, 0.020], [0.104, 0.015]]]
    ];
    var html = '<thead><tr><th>Method</th>' + cols.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead><tbody>';
    rows.forEach(function (r) {
      html += '<tr class="' + r[2] + '"><td class="m">' + r[0] + (r[1] ? '<small>' + r[1] + '</small>' : '') + '</td>';
      r[3].forEach(function (v) {
        html += '<td><div class="cell"><div class="n"><b>' + v[0].toFixed(3) + '</b> <small>± ' + v[1].toFixed(3) + '</small></div>' +
          '<div class="bar"><i style="width:' + (v[0] * 100) + '%"></i><u style="left:' + ((v[0] - v[1]) * 100) + '%;width:' + (v[1] * 200) + '%"></u></div></div></td>';
      });
      html += '</tr>';
    });
    tb.innerHTML = html + '</tbody>';
  })();

  /* ---------------------------------------------------------------- Fig. 3: selective retention */
  (function sr() {
    var svg = $('#sr-svg'); if (!svg) return;
    var small = window.innerWidth < 560;                    /* phones get a tighter viewBox so labels stay legible */
    var W = small ? 380 : 620, Hh = small ? 330 : 392, m = small ? { l: 40, r: 10, t: 10, b: 40 } : { l: 48, r: 16, t: 12, b: 42 };
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
    var sx = lin(0, 100, m.l, W - m.r), sy = lin(0, 100, Hh - m.b, m.t);
    [0, 25, 50, 75, 100].forEach(function (v) {
      S('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), 'class': 'gridline' }, svg);
      var g = S('g', { 'class': 'ax' }, svg);
      S('text', { x: m.l - 8, y: sy(v) + 4, 'text-anchor': 'end' }, g).textContent = v + '%';
      S('text', { x: sx(v), y: Hh - m.b + 17, 'text-anchor': 'middle' }, g).textContent = v + '%';
    });
    S('text', { x: (m.l + W - m.r) / 2, y: Hh - 6, 'text-anchor': 'middle', 'class': 'ax-label' }, svg).textContent = 'Candidates discarded';
    S('text', { x: 13, y: (m.t + Hh - m.b) / 2, 'text-anchor': 'middle', 'class': 'ax-label', transform: 'rotate(-90 13 ' + (m.t + Hh - m.b) / 2 + ')' }, svg).textContent = 'Top-5% proposals kept';
    var drop = S('rect', { x: m.l, y: m.t, width: 0, height: Hh - m.b - m.t, 'class': 's-drop' }, svg);
    var gl = S('g', {}, svg);
    var band = S('path', { 'class': 's-band' }, gl);
    S('path', { d: linePath([[0, 100], [100, 0]], sx, sy), 'class': 's-line s-rand' }, gl);
    var zs = [0, 1, 2, 3].map(function () { return S('path', { 'class': 's-line s-zs' }, gl); });
    var icl = S('path', { 'class': 's-line s-icl' }, gl), sft = S('path', { 'class': 's-line s-sft' }, gl), srl = S('path', { 'class': 's-line s-sr' }, gl);
    var lab1 = S('text', { 'class': 'end-label', fill: 'var(--sr-deep)' }, svg); lab1.textContent = 'Selective retention';
    var lab2 = S('text', { 'class': 'end-label', fill: 'var(--ink-3)', x: sx(4), y: sy(57) }, svg); lab2.textContent = 'Random discarding';
    var cursor = S('line', { y1: m.t, y2: Hh - m.b, 'class': 's-cursor' }, svg);
    var marks = { rand: S('circle', { r: 4, 'class': 's-dotm', fill: 'var(--ink-3)' }, svg), icl: S('circle', { r: 4, 'class': 's-dotm', fill: 'var(--base)' }, svg), sft: S('circle', { r: 4.5, 'class': 's-dotm', fill: 'var(--sr-soft)' }, svg) };
    var knob = S('circle', { r: 6.5, 'class': 's-knob' }, svg);
    var key = 'creativity', range = $('#sr-range'), list = $('#sr-list'), say = $('#sr-say');

    function draw() {
      var d = D.sr[key];
      band.setAttribute('d', linePath(d.sr_band, sx, sy) + 'Z');
      d.zeroshot.forEach(function (z, i) { zs[i].setAttribute('d', linePath(z, sx, sy)); });
      icl.setAttribute('d', linePath(d.icl, sx, sy)); sft.setAttribute('d', linePath(d.sft, sx, sy)); srl.setAttribute('d', linePath(d.sr, sx, sy));
      var lx = small ? 60 : 67; lab1.setAttribute('x', sx(lx)); lab1.setAttribute('y', sy(interp(d.sr, lx) + (small ? 12 : 9)));
      update();
    }
    function pct(v) { return Math.round(v) + '%'; }
    function update() {
      var d = D.sr[key], x = +range.value;
      var v = { sr: interp(d.sr, x), sft: interp(d.sft, x), icl: interp(d.icl, x), rand: 100 - x };
      var z = d.zeroshot.map(function (c) { return interp(c, x); });
      var zmin = Math.min.apply(null, z), zmax = Math.max.apply(null, z);
      var px = sx(x);
      cursor.setAttribute('x1', px); cursor.setAttribute('x2', px);
      drop.setAttribute('width', px - m.l);
      knob.setAttribute('cx', px); knob.setAttribute('cy', sy(v.sr));
      ['rand', 'icl', 'sft'].forEach(function (k) { marks[k].setAttribute('cx', px); marks[k].setAttribute('cy', sy(v[k])); });
      $('#sr-out').textContent = x + '%'; setFill(range);
      list.innerHTML =
        '<div class="row lead"><i style="background:var(--sr-deep)"></i><span>Selective retention</span><b>' + pct(v.sr) + '</b></div>' +
        '<div class="row"><i style="background:var(--sr-soft)"></i><span>SFT regression</span><b>' + pct(v.sft) + '</b></div>' +
        '<div class="row"><i style="background:var(--base)"></i><span>In-context learning</span><b>' + pct(v.icl) + '</b></div>' +
        '<div class="row"><i style="background:var(--base-soft)"></i><span>Zero-shot judges <small style="color:var(--ink-3)">(4)</small></span><b>' + (Math.round(zmin) === Math.round(zmax) ? pct(zmax) : Math.round(zmin) + '–' + pct(zmax)) + '</b></div>' +
        '<div class="row"><i style="background:none;border:2px dotted var(--ink-3)"></i><span>Random discarding</span><b>' + pct(v.rand) + '</b></div>';
      var loss = 100 - v.sr, msg;
      if (x <= 0) msg = 'Nothing is discarded, so every candidate still has to be verified.';
      else if (x >= 100) msg = 'Everything is discarded.';
      else if (loss < 0.5) msg = 'SR skips <b>' + x + '%</b> of the experiments without losing any top proposal. Random discarding would already have lost <b>' + x + '%</b> of them.';
      else msg = 'SR skips <b>' + x + '%</b> of the experiments and loses <b>' + pct(loss) + '</b> of the top proposals. For the same loss, random discarding could skip only <b>' + pct(loss) + '</b>: SR saves <b>' + (x / loss).toFixed(1) + '×</b> more verification.';
      say.innerHTML = msg;
    }
    range.addEventListener('input', update);
    var drag = false;
    function at(e) { range.value = Math.round(clamp((vbX(svg, e) - m.l) / (W - m.r - m.l) * 100, 0, 100) / 2.5) * 2.5; update(); }
    svg.addEventListener('pointerdown', function (e) { drag = true; svg.setPointerCapture(e.pointerId); at(e); });
    svg.addEventListener('pointermove', function (e) { if (drag || e.pointerType === 'mouse') at(e); });
    svg.addEventListener('pointerup', function () { drag = false; });
    svg.addEventListener('pointercancel', function () { drag = false; });
    Array.prototype.forEach.call(document.querySelectorAll('#sr-seg button'), function (b) {
      b.addEventListener('click', function () {
        key = b.dataset.k;
        Array.prototype.forEach.call(document.querySelectorAll('#sr-seg button'), function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
        draw();
      });
    });
    draw();
  })();

  /* ---------------------------------------------------------------- shared: search-trajectory panel */
  var SERIES = {
    base: { cls: 'l-base', name: 'SimpleTES', color: 'var(--base)' },
    bv: { cls: 'l-bv', name: '+ BV', color: 'var(--bv)' },
    sr: { cls: 'l-sr', name: '+ SR', color: 'var(--sr)' },
    bvsr: { cls: 'l-bvsr', name: '+ BVSR', color: 'var(--bvsr)' }
  };
  function trajPanel(host, d, opt) {
    var W = opt.W, Hh = opt.H, m = opt.m;
    var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, role: 'img', 'aria-label': d.name + ': best score versus proposals verified' }, host);
    var sx = lin(0, 1500, m.l, W - m.r), sy = lin(d.ylim[0], d.ylim[1], Hh - m.b, m.t);
    opt.yticks.forEach(function (v) {
      S('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), 'class': 'gridline' }, svg);
      S('text', { x: m.l - 6, y: sy(v) + 3.5, 'text-anchor': 'end' }, S('g', { 'class': 'ax' }, svg)).textContent = opt.fmtTick(v);
    });
    var gx = S('g', { 'class': 'ax' }, svg);
    S('line', { x1: m.l, x2: W - m.r, y1: Hh - m.b, y2: Hh - m.b }, gx);
    [0, 750, 1500].forEach(function (v) {
      S('text', { x: sx(v), y: Hh - m.b + 14, 'text-anchor': v === 0 ? 'start' : v === 1500 ? 'end' : 'middle' }, gx).textContent = v.toLocaleString('en-US');
    });
    var paths = {}, pts = {};
    var cursor = S('line', { y1: m.t, y2: Hh - m.b, 'class': 'l-cursor', visibility: 'hidden' }, svg);
    opt.keys.forEach(function (k) {
      paths[k] = S('path', { d: linePath(d[k], sx, sy), 'class': SERIES[k].cls }, svg);
    });
    opt.keys.forEach(function (k) { pts[k] = S('circle', { r: 3.6, fill: SERIES[k].color, 'class': 'l-pt', visibility: 'hidden' }, svg); });
    var api = {
      svg: svg, sx: sx, sy: sy, paths: paths, d: d,
      at: function (x) {
        var out = {};
        cursor.setAttribute('x1', sx(x)); cursor.setAttribute('x2', sx(x)); cursor.setAttribute('visibility', 'visible');
        opt.keys.forEach(function (k) {
          var v = interp(d[k], x); out[k] = v;
          var show = v != null && paths[k].style.display !== 'none';
          pts[k].setAttribute('visibility', show ? 'visible' : 'hidden');
          if (show) { pts[k].setAttribute('cx', sx(x)); pts[k].setAttribute('cy', sy(v)); }
        });
        return out;
      },
      clear: function () { cursor.setAttribute('visibility', 'hidden'); opt.keys.forEach(function (k) { pts[k].setAttribute('visibility', 'hidden'); }); },
      xOf: function (e) { return clamp(Math.round(sx.inv(vbX(svg, e)) / 20) * 20, 0, 1500); }
    };
    sx.inv = lin(m.l, W - m.r, 0, 1500);
    return api;
  }

  /* ---------------------------------------------------------------- Fig. 4: SimpleTES ± BVSR */
  (function tes() {
    var box = $('#tes'); if (!box) return;
    var meta = [
      { t: 'Gene expression analysis', s: 'spatially variable genes', x: 3.8, ticks: [0.86, 0.88, 0.90], dec: 3 },
      { t: 'Drug response prediction', s: 'perturbation prediction', x: 1.7, ticks: [0.35, 0.40, 0.45], dec: 3 },
      { t: 'Anomaly detection', s: 'MLS-Bench', x: 1.9, ticks: [0.4, 0.5, 0.6], dec: 3 },
      { t: 'Scaling-law discovery', s: 'SLD-Bench', x: 3.1, ticks: [0.85, 0.90, 0.95], dec: 3 },
      { t: 'Brain activity prediction', s: 'ZAPBench, horizon 16', x: 1.6, ticks: [-0.028, -0.024, -0.020], dec: 4 }
    ];
    var panels = D.tes.map(function (d, i) {
      var mt = meta[i];
      var el = H('div', 'tpanel', '<div class="t-top"><div class="t-name">' + mt.t + '<span class="t-task">' + mt.s + '</span></div><div class="t-x">' + mt.x.toFixed(1) + '×<small>fewer proposals</small></div></div>', box);
      var read = H('div', 't-read', '', el);
      var p = trajPanel(el, d, { W: 250, H: 168, m: { l: 40, r: 8, t: 8, b: 22 }, keys: ['base', 'bvsr'], yticks: mt.ticks, fmtTick: function (v) { return v.toFixed(mt.dec === 4 ? 3 : 2); } });
      /* dotted reference: SimpleTES's final score, from where BVSR first reaches it */
      var fin = d.base[d.base.length - 1][1], cross = 1500, j;
      for (j = 1; j < d.bvsr.length; j++) {
        if (d.bvsr[j][1] >= fin) { var a = d.bvsr[j - 1], b = d.bvsr[j]; cross = a[0] + (fin - a[1]) / ((b[1] - a[1]) || 1) * (b[0] - a[0]); break; }
      }
      var ref = S('line', { x1: p.sx(cross), x2: p.sx(1500), y1: p.sy(fin), y2: p.sy(fin), 'class': 'l-ref' }, p.svg);
      p.svg.insertBefore(ref, p.paths.base);
      p.read = read; p.mt = mt;
      return p;
    });
    H('div', 'tkey', '<div class="legend"><span class="key" style="color:var(--bvsr)"><i class="swatch" style="height:4px"></i><span style="color:var(--ink-2)">SimpleTES + BVSR</span></span>' +
      '<span class="key" style="color:var(--base)"><i class="swatch"></i><span style="color:var(--ink-2)">SimpleTES</span></span>' +
      '<span class="key" style="color:var(--base)"><i class="swatch dotted"></i><span style="color:var(--ink-2)">SimpleTES final score</span></span></div>' +
      '<p>Horizontal: proposals verified, 0 to 1,500. Vertical: best task score so far, each task on its own scale.</p>', box);
    function fmt(v, dec) { return v == null ? '–' : v.toFixed(dec).replace('-', '−'); }
    function show(x) {
      panels.forEach(function (p) {
        var v = x == null ? { base: p.d.base[p.d.base.length - 1][1], bvsr: p.d.bvsr[p.d.bvsr.length - 1][1] } : p.at(x);
        if (x == null) p.clear();
        p.read.innerHTML = '<span style="color:var(--ink-3)">' + (x == null ? 'final score' : 'after ' + x.toLocaleString('en-US') + ' verified') + ':</span> ' +
          '<i style="background:var(--bvsr)"></i><b>' + fmt(v.bvsr, p.mt.dec) + '</b> &nbsp;<i style="background:var(--base)"></i>' + fmt(v.base, p.mt.dec);
      });
    }
    panels.forEach(function (p) {
      p.svg.addEventListener('pointermove', function (e) { show(p.xOf(e)); });
      p.svg.addEventListener('pointerdown', function (e) { show(p.xOf(e)); });
      p.svg.addEventListener('pointerleave', function () { show(null); });
    });
    show(null);
  })();

  /* ---------------------------------------------------------------- Fig. 5: ablation */
  (function abl() {
    var box = $('#abl'); if (!box) return;
    var keys = ['base', 'bv', 'sr', 'bvsr'], on = { base: true, bv: true, sr: true, bvsr: true };
    var ticks = [[0.86, 0.88, 0.90], [0.85, 0.90, 0.95]];
    var panels = D.ablation.map(function (d, i) {
      var el = H('div', 'apanel', '<div class="t-name">' + ['Gene expression analysis', 'Scaling-law discovery'][i] + '</div>', box);
      var p = trajPanel(el, d, { W: 380, H: 232, m: { l: 42, r: 64, t: 10, b: 34 }, keys: keys, yticks: ticks[i], fmtTick: function (v) { return v.toFixed(2); } });
      S('text', { x: (42 + 316) / 2, y: 228, 'text-anchor': 'middle', 'class': 'ax-label' }, p.svg).textContent = 'Proposals verified';
      p.labels = {};
      keys.forEach(function (k) {
        var g = S('g', {}, p.svg);
        S('circle', { cx: 380 - 58, cy: -3.5, r: 3.4, fill: SERIES[k].color }, g);
        S('text', { x: 380 - 51, y: 0, 'class': 'end-name' }, g).textContent = SERIES[k].name;
        p.labels[k] = g;
      });
      return p;
    });
    function layout() {
      panels.forEach(function (p) {
        var items = keys.filter(function (k) { return on[k]; }).map(function (k) { return { k: k, y: p.sy(p.d[k][p.d[k].length - 1][1]) + 4 }; });
        items.sort(function (a, b) { return a.y - b.y; });
        for (var i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < 13) items[i].y = items[i - 1].y + 13;
        keys.forEach(function (k) { p.labels[k].style.display = on[k] ? '' : 'none'; p.paths[k].style.display = on[k] ? '' : 'none'; });
        items.forEach(function (it) { p.labels[it.k].setAttribute('transform', 'translate(0,' + it.y.toFixed(1) + ')'); });
      });
    }
    var legend = $('#abl-legend');
    keys.forEach(function (k) {
      var b = H('button', 'key', '<i class="swatch" style="color:' + SERIES[k].color + '"></i>' + (k === 'base' ? 'SimpleTES' : SERIES[k].name), legend);
      b.type = 'button'; b.setAttribute('aria-pressed', 'true');
      b.addEventListener('click', function () { on[k] = !on[k]; b.setAttribute('aria-pressed', on[k] ? 'true' : 'false'); layout(); });
    });
    panels.forEach(function (p) {
      function move(e) {
        var x = p.xOf(e), v = p.at(x);
        var rows = keys.filter(function (k) { return on[k] && v[k] != null; }).sort(function (a, b) { return v[b] - v[a]; }).map(function (k) {
          return '<div class="row"><span><i style="background:' + SERIES[k].color + '"></i>' + SERIES[k].name + '</span><b>' + v[k].toFixed(3) + '</b></div>';
        }).join('');
        showTip('<div class="t-h">' + x.toLocaleString('en-US') + ' proposals verified</div>' + rows, e.clientX, e.clientY);
      }
      p.svg.addEventListener('pointermove', move);
      p.svg.addEventListener('pointerdown', move);
      p.svg.addEventListener('pointerleave', function () { p.clear(); hideTip(); });
    });
    layout();
  })();

  /* ---------------------------------------------------------------- case studies */
  (function cases() {
    var host = $('#cases'); if (!host) return;
    var art = {
      neb: '<svg viewBox="0 0 200 104"><path d="M12 70C45 20 80 22 104 52s52 34 84-14" fill="none" stroke="var(--base-soft)" stroke-width="1.5"/><path d="M12 84C45 34 80 36 104 66s52 34 84-14" fill="none" stroke="var(--base-soft)" stroke-width="1.5"/><path d="M14 76C46 28 80 30 104 59s52 34 82-12" fill="none" stroke="#6a58c4" stroke-width="2.4"/><g fill="#6a58c4" stroke="var(--surface-2)" stroke-width="1.5"><circle cx="14" cy="76" r="5"/><circle cx="36" cy="49" r="5"/><circle cx="60" cy="35" r="5"/><circle cx="84" cy="40" r="5"/><circle cx="104" cy="59" r="5"/><circle cx="124" cy="76" r="5"/><circle cx="148" cy="80" r="5"/><circle cx="170" cy="66" r="5"/><circle cx="186" cy="47" r="5"/></g></svg>',
      lattice: (function () {
        var s = '<svg viewBox="0 0 200 104">', r, c;
        for (r = 0; r < 5; r++) for (c = 0; c < 11; c++) {
          var x = 20 + c * 16 + (r % 2 ? 8 : 0), y = 16 + r * 18, dd = Math.abs((c + r * 0.9) - 6.5), o = Math.max(0.12, 1 - dd / 3.2);
          s += '<circle cx="' + x + '" cy="' + y + '" r="5.5" fill="#4f9a4c" opacity="' + o.toFixed(2) + '"/>';
        }
        return s + '</svg>';
      })(),
      waves: (function () {
        var p1 = '', p2 = '', x;
        for (x = 10; x <= 190; x += 3) { p1 += (x === 10 ? 'M' : 'L') + x + ' ' + (52 + 26 * Math.sin(x / 12)).toFixed(1); p2 += (x === 10 ? 'M' : 'L') + x + ' ' + (52 + 26 * Math.sin(x / 12 + 1.9)).toFixed(1); }
        return '<svg viewBox="0 0 200 104"><path d="' + p2 + '" fill="none" stroke="#6a58c4" stroke-width="2" opacity=".4"/><path d="' + p1 + '" fill="none" stroke="#6a58c4" stroke-width="2.6"/></svg>';
      })(),
      anomaly: '<svg viewBox="0 0 200 104"><g fill="var(--base)"><circle cx="86" cy="50" r="3.500"/><circle cx="97" cy="42" r="3.500"/><circle cx="104" cy="56" r="3.500"/><circle cx="93" cy="62" r="3.500"/><circle cx="112" cy="46" r="3.500"/><circle cx="118" cy="60" r="3.500"/><circle cx="101" cy="70" r="3.500"/><circle cx="80" cy="60" r="3.500"/><circle cx="108" cy="36" r="3.500"/><circle cx="124" cy="52" r="3.500"/><circle cx="90" cy="72" r="3.500"/><circle cx="113" cy="70" r="3.500"/></g><g fill="#5a6677"><circle cx="36" cy="26" r="4"/><circle cx="168" cy="80" r="4"/></g><g fill="none" stroke="#5a6677" stroke-width="1.8"><circle cx="36" cy="26" r="10"/><circle cx="168" cy="80" r="10"/></g></svg>',
      assign: '<svg viewBox="0 0 200 104"><g stroke="var(--base-soft)" stroke-width="1.3"><path d="M52 24 140 52M52 24 140 80M52 52 140 80M52 80 140 24M52 80 140 52"/></g><g stroke="#a84d7b" stroke-width="2.6"><path d="M52 24 140 24M52 52 140 52M52 80 140 80"/></g><g fill="#a84d7b"><circle cx="52" cy="24" r="6"/><circle cx="52" cy="52" r="6"/><circle cx="52" cy="80" r="6"/></g><g fill="var(--surface-2)" stroke="#a84d7b" stroke-width="2"><rect x="134" y="18" width="12" height="12" rx="2"/><rect x="134" y="46" width="12" height="12" rx="2"/><rect x="134" y="74" width="12" height="12" rx="2"/></g><g fill="#a84d7b" font-size="14" font-weight="700"><text x="156" y="29">$</text><text x="156" y="57">$</text><text x="156" y="85">$</text></g></svg>',
      scaling: '<svg viewBox="0 0 200 104"><path d="M30 14v72h152" fill="none" stroke="var(--ink-3)" stroke-width="1.5"/><path d="M44 24c22 46 44 56 66 56s44-18 62-50" fill="none" stroke="#5a6677" stroke-width="2.6"/><g fill="#5a6677"><circle cx="52" cy="44" r="3.500"/><circle cx="68" cy="60" r="3.500"/><circle cx="86" cy="76" r="3.500"/><circle cx="108" cy="78" r="3.500"/><circle cx="128" cy="76" r="3.500"/><circle cx="148" cy="60" r="3.500"/><circle cx="164" cy="46" r="3.500"/></g></svg>'
    };
    var cases = [
      { tab: 'Gene expression analysis', title: 'Spatially variable genes via the nudged elastic band',
        from: ['phys', 'Physics', 'Nudged elastic band', 'transition-state theory', art.neb], to: ['bio', 'Biology', 'Spatial expression', 'ranking genes on a tissue section', art.lattice],
        src: 'Mechanism from Jónsson, Mills & Jacobsen (1998)',
        body: ['The task is to rank genes by how strongly their expression is organized in space on a tissue section. The program borrows the nudged elastic band method, which finds a minimum-energy path by relaxing a chain of images under two projected forces: the true force perpendicular to the path and a spring force along it.',
          'It follows the heat-flow trajectory of each gene\'s expression field on the tissue graph, from the observed field to the uniform one, and measures at every image the component of a robust energy gradient perpendicular to the path. For pure noise the heat-flow path already is the minimum-energy path and this component vanishes. Spatial patterns bend the path, so the perpendicular force grows with pattern strength and becomes the gene\'s score.'],
        scores: [['Best baseline · SPARK-X', 0.853], ['SimpleTES', 0.906], ['BVSR', 0.918]], range: [0.84, 0.93] },
      { tab: 'Anomaly detection', title: 'Anomaly detection via finite-momentum pairing',
        from: ['phys', 'Physics', 'Finite-momentum pairing', 'the FFLO superconducting state', art.waves], to: ['cs', 'CS / ML', 'Anomaly scores', 'tabular outlier detection', art.anomaly],
        src: 'Mechanism from Larkin & Ovchinnikov (1964)',
        body: ['The task asks for an anomaly detector with at most 10,000 parameters, scored on four tabular datasets. The program transplants the nonuniform (FFLO) superconducting state, in which two species with mismatched Fermi surfaces still pair by giving each pair a nonzero momentum, so that the order parameter oscillates in space.',
          'Its two "species" are two scores that often disagree: marginal tail evidence from empirical distribution functions, and geometric evidence from a robust covariance and a Gaussian mixture of the normal mass. Instead of averaging them, it pairs them with an oscillating modulation along the direction in which they disagree most, relaxes the recovered pairing signal on a nearest-neighbour graph, and adds it to the combined score. No network is trained.'],
        scores: [['Best baseline · Isolation Forest', 0.491], ['SimpleTES', 0.629], ['BVSR', 0.679]], range: [0.45, 0.70] },
      { tab: 'Scaling-law discovery', title: 'A scaling law from assignment-game prices',
        from: ['econ', 'Economics', 'Assignment-game prices', 'competitive prices of a matching market', art.assign], to: ['cs', 'CS / ML', 'Scaling-law fit', 'final loss vs. training setup', art.scaling],
        src: 'Mechanism from Shapley & Shubik (1971)',
        body: ['The task is to find a formula with at most 26 parameters for the final training loss as a function of learning rate, batch size, data size and model size. The program draws on the assignment game, whose optimal matching is a linear program with competitive prices as dual variables.',
          'It solves one fixed, entropy-regularized assignment problem between two grids in the (log learning rate, log batch size) plane and uses the resulting price field as a fixed surface that reshapes the loss basin near the optimal learning rate and batch size. The law adds power laws in data and model size and a convex bowl around a scale-dependent optimum, and fits only two parameters.'],
        scores: [['Best baseline', null], ['SimpleTES', 0.940], ['BVSR', 0.967]], range: [0.93, 0.975] }
    ];
    var tabs = $('#case-tabs'), els = [], btns = [];
    cases.forEach(function (c, i) {
      var b = H('button', 'chip', c.tab, tabs); b.type = 'button'; b.setAttribute('role', 'tab'); btns.push(b);
      var sc = c.scores.map(function (s, j) {
        if (s[1] == null) return '<div class="score na"><span>' + s[0] + '</span><div class="track"></div><b style="color:var(--ink-3)">n/a</b></div>';
        var pos = (s[1] - c.range[0]) / (c.range[1] - c.range[0]) * 100;
        return '<div class="score dot' + (j === 2 ? ' win' : '') + '"><span>' + s[0] + '</span><div class="track"><i style="left:' + pos.toFixed(1) + '%"></i></div><b>' + s[1].toFixed(3) + '</b></div>';
      }).join('');
      function tb(t) { return '<div class="tbox"><span class="ftag ' + t[0] + '">' + t[1] + '</span>' + t[4] + '<div class="n">' + t[2] + '</div><div class="s">' + t[3] + '</div></div>'; }
      var el = H('div', 'case', '<div><div class="transfer">' + tb(c.from) +
        '<svg class="t-arrow" viewBox="0 0 30 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8h25M21 2.500 27 8l-6 5.500"/></svg>' + tb(c.to) + '</div>' +
        '<div class="scores">' + sc + '</div><div class="axis-note">Task score, higher is better. Dots are positioned on a ' + c.range[0] + ' – ' + c.range[1] + ' scale.</div></div>' +
        '<div class="case-body"><h4>' + c.title + '</h4><div class="src">' + c.src + '</div>' + c.body.map(function (p) { return '<p>' + p + '</p>'; }).join('') + '</div>', host);
      el.setAttribute('role', 'tabpanel'); els.push(el);
      b.addEventListener('click', function () { pick(i); });
    });
    function pick(i) {
      els.forEach(function (e, j) { e.classList.toggle('on', i === j); });
      btns.forEach(function (b, j) { b.setAttribute('aria-pressed', i === j ? 'true' : 'false'); b.setAttribute('aria-selected', i === j ? 'true' : 'false'); });
    }
    pick(0);
  })();

  /* ---------------------------------------------------------------- domains of inspiration */
  (function domains() {
    var host = $('#domains'); if (!host) return;
    var sets = [
      ['Top creativity', [['Mathematics', 1.33], ['Physics', 1.10], ['CS / ML', 0.97], ['Biology / Medicine', 0.88], ['Engineering / Signal', 0.85], ['Chemistry / Materials', 0.58], ['Astronomy', 0.55], ['Statistics', 0.37], ['Economics / Finance', 0.34]]],
      ['Top utility', [['Biology / Medicine', 1.25], ['Mathematics', 1.09], ['CS / ML', 1.08], ['Statistics', 1.04], ['Astronomy', 1.02], ['Physics', 0.93], ['Economics / Finance', 0.86], ['Engineering / Signal', 0.83], ['Chemistry / Materials', 0.53]]]
    ];
    var MAX = 1.5;
    sets.forEach(function (s) {
      var col = H('div', '', '<h4>' + s[0] + '</h4>', host), wrap = H('div', 'dwrap', '', col);
      s[1].forEach(function (r) {
        var row = H('div', 'dbar' + (r[1] > 1 ? ' up' : ''), '<span class="lab">' + r[0] + '</span><div class="track"><i data-w="' + (r[1] / MAX * 100) + '"></i><b data-w="' + (r[1] / MAX * 100) + '">' + r[1].toFixed(2) + '×</b></div>', wrap);
        row.title = r[0] + ': ' + r[1].toFixed(2) + '× the average top-1% rate';
      });
      H('div', 'dref', '', wrap);
      H('div', 'daxis', '<div></div><div>' + [0, 0.5, 1, 1.5].map(function (v) { return '<span style="left:' + (v / MAX * 100) + '%">' + v + '×</span>'; }).join('') + '</div>', wrap);
    });
    function grow() {
      Array.prototype.forEach.call(host.querySelectorAll('[data-w]'), function (e) {
        if (e.tagName === 'I') e.style.width = e.dataset.w + '%'; else e.style.left = e.dataset.w + '%';
      });
    }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) { if (en[0].isIntersecting) { grow(); io.disconnect(); } }, { threshold: 0.25 });
      io.observe(host);
    } else grow();
  })();
})();
