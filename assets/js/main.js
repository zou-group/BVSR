/* BVSR project page: page chrome (links, theme, navigation, copy button). */
(function () {
  'use strict';

  /* ------------------------------------------------------------------
     EDIT ME: external links. Leave `paper` / `huggingface` empty until the link exists;
     that button then shows "(coming soon)" and is not clickable.
     ------------------------------------------------------------------ */
  var LINKS = {
    paper: '',                                   // e.g. 'https://arxiv.org/abs/2611.01234'
    code: 'https://github.com/zou-group/BVSR',
    huggingface: 'https://huggingface.co/collections/Xkev/bvsr'
  };

  var paper = document.getElementById('link-paper');
  if (paper && LINKS.paper) {
    paper.href = LINKS.paper; paper.target = '_blank'; paper.rel = 'noopener';
    paper.removeAttribute('aria-disabled');
    var soon = document.getElementById('paper-soon'); if (soon) soon.remove();
  }
  var hf = document.getElementById('link-hf');
  if (hf && LINKS.huggingface) {
    hf.href = LINKS.huggingface; hf.target = '_blank'; hf.rel = 'noopener';
    hf.removeAttribute('aria-disabled');
    var hfSoon = document.getElementById('hf-soon'); if (hfSoon) hfSoon.remove();
  }
  var code = document.getElementById('link-code');
  if (code && LINKS.code) code.href = LINKS.code;

  /* theme toggle */
  var root = document.documentElement;
  document.getElementById('theme-toggle').addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('bvsr-theme', next); } catch (e) {}
  });

  /* reading progress + active section in the top nav */
  var bar = document.getElementById('progress');
  var links = Array.prototype.slice.call(document.querySelectorAll('.topnav a'));
  var targets = links.map(function (a) { return document.querySelector(a.getAttribute('href')); });
  var ticking = false;
  function onScroll() {
    if (ticking) return; ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = 'scaleX(' + (max > 0 ? Math.min(1, window.scrollY / max) : 0) + ')';
      var cur = -1;
      targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top < 140) cur = i; });
      links.forEach(function (a, i) { a.classList.toggle('active', i === cur); });
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();

  /* reveal on scroll */
  var items = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    Array.prototype.forEach.call(items, function (el) { io.observe(el); });
  } else {
    Array.prototype.forEach.call(items, function (el) { el.classList.add('in'); });
  }

  /* copy BibTeX */
  var copy = document.getElementById('copy-bib');
  if (copy) copy.addEventListener('click', function () {
    var text = document.getElementById('bibtex').textContent;
    function done() { copy.textContent = 'Copied'; setTimeout(function () { copy.textContent = 'Copy'; }, 1600); }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () {});
    else {
      var ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); done(); } catch (e) {}
      ta.remove();
    }
  });
})();
