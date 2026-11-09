# BVSR project page

Static website for **Blind Variation, Selective Retention: Improving Creative Idea Generation in AI Scientists**.
No build step: open `index.html`, or serve the folder with any static host (GitHub Pages, Netlify, a lab web server).

```bash
python3 -m http.server 8000
```

## Things to edit before publishing

| What | Where |
|---|---|
| Paper / arXiv link (the button shows "coming soon" until this is set) | `LINKS.paper` at the top of `assets/js/main.js` |
| Hugging Face link (same "coming soon" behaviour) | `LINKS.huggingface` in `assets/js/main.js` |
| Code link | `LINKS.code` in `assets/js/main.js`, plus the footer of `index.html` |
| BibTeX (add the arXiv ID) | the `<pre id="bibtex">` block in `index.html` |
| Author homepages | wrap the names in the `.authors` block of `index.html` in `<a href="...">` |

## Layout

```
index.html                 page content
assets/css/style.css       styles (light + dark theme tokens at the top)
assets/js/data.js          chart data, recovered from the paper's vector figures (Figs. 2-5)
assets/js/charts.js        all figures and interactive widgets (plain SVG, no chart library)
assets/js/main.js          links, theme toggle, navigation, copy-BibTeX
assets/video/              walkthrough video + poster frame
assets/img/                favicon, social preview image, Stanford wordmark (from the paper sources)
tools/walkthrough-video/   source of the video (not needed by the site; safe to leave out when uploading)
```

External resources: Google Fonts (Inter, Source Serif 4, JetBrains Mono), KaTeX for equations, and the Hugging Face logo
(loaded from huggingface.co, with an emoji fallback if it cannot be fetched).

## Regenerating the walkthrough video

The video is an HTML animation (`tools/walkthrough-video/scene.html`, open it with `?play` to preview) rendered
frame by frame in headless Chrome and encoded with macOS AVFoundation, so it needs no ffmpeg.

```bash
python3 tools/walkthrough-video/capture.py "$PWD/tools/walkthrough-video" /tmp/bvsr-video video 30
```

```bash
swiftc -O -o /tmp/bvsr-encode tools/walkthrough-video/encode.swift
```

```bash
/tmp/bvsr-encode /tmp/bvsr-video/frames assets/video/bvsr-walkthrough.mp4 30 2400000
```

`capture.py` needs the `websocket-client` Python package and Google Chrome; it resumes if interrupted.
