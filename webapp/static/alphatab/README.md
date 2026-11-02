# Vendored alphaTab

This directory contains the alphaTab 1.9.0 browser runtime used by the Tabula
workbench. alphaTab is licensed under the Mozilla Public License 2.0; see
`LICENSE` and `LICENSE.header`.

Source: <https://github.com/CoderLine/alphaTab>

Files used by the app:

- `alphaTab.min.mjs`, `alphaTab.core.min.mjs` — ES module build. The app imports
  `alphaTab.min.mjs` directly; the worker and worklet below import the core.
- `alphaTab.worker.min.mjs`, `alphaTab.worklet.min.mjs` — renderer worker and
  audio worklet. Their URLs resolve relative to `import.meta.url`.
- `font/Bravura.*` — Bravura music font (SIL Open Font License 1.1).
- `soundfont/sonivox.sf2` — Sonivox GM soundfont; see
  `soundfont/README.md` and `soundfont/LICENSE` for its license.
- `LICENSE`, `LICENSE.header` — alphaTab license notices.

The player is configured in `webapp/static/app.js`. The rendered score keeps
alphaTab's attribution visible; do not hide it. The source tempo map comes from
the job's own `final.mid`, and the score uses guitar program 24 (nylon).
