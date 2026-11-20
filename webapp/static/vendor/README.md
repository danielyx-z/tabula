# Vendored PDF export libraries

Loaded lazily by `webapp/static/app.js` the first time the user clicks the PDF
export link, so the initial page payload is unchanged.

- `jspdf.umd.min.js` — jsPDF 2.5.2, MIT licence.
  Source: <https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js>
- `svg2pdf.umd.min.js` — svg2pdf.js 2.2.3, MIT licence.
  Source: <https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js>
- `tabula-music.ttf` — the music font used by the exported score. jsPDF only
  embeds TrueType outlines, so alphaTab's `font/Bravura.otf` was converted with
  `otf2ttf` (`fonttools` + `cu2qu`); nothing else was changed. The SIL Open
  Font License reserves the name "Bravura" for modified versions, so the
  derived font is named "Tabula Music". Bravura is
  Copyright © 2020 Steinberg Media Technologies GmbH and licensed under the
  SIL Open Font License 1.1 (<https://scripts.sil.org/OFL>); the copyright and
  licence records are kept in the font's `name` table.
