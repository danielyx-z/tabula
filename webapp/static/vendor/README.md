# Vendored PDF export libraries

Loaded lazily by `webapp/static/app.js` the first time the user clicks the PDF
export link, so the initial page payload is unchanged.

- `jspdf.umd.min.js` — jsPDF 2.5.2, MIT licence.
  Source: <https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js>
- `svg2pdf.umd.min.js` — svg2pdf.js 2.2.3, MIT licence.
  Source: <https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js>

The global names are `window.jspdf.jsPDF` and `window.svg2pdf.svg2pdf`.
alphaTab renders every system as one `svg.at-surface-svg` of pure vector
geometry, so the export needs no music font: the page clones those SVGs, adds
the missing `viewBox` and draws them into the PDF.
