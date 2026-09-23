# Paper figures

> **Prepublication artifact status:** Model-dependent result figures in this directory use the archived pre-correction Phase 95 seed-004 policy. They are retained for provenance and will be regenerated together after final corrected policy selection. System, architecture, and supporting-method figures remain part of the working paper set.

Each figure has its own directory with a builder or editable source, inputs needed for rebuilding, exported PDF/PNG files where applicable, and a short provenance README. The catalog is `paper/figure_catalog.json`; shared plotting code is under `paper/figure-support/`.

Run all eleven cataloged Python-built main-paper figures with:

```powershell
python -B paper/build-figures.py
```

The study-system map, objective-location schematic, and spring-peak flowchart have manually refined PowerPoint sources. The batch command leaves those exports unchanged.

Portable Appendix A--C builders and exports are under `paper/figures/appendix/`. Their README files identify inputs, calculations, and limits.

The current working set contains fourteen inherited main/supporting figure directories plus six new appendix figures. Final model-dependent exports must be refreshed only after one corrected policy is explicitly selected so that checkpoint, metrics, tables, and figures stay internally consistent.
