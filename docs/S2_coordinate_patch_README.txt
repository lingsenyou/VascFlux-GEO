HISTORICAL S2 REPAIR NOTES - INCLUDED IN AUTHORIZED CODE RELEASE v1.0.0
Prepared 27 October 2026

Purpose
This patch repairs Figure S2C/D redraw from historical fixed out-of-fold
predictions without including subject identifiers, sample accessions,
treatment labels, module labels, or row identifiers. It exports precisely
four numeric plot-coordinate columns for the 576 original statin rows.
It is not a de-identification certificate or permission to publish the data.

Application to the existing 65-member code-and-aggregate archive
1. Replace figures/redraw_supplement_figures.py with the patched copy.
2. Replace figures/figure_specs.json with the patched copy. Only the S2
   source-file list is changed; it must point to the coordinate CSV so
   redraw_main_figures.save() no longer looks for the omitted old anchor.
3. Add reproducibility/results/figureS2_plot_coordinates.csv.
4. Retain reproducibility/results_extended/repeat_metrics.csv. The copy in
   this patch is unchanged and was already included in the existing archive.
5. Update the archive README, attachment/path manifest, checksums, and any
   claims about the excluded S2 historical drawing dependency. Recompute
   manifests from the final assembled package; do not reuse stale checksums.
6. Keep the existing final journal figure-number crosswalk. This S2 patch
   does not change main-figure numbering or any manuscript result.

Drawing
Use the original package layout with figures/ beside reproducibility/ and
its recorded figure environment. From figures/, call:
    python -c "import redraw_supplement_figures as s; s.outcome()"
Running the full supplementary-figure script also requires reconstructing
any other sample-level prediction inputs omitted from the aggregate package;
this patch alone makes no claim to provide those inputs.

Validation actually completed
- The new CSV round-trips all four float columns without changing a bit.
- Original and patched S2 PNGs are pixel-identical at 600 dpi.
- S2C maximum absolute residual difference: 2.220446049250313e-16.
- S2D maximum prediction difference: 0.0023220995754172913.
- Patched PDF was rendered with Poppler and visually checked for clipping,
  overlap, glyph loss and label placement.
- Formulas, source hash, row count and reproduction limits are documented
  in S2_coordinate_provenance.json. The raw linked anchor is not included.

Scope and remaining work
This repairs a plot-input gap using fixed historical predictions. It does
not regenerate the historical fitted analysis, independently validate any
biology, or establish that every excluded sample-level result is available.
The original software is now licensed under MIT for release v1.0.0; this does
not relicense GEO data. Repository: https://github.com/lingsenyou/VascFlux-GEO.
No DOI is claimed. The historical repair did not alter manuscript results.
