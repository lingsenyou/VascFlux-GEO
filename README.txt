VascFlux-GEO analysis and figure code - v1.0.0
Release prepared 27 October 2026.
Repository: https://github.com/lingsenyou/VascFlux-GEO
Release: https://github.com/lingsenyou/VascFlux-GEO/releases/tag/v1.0.0

LICENCE AND SOURCE RIGHTS
The authors confirmed their rights and authorized public release of their
original analysis and plotting code under the MIT License on 27 October 2026.
LICENSE contains the standard MIT terms for that code and its original source
documentation. The copyright notice identifies the VascFlux-GEO contributors;
the recorded author list is provided in CITATION.cff.

The MIT grant does not relicense third-party GEO data, source-study metadata,
or third-party dependencies. Original GEO expression/platform payloads and
source-linked individual records are not included. The included aggregate,
synthetic, and unlinked individual-derived plotting tables are research outputs
with the provenance and limits below; this software licence does not grant
additional rights over underlying third-party data. Consult original source
conditions and citations before reuse or redistribution of source-derived data.
No CC0 declaration for GEO data is made by this release. Dependency licences
remain those of their respective packages.

This is a versioned source-code release with included reporting inputs; no
archive DOI or GigaDB deposit is claimed. Journal submission and publication
status are separate from this repository release.

CONTENTS AND DATA SCOPE
This release retains the original analysis/acquisition/verification code,
dependency records, frozen modules, study-level registry and 35 result CSVs
(34 current aggregate or synthetic tables plus the historical permutation
diagnostic). It additionally contains two derived plotting tables:
* results/figureS2_plot_coordinates.csv: 576 rows, four numeric plot coordinates
  from fixed historical statin predictions; no row or cross-module identifiers.
* figures/small_group_reporting/small_group_point_values.csv: 416 derived values,
  labelled only by descriptive context, grade/change group, module and value;
  sorted independently within each context/group/module, with no point IDs.
These are individual-derived plot values, not aggregate-only data. Their lack
of original identifiers is not a de-identification certificate or a new consent
determination. Original subject/sample/source IDs, ages, linked folds, individual
prediction records and source expression/platform payloads are excluded. Do not
upload locally reconstructed individual records with this release.

SCIENTIFIC BOUNDARY
The focal cohort has 72 rosuvastatin pairs; PAD has 14 separately fitted pairs.
Other sources are descriptive. Do not sum heterogeneous counts as independently
confirmed participants. Simulations are synthetic. Extensions are post hoc and
nonconfirmatory. Bootstrap ranges resample fixed losses; their nominal level is
not validated population coverage. Modules overlap. See the data dictionary.

NUMERICAL REBUILDING
Keep reproducibility/ and figures/ as siblings. Recorded numerical environment:
Python 3.12.14 and reproducibility/requirements-analysis.txt. From reproducibility/:
  python -m pip install -r requirements-analysis.txt
  python inputs/download_geo.py --out geo_cache
  python inputs/acquire_canonical_platforms.py --source-cache geo_cache --out canonical_cache
  python reproduce.py --full --data-dir canonical_cache --source-manifest canonical_cache/canonical_manifest.json
Acquisition needs the network and exact complete-file hash checks. The official
family streams are checked through the first complete platform table, not the
entire family archive. A previously truncated GPL570 cache is not valid. Do not
replace expected hashes merely to make acquisition pass. Source visibility is
not a new licence for redistributing third-party data.
After full rebuilding, python reproduce.py --verify-only checks generated local
outputs. The distributed subset alone does not pass --verify-only and does not
claim to. Rebuilding generates source-linked local files which remain excluded.
The optional original_fixed_lambda_OOF_reproduction audit row is skipped when
the original historical results/oof_subject_errors.csv anchor is absent. The new
S2 plot coordinates do not replace that historical fitting/audit anchor.

PLOTTING AND THE TWO REPORTING REPAIRS
Recorded figure environment: Python 3.11.9 and reproducibility/requirements-figures.txt.
Main figures retain historical code filenames; docs/Figure_number_crosswalk.txt
maps original 7, 8, 6 to current manuscript 6, 7, 8. Do not relabel scientific outputs.
From figures/, Figure S2 can now be redrawn offline from this release:
  python -c "import redraw_supplement_figures as s; s.outcome()"
Both its script and figure_specs.json use the coordinate file, not the omitted
historical anchor. This repairs S2C/D plotting only; it does not refit that anchor.
Full plotting still requires other sample-level predictions/JSON summaries from
a complete numerical rebuild. No claim is made that this subset alone redraws
all 16 main-plus-supplementary figures.
To redraw Figure S8 and its point-table addendum, from figures/small_group_reporting/:
  python -m pip install -r requirements-reporting.txt
  python plot_reporting_points.py
This reads the included point table only. The exercise display has five paired
participants. External values are restriction-specific means within literal
source record ID and grade; distinct participant identity is unconfirmed. The
four restrictions overlap, and title proxies are not confirmed technical batches.
Counts cannot be summed. No new model training, tests, intervals or conclusions
were introduced by either reporting repair.

HISTORICAL SUBSET
reproducibility/results/summary.json is a declared derived aggregate subset, not
the original byte-identical summary. Its original source hash/selected keys are
preserved in provenance. Historical permutation values are diagnostic, not a
calibrated hypothesis test. Corrected cross-context outputs use complete GPL570.

VALIDATION AND REPRODUCTION LIMITS
The release audit records byte comparisons, Python syntax/import smoke checks,
S2 pixel identity and S8 redraw. Passing them does not establish independent
scientific validation, a new consent determination or complete de-identification.
The 35 original result CSVs are unchanged; two additional plotting CSVs repair
reporting-input gaps. Seventeen Python files passed syntax checks. Offline
S2 and both S8 pages were redrawn in a relocated directory and were pixel-
identical to the validated reference plots. This release task did not rerun
GEO acquisition or a full numerical reconstruction, and it does not claim that
the aggregate subset alone regenerates every source-linked output or figure.
See docs/SELECTED_REDRAW_VALIDATION.json and the original reproduction notes.
