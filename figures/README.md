# Public figure set

This directory contains the figures used by the public Do ≠ See release.
`*.svg` files are the primary, publication-ready vector artifacts; matching
`*.png` files are provided for previews and slide decks. The two `*.drawio`
files are editable conceptual sources for diagrams 1 and 3.

## Provenance

- `fig1_do_not_equal_see`: conceptual causal model; it is not a behavioral
  result.
- `fig2_p11b_opportunity_funnel`: generated from
  `results/p11b_public_aggregate_v1.json`.
- `fig3_p12_fault_conditioned_design`: generated from
  `configs/p12_fault_conditioned_public_plan_v1.json`.
- `figure_manifest_v1.json`: SHA-256 hashes of the generated SVG artifacts.

Run `python3 scripts/generate_public_figures.py` to rebuild all outputs. The
generator is deterministic given the two public JSON inputs and deliberately
does not read private traces, credentials, or provider responses.
