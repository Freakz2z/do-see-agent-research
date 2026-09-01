# Public figure set

This directory contains the figures used by the public Do ≠ See release.
The three primary `fig*.png` files are complete GPT-image renderings with their
titles, labels, numbers and captions embedded directly in the image. No
post-processing or Matplotlib text overlay is applied to those primary assets.

## Provenance

- `fig1_do_not_equal_see.png`: complete conceptual causal model; it is not a
  behavioral result.
- `fig2_p11b_opportunity_funnel.png`: complete P11-B result figure; exact
  values originate in `results/p11b_public_aggregate_v1.json`.
- `fig3_p12_fault_conditioned_design.png`: complete P12 design figure; exact
  design metadata originates in `configs/p12_fault_conditioned_public_plan_v1.json`.
- `figure_manifest_v2.json`: SHA-256 hashes of the three primary GPT-image
  assets.
- `gpt_image_provenance_v2.json`: prompt summaries and claim-boundary notes
  for the three GPT-image assets.

The GPT-image assets are intentionally fixed renderings rather than a
deterministic local re-generation step. The machine-readable JSON inputs are
the source of truth for numbers and design metadata; no private traces,
credentials or provider responses are required to inspect the figures.
