# Fossil AI / Window Inference — private priority manifest

Created: 2026-11-02 (America/New_York), by Cruffalo Crill for Rob.
Purpose: record SHA-256 hashes of the project's key unpublished findings as of
this date, so the state of the work on Nov 2, 2026 can be verified later
without any public disclosure. Publication of any of this remains Rob's
explicit decision — this manifest publishes nothing.

## Hashed files (SHA-256)

- UE8M0_DRIFT_CHARACTERISATION.md — fe97526ff659a68cfd2901efe111dbe2e9898375e8b0f85d2c671d1f71b5d042
  (ue8m0 act_quant drift characterisation of DeepSeek-V4-Flash: layers 21/22
  individually catastrophic, layer-37 router flip, layer-38 collapse;
  internal draft dated Nov 1, 2026, marked "Not for release")
- DRIFT_INJECTION_REPORT.md — 11afd233a58743d024edaaa6d1a9d5e9b6fe1c3274597c8842d042d1f9d17764
  (per-layer drift-injection sensitivity map v2, validated Nov 2, 2026)
- ACTQUANT_AB_REPORT.md — 165a9185e5ebb6baa7b54911c9aeb90fa651976433e1026f5fef810d73f22262

## Notes

- A local hash proves content existed in this form by the date another trusted
  record references the hash; on its own it is weaker than a public timestamp.
  If Rob later chooses to harden priority, options in increasing strength:
  (1) commit these hashes into a dated private git history, (2) email the
  manifest to a third party, (3) publish the characterisation itself.
- The governor design described in the characterisation is still unbuilt —
  priority for the *design* is what this manifest covers, not a working system.
