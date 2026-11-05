# Silent drift: ue8m0 activation-scale rounding destabilises DeepSeek-V4-Flash

**Update — November 5, 2026.** The full synthesis is published as
[WRITEUP.md](WRITEUP.md), with data tables in [data/](data/) (published to this repo on Nov 5, 2026). It extends
the Nov 2 characterisation below with: the per-layer sensitivity map's
position/state structure (exactly layers 21 and 22 are individually
catastrophic, at prompt onset only), a deep-generation extension, the
fidelity governor's validation *and* its shadow-test failure on live
traffic (the governor has since been built; its detection bands did not
generalise, so the "designed, not yet built" status below is superseded),
and the baseline battery's relative-bound re-file. Where this release
note and the synthesis differ, the synthesis is the current record.

---

Public release — Nov 2, 2026. Fossil AI / Window Inference project.

**Scope.** This is an engineering technical note about one kernel
integration on one checkpoint — not a general claim about ue8m0
quantisation or about DeepSeek-V4-Flash as shipped. DeepSeek's own
stack uses ue8m0 successfully; what is documented here is how this
rounding choice accumulated fatally in *our* streaming integration,
where it bites layer by layer, and what that teaches the design of a
fidelity governor. The evidence limits are stated plainly below and in
the characterisation itself.

## The finding

Integrating fast Tilelang FP4 expert kernels into a verified streaming
implementation of DeepSeek-V4-Flash-0731 produced fluent-looking failure:
the model's greedy output flipped and its hidden-state norm grew ~100x,
with no error anywhere. The cause is a single rounding decision —
`act_quant`'s ue8m0 (E8M0, power-of-two) activation scales.

Characterised as a three-act process:

1. A small, systematic per-layer shrink of the hidden state (norm ratio
   ~0.98) accumulates through layers 20–29.
2. The accumulated drift — not any layer's local error — flips a
   borderline router at layer 37.
3. Layer 38's router collapses onto a single expert and the
   representation explodes.

Per-layer drift injection (fresh process per layer) localises the
catastrophic sensitivity to layers 21 and 22 individually: Tilelang
act_quant at either layer alone destroys the forward (final hidden norm
143x / 159x baseline), while the other 41 layers are norm-benign
single-layer (several still flip near-tie greedy choices).

Replacing only the power-of-two scales with exact FP32 scales restores
baseline behaviour exactly — identical 10-token greedy sequence — at a
net speed *loss*. The fast path's speed and its drift are the same
rounding decision.

The characterisation motivates a closed-loop fidelity governor (offline
per-layer drift-sensitivity map + live hidden-norm envelope guardrail
with exact-path fallback). The governor is **designed, not yet built** —
this release covers the characterisation and the sensitivity map, not a
working governor.

## Contents

- `UE8M0_DRIFT_CHARACTERISATION.md` — the main characterisation.
- `DRIFT_INJECTION_REPORT.md` — per-layer drift-injection sensitivity
  map. Includes the invalidated v1 run and its retraction, with the
  corrected fresh-process v2 results the conclusions rest on.
- `ACTQUANT_AB_REPORT.md` — the act_quant A/B mechanism-isolation tests.
- `PRIORITY_MANIFEST_2026-11-02.md` — SHA-256 hashes of the three
  reports, recorded before publication.

## Provenance note

The characterisation was drafted Nov 1, 2026 as an internal draft and
still carries its original "not for release" header. It is published
verbatim — byte-identical to the hashed draft — by the author's
explicit decision on Nov 2, 2026, so the public file and the priority
record match exactly.

SHA-256:
- UE8M0_DRIFT_CHARACTERISATION.md — fe97526ff659a68cfd2901efe111dbe2e9898375e8b0f85d2c671d1f71b5d042
- DRIFT_INJECTION_REPORT.md — 11afd233a58743d024edaaa6d1a9d5e9b6fe1c3274597c8842d042d1f9d17764
- ACTQUANT_AB_REPORT.md — 165a9185e5ebb6baa7b54911c9aeb90fa651976433e1026f5fef810d73f22262

## Limitations (summary — full list in the characterisation)

- End-to-end evidence is one prompt family at position 0 plus a 10-token
  greedy continuation; the mechanism A/B tests span 4 layers. A ~200-token
  trace corpus is built and queued to widen the evidence.
- Greedy decoding on a near-tie baseline is a deliberately sensitive
  drift detector; sampled decoding would show graded, not binary, damage.
- One checkpoint (DeepSeek-V4-Flash-0731), one hardware target
  (RTX 4060), one kernel-library version (Tilelang 0.1.15, CUDA 12.8.1).
