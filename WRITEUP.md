# Silent drift: ue8m0 activation-scale rounding destabilises DeepSeek-V4-Flash under approximate FP4 expert kernels

*Characterisation notes, November 2026. All claims below are scoped to the
measurements reported here; the data tables behind the headline numbers are
in [`data/`](data/).*

## Summary

Integrating fast Tilelang FP4 expert kernels into a verified, bit-exact FP32
streaming implementation of DeepSeek-V4-Flash produced fluent-looking
failure: the model's greedy output flipped and its hidden-state norm grew
~100×, with no error anywhere. We localise the cause to a single rounding
decision — `act_quant`'s ue8m0 (E8M0, power-of-two) activation scales — and
characterise the failure as a three-act process: (1) a small, systematic
per-layer shrink of the hidden state (norm ratio ~0.98) accumulates through
layers 20–29; (2) the drift, not any layer's local error, flips a borderline
router at layer 37; (3) layer 38's router collapses onto a single expert and
the representation explodes. Replacing only the power-of-two scales with
exact FP32 scales restores baseline behaviour exactly (identical 10-token
greedy sequence), at a net speed *loss* — the fast path's speed and its
correctness turn out to be the same rounding decision.

Follow-up measurements give the failure a sharp structure. Under a
calibrated single-layer insult of the same kind, exactly two of the model's
43 layers — 21 and 22 — are individually catastrophic, and catastrophic-ness
is a property of (layer × position × state), not of a layer alone: the same
insult at layers 21/22 is fully catastrophic at prompt onset and completely
absorbed later in prefill and at deep-generation low-margin positions. The
re-arming axis is prompt-onset position/state, not how contested the
next-token decision is. A closed-loop fidelity governor built on that map
passed every correctness test we filed, but its detection bands did not
generalise to live traffic — healthy generations produce norm excursions
larger than the real blow-up — so it survives only as an offline tripwire.
That negative result is part of the record and is reported in §7.

## Setup

- **Checkpoint:** `deepseek-ai/DeepSeek-V4-Flash-0731` (MIT weights; 284B
  MoE, 43 layers, 256 routed experts/layer, top-6; FP4 experts with E8M0
  scales; other weights FP8 e4m3 dynamic, scale_fmt ue8m0).
- **Reference implementation:** a hand-written streaming forward pass that
  runs the model in FP32 on a single consumer machine (RTX 4060, 8 GB VRAM,
  16 GB system RAM), streaming weights slice by slice. Its dequant path was
  verified against DeepSeek's Tilelang reference kernels per expert
  (cosine 0.99979 and 0.99907 for two routed experts, 0.999997 for the
  shared expert, at layer 5; normalisation agreement ~1e-7).
- **Baseline (exact path, "Hello", position 0):** greedy token 14,
  top-1 logit 17.32, final hidden norm 218.658, top-3 [14, 3362, 21133] —
  a near-tie (top-2 margin 0.0227), which makes greedy flips a sensitive
  drift detector.
- **Approximate path:** Tilelang FP4/FP8 expert GEMM with `act_quant` as
  shipped in DeepSeek's kernel code (block 128, ue8m0 scales; Tilelang
  0.1.15, CUDA 12.8.1), all other components identical.

## 1. A red herring, disposed of: the layer-5 spike is the model, not a bug

The exact path shows a hidden-norm jump at layer 5 (48 → 540, growing to
~8,947 by layer 42), driven by routed expert 35 (output norm 228.5 vs ~6–7.5
for its peers; routing weight 0.888–0.948 across prompts). Tilelang's own
kernels reproduce it (224.54, cosine 0.99979 against the dequant path), as
does a nibble-order control (swapping FP4 nibble order collapses expert 35
to 6.47, and DeepSeek's converter stacks low-nibble-first, against the bug
hypothesis). Conclusion: large-magnitude "load-bearing" experts are genuine
behaviour of this checkpoint. Any drift analysis must be differential
against the exact path, never absolute — the model's healthy state is
already extreme. This lesson returns in §7 and §8.

## 2. The integration failure

With Tilelang experts everywhere: greedy 201 vs baseline 14; final hidden
norm 23,844.6 vs 218.658. Per-layer norms track the exact path within
~0.5–3% through layer 35, then +4.3% (L36), +12.9% (L37), +836% (L38),
remaining exploded through L42. Routing is identical to the exact path
through layer 36; at layer 37 the selected set changes; at layer 38 the
router collapses (one expert at weight 1.489, all others ≤ 0.009) and the
norm reaches 69,486 vs the exact path's 7,424.

## 3. Localisation: drift through layers 20–29 is sufficient; the flip is downstream

Hybrid sweep (approximate kernels below a cutoff layer, exact above; one
token per run; cutoff 0 as control reproduces the baseline bit-for-bit):

- **Cutoff 20: stable.** Final norm 217.9; per-layer ratios 0.971–0.996;
  greedy 21133 — a member of the baseline top-3 near-tie.
- **Cutoffs 24, 27, 30, 37, 38: all blown**, final norms ~23.8–24.5k, with
  the identical layer-37 flip and layer-38 collapse — *including cutoff 37,
  where layers 37+ run the exact path*. The strong prior prediction ("exact
  layers downstream of the drift will recover the baseline") failed:
  accumulated drift through layer 36 alone flips layer 37's router, and
  layer 38's collapse is path-independent given the diverged state.
- Bisect (24/27) narrows the fatal accumulation band to layers 20–29, where
  blown runs show a systematic norm ratio ~0.979–0.984 per layer vs the
  stable run's 0.985–0.993: a consistent ~1.5–2% per-layer shrink.

Data: [`data/hybrid_sweep.csv`](data/hybrid_sweep.csv).

## 4. Mechanism isolation: it is the power-of-two scale rounding

A/B tests on identical real inputs at layers 5/22/30/38 (paths: A exact
dequant; B Tilelang as shipped; C Tilelang GEMM on BF16 activations, no
`act_quant`; D BF16-cast control; E `act_quant` with exact FP32 scales):

- B vs A per-expert cosine 0.9993–0.9999 — locally excellent, globally
  fatal. The failure is accumulation, not local error size.
- Path D is cosine 1.0 everywhere: the BF16 cast is ruled out.
- The `act_quant` round-trip alone reproduces B's deviation; path E (exact
  scales) collapses it: layer 5 cosine 1.0 / norm ratio 0.99871 (= D
  exactly), layer 30 norm ratio 1.00121 (vs B's 0.99355), layer 22 max
  absolute error 0.243 → 0.088.
- **Mechanism:** ue8m0 scales are powers of two, so each block's activation
  scale is rounded to a coarse grid, biasing block magnitudes (here,
  systematically downward); 43 layers of experts compound the bias until it
  exceeds a router margin. **FP8 quantisation per se is exonerated** — the
  FP8 shared expert and FP8 paths agree to cosine ≥ 0.9999.

## 5. Resolution and its price

Tilelang + exact FP32 scales: one-token validation greedy 14 = baseline,
final norm 219.37 vs 218.66, per-layer norms within ±0.3%, routing
identical at *all* layers; full 10-token generation identical to the exact
path token-for-token. But the exact-scale variant is ~22% *slower* than the
exact dequant path (29.5 vs 24.0 s/token in the build at the time): the
exact scales remove the kernel's speed advantage. The fast path's speed and
its drift are one and the same rounding decision; there is no free
configuration of it.

## 6. The structure of catastrophic sensitivity

The drift motivated a per-layer sensitivity map: inject the same ue8m0
insult at a *single* layer in an otherwise exact forward (blend factor
α = 1.0; fresh process per cell) and score behaviour against the exact
baseline. A cell is **catastrophic-class** iff the final-norm ratio ≥ 10×,
or the greedy flip is non-near-tie (the flipped token outside the control's
top-3). Findings:

- **Exactly two of 43 layers are individually catastrophic: 21 and 22.**
  At prompt position 0, layer 21 gives final-norm ratio 142.9× and layer 22
  gives 159.1×, both flipping greedy 14 → 201 (non-near-tie). Every other
  layer tested at that position is absorbed or produces only a near-tie
  flip (e.g. layer 38: ratio 0.991×, flip to the baseline's own top-3
  member 21133).
- **Dose:** layer 21 at α = 0.5 remains fully catastrophic (117.7×). Halving
  the insult does not change the class — the position-0 catastrophe is not
  a marginal effect.
- **Position/state dependence:** at late-prefill position 35 of a longer
  prompt (control top-2 margin 1.473), the identical insult at layers 21
  and 22 is completely absorbed (norm ratios 1.002× and 1.000×; no flip;
  neighbouring layers 20/23/40/42 likewise ~1.000×).
- **Deep generation:** at the lowest-margin position of a deep-generation
  trajectory (position 24; top-2 margin 0.0627 — the most contested forward
  on that trajectory), no catastrophic cell occurred at any layer outside
  {21, 22}, and layers 21/22 themselves were harmless there (their
  final-norm deviation was orders of magnitude below their position-0
  effect; layer 38, the only other layer ever observed to flip anywhere,
  did not flip at all).
- **The axis is onset, not margin.** Position 0 (margin 0.0227,
  catastrophic) and deep position 24 (margin 0.0627, absorbed) are the same
  near-tie class with opposite outcomes; decision margin does not re-arm
  the insult. What distinguishes the catastrophic cells is prompt-onset
  position/state.

Data: [`data/sensitivity_cells.csv`](data/sensitivity_cells.csv). Scope is
stated cell-by-cell in the table; the map's claim is an *individually
catastrophic* claim (single-layer insults) at the tested positions, not a
claim about combined multi-layer insults beyond §3's accumulation result.

## 7. The governor: correct machinery, a sensor that does not generalise

A natural response to §6 is a closed-loop fidelity governor: run the
approximate path only at insensitive layers, watch the live hidden state
against an exact-path envelope, and fall back to exact recompute on breach.
We built and validated one (bands from a 241-forward exact-path capture
across 7 prompts; layers 21/22 exact unconditionally; the final hidden norm
added as a detection channel after an earlier version's sensor failed both
ways and was rejected).

- **Correctness: passed every filed test.** Governed baseline reproduces
  the exact sequence with zero governor events; a forced layer-21 insult is
  caught (final norm 23,853 vs band top 5,569), recomputed exactly, and the
  committed sequence is identical to the exact path.
- **Benefit: missed its filed bar.** Governed speed 13.132 s/token was a
  4.91% improvement against a pre-registered ≥ 5% requirement. Validated,
  not promoted.
- **Shadow test: the bands do not generalise.** Run in shadow over recorded
  healthy traffic and a live leg, the sensor fired repeatedly on healthy
  generations: healthy late-layer norms reach ~3× the real blow-up's
  values, and healthy final-norm excursions exceed the blow-up's final
  norm, so no magnitude channel separates the catastrophe class from
  ordinary traffic (two of three pre-registered trust criteria failed;
  the would-have-fired cost was 5.17% of wall against a < 2% bar). In the
  live leg every firing was a false positive — ~38% added wall time for
  zero committed-token benefit. The surviving role for the sensor is an
  offline tripwire over recorded traffic, not a live guardrail.

The mechanistic reason is §1's lesson at scale: this checkpoint's healthy
state is already extreme, and norm magnitude is a property of the traffic,
not of the failure.

## 8. Methodological notes

Three lessons from this programme that we believe transfer:

1. **Absolute norm bounds are falsified as a fidelity gate.** A
   pre-registered baseline battery across 725 compared positions passed
   behaviour completely (0 greedy mismatches, 0 routing mismatches,
   generation leg identical) but *failed* its absolute norm bounds: late
   layers legitimately reach norms of 1e4–1e5, where summation-order noise
   is large in absolute terms and tiny relatively. The battery was
   re-filed — as a new test, with the original failure left standing on
   its own record — under relative bounds filed in advance (max relative
   deviation ≤ 2.0e-4, chosen at ~5× the worst observed value in the
   original battery's own published figures). It passed: worst relative
   final-norm deviation 2.38e-5, worst per-layer 3.70e-5. Gate on relative
   deviation, or do not gate on norms at all.
   Data: [`data/battery_relative.csv`](data/battery_relative.csv).
2. **Single-position statistics are not traffic statistics.** The original
   sensitivity map was measured at one prompt, one position; the governor
   bands built partly on such data failed on live traffic (§7), while the
   map's *structural* claim survived being re-tested at new positions and
   in a new regime (§6) precisely because those tests were run. Treat any
   n=1 cell as a hypothesis with a scope label, not a constant.
3. **File the bar before the run, and do not move it.** Every verdict
   above — including the failures (the original battery, the governor's
   speed bar, the shadow criteria) — was scored against criteria
   registered before the measurement ran. Two of the most useful results
   in this note (the battery re-file and the governor's non-promotion)
   exist only because the failed bars were left standing.

## Limitations

- One checkpoint, one hardware class (consumer RTX 4060), one kernel
  library version.
- The mechanism isolation (§4) spans 4 layers; the hybrid sweep (§3) is one
  prompt at position 0 plus a 10-token continuation; the sensitivity map
  (§6) covers the positions and prompts listed in its data table — the
  claim extends exactly that far and no further.
- Greedy decoding on a near-tie baseline is a deliberately sensitive
  detector; sampled decoding would show graded, not binary, damage.
- The governor shadow result (§7) is stated for norm-magnitude detection
  channels; it does not rule on detection signals of other kinds.

## Reproducibility

The checkpoint is public (`deepseek-ai/DeepSeek-V4-Flash-0731`, MIT). The
experiments require (a) an exact FP32 reference path to diff against —
absolute measurements are meaningless on this model, per §1 — and (b) the
ability to run the vendor's Tilelang expert kernels with `act_quant`
enabled and disabled per layer. All numbers in the tables are taken from
the runs described; where a figure is a range across runs it is labelled
as such.
