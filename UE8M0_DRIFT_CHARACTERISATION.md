# Silent drift: ue8m0 activation-scale rounding destabilises DeepSeek-V4-Flash under approximate FP4 expert kernels

**INTERNAL DRAFT — Nov 1, 2026. Not for release. Publication is Rob's explicit decision.**
Authors: Window Inference project (Rob / Cruffalo Crill). Status: characterisation
complete on the evidence below; the runtime governor it motivates is designed,
not yet built.

## Abstract
Integrating fast Tilelang FP4 expert kernels into a verified streaming
implementation of DeepSeek-V4-Flash-0731 produced fluent-looking failure: the
model's greedy output flipped and its hidden-state norm grew ~100x, with no
error anywhere. We localise the cause to a single rounding decision —
`act_quant`'s ue8m0 (E8M0, power-of-two) activation scales — and characterise
the failure as a three-act process: (1) a small, systematic per-layer shrink
of the hidden state (norm ratio ~0.98) accumulates through layers 20-29;
(2) the drift, not any layer's local error, flips a borderline router at
layer 37; (3) layer 38's router collapses onto a single expert and the
representation explodes. Replacing only the power-of-two scales with exact
FP32 scales restores baseline behaviour exactly (identical 10-token greedy
sequence), at a net speed LOSS — the fast path's speed and its correctness
are the same rounding decision. We propose a closed-loop fidelity governor:
an offline per-layer drift-sensitivity map plus a live hidden-norm envelope
guardrail that falls back to the exact path per layer, per token.

## Setup
- Checkpoint: deepseek-ai/DeepSeek-V4-Flash-0731 (284B MoE, 43 layers, 256
  routed experts/layer, top-6, FP4 experts with E8M0 scales over blocks of 32;
  config: quant_method fp8 e4m3 dynamic for other weights, scale_fmt ue8m0).
- Reference implementation: hand-written streaming forward pass ("Window
  Inference"), GPU FP32 dequant path, verified against DeepSeek's Tilelang
  reference kernels per-expert (cosine 0.99979 expert 35, 0.99907 expert 59,
  0.999997 shared expert, layer 5; HC Sinkhorn agreement ~1e-7).
- Baseline (exact path, token 19923 "Hello", pos 0): greedy 14 (","),
  logit 17.32, final hidden norm 218.658, top-3 [14, 3362, 21133] within
  0.03 (a near-tie that makes greedy flips a sensitive drift detector).
- Approximate path: Tilelang FP4/FP8 expert GEMM with act_quant as shipped
  in DeepSeek's kernel.py (block 128, ue8m0), all other components identical.

## 1. A red herring, disposed of: the layer-5 spike is the model, not a bug
The exact path shows a hidden-norm jump at layer 5 (48 -> 540, growing to
~8,947 by layer 42), driven by routed expert 35 (output norm 228.5 vs ~6-7.5
for its peers; routing weight 0.888-0.948 across prompts). Tilelang's own
kernels reproduce it (224.54, cosine 0.99979 vs dequant), as does a
nibble-order control (swapping FP4 nibble order collapses expert 35 to 6.47,
and DeepSeek's converter stacks low-nibble-first, against the bug
hypothesis). Conclusion: large-magnitude "load-bearing" experts are genuine
behaviour of this checkpoint. Any drift analysis must be differential
against the exact path, never absolute — the model's healthy state is
already extreme.

## 2. The integration failure
With Tilelang experts everywhere: greedy 201 ("\n") vs baseline 14; final
hidden norm 23,844.6 vs 218.658. Per-layer norms track the exact path
within ~0.5-3% through layer 35, then +4.3% (L36), +12.9% (L37), +836%
(L38), remaining exploded through L42. Routing is identical to the exact
path through layer 36; at L37 the selected set changes (exact
[93,168,166,214,237,136] -> approx [93,99,2,237,80,227]); at L38 the
router collapses (expert 152 at weight 1.489, all others <=0.009) and the
norm reaches 69,486 vs the exact path's 7,424.

## 3. Localisation: drift through layers 20-29 is sufficient; the flip is downstream
Hybrid sweep (Tilelang below a cutoff layer, exact above; one token per
run; cutoff 0 control reproduces the baseline bit-for-bit):
- Cutoff 20: STABLE. Final norm 217.9; per-layer ratios 0.971-0.996;
  greedy 21133, a member of the baseline top-3 near-tie.
- Cutoffs 24, 27, 30, 37, 38: ALL BLOWN, final norms ~23.8-24.5k,
  identical L37 flip and L38 collapse — including cutoff 37, where layers
  37+ run the exact path. The strong prior prediction ("exact layers
  downstream of the drift will recover the baseline") FAILED: accumulated
  drift through layer 36 alone flips layer 37's router, and layer 38's
  collapse is path-independent given the diverged state.
- Bisect (24/27) narrows the fatal accumulation band to layers 20-29,
  where blown runs show a systematic norm ratio ~0.979-0.984 vs the
  stable run's 0.985-0.993: a consistent ~1.5-2% per-layer shrink.

## 4. Mechanism isolation: it is the power-of-two scale rounding
A/B tests on identical real inputs at layers 5/22/30/38 (paths: A exact
dequant; B Tilelang as shipped; C Tilelang GEMM on BF16 activations,
no act_quant; D BF16-cast control; E act_quant with exact FP32 scales):
- B vs A per-expert cosine 0.9993-0.9999 — locally excellent, globally
  fatal. The failure is accumulation, not local error size.
- Path D is cosine 1.0 everywhere: the BF16 cast is ruled out.
- The act_quant round-trip alone reproduces B's deviation; path E
  (exact scales) collapses it: layer 5 cosine 1.0 / norm 0.99871
  (= D exactly), layer 30 norm ratio 1.00121 (vs B's 0.99355),
  layer 22 max_abs_err 0.243 -> 0.088.
- Mechanism: ue8m0 scales are powers of two, so each block's activation
  scale is rounded to a coarse grid, biasing block magnitudes (here,
  systematically downward); 43 layers of experts compound the bias until
  it exceeds a router margin. FP8 quantisation per se is exonerated —
  the FP8 shared expert and FP8 paths agree to cosine >=0.9999.

## 5. Resolution and its price
Tilelang + exact FP32 scales ("v8e"): one-token validation greedy 14 =
baseline, final norm 219.37 vs 218.66, per-layer norms within ±0.3%,
routing identical at ALL layers; full 10-token generation identical to
the exact path token-for-token. But v8e is ~22% SLOWER than the exact
dequant path (29.5 vs 24.0 s/token in the then-current build): the exact
scales remove the kernel's speed advantage. The fast path's speed and
its drift are one and the same rounding decision; there is no free
configuration of it. (Production therefore remains the exact path —
since further optimised by partial residency to 13.81 s/token with
bit-identical output.)

## 6. Proposal: a closed-loop fidelity governor
Static sensitivity allocation (the quantisation literature's answer)
cannot see the input-dependent quantity that actually predicts failure:
the live hidden state's deviation from its exact-path envelope, which in
our data is a clean signal (stable 0.971-0.996; blow-up ~100x). Design:
(1) offline per-layer drift-sensitivity map, built by calibrated drift
injection (the A/B rig) scored by downstream flip probability (the sweep
rig); (2) allocate the approximate kernel only to insensitive layers;
(3) live guardrail — track the running hidden-norm ratio against the
envelope; on breach, recompute the current layer exactly and mark the
layer sensitive for the rest of the session. Every component of this
design is falsifiable with the rigs that produced sections 3-4.

## 7. Supporting observation: near-single-expert late routing
In the exact model, layers 29-41 routinely route 92-98% of weight to a
single expert (top-1 share median 0.617 across score layers), the other
five carrying <=0.018 each. "Passenger" experts below weight 0.05 are
18% of streamed expert bytes in the pilot trace (39% below 0.1) — the
basis of a separate pruning experiment, and part of why single-expert
router collapse at L38 is so destructive when it picks the WRONG expert.

## Limitations
- All end-to-end evidence is one prompt family at position 0 (token
  19923) plus a 10-token greedy continuation; the A/B mechanism tests
  span 4 layers. A multi-token trace corpus (~200 tokens, top-16 router
  scores) is built and queued to widen sections 3 and 7.
- Greedy decoding on a near-tie baseline is a deliberately sensitive
  detector; sampled decoding would show graded, not binary, damage.
- One checkpoint, one hardware target (RTX 4060), one kernel library
  version (Tilelang 0.1.15, CUDA 12.8.1).

## Evidence inventory (internal)
CUDA_TILELANG_REPORT.md; V8_VALIDATION_REPORT.md; V8_ROUTING_DIVERGENCE_
DIAGNOSIS.md; V8_SWEEP_REPORT.md; V8_BISECT_REPORT.md; ACTQUANT_AB_
REPORT.md; validate_sweep_{0,20,24,27,30,37,38}_result.json;
actquant_ab{,2,3}_result.json; validate_v8e_result.json;
INVENTION_PILOT_REPORT.md — all in ~/workspace/window-inference-real/.
