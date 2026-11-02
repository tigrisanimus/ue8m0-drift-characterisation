# act_quant A/B + v8e exact-scale fix — mechanism found, one-token validation PASSED
## Nov 1, 2026, 18:45–18:53 EST, Windows PC idle

Scripts: actquant_ab.py (paths A/B/C/D), actquant_ab2.py (path C with
exact FP32 scales), actquant_ab3.py (adds path E), phase4_generate_v8e.py.
Tasks (all exit 0): WindowInferenceActQuantAB, WindowInferenceActQuantAB2,
WindowInferenceActQuantAB3, WindowInferenceValidateV8E.
Raw: actquant_ab_result.json, actquant_ab2_result.json,
actquant_ab3_result.json, validate_v8e_result.json (local + PC).

## Paths (identical real inputs/weights per layer; layers 5, 22, 30, 38)

- A: production dequant FP32 (baseline)
- B: Tilelang as shipped in v8 (act_quant ue8m0 + fp4/fp8 GEMM)
- C: dequant FP32 weights + activations round-tripped through act_quant
- D: dequant FP32 + activations only rounded to BF16
- E (ab3): Tilelang GEMM with EXACT FP32 activation scales and weight
  scales converted E8M0 -> FP32 (scale_dtype=float32 for both)

## Findings

1. act_quant reproduces Tilelang's deviation in the fatal band.
   Layers 22/30/38: B_vs_C cosine 0.99983–0.99992; at layer 30 the
   norm shrink matches (B 0.99355 vs C 0.99377). Path D (BF16 only)
   is cosine 1.0 everywhere — the BF16 cast is ruled out.
   (Layer 5 is a partial exception: C overshoots, norm 1.025 vs B's
   0.993; act_quant is the dominant mechanism in layers 20–38, not
   necessarily the only contributor at layer 5.)
2. The specific culprit is the ue8m0 power-of-2 scale rounding, not
   FP8 quantisation itself. With exact FP32 scales (ab2 path C):
   layer 5 C_vs_A cosine 1.0 / norm 0.99871 (= path D exactly);
   layer 30 norm ratio 1.00121 — the systematic shrink is gone;
   layer 22 max_abs_err falls 0.243 -> 0.088.
3. Path E (real Tilelang GEMMs, exact scales both sides) confirms it
   end-to-end at expert level: layer 5 cosine 1.0, norm 0.99736
   (B: 0.99988 / 0.99317); layer 30 norm 1.00145 (B: 0.99355);
   layer 38 cosine 0.99979 (B: 0.99958). No shrink anywhere.

## v8e one-token validation (token 19923, pos 0, ALL 43 layers Tilelang)

phase4_generate_v8e.py = v8 sweep generator with _tl_linear switched
to exact FP32 scales (act_quant scale_fmt=None/FP32; weight scales
.float(); gemm scale_dtype=float32), cutoff 43.

- greedy_next 14 (",") — EXACT v7 baseline (v8 ue8m0: 201).
- Final hidden norm 219.367 vs v7 218.658 (ratio 1.003).
- Per-layer hidden norms track v7 within ±0.3% at every layer
  (worst: layer 4, 0.9973). v8 ue8m0 blew up to 23,844.
- Routing expert SETS identical to v7 at all 43 layers, including
  layer 37 ([93, 168, 166, 214, 237, 136]) and layer 38 — no flip,
  no collapse.
- Top-3 logits: 14 (17.316), 21133 (17.141), 3362 (17.122) vs v7
  14 (17.316), 3362 (17.293), 21133 (17.291) — same set, near-tie
  ordering differs by <=0.17 logit.
- Wall 30.1 s for the single token vs v7 26.2 s — this INCLUDES
  first-call JIT of the new FP32-scale kernel variants, so it is not
  a speed verdict. The MoE-step saving measured for the ue8m0 path
  was ~1.6 s/token; whether the FP32-scale GEMM keeps that speed is
  measured by the multi-token generation comparison (running).

## Generation comparison (task WindowInferenceGenCompare, 18:55–19:05 EST)

"Hello world" + 8 greedy tokens, v7 control vs v8e, same code path.

- v7 control: FULL SEQ [19923, 2058, 3, 3955, 361, 582, 28411,
  12539, 3362, 582], text 'Hello world!",\n    "greeting": "',
  steady ~24.0 s/token, wall 246.7 s.
- v8e: greedy choice IDENTICAL to v7 at both prefill positions and
  GEN 0–6 (3955, 361, 582, 28411, 12539, 3362, 582). It then crashed
  printing the GEN 7 line: UnicodeEncodeError, cp1252 console under
  the hidden task could not encode a character in GEN 7's top-3 text.
  That is a print-path encoding bug, not a model failure; GEN 7's
  token is not in the logs (print precedes the log write). The UTF-8
  rerun (WindowInferenceGenV8ERerun, 19:06–19:11 EST, exit 0)
  completed cleanly: v8e FULL SEQ [19923, 2058, 3, 3955, 361, 582,
  28411, 12539, 3362, 582] — IDENTICAL to the v7 control, same full
  text 'Hello world!",\n    "greeting": "'. Wall 304.4 s vs v7
  246.7 s. Multi-token sequence agreement: PROVEN.
- SPEED VERDICT (steady state, GEN tokens): v8e ~29.2–29.8 s/token
  vs v7 ~24.0 s/token. v8e is ~22% SLOWER per token in production
  conditions despite the isolated MoE GEMM being faster — the
  FP32-scale kernel path plus per-call scale conversions cost more
  at batch 1 than the GEMM saves. Correctness fix proven; material
  speedup NOT proven — the opposite. v8e therefore fails the
  promotion test in body step 5 and is NOT promoted.

## Status / guardrails

- v8e is NOT promoted. Production phase4_generate.py remains v7
  dequant. Promotion rule stands: not on single-token evidence.
- Next: short generation comparison ("Hello world" + 8 greedy tokens,
  v7 control vs v8e, task WindowInferenceGenCompare) for sequence
  agreement + steady-state token times. If that passes, a harness
  run on v8e is the fuller validation before any promotion decision.
