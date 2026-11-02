# Drift-Injection Report — v1 INVALID (in-process carryover); corrected v2 run launched

Date: 2026-11-02. Task WindowInferenceDriftInj: START 09:13:49 EST, END 09:31:40 EST, exit 0, INJ DONE. 43 single-layer Tilelang act_quant injections, token 19923 pos 0, vs exact baseline (validate_v7_result.json: greedy 14, final norm 218.658).

## Raw v1 result (as logged)
- Layer 0: greedy 14 preserved, norm 217.485 (-0.5%).
- Layers 1–7: greedy flipped to near-tie tokens 21133 / 3362, final norms 214–225 (close to baseline).
- Layer 8: greedy 14 preserved, norm 210.361.
- Layers 9–42: greedy flipped (5 / 201 / 7249 / 666 / 42498), final norms 26,766–60,067 (120–275× baseline).

## Why v1 is NOT a valid per-layer sensitivity map
The driver (inj_driver.py) ran all 43 forwards in ONE process, mutating gen.TILELANG_ONLY between calls, with no state reset. Proof of carryover from the per-layer records themselves:

- Layers BEFORE the injected layer must be identical to baseline in a clean single-layer experiment. They are not, and the deviation grows monotonically with run order, not with the injected layer: hidden norm at layer 1 (injected layers 6→42) drifts 41.9 → 41.5 → 40.8 → 39.0 → 38.2 vs baseline 47.0; layer-3 routing set changes progressively across runs (baseline [174,149,203,73,67,69] → [174,217,203,73,201,231] by run 8–9 → [174,201,142,217,143,112] by run 42).
- Layer 0 — the first run in the process, hence uncontaminated — tracks baseline within ~1.5% at every early layer and preserves greedy. That is the only v1 row usable as-is.
- The catastrophic norms from run 9 onward therefore track cumulative in-process state, not layers 9–42 individually. The exact state mechanism (which global/KV/attention state run_one_token leaves behind) is NOT proven and is not claimed here.

Conclusion: governor sensitivity map v2 must NOT be built from v1. The earlier checkpoint reading ("hash layers 1–2 are individually sensitive") is retracted — layers 1–2 flips were near-tie flips in already-drifting runs.

## Corrected experiment (launched this run)
- inj_single.py: one layer per FRESH process; inj_driver2.py spawns a subprocess per layer (0–42), writes inj_results2/inj_L<n>.json + incremental inj_summary2.json on the PC; task WindowInferenceDriftInj2 via run-hidden.vbs, py_compile exit 0, State Running verified ~09:47 EST, PC clear at launch (free RAM 9,993,312 KB, no UnrealEditor, PID 6684 untouched).
- Control expectation: a fresh-process run with no effective drift should reproduce greedy 14 / norm 218.658 at pre-injection layers exactly, as the prune-sweep control (separate process) did.
- Next: capture inj2 terminal state → build governor_sensitivity_map_v2.json from inj_results2 only → then compressed-attention frontier (v12 / window-only 128).

Raw evidence: inj_summary.json + inj_L0…42.json copied locally to window-inference-real/inj/ (v1, retained as evidence of the confound, not as a map).

## Early confirmation from the corrected run (09:52 EST)
WindowInferenceDriftInj2 Running; first subprocesses completed rc 0 (~32 s/layer wall including process start). Fresh-process layer 1: greedy 14 PRESERVED, norm 218.390 — directly contradicting v1's in-process layer-1 row (greedy 21133, norm 223.806) and confirming the carryover diagnosis. Layer 1 alone is, on this evidence, safe.

## v2 corrected run — TERMINAL (exit 0), map v2 built
Task WindowInferenceDriftInj2: START 09:43:59 EST, END 10:07:12 EST, exit 0, INJ2 DONE. 43 fresh-process single-layer injections (inj_single.py subprocess per layer), token 19923 pos 0, baseline greedy 14 / final norm 218.658. Raw: inj_results2/inj_summary2.json (PC + local copy in window-inference-real/inj2/).

### Result
- CATASTROPHIC (2 layers): layer 21 → greedy 201, norm 31,241 (143× baseline); layer 22 → greedy 201, norm 34,790 (159×). Single-layer Tilelang act_quant at 21 or 22 alone destroys the forward. Consistent with the earlier bisect (cutoff 20 stable, cutoff 24 blown — cumulative runs include 21/22).
- NEAR-TIE FLIPS (11 layers, norms within ~0.6% of baseline): to 21133 — layers 6, 8, 20, 24, 27, 38; to 3362 — layers 23, 29, 30, 35, 42. Drift is real but small; the greedy winner flips among near-tied candidates (matches the trace-corpus finding that margins are near-tied everywhere).
- PRESERVED (30 layers): greedy 14, norms within ~0.5%.

### Conclusions
- v1's "layers 9–42 catastrophic" was in-process carryover, confirmed: in fresh processes only 21–22 blow up.
- Governor rule from evidence: Tilelang act_quant path must NEVER be applied at layers 21–22; elsewhere single-layer drift is norm-benign but can flip near-tie greedy choices, so exact-output validation remains per-configuration.
- Map: governor_sensitivity_map_v2.json (validated for this token/position; corpus flip-rate still owed).
