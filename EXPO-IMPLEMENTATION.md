# EXPO-FT on the Ultra GEN2: implementation design (phase 1)

EXPO-FT (Dong, Hung et al. 2026, arXiv 2605.25477) RL-finetunes a VLA with an
**on-the-fly (OTF) policy**: sample N base action chunks from the VLA, add N learned
*edits*, and execute whichever of the 2N the critic scores highest. This document
describes the port to our π0.5 policies (trained with train-time RTC in lair-openpi) on
the GEN2 bimanual robot. Phase 1 freezes the VLA and trains only the critic and the edit
policy. VLA BC updates come later, behind a flag.

Everything new is in `expo_ft/ultra/` (this repo). The server change is a small hook in
`~/projects/policy-inference-spec/server/server_openpi.py`.

---

## 1. Module map

| File | What it is | Reused from the authors' code |
|---|---|---|
| `expo_ft/ultra/config.py` | `ExpoFtConfig`: every hyperparameter, the 25-D layout, edit scales and mask | Table 3 defaults |
| `expo_ft/ultra/edits.py` | Knot-parameterized edits: interpolation, prefix envelope, carry-over | — |
| `expo_ft/ultra/networks.py` | Critic encoder (swappable), REDQ critic, edit policy, tanh-Gaussian | `ResNetV2Encoder`, `MLP`, `Ensemble`, `StateActionValue` |
| `expo_ft/ultra/learner.py` | Pure-JAX `select` (OTF), `update` (critic TD, edit actor, temperature) | Structure of `EXPOLearner._update_jit` / `sample_batch_actions` / `update_edit_actor` |
| `expo_ft/ultra/sampler.py` | `RtcCandidateSampler`: N VLA chunks sharing one prefix pass, RTC prefix pinned | `Pi0.sample_actions`, `server/openpi_latent.py` split |
| `expo_ft/ultra/buffer.py` | Decision-point replay buffer with the per-observation candidate cache | — |
| `expo_ft/ultra/episode.py` | `EpisodeRecorder`: server events → transitions, human-takeover splicing | — |
| `expo_ft/ultra/rewards.py` | Outcome labels (dsrl_pi0 keypress), `SuccessDetector` interface | `read_outcome_keypress` from `dsrl_pi0/examples/train_agent_server.py` |
| `expo_ft/ultra/normalize.py` | Robot ↔ normalized action space (same formula as openpi `Normalize`) | — |
| `expo_ft/ultra/demos.py` | Minimal LeRobot v3 reader, demos → decision windows | — |
| `expo_ft/ultra/runtime.py` | `ExpoRuntime`: the in-server actor and learner, persistence | — |
| `expo_ft/ultra/server_hook.py` | CLI flags, runtime construction, warmup | — |
| `scripts/expo_dry_run.py` | Full loop on recorded data through the real server | Seam metric idea from `rtc_rollout_replay.py` |
| `tests/ultra/*` | Unit tests per component, plus checkpoint tests for the sampler | — |

The authors' `expo_ft/distributions` (tensorflow_probability) and `utils/augmentation.py`
(augmax) are **not** used: neither package is in the lair-openpi venv. `networks.py` has a
self-contained tanh-Gaussian (tested against the change-of-variables density), and
`learner.random_shift` is a DrQ random shift.

---

## 2. Process layout

There is one process and one GPU. The EXPO-FT actor and learner live inside
`server_openpi`, so the VLA is loaded once and everything is JAX. Training is synchronous
and per episode: the server trains while it handles the `done` frame and acknowledges it
only afterwards, so the robot cannot start the next episode until the updates are done.

```
robot (GEN2 client, fixed-step async, steps/inference = C = 20)
   │ request every C steps: obs + action_prefix + prefix_change_start=d
   ▼
server_openpi (unchanged: validation, _resolve_prefix, action limiter, truncation, chunk_id)
   │  policy.predict  ──replaced by──►  ExpoRuntime.predict
   │                                      ├─ RtcCandidateSampler: N=8 base chunks, d rows pinned  (~93 ms)
   │                                      └─ ExpoLearner.select: + 8 edits, argmax min-of-2 target Q
   │  _respond (limiter)  ─────────────►  ExpoRuntime.record_sent(chunk as sent)
   │  endpoint=intervention  ──────────►  ExpoRuntime.on_intervention
   │  endpoint=done  ──────────────────►  ExpoRuntime.on_done: label (keypress) → buffer →
   │                                      candidate fill → 4 × update(UTD 20, B 64) → ack
```

Lock order: the server's `_PREDICT_LOCK` is always taken before the runtime's own lock,
both around `predict` and inside `on_done`. Training therefore never overlaps a forward
pass, and the two locks cannot deadlock.

---

## 3. Chunks, RTC prefix and what "the action" is

* H = 50 rows per VLA chunk; **C = 20** rows executed per decision. This matches the
  client's "Fixed step Asynchronous" mode with Steps per inference = 20 (0.4 s per
  decision at 50 Hz). The paper uses C = 4 or 8 at 10 Hz (0.4 or 0.8 s), so this is the
  same wall-clock replanning as their C = 4.
* At decision k the observation is taken at step t_k. Rows [0, d) of the new chunk are
  the **committed prefix**: rows of the previous chunk that execute while inference runs.
  `_resolve_prefix` provides them, unchanged. All N candidates are sampled with those d
  rows pinned (train-time RTC, exactly as in `Pi0.sample_actions`). d = 0 on the first
  request, after an intervention, and after a resume gap.
* **The critic's action is the C executed rows [0, C)**, including the prefix rows. It is
  flattened to 20 × 25 = 500 normalized values. Under fixed-step async these rows are
  exactly what the robot runs between o_k and o_{k+1}, because the prefix rows of chunk
  k+1 equal rows [C, C+d′) of chunk k.
* The next observation's prefix (d′ rows, committed at decision k+1) is stored with that
  observation. The TD target samples its candidates with the same prefix pinned.
* **Normalized space everywhere.** Base chunks, edits, the critic's inputs and the buffer
  are all in the VLA's quantile-normalized space, computed from the checkpoint's norm
  stats. Human rows and sent chunks are normalized with `ActionNormalizer`, whose
  agreement with openpi's transform pipeline is tested on the real checkpoint. The chunk
  that goes to the robot is unnormalized through the policy's own output transforms.

---

## 4. On-the-fly selection (paper Sec. 3, Eq. 3)

`ExpoLearner._otf_choose` runs the same code at rollout time and inside the TD target:

1. Take N = 8 base chunks (pinned prefix).
2. Make N_edit = 8 edited copies of the first 8 base chunks, one fresh edit each.
3. Q = REDQ min over a random 2 of the 10 **target** critics, for all 16.
4. Execute (or bootstrap from) the argmax.

At rollout, the edit policy's input features come from the target critic's encoder too.
That way one encoder pass on one observation serves the whole decision.

---

## 5. Edit policy

**Distribution.** A tanh-Gaussian over u ∈ [-1, 1]^(K·E). Its input is the critic's
image features plus the state embedding, under `stop_gradient` (paper B.2: reusing critic
features works best), concatenated with the base chunk's executed rows (500-D). The MLP
is (256, 256, 256).

**Knot parameterization** (the default; `edit_knot_stride=1` gives one offset per row):
* K = 5 knots at rows `linspace(0, C-1, K)` = 0, 4.75, 9.5, 14.25, 19, and E = 16
  edited dims. The edit is 80-D.
* Rows are linearly interpolated between knots. **Every row ≥ C−1 holds the last
  knot's offset** (carry-over), so the next decision's prefix carries the offset the
  robot is executing and the next chunk starts from there instead of snapping back.
  Tested: edited − base is constant on rows ≥ C and equal to row C−1.

**Scale and mask** (per dimension, in normalized units; 0 means the dimension is not
part of the distribution at all):

| Dims | Indices | Default scale |
|---|---|---|
| left/right arm joints | 0:7, 8:15 | `edit_scale_arm` = 0.1 |
| left/right gripper | 7, 15 | `edit_scale_gripper` = 0.2 |
| chest | 16:22 | `edit_scale_chest` = 0 (masked) |
| neck | 22:25 | `edit_scale_neck` = 0 (masked) |

The paper's range is 0.05–0.2. In normalized units 0.1 is 5 % of a joint's q01–q99 range.

**Prefix envelope.** The edit is multiplied by 0 on the committed rows [0, d), then fades
in linearly over `edit_ramp_rows` = 5 rows (100 ms). Without the ramp, the first free row
would step by up to the full scale in one 20 ms tick. The ramp also applies when d = 0.

**Loss** (Eq. 4): `E[α·log π(û|s,a) − mean_ensemble Q(s, a + û)]`, with a ~ one of the
cached base samples at s. log π includes `−K·Σ log scale_e`, the change of variables
u → scale·u, the same correction as the authors' `log_probs -= dim·log(edit_scale)`.

**Temperature.** α0 = 1. The target entropy defaults to −edit_dim/2 (the authors'
default, `adjust_target_entropy=False`). The scaled edit's entropy sits far below that
target, so α decays steadily, which is the authors' behaviour. `target_entropy` is
configurable.

---

## 6. Critic

**Encoder** (Appendix D.1). A ResNetV2 trunk with stage sizes (3,4,6,3) and 64 filters
(the authors' `ResNetV2Encoder` with GroupNorm), trained from scratch. **One trunk with
shared weights** runs on each of the 3 cameras. Each camera then goes through Dense 512
→ LayerNorm → tanh, and the three are concatenated. The state passes through Dense 64 →
LayerNorm → tanh. Images are the cameras letterboxed to 224 with the VLA's own
`resize_with_pad`, stored as uint8, with a ±8 px random-shift augmentation during
training. The encoder is the `CriticEncoder` protocol: a pretrained backbone (e.g. the
value model's) plugs into `build_critic(cfg, encoder=...)`.

**Heads.** A REDQ ensemble of 10 MLPs (256, 256, 256) with LayerNorm, over
[features, 500-D chunk].

**TD target** (Eq. 5): `y = r + γ^n · mask · min_{2 of 10} Q′(s′, a*)`, where a* is the
OTF argmax at s′ over the cached base candidates at s′ plus 8 fresh edits. No entropy
term in the backup, as in the paper. Normally n = C = 20, so γ^n = 0.999^20 = 0.980 per
decision. n differs only around human takeovers and the episode end (§8).

**Target network.** Deviation from the authors' code: the encoder sits inside the
critic's parameters, so the target critic has its own Polyak-averaged encoder
(τ = 5e-3, updated every critic step). The authors re-encode s′ with the online encoder.
With the frozen-target encoder, the TD target and the rollout run entirely on the target
network.

---

## 7. Update schedule

`ExpoLearner.update` is one jitted call:
* `utd_ratio` = 20 critic steps (a `lax.scan`) over 20 minibatches of 64.
* Then 1 edit-actor step and 1 temperature step on the last minibatch.

This is the authors' order. The runtime calls `update` `updates_per_episode` = 4 times
in `on_done` (Table 4: 1–6), after the episode and any missing candidates are in the
buffer. Training starts once the buffer holds ≥ 64 transitions; with demos that is
immediate. Adam with lr 3e-4 is used everywhere.

---

## 8. Episodes, rewards and human takeovers

**Labels.** This is dsrl_pi0's flow (`read_outcome_keypress`). On `done` the server
blocks on the operator's terminal: **2 = success, 1 = failure, 0 = abort**. A
disconnect discards an unlabelled episode.

One change of meaning from DSRL: abort now means "not the policy's fault" (reset problem,
hardware, operator stop). Unsafe or degenerate behaviour is a **failure**. The keypress
prompt says this.

**Reward.** Sparse, terminal-only, binary {0, 1}. Every return is ≥ 0, so no early
ending can outrank a success; the length problem DSRL hit with −1 per chunk cannot occur.

| Outcome | Last transition |
|---|---|
| success | r = 1, mask = 0 |
| failure | r = 0, mask = 0 |
| abort (`abort_mode="truncate"`) | r = 0, **bootstraps** from the terminal observation with γ^n (its candidates are sampled at episode end with d = 0). With no terminal observation the last transition is dropped. |
| abort (`abort_mode="discard"`) | episode dropped |

`SuccessDetector` is the interface for automatic labels. `HumanLabel` (the default) and
`FixedLabel` (dry runs) implement it.

**Recorder.** `EpisodeRecorder` receives three kinds of events in arrival order:
* policy chunks, as sent after the limiter;
* intervention frames: an observation plus the human's rows;
* `done`, with the terminal observation when the robot sends one.

It lays the executed rows end to end into one 50 Hz timeline. How many rows of an event
ran is taken from the next event's `prev_skipped_action_start` when present. Otherwise
it is C for a policy chunk (fixed-step async) and all rows for a human frame.

**Anchors** are the observations that become buffer slots:
* every policy decision;
* the first frame of a takeover, then one every `hil_anchor_stride` (= C) rows of it;
* the terminal observation.

A transition starts at each anchor (t). Its action is the timeline rows [t, t+C). It
bootstraps at the latest anchor at or before t + C, with discount γ^n for the n rows in
between. Between two policy decisions n = C, which is Eq. 5 exactly.

**The splice.** If the human takes over k rows into a chunk, that chunk's transition
gets k policy rows followed by the C−k human rows that overrode the rest, and bootstraps
at the takeover observation with γ^k. Human anchors have d = 0, and their base candidates
are sampled at episode end ("candidate fill"). The first request after a takeover carries
no prefix; the server already drops it.

**Assumption to confirm against the GEN2 client:** the intervention frame's observation
is taken when its first human row starts executing, and its `action` rows are what the
robot then executed until the next event. If the client sends different spans, only
`EpisodeRecorder._lay_out_timeline` needs to change.

---

## 9. Replay buffer and the candidate cache

**Storage** is per observation slot. Each slot holds:
* 3 × 224 × 224 × 3 uint8 images;
* the state, normalized and in robot space;
* the prefix d and its robot-space rows;
* **N = 8 cached base chunks, rows [0, C)**.

A transition is stored in its source slot and points to its bootstrap slot, so
next-observation images are never duplicated. Memory is about 0.45 MB per slot; the
default capacity of 60 k slots is about 27 GB, allocated lazily.

**Candidate cache (frozen VLA only).** The N base samples drawn at s_{t+C} during the
rollout are exactly the TD-target base candidates for the transition that ends there,
since they were drawn at that observation with its prefix pinned. They are kept, not
redrawn. Edits are resampled fresh on every update.

Slots without candidates are filled in one pass at episode end or at seeding, through
the same sampler: human anchors, abort-terminal observations and demos.
`cache_base_candidates` must be False once `train_vla` is on; the config asserts this.

**The same cache supplies the edit policy's training base:** a random one of the 8
cached chunks at s. Following Eq. 4 this is a sample from π_base, not the executed action
the authors' code uses.

**Eviction.** Demos sit in a protected region at the front of the buffer. Online
episodes fill a ring after them, and the oldest whole episode is evicted first.
`demo_ratio` (default 0: uniform) can oversample demos.

---

## 10. Demos

`--expo-ft-demos <LeRobot v3 root>` seeds `--expo-ft-num-demos` (default 20; paper
10–40) successful episodes of the task, up to `--expo-ft-demo-max-len` frames.

* Each demo is cut into windows at stride C: the observation at the window start, the
  next C recorded actions as the chunk, and the episode end as a success.
* The prefix d is drawn from the online distribution: U[0, 15), and d = 0 on the first
  window. The window's first d recorded rows act as the committed prefix, so the
  candidate fill samples the VLA with those rows pinned.
* Demos go through the same `EpisodeRecorder` as online data.

The reader is a minimal parquet + PyAV implementation, because the lerobot package in
the lair-openpi venv predates v3. On cubestack, 911 of 989 episodes are `success`;
"stack cubes" accounts for 460 of them, with a median length of 556 frames.

---

## 11. Running it

Server (lair-openpi venv; both repos on `PYTHONPATH`):

```bash
cd ~/projects/policy-inference-spec
PYTHONPATH=.:$HOME/projects/expo-ft XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
python -m server.server_openpi \
  --checkpoint-path models/openpi-ultra-cubestack-pi05-rtctrain-15000 \
  --rtc-train --task "stack cubes" --host 0.0.0.0 \
  --expo-ft \
  --expo-ft-demos ~/projects/ultra-tools/data/cubestack/lerobot --expo-ft-num-demos 20 \
  --expo-ft-set exec_steps=20 edit_scale_arm=0.1
```

Client: Rollout Hz 50, Fixed step Asynchronous, **Steps per inference = 20** (must equal
`exec_steps`), Chunk execution limit 50.

| Flag | Meaning |
|---|---|
| `--expo-ft-set KEY=VALUE ...` / `--expo-ft-config file.json` | `ExpoFtConfig` overrides |
| `--expo-ft-label human\|reason\|success\|failure\|abort` | keypress (default), the client's done_reason, or fixed |
| `--expo-ft-base-only` | execute base sample 0 (no selection, no edits) but still record and train |
| `--expo-ft-run-dir` | default `~/exp/expo-ft/<timestamp>`: `config.json`, `metrics.jsonl` (one line per episode, with every update's losses), `episodes/episode_*.npz`, `checkpoints/state_*.msgpack` every 10 episodes |
| `--expo-ft-resume state.msgpack`, `--expo-ft-resume-episodes <run>/episodes` | continue a run |

---

## 12. Dry run (before the robot)

`scripts/expo_dry_run.py` starts the **real** server with `--expo-ft` and drives it over a
local websocket with a simulated fixed-step-async client. The client replays recorded
LeRobot episodes:

* a request every C steps, with `action_prefix` = the previous chunk past row C and a
  simulated delay d ~ U[4, 12];
* scripted takeovers: intervention frames of the recording's own actions, 7 rows into a
  chunk, followed by a resume with no prefix;
* `done` with the terminal observation and a scripted label (success/failure cycle).

The server stores each episode, fills candidates and runs the 4 updates, exactly as on
the robot. Observations come from the recording (open loop), so the dry run tests the
plumbing, timing, learning dynamics on real images and chunk seams, not policy quality.

`--compare` runs a second pass with `--expo-ft-base-only` and prints both **seam
ratios**. The seam ratio is the mean |Δ| at the first freshly sampled row of each chunk
divided by the mean |Δ| elsewhere, over arms and grippers. It is the offline check that
the edits and their carry-over do not add jumps.

Results of the first run are in §15.

---

## 13. Tests

Run with `PYTHONPATH=.:$HOME/projects/policy-inference-spec pytest tests/ultra` (lair-openpi venv).

| File | Covers |
|---|---|
| `test_edits.py` | config geometry (80-D edit, masks, γ^C); interpolation weights; prefix envelope/ramp; **edit bounds** (never beyond the per-dim scale), **mask** (chest/neck untouched), **prefix rows untouched**, **carry-over** (rows ≥ C equal row C−1); stride 1 = per-row; log-scale correction |
| `test_learner.py` | tanh-Gaussian log-prob vs. change of variables; network **shapes**; `select` shapes and candidate structure (base copies, edit bounds, pinned prefix, carry-over); **OTF argmax** (planted Q with a known best candidate; an edit wins when it is better); **TD target** (constant Q; argmax candidate + REDQ min-of-2 over a planted ensemble; terminal masking); `update` fits rewards; temperature direction; target-network Polyak |
| `test_episode_buffer.py` | recorder: policy-only episodes, **HIL splice** (k policy rows + human rows, γ^k), a takeover window running into the resume chunk, dense frames subsampled to the anchor stride, abort truncate/discard, zero-row done; buffer: next-observation linking, terminal handling, missing-candidate fill, FIFO eviction with protected demos; normalizer formula |
| `test_runtime.py` | the runtime loop with a fake VLA: predict → record → takeover → done → fill → train; the prefix stays pinned through selection and edits; metrics log; episode files round-trip; abort bootstrap; disconnect discards; base-only mode |
| `test_sampler.py` (checkpoint) | candidates equal `Policy.infer` for the same noise (batch-N KV-shared vs batch 1); **pinned prefix**; `ActionNormalizer` equals openpi's transforms; latency (N = 8: **93 ms median, 109 ms max**) |

The existing `policy-inference-spec/tests/test_server_openpi.py` still passes with the
hook in place (14 passed, 3 skipped).

---

## 14. Deviations, assumptions, open items

**Deliberate deviations from the paper or the authors' code:**
1. **Chunk length.** C = 20 at 50 Hz, with knot edits (80-D) instead of a full per-row
   edit (C × 7 dims at 10 Hz in the paper; here a full edit would be 400-D). Per-row
   edits remain available with `edit_knot_stride=1`.
2. **Edit carry-over.** The last offset is held on rows ≥ C, and the edit is zeroed on
   the RTC prefix and ramped in after it. The paper has no RTC.
3. **Target encoder.** The critic encoder is part of the target network (Polyak),
   rather than re-encoding with the online encoder.
4. **Edit actor base.** A cached π_base sample at s (as in Eq. 4) rather than the buffer
   action.
5. **Discount.** γ = 0.999 per 50 Hz step, about a 20 s horizon. Our episodes run
   10–25 s, longer than the paper's (≈ 10 s at γ = 0.99, 10 Hz).
6. **Abort handling.** Abort is a truncation, not a terminal (see §8).
7. **Cameras.** All 3 cameras through one shared trunk. The paper has 2 cameras.

**Assumptions to verify on the robot:**
* The intervention frame semantics (§8).
* That "Steps per inference" counts executed rows between requests, so policy → policy
  transitions are exactly C rows. The recorder also accepts `prev_skipped_action_start`
  on requests if the client sends it.
* Latency: sampling takes 93 ms. Selection plus host overhead was measured in the dry
  run (§15). Everything must stay well under d_max × 20 ms = 280 ms.

**Phase 2 (not implemented):** `train_vla=True` (asserts `cache_base_candidates=False`)
is where π0.5 BC updates on successful and intervention data would go. `tau_actor` is
already in the config.

---

## 15. Dry-run results (2026-10-22, cubestack checkpoint, 4 replayed episodes, 3 seeded demos)

Both passes completed end to end with no errors. Each pass covered:
* demo seeding: 3 demos became 172 transitions, with 172 candidate sets filled at startup;
* 151 policy decisions;
* 2 episodes with a takeover (40 human rows each, 2 anchors filled per episode);
* success and failure labels;
* 4 updates per episode.

| | EXPO-FT (edits on) | base-only |
|---|---|---|
| request round trip, client side (median) | 124–167 ms | 98–170 ms |
| request round trip (max, excluding the one-off below) | 194 ms | 189 ms |
| per-episode training: 4 × (20 critic + 1 actor step) | 17–19 s | 19 s |
| critic loss after episodes 1 → 4 | 0.125 → 0.0095 | 0.132 → 0.007 |
| edit selected (fraction of decisions) | 0.92–0.98 | — |
| seam ratio (mean of 4 episodes) | 1.52 | 1.33 |
| mean \|Δ\| per step inside chunks (normalized) | 0.0065 | 0.0026 |

**One-off stalls (fixed).** The first `update` took about 60 s and one request took
2.4 s. The cause was that the train state's leaf types changed after the first update
(`TrainState.step` started as a Python int, the temperature as a weak float), so
`update` and `select` compiled a second time. `init` now normalizes the leaf types, and
`test_update_compiles_once` checks it. A second dry run after the fix (3 episodes, one
takeover) gave a first-episode training time of 16 s, a maximum request round trip of
211 ms, an update compile of 29 s at warmup, and a seam ratio of 1.37.

**Seams.** The seam ratio with edits (1.52) is close to the base policy's own (1.33),
and both vary 0.8–1.9 across episodes. The carry-over therefore does not create a jump at
the start of each chunk. What edits do add is motion *inside* chunks: 2.5× the per-step
motion. This is the **untrained** edit policy drawing nearly uniform knot offsets
(±0.1 normalized, re-drawn every ~5 rows) for 98 % of decisions. The action limiter
clipped a handful of steps by 0.2–0.6°.

**Things to watch on the robot.**
1. **Early edit jitter.** Until the edit policy's entropy drops, executed chunks carry
   random offsets of up to the edit scale. To be conservative for the first episodes,
   start with `edit_scale_arm=0.05`, or run a few episodes with `--expo-ft-base-only`
   (they still train the critic and fill the buffer).
2. **Edits picked ~95 % of the time.** With a barely trained critic, the argmax over 16
   candidates favours the higher-variance edited ones. `edit_selected` in
   `metrics.jsonl` should fall toward a task-dependent level as the critic learns.
3. **Q is still negative** (≈ −0.7 after 16 updates; the returns are in [0, 1]). It
   starts from the random initialization and moves slowly with τ = 5e-3. Watch `q` and
   `target_q` in `metrics.jsonl`. Clipping the target to [0, 1] would be a cheap,
   non-paper fix if it lingers.

Run directory layout: `<run>/{expo-ft,base-only}/{server.log,metrics.jsonl,config.json,episodes/}`.
