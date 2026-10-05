# CT2MRI-DTE — Project Handoff

**Last updated:** 2026-10-04
**Scope:** everything a new person needs to pick this up — what the project is, what has
been tried, what is true right now, and the traps that have cost us time.

> Dates are given throughout because several facts below have a shelf life. Where a
> number could have moved, the source of truth is named. **Do not trust this document
> over a fresh log.**

---

## 1. What this project is

A CT→MRI brain image-synthesis research fork of **BBDM** (Brownian Bridge Diffusion
Model). The clinical motivation is **deep brain stimulation targeting**: we want a
synthetic MR from a CT good enough to localise small subcortical structures (STN, GPi,
thalamus) when a real MR is unavailable or unsafe to acquire.

Stock BBDM is a generic image-to-image diffusion translator. This fork extends it along
five axes:

1. **Losses (fidelity)** — VGG perceptual loss + differentiable SSIM + focal-frequency
   loss on top of base L1. Computed in fp32 with NaN/Inf guards so AMP doesn't break
   them. `model/losses.py`; weights in `configs/fine-tune.yaml`.
2. **Histogram conditioning (modality-awareness)** — the biggest conceptual change.
   Replaces BBDM's `SpatialRescaler` with a global 128-bin intensity histogram of the
   target MR (histogram + CDF + gradient), fed via cross-attention while the CT is
   channel-concatenated. `condition_key: hist_context_y_concat`, `context_dim: 128`.
   See §7 for why this matters and what it costs us.
3. **ISTA / averaging sampling (consistency)** — beyond DDIM, adds `ISTA_average` /
   `ISTA_mid` plus slice-volume averaging for inter-slice consistency.
   `model/BrownianBridge/BrownianBridgeModel.py`.
4. **Training infrastructure (scale)** — AMP + GradScaler, DDP, gradient accumulation
   and clipping, EMA, `torch.compile`, and a **separate cross-attention learning rate**
   (`cross_attn_lr`) so the conditioning path trains at a different pace than the
   backbone.
5. **Medical data pipeline** — HDF5-backed float16 volume storage, multi-plane
   (axial/coronal/sagittal), BraTS T2F→T1N support alongside the primary CT→MRI task.

---

## 2. Current state in one paragraph

**Tier 3 `top_model_epoch_472.pth` is SOTA** and has been since 2026-08-05. Tier 4
(z-context) was a negative result. The UQ ensemble is not a new SOTA but produces usable
uncertainty maps. Phase 3 (structure-weighted loss, three arms) finished training
2026-09-12 and is **waiting on evaluation** — that is the live piece of work. A
per-structure evaluation harness exists and has been run against Tier 3, giving the
first per-structure numbers (§5).

---

## 3. Results history

All metrics: N=36 test subjects, sampler `normal_200`, η=0, unless stated.

### The tier ladder

Each tier warm-starts from the previous best, **weights only, fresh optimizer**.

| | change | SSIM | masked SSIM | LPIPS |
|---|---|---|---|---|
| baseline ep_235 | histogram BBDM | 0.741 | 0.458 | 0.189 |
| **Tier 1** ep_390 | + perceptual λ=0.05, focal-frequency λ=0.1 | 0.759 | 0.496 | 0.172 |
| Tier 2 ep_412 | + brain-mask loss (3× in-brain) | 0.760 | 0.498 | 0.174 |
| **Tier 3** ep_472 | + SSIM λ=0.1, **morphological closing on the mask** (`mask_close_kernel: 9`) | **0.774** | **0.533** | **0.158** |
| Tier 4 ep_542 | z-context 3→7 slices | 0.759 | 0.515 | 0.201 |

**Only two changes ever moved the needle:** Tier 1's perceptual + frequency losses, and
Tier 3's morphological closing of the brain mask (which fills CSF/ventricle holes that a
bare intensity threshold marks as background). Tier 2 added mask weighting and got
nothing — because the mask it was weighting was full of holes exactly where the deep gray
lives. Total masked-SSIM gain across the ladder: **+16%**.

### Negative and null results — read these before proposing work

- **Tier 4 / z-context 7 lost on every metric** (job 39425432 trained, 39688697 /
  39689316 evaluated). *But* within Tier 4, `average` sampling beat `normal` by +3.2%
  masked SSIM where in Tier 3 the two were tied — so the overlap-averaging mechanism is
  real, it just starts from a worse model. **Confounded:** base LR went 1e-6 → 5e-6
  alongside the z-context change, so it is two variables, not one. Do not conclude
  "z-context doesn't work".
- **UQ ensemble (5 members)** gave +16% masked SSIM over a single member but is roughly a
  wash against Tier 3 for ~25% more compute. Its deliverable is the uncertainty map:
  predictive std is **2.5× higher in-brain** (the signal is real) but only **~38% of
  actual error** — substantially over-confident. Raw std is not usable as an error bar;
  that is what Conformal Risk Control in the roadmap is for.
- **Sampler choice is not a quality lever.** An early sweep did find a free win —
  switching from `ISTA_average/200/η=0` to `normal/50/η=0` took SSIM 0.669 → 0.733 with
  no retraining, because the old sampler was spatially averaging at every step and
  masking the model's real quality. Beyond that fix, sampler changes do nothing.
- **Recurring signature:** mean-averaging cancels sampling noise *and* genuine texture
  together — pixel metrics improve while LPIPS worsens. Seen in the UQ ensemble and in
  Tier 4 `average`.

---

## 4. The SynthSeg structure-aware work (current thread)

### Phase 1 — labels (complete 2026-09-05)

Built ground-truth subcortical labels for all 180 subjects on the training grid. Two live
bugs found and fixed:

- **The training brain mask was excluding half the deep gray.** `mask_threshold: 0.05`
  was compared against `x0` in `[-1,1]`, which is equivalent to thresholding at 0.525 in
  `[0,1]` intensity. Deep gray is darker than white matter on T1, so the broken threshold
  kept bright WM and dropped the structures the project exists to get right. Measured
  deep-gray coverage **0.556 → 1.000**. Fixed in `BrownianBridgeModel.p_losses`.
  **Tier 3 SOTA was trained with this bug**, so some headroom may come back for free.
- `index_dataset` was `uint8`, truncating at 255. Subject `1BB152` has 260 slices,
  desynchronising 53 training subjects. Now `int32`. Rebuilt into `hdf5s/fine_v2`
  (**not** over `hdf5s/fine`).

New pipeline: `brain_dataset_utils/geometry.py` (single source of truth for the transform
chain), `make_labels.py`, `verify_labels.py`, `shell/data/synthseg_array.sh`,
`synthseg_robust.sh`, `label_pipeline.sh`, plus regression tests in
`brain_dataset_utils/tests/`.

**QC manifest** `datasets/label_qc/label_manifest.csv` — 10 of 2160 structure-subject
pairs excluded (0.46%). No subject dropped; test split stays N=36. The manifest is
**frozen** — iterative outlier removal with recomputed statistics never terminates.

### Phase 2 — per-structure baseline (complete 2026-09-11, re-run 2026-10)

Per-structure harness: `runners/export_eval_volumes.py` → SynthSeg eval array →
`runners/eval_structures.py`. Baselined on Tier 3 ep_472, all 432 structure-subject pairs
scored, zero failures. Numbers in §5.

### Phase 3 — structure-weighted loss (trained 2026-09-12, **eval pending**)

Three arms, all fine-tuning from ep_472 with a fresh optimizer, `n_epochs: 532`:

| arm | config | weighting |
|---|---|---|
| **A** control | `phase3_armA_control.yaml` | none — matched budget, isolates the mask-threshold fix |
| **B** uniform | `phase3_armB_uniform5x.yaml` | 5× over the whole structure |
| **E** shell | `phase3_armE_shell5x.yaml` | 5× on a dilate(2)/erode(1) boundary band |

`lambda_deepgray_weight` is **relative to in-brain tissue** and composes multiplicatively
with `lambda_mask_weight`: absolute weights are 1.0 background / 3.0 in-brain / 15.0 deep
gray.

**Results so far:** all three ran 60/60 epochs, zero preemptions, improved at epochs 473
and 476 only, then flat for 56 epochs. Best = `top_model_epoch_476.pth`.

| arm | best val loss | vs A |
|---|---|---|
| A | 0.13670 | — |
| B | 0.14090 | +3.1% |
| E | 0.14159 | +3.6% |

**Val loss cannot rank these arms.** The weight map is not normalised
(`recloss = (diff * loss_weight).mean()`), so adding deep-gray weight shifts the global
loss scale. The preflight predicted shifts of +2.8% (B) and +2.7% (E); the observed
gaps are essentially that. Only evaluation can rank them.

**Arm E is close to a duplicate of arm B.** The preflight reports deep gray as a fraction
of in-brain at `uniform 0.0117` vs `shell 0.0115` — the shell covers **98% of the whole
structure**. Deep-gray structures are only ~5 px across at 256², so a 2-out/1-in band is
the entire structure with no interior left to exclude. A genuine boundary-vs-interior
contrast needs a thinner shell or higher resolution.

**Next step:** Phase 4 sampling, already written and parameterized.

```bash
sbatch --export=ALL,ARM=armA_control   ct2mri_test_phase4.sh
sbatch --export=ALL,ARM=armB_uniform5x ct2mri_test_phase4.sh
sbatch --export=ALL,ARM=armE_shell5x   ct2mri_test_phase4.sh
```

`shell/test/phase4_arm.sh` deliberately evaluates **epoch 532, not 476** — 476 was
selected on a 0.38% validation difference that is plausibly noise and carries only 4
epochs of exposure to the deep-gray term, while 532 has all 60, and all three arms
overfit by the same margin so the comparison stays matched. Override with `EPOCH=`.

Run arm A first; it is the only thing that makes B and E interpretable.

---

## 5. Per-structure numbers (Tier 3 ep_472, N=36, 432 pairs, 0 failures)

| structure | Dice | noise floor | centroid (mm) | volume ratio | Δ voxels | SSIM | CW-SSIM | PSNR |
|---|---|---|---|---|---|---|---|---|
| thalamus | 0.909 ± 0.022 | 0.977 | 0.87 ± 0.34 | 0.999 | +26 | 0.680 | 0.658 | 26.12 |
| caudate | 0.865 ± 0.046 | 0.971 | 1.00 ± 0.63 | 0.948 | −276 | 0.728 | 0.689 | 24.01 |
| putamen | 0.831 ± 0.046 | 0.967 | 1.08 ± 0.47 | 0.874 | −889 | 0.646 | 0.652 | 26.21 |
| **pallidum** | **0.731 ± 0.087** | 0.959 | **1.50 ± 0.61** | **0.839** | −344 | 0.655 | 0.662 | 28.38 |
| hippocampus | 0.818 ± 0.046 | 0.966 | 1.23 ± 0.67 | 0.908 | −545 | 0.583 | 0.608 | 20.95 |
| amygdala | 0.835 ± 0.051 | 0.959 | 0.98 ± 0.45 | 0.899 | −248 | 0.554 | 0.576 | 23.74 |

Six bilateral structures, scored left and right separately (12 labels × 36 subjects =
432). Voxels ≈ 0.75 mm³. Cohort volume ratio **0.911**, median 0.921, **345/432
under-segmented** against a chance expectation of 216.

### What these say together

1. **Dice is not saturated.** The noise floor — real MR re-segmented against the Phase 1
   labels, same anatomy, no synthesis — is 0.966 ± 0.01 against dice_syn 0.831. Mean
   headroom 0.135. An earlier read that Dice was at ceiling was wrong.
2. **Placement is correct.** Centroids are ~1 voxel, and the **CW-SSIM gap is zero**
   (0.641 vs 0.641). CW-SSIM is verified translation-tolerant (0.765 vs 0.636 under a
   2px shift), so a zero gap means there is no misalignment for it to forgive. The
   residual error is intensity and texture *inside* the structure.
3. **Structures are systematically under-filled, and the deficit scales inversely with
   boundary contrast.** Thalamus (most distinct borders) loses nothing; pallidum and
   putamen, which share a weak low-contrast boundary with each other, lose the most.
4. **This explains the pallidum anomaly** — worst Dice and worst centroid yet the *best*
   PSNR (28.38 dB). Mild contrast compression moves where SynthSeg puts a low-contrast
   boundary while pixel error stays small. Clinically the worst place to be weak: GPi is
   a DBS target.

**Use `vol_ratio` as the primary readout for Phase 4.** It is directional, interpretable,
and has a known target of 1.00; Dice moves slowly and mixes two error modes.

---

## 6. Operational gotchas — these have all bitten us

### HiPerGator `hpg-b200` kills jobs at random

Nodes SIGTERM jobs mid-run as part of a node drain. Confirmed **not** walltime (limit is
14 days), **not** preemption (`PreemptMode=OFF`), **not** fairshare. Five consecutive
test jobs died this way before one completed.

- **Training auto-resume works** (as of 2026-09-11). `BaseRunner.find_latest_checkpoint`
  scans for the newest `latest_model_<n>.pth` that has a matching
  `latest_optim_sche_<n>.pth`; requiring both halves means a job killed mid-`torch.save`
  won't resume weights without their optimizer.
- **Inference resume works** — `sample_to_eval` skips patients whose `{pid}.nii` already
  exists. This turned a 4 h re-run into 14 min.
- **The SIGTERM trap only fires if python is backgrounded**: `python ... &` then
  `wait $!`. In the foreground python absorbs the signal before bash sees it.
- Any variable the resubmit depends on (e.g. `ARM`) **must be re-exported** in the
  `sbatch --export` inside the trap, or the requeued job dies on the `${VAR:?}` guard.

### `n_epochs` is absolute, not additional

The loop is `range(start_epoch, n_epochs)`. Resuming from epoch 472 with `n_epochs: 532`
runs 60 epochs. A job that resumes at epoch == `n_epochs` trains **nothing** and exits in
minutes — we have logs that look like runs but aren't (jobs 43388428 / 43388985 /
43388986).

### One `exp_name` per experimental arm

Auto-resume keys off the checkpoint dir, which derives from `exp_name`. Two arms sharing
an `exp_name` will resume from each other's checkpoints and silently contaminate the
comparison.

### `--HW` overrides the YAML `image_size`

`main.py` applies `--HW` to both data and UNet. The config says `image_size: 160` but
`HW=256` makes the run 256. **The YAML value is effectively dead** — don't trust it when
reading a config.

### Output dir collisions can delete your baseline

Output dir = `result_path / (dataset_name + _{HW}) / exp_name`. The default fine-tune
script's `exp_name` once collided with the baseline's own directory, where `--save_top`
could prune `epoch_235.pth`. Always set a fresh `prefix`/`exp_name`.

`BaseRunner` rewrites `config_backup.yaml` into `ckpt_path` on **every** launch — so a
checkpoint's backup config may describe a later failed launch, not the run that produced
it.

### Sync before you submit

Job 37492440 died because HiPerGator had a stale `configs/broken-fine.yaml` with a YAML
indent error, wrong architecture, and no `model_load_path`. **rsync configs and shell
scripts before every resubmit.**

### Reading training logs

`save top model start...` prints **unconditionally**, before the improvement check —
counting it tells you nothing. The real signal is
`saving top checkpoint: average_loss=<x> epoch=<n>` (`runners/BaseRunner.py` ~line 655).

### wandb

1. Always pass `self.global_step` as `step=`. Never `epoch`. wandb's step must be
   monotonically non-decreasing; one out-of-order call poisons the whole run and every
   subsequent `wandb.log` raises ("Could not log loss to wandb" spam).
2. Don't use `wandb.watch(model, log='all')` under AMP — grad hooks fire before
   `GradScaler.unscale_`, so you log scaler-scaled (often Inf/NaN) gradients. Use
   `log='parameters'` and log post-clip grad norms manually.
3. Filter non-finite scalars before logging.
4. Never use a bare `except:` around wandb calls — it hid the root cause for weeks.

### FreeSurfer / SynthSeg

- FreeSurfer 8.1.0 is an `apptainer exec` wrapper. Loading python/conda alongside it sets
  `PYTHONHOME` and kills `mri_synthseg` with `No module named 'encodings'`. Load
  FreeSurfer **alone** and use `env -u PYTHONHOME -u PYTHONPATH`.
- `--keepgeom` is **required** on the eval SynthSeg array, or it writes on its own 1 mm
  grid (256×256×173 → 234×213×173).
- `sample_to_eval` saves `.nii` with `np.eye(4)`, discarding spacing.
  `export_eval_volumes.py` rebuilds the true affine from `geometry.json`. A plain diagonal
  affine made SynthSeg return cortex+CSF and **zero subcortical structures**.
- Use `det(affine)` for voxel volume and you inflate volumes 15–50%. Use `spacing_mm`.
  `crop_affine` is deliberately the PRE-resize affine.
- Histogram pkls are **not** reusable across HDF5 builds — they're keyed by subject name
  but built from a specific `data.csv`. Stale copies cause `KeyError`. Regenerate with
  `pkl_fine_v2.sh`.
- Segment each synthetic MR with the **same `synthseg_mode` the manifest assigns that
  subject**, or Dice measures model disagreement rather than synthesis error.

---

## 7. Known caveats that affect how results should be reported

### The evaluation protocol is optimistic

`hist_type` is unset in `configs/fine-tune.yaml`, so at test time the model is handed the
**ground-truth target MR's own intensity histogram** — a summary of the image it is
synthesizing. This is defensible if framed as operator-supplied scanner calibration, but
it is not a deployment condition, and it inflates every number downstream.

The fix already exists and is unused: `generate_total_hist_global.py` has three builders —
per-subject, `avg` (mean over the **train** split, correctly avoiding leakage), and
`colin` (template). Setting `hist_type: 'avg'` or `'colin'` and re-running eval is **one
job** and yields the deployment-realistic number. This is also the only ablation that
would show what histogram conditioning contributes at all — there is no
conditioned-vs-unconditioned comparison anywhere in the record.

### Structure metrics measure a segmenter, not anatomy

Every per-structure figure compares SynthSeg on the real volume against SynthSeg on the
synthetic one. The noise floor bounds the tool's contribution at 0.966 ± 0.01, so ~3–4
points of the Dice deficit is the segmenter. "The model under-fills pallidum by 16%" and
"SynthSeg is conservative on synthetic contrast" are **not separable** from these numbers.
The defensible phrasing is that synthetic volumes *segment* smaller.

### Precision

Left and right instances within one brain are correlated, so standard errors should use
n=36, not n=72. Pallidum Dice is 0.731 ± 0.015 and thalamus 0.909 ± 0.004 — the gap is
~12 SE, so rankings are not fragile.

### Tier 3 was trained with a known bug

The mask-threshold unit mismatch (§4, Phase 1) means SOTA was trained emphasising the
wrong voxels. Phase 3 arm A is the control that measures what fixing it alone buys.

---

## 8. Where things live

### HiPerGator

```
/blue/neurology-dept/jlabasbas/
  hdf5s/fine_v2/                 # current training data (has LABEL_dataset)
  new-fine/fine-tune_256/        # training checkpoints, one dir per exp_name
    241213_256_BBDM_axial_DDIM_MR_global_hist_context/   # baseline ep_235
    ..._MR_tier1_freq/           # Tier 1 ep_390
    ..._MR_tier2_ssim/           # Tier 2 ep_412
    ..._MR_tier3_morphmask/      # Tier 3 ep_472  <- SOTA
    ..._MR_tier4_zcontext7/      # Tier 4 ep_542  (negative result)
    ..._MR_p3_armA_control/      # Phase 3 arms
    ..._MR_p3_armB_uniform5x/
    ..._MR_p3_armE_shell5x/
  8-8-26/                        # Tier 4 eval outputs
  phase2/export/                 # per-structure eval (structure_metrics.csv)
  phase2/structures/             # 864 per-structure NIfTI masks
  phase4/                        # Phase 3 arm eval outputs
```

Repo on HPC: `~/CT2MRI-DTE` (user `joshua.labasbas`). SSH needs password + Duo; key auth
is not set up.

### Repo layout worth knowing

```
configs/                         # fine-tune.yaml is the live one; phase3_arm*.yaml
model/losses.py                  # perceptual, SSIM, frequency losses
model/BrownianBridge/BrownianBridgeModel.py   # p_losses, mask construction, samplers
runners/BaseRunner.py            # training loop, auto-resume, checkpointing
runners/eval_structures.py       # per-structure Dice/centroid/volume/CW-SSIM
runners/export_eval_volumes.py   # rebuilds true affines for SynthSeg
runners/phase3_preflight.py      # label presence + loss-scale shift check
brain_dataset_utils/geometry.py  # the transform chain — single source of truth
brain_dataset_utils/tests/       # 11 offline regression guards, run_all.sh
shell/{train,test,data}/         # inner scripts; ct2mri_*.sh at root are the sbatch wrappers
paper/structure_metrics.tex      # per-structure results, compiles standalone
```

### Running the per-structure evaluation

```bash
python runners/eval_structures.py \
  --export_dir /blue/neurology-dept/jlabasbas/phase2/export \
  --dump_structures /blue/neurology-dept/jlabasbas/phase2/structures
```

`--dump_structures` writes each structure as its own binary `uint8` mask carrying the
SynthSeg affine (12 structures × 2 × 36 = 864 files). Omit it to just regenerate the CSV.

---

## 9. Roadmap

From a verified research pass (2026-06-29; 25 claims verified, 21 killed). **Foundation
models, federated learning, and physics-informed synthesis produced zero verified claims
— treat as frontiers to monitor, not plans.**

| Phase | What | Why |
|---|---|---|
| now | Phase 3/4 structure-weighted loss | first anatomy-aware attempt at the in-brain gap |
| next | Heteroscedastic UQ on the 2D model | low cost, clinical safety framing, current arch |
| then | Conformal Risk Control calibration | distribution-free coverage across scanners; fixes the 38% over-confidence |
| Tier 5 | 3D volumetric, single-modality first (~24 GB) | fixes documented inter-slice inconsistency |
| Tier 6 | Multi-modality 3D (~81 GB, needs a full A100 80 GB node) | only after single-modality 3D validates |

**2D inter-slice inconsistency is a documented structural defect** (Make-A-Volume, MICCAI
2023; CT-to-MRI BBDM, MICCAI 2024) — each axial slice is generated independently, so
anatomy drifts across the volume. For DBS this is a safety issue. The claim that
ISTA-style inter-slice alignment fixes it without a full 3D architecture was **refuted**.

**Open questions nobody has answered:**
1. Does fine-tuning a neuroimaging foundation model beat BBDM from scratch at <1000
   CT-MR pairs?
2. What minimum masked SSIM / Dice on STN/GPi is clinically acceptable for DBS
   targeting? **No published threshold found.**
3. Does conformal coverage hold under multi-site scanner shift in this domain?
4. What 3D strategy (pseudo-3D, latent 3D, sliding-window patch) fits in 40 GB while
   keeping volumetric consistency in deep brain structures?

---

## 10. Working agreements

- **Change one variable at a time.** Tier 4 changed z-context *and* LR, so its negative
  result can't be cleanly attributed. Phase 3 arm A exists specifically to avoid
  repeating this.
- **Pause at phase boundaries.** Training and evaluation run on HiPerGator via `sbatch`
  and are launched by hand. Hand over the exact commands and wait for results rather than
  chaining unvalidated phases.
- **Report masked / per-structure metrics, not whole-volume.** Whole-image SSIM is
  inflated by black background and by skull; it moves barely at all when the deep gray
  changes.
- **Run the offline guards before launching anything.** `brain_dataset_utils/tests/run_all.sh`
  — 11 tests, synthetic data, no GPU, seconds. They encode most of §6.

---

## 11. Repo hygiene — needs attention

- Much of the Phase 1–3 work was developed **uncommitted** on `main` against a short
  commit history. Branch `worktree-structure-export-voxel-change` carries a snapshot
  commit plus the per-structure export work. Getting this onto `main` is outstanding.
- `datasets/MR_hists_global_{180,256}/` hold ~1080 generated histogram QC PNGs (~50 MB)
  that make an HTTPS push fail with `RPC failed; HTTP 408`. Now gitignored; regenerate
  from `generate_total_hist_global.py`.
- **A live `WANDB_API_KEY` is committed** in `ct2mri_train.sh` (and other wrappers). It is
  in git history. Rotate it at wandb.ai/settings and move to the `~/.netrc` approach
  `ct2mri_train_phase3.sh` already uses.
- A failed `git push` can still exit 0. Verify with
  `git ls-remote --heads origin <branch>`.
