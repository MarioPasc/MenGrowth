# Registration Debugging Log (v2 Pipeline)

**Date:** 2026-02-26
**Patient:** MenGrowth-0002 (study 000)
**Pipeline version:** v2 (skull-strip-first: skull_stripping before registration)

---

## Problem Statement

After switching to the v2 pipeline order (skull stripping before registration), two issues appeared:
1. **Skull visible in registration visualizations** despite correct skull stripping
2. **Catastrophic atlas registration failure** (MI ~ -0.006, Corr ~ 0.007)

---

## Issue 1: Skull Visible in Post-Skull-Stripping Registration

### Root Cause: Consensus Mask on Unregistered Volumes

In the v2 pipeline, skull stripping runs **before** registration. The consensus masking feature was designed for the v1 pipeline where all modalities were already co-registered. In v2:

- Modalities have very different original shapes:
  - t1c/t1n: 312x512x512 at 0.5mm spacing -> 156x256x256 after resampling
  - t2w/t2f: 512x512x24 at 0.43x0.43x6mm -> 220x220x144 after resampling
- After cubic padding to 256^3, brains are at different positions
  - t2f mask center-of-mass at Z=79.5
  - t1n mask center-of-mass at Z=47.5
  - **32-voxel Z-axis offset between modalities**
- Consensus voting (majority vote) on these misaligned masks includes wrong voxels, leaving skull fragments

### Fix

Disabled consensus masking in `configs/local/preprocessing_local.yaml`:
```yaml
consensus_masking: false  # Modalities not co-registered when skull stripping runs
```

Each modality now gets its own independent HD-BET brain mask.

### File Changed
- `configs/local/preprocessing_local.yaml`: `consensus_masking: true` -> `false`

---

## Issue 2: Checkpoint System Not Respecting `overwrite: true`

### Root Cause

`CheckpointManager` only tracks per-modality steps. When `overwrite=True`, checkpoints were NOT cleared, so:
- Per-modality steps (data_harmonization, bias_field_correction, resampling) were **skipped** by stale checkpoints
- Study-level steps (skull_stripping, registration) always re-executed
- On re-runs, registration operated on already-registered files (double-registration)

### Fix

Added checkpoint clearing in `run_patient()` when `overwrite=True`:
```python
if self.config.overwrite and self.checkpoint_manager:
    cleared = 0
    for study_dir in study_dirs:
        for modality in self.config.modalities:
            if self.checkpoint_manager.clear_checkpoint(patient_id, study_dir.name, modality):
                cleared += 1
    if cleared:
        self.logger.info(f"Cleared {cleared} checkpoint(s) for {patient_id} (overwrite=True)")
```

### File Changed
- `mengrowth/preprocessing/src/preprocess.py`: Added checkpoint clearing logic in `run_patient()`

---

## Issue 3: Atlas Registration Catastrophic Failure

### Background

After fixing the consensus mask (Issue 1), registration no longer showed skull. However:
- t2f -> t1n intra-study: Corr = 0.0009 (catastrophic)
- All modalities -> atlas: MI ~ -0.006, Corr ~ 0.007 (catastrophic)

### Investigation: `moving_mask` Parameter

Initial hypothesis: sparse skull-stripped volumes (~7.5% nonzero in 256^3 padded cube) need a `moving_mask` to focus the MI metric on brain tissue.

Added `moving_mask = ants.get_mask(reference_img, low_thresh=1e-6)` to both:
- `antspyx_intra_study_to_atlas.py`
- `antspyx_multi_modal_coregistration.py`

### Test Results

Tested with study 001 (246x246x150, not 256^3 padded):

| Configuration | MI |
|---|---|
| Full-head -> atlas | -0.31 (OK) |
| Skull-stripped, no mask, no COM | -0.30 (OK) |
| **Skull-stripped + moving_mask + NO COM** | **-0.009 (CATASTROPHIC)** |
| Skull-stripped + moving_mask + COM | -0.29 (OK) |

Tested with simulated sparse 256^3 volume (24% nonzero):

| Configuration | MI |
|---|---|
| With mask + COM | -0.285 (OK) |
| Without mask + COM | -0.316 (OK) |
| Without mask, no COM | -0.310 (OK) |

### Root Cause

`moving_mask` without COM initialization causes catastrophic failure at coarse multi-resolution levels. At 8x shrink factor, the masked brain becomes too small (~0.9% of downsampled volume) for the MI optimizer to find a useful gradient. The optimizer gets stuck in a local minimum.

With COM initialization, brains are pre-aligned so the mask is viable. Without COM, the mask restricts the metric to a region that doesn't overlap with the atlas at coarse resolutions.

For the intra-study case (t2f -> t1n), the moving_mask restricted sampling to t2f's brain region which is at a completely different position than t1n's brain in the 256^3 cube (32-voxel offset from different original shapes).

### Fix

**Removed `moving_mask`** from both registration implementations. Registration works reliably without the mask when COM initialization is present:

- COM init handles spatial pre-alignment (centers of mass are aligned before optimization)
- MI metric is robust to zero-background (zero-zero pairs contribute no gradient)
- The 7.5% brain fraction is sufficient for MI to converge

### Files Changed
- `mengrowth/preprocessing/src/registration/antspyx_intra_study_to_atlas.py`: Removed `moving_mask` creation and `moving_mask=moving_mask` from `reg_kwargs`
- `mengrowth/preprocessing/src/registration/antspyx_multi_modal_coregistration.py`: Same removal

---

## Issue 3b: Custom COM Init Causes Atlas Registration Divergence

### Observation

After removing `moving_mask` (Issue 3), re-running the pipeline showed:
- **Study 000**: atlas MI=-0.016, Corr=0.015 (catastrophic — brain pushed to corner of atlas)
- **Study 001**: atlas MI=-0.43, Corr=0.86 (excellent)

Same code, same config, same patient. The difference: study 000 had a 65mm COM offset, study 001 had 30mm.

### Investigation

Synthetic A/B test with the atlas brain placed in a 256^3 padded cube (8.7% nonzero):

| Configuration | MI | Status |
|---|---|---|
| With custom COM init (81mm offset) | -0.010 | CATASTROPHIC |
| Without COM init (antspy default) | -0.306 | OK |
| With custom COM, 7.6% eroded | -0.299 | OK |

### Root Cause

Our custom Euler3DTransform COM initialization causes the ANTs optimizer to diverge when the COM offset is large (>60mm). The transform has a `center` parameter set to the subject's center of mass (far from the atlas origin). When ANTs optimizes rotation/affine parameters centered at this point, the optimization landscape becomes unstable for large offsets.

When `initial_transform=None`, antspy uses ANTs' built-in center-of-mass initialization (`-r [fixed,moving,1]`), which handles this correctly regardless of offset magnitude. The built-in initialization applies the alignment BEFORE optimization starts, in a way that properly conditions the optimizer's parameter space.

### Fix

Removed the custom COM initialization from atlas registration. When `initial_transform` is not provided, antspy automatically uses ANTs' built-in `[fixed,moving,1]` center-of-mass initialization.

The custom COM init is kept for **intra-study** coregistration where offsets are smaller and inter-subject shape differences don't exist.

### File Changed
- `mengrowth/preprocessing/src/registration/antspyx_intra_study_to_atlas.py`: Replaced custom COM init block with `initial_transform=None` (antspy's default)

---

## Issue 4: Diagnostic Logging Added

Added `[DIAG]` logging to verify file state at critical points:

### Skull Stripping Diagnostics
After Phase 3 (mask application), logs shape, nonzero%, intensity range, and path for each modality.

### Registration Diagnostics
Before registration reads files and after atlas registration completes, logs the same diagnostic info.

### Files Changed
- `mengrowth/preprocessing/src/steps/skull_stripping.py`: Added post-Phase-3 diagnostic block
- `mengrowth/preprocessing/src/steps/registration.py`: Added pre-registration and post-atlas diagnostic blocks

---

## Summary of All Changes

| File | Change | Reason |
|---|---|---|
| `configs/local/preprocessing_local.yaml` | `consensus_masking: false` | Consensus mask invalid on unregistered modalities |
| `mengrowth/preprocessing/src/preprocess.py` | Clear checkpoints when `overwrite=True` | Prevent stale checkpoint skipping |
| `mengrowth/preprocessing/src/registration/antspyx_intra_study_to_atlas.py` | Remove `moving_mask` + Remove custom COM init | Mask + custom COM cause catastrophic failures |
| `mengrowth/preprocessing/src/registration/antspyx_multi_modal_coregistration.py` | Remove `moving_mask` | Mask causes failure on sparse volumes |
| `mengrowth/preprocessing/src/steps/skull_stripping.py` | Add `[DIAG]` logging | Verify skull stripping output |
| `mengrowth/preprocessing/src/steps/registration.py` | Add `[DIAG]` logging | Verify registration input/output |

---

## Final Results (After All Fixes)

Pipeline re-run on MenGrowth-0002 with all fixes applied:

### Study 000
| Registration | Metric | Status |
|---|---|---|
| t1c → t1n (intra-study) | Corr = 0.97 | Excellent |
| t2w → t1n (intra-study) | Corr = 0.63 | Good (expected for T2w↔T1n) |
| t2f → t1n (intra-study) | Corr = 0.84 | Good |
| t1n → atlas | MI = -0.434, Corr = 0.881 | Excellent |

### Study 001
| Registration | Metric | Status |
|---|---|---|
| t1c → t1n (intra-study) | Corr = 0.97 | Excellent |
| t2w → t1n (intra-study) | Corr = 0.75 | Good |
| t2f → t1n (intra-study) | Corr = 0.94 | Excellent |
| t1n → atlas | MI = -0.426, Corr = 0.863 | Excellent |

Pipeline completed: 16 processed, 0 errors.

### Config Propagation

Changes applied to both `configs/local/preprocessing_local.yaml` and `configs/picasso/preprocessing_v2.yaml`:
- `consensus_masking: false`
- `fill_value_mode: "zero"`
- `resampling.method: bspline`
- `interpolation: "Linear"` (intra-study + atlas)
- `atlas_path: T1_brain.nii` (skull-stripped atlas)
- `intensity_normalization.method: null`

---

## Key Takeaways

1. **Consensus masking requires co-registered modalities.** In any pipeline where skull stripping precedes registration, consensus masking must be disabled.

2. **`moving_mask` in ANTs registration is dangerous for sparse volumes.** At coarse multi-resolution levels (8x shrink), the masked region becomes too small for optimization.

3. **Checkpoint systems must respect overwrite flags.** A checkpoint system that doesn't clear on overwrite can cause subtle double-processing bugs.

4. **Custom COM initialization can hurt atlas registration.** For inter-subject registration with large spatial offsets (>60mm), a custom Euler3DTransform COM init causes the ANTs optimizer to diverge. ANTs' built-in `[fixed,moving,1]` initialization handles all offset magnitudes correctly. Custom COM init is fine for intra-study registration where offsets are smaller.

5. **Always validate on multiple studies.** Study 001 worked perfectly while study 000 failed catastrophically — testing on a single study can mask systematic issues.
