# Step 6: Skull Stripping

## Theory

Skull stripping (brain extraction) removes non-brain tissue — skull, scalp, eyes, neck — from MRI volumes, producing a binary brain mask and a skull-stripped volume where non-brain voxels are set to zero (or a configurable fill value).

Two deep learning methods are available:

### HD-BET (Isensee et al., 2019)

HD-BET (High Definition Brain Extraction Tool) uses an ensemble of U-Net models trained on diverse MRI data. It operates in two modes:
- **Accurate mode:** Runs all 5 ensemble members with test-time augmentation (mirroring). Highest quality but slowest (~30s/volume on GPU).
- **Fast mode:** Single model, no augmentation. ~5s/volume. Adequate for most clinical data.

HD-BET is robust to pathology (tumors, edema, resection cavities) — critical for meningioma datasets where large space-occupying lesions may confuse simpler methods.

### SynthStrip (Hoopes et al., 2022)

SynthStrip uses a contrast-agnostic U-Net trained on synthetic data from anatomical label maps. It generalizes across MRI contrasts without fine-tuning. The `border` parameter controls how tightly the mask hugs the brain surface (positive values expand, negative values contract).

Both methods process the reference modality of the study (typically T1c after registration) and produce a single brain mask that is applied to all modalities.

## Motivation

Skull stripping is essential for meningioma preprocessing because:
- Intensity normalization must exclude skull/scalp to compute meaningful brain tissue statistics
- Longitudinal registration is more stable when non-brain structures are excluded
- Deep learning models should not waste capacity modeling skull/air boundaries
- Meningiomas are extra-axial tumors (outside the brain parenchyma but inside the skull), so the mask must include the tumor

## Code Map

- **Step handler:** `mengrowth/preprocessing/src/steps/skull_stripping.py`
- **HD-BET wrapper:** `mengrowth/preprocessing/src/skull_stripping/hdbet.py` → `HDBetSkullStripper`
- **SynthStrip wrapper:** `mengrowth/preprocessing/src/skull_stripping/synthstrip.py` → `SynthStripSkullStripper`
- **Base class:** `mengrowth/preprocessing/src/skull_stripping/base.py`
- **Config class:** `mengrowth/preprocessing/src/config.py` → `SkullStrippingStepConfig`, `SkullStrippingConfig`
- **YAML key:** `step_configs.skull_stripping`

## Config Reference

```yaml
skull_stripping:
  save_visualization: true         # Generate before/after PNG
  save_mask: true                  # Save brain mask to artifacts/
  skull_stripping:
    method: "hdbet"                # "hdbet" | "synthstrip" | null
    fill_value: 0.0                # Value for non-brain voxels

    # HD-BET parameters
    hdbet_mode: "accurate"         # "fast" | "accurate"
    hdbet_device: 0                # GPU id (int) or "cpu"
    hdbet_do_tta: true             # Test-time augmentation (5 folds)

    # SynthStrip parameters
    synthstrip_border: 1           # Border expansion in mm
    synthstrip_device: 0           # GPU id (int) or "cpu"
```

## Inputs / Outputs

- **Input:** `{output_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/*.nii.gz` (registered volumes)
- **Output:** Same paths (skull-stripped, in-place update)
- **Artifact:** `{artifacts}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/{modality}_brain_mask.nii.gz`
- **Visualization:** `{viz_root}/MenGrowth-XXXX/MenGrowth-XXXX-YYY/step6_skull_stripping_*.png`
- **Execution level:** per-study

## Common Tasks

| Task | How |
|------|-----|
| Switch to SynthStrip | Set `method: "synthstrip"` |
| Run on CPU (no GPU) | Set `hdbet_device: "cpu"` or `synthstrip_device: "cpu"` |
| Faster HD-BET | Set `hdbet_mode: "fast"` and `hdbet_do_tta: false` |
| Skip skull stripping | Set `method: null` |
| Expand mask boundary | Increase `synthstrip_border` (SynthStrip only) |
| Inspect brain mask | Load `{artifacts}/.../t1c_brain_mask.nii.gz` in viewer |
| Debug poor extraction | Check visualization PNG; ensure registration was successful first |
