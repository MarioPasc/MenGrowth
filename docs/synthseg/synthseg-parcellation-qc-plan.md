# SynthSeg Parcellation & Preprocessing QC Analysis — Agent Plan

**Date:** 2026-04-14
**Scope:** Run SynthSeg brain parcellation on MenGrowth preprocessed T1n volumes; perform downstream analysis to validate preprocessing quality and deliver regional brain volumetrics as a dataset feature.
**Priority:** Medium-high. This is both a QC validation tool and a dataset deliverable.

---

## 1. Motivation & Theoretical Background

### 1.1 Why SynthSeg

SynthSeg (Billot et al., "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining," *Medical Image Analysis*, 2023, DOI: `10.1016/j.media.2023.102789`) is a contrast-agnostic, resolution-agnostic brain parcellation tool trained via domain randomization on synthetic images generated from healthy brain label maps. It outputs:

1. A **32-region parcellation** following the FreeSurfer DKT atlas (cortical + subcortical structures).
2. A **per-volume QC score** $q \in [0, 1]$, computed as a soft Dice between the segmentation of the input and a resampled-then-re-segmented version. This is a self-consistency measure: degraded images (motion, low resolution, artifacts) yield low $q$.
3. **Per-region volumes** in mm³.

### 1.2 Why T1n Only

- **T1n** (native, pre-contrast T1-weighted) has the tissue contrast profile closest to what SynthSeg's synthetic training generator approximates. Gadolinium enhancement (T1c) creates bright regions in tumors/meninges that are out-of-distribution for SynthSeg.
- T1n is typically acquired at the best native resolution (~1mm isotropic or near-isotropic).
- **Critical design decision:** The SynthSeg parcellation from T1n serves as a **structural ROI template** that, because all modalities are co-registered after preprocessing, can be applied to T2w, T2-FLAIR, and T1c without additional transformation. This avoids relying on SynthSeg's weaker performance on T2-weighted contrasts.

### 1.3 The `--robust` Flag

SynthSeg ships with a `--robust` mode (Billot et al., "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets," *PNAS*, 2023) that applies different posterior regularization tolerant of pathological out-of-distribution regions. **This flag is mandatory for MenGrowth** — the cohort contains meningiomas with variable mass effect.

### 1.4 Meningioma Interaction & Mitigation

Meningiomas are extra-axial (they compress rather than infiltrate brain parenchyma), so:
- Tumor-distant regions retain their identity and SynthSeg can parcellate them reliably.
- Tumor-adjacent regions will show distorted boundaries due to mass effect.
- The QC score $q$ may be slightly depressed by tumor presence, but this is approximately constant per-subject across timepoints.

**Mitigation:** All downstream analyses must stratify regions by proximity to the tumor segmentation mask. BraTS segmentations (`seg.nii.gz`) are already available in the study directories.

---

## 2. Existing Code & Infrastructure

### 2.1 SynthSeg Tool Installation (Picasso)

```
~/fscratch/tools/SynthSeg/        # SynthSeg repository (scripts, models, etc.)
~/fscratch/tools/synthseg_env/    # Dedicated conda environment
```

### 2.2 Existing MenGrowth SynthSeg Code (Copied from Another Project)

**Read these files first — they are tested and working:**

```
mengrowth/synthseg/               # Python code (runner, config, postprocessing)
slurm/synthseg/                   # SLURM launcher scripts
```

The agent MUST read all files in both directories before making any changes. The existing implementation was copied verbatim from a working project. The task is to **refactor and adapt** it to MenGrowth's conventions, not to rewrite from scratch. Preserve the core SynthSeg invocation logic that is known to work.

### 2.3 MenGrowth Preprocessed Data Layout

```
{dataset_root}/MenGrowth-2025/
  MenGrowth-{XXXX}/                          # Patient directory
    MenGrowth-{XXXX}-{YYY}/                  # Study directory (timepoint)
      t1n.nii.gz                             # ← SynthSeg input
      t1c.nii.gz
      t2w.nii.gz
      t2f.nii.gz
      seg.nii.gz                             # BraTS tumor segmentation (if available)
```

On Picasso, `{dataset_root}` is:
```
/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/dataset/preprocecessed
```

Locally (for testing/analysis):
```
/media/mpascual/PortableSSD/Meningiomas/MenGrowth/v5_final
```

### 2.4 Project Conventions (MUST Follow)

Read `.claude/CLAUDE.md` for the full reference. Key constraints:

- **Config:** Pure `@dataclass` trees, parsed from YAML. No mutable defaults.
- **Typing:** Full type annotations everywhere.
- **Logging:** Python `logging` module. Structured log messages.
- **Error handling:** Per-file exception catching; never crash the pipeline on a single failure.
- **Temp files:** Atomic writes via `.tmp` suffix, then rename.
- **Modular code:** OOP with base classes where applicable; low cyclomatic complexity; atomic functions.
- **No symlinks for SLURM/Singularity paths** — use `shutil.copy2()` if copies are needed.

### 2.5 Reference: BraTS Segmentation Pipeline (Parallel Pattern)

The existing BraTS segmentation pipeline follows a pattern this implementation should mirror:

```
mengrowth/cli/segment.py           # CLI entry point (prepare, postprocess, run subcommands)
mengrowth/segmentation/prepare.py  # Study discovery, validation, BraTS-format staging
mengrowth/segmentation/postprocess.py  # Output remapping
configs/picasso/segmentation.yaml  # YAML config
slurm/segmentation/                # SLURM launcher + worker scripts
```

Study discovery uses `discover_studies()` which scans the preprocessed directory tree. Adopt the same pattern.

---

## 3. Implementation Plan

### Phase 1: SynthSeg Execution Pipeline

#### 3.1 Configuration (`configs/picasso/synthseg.yaml`)

Create a YAML config file. Required fields:

```yaml
synthseg:
  # Paths
  input_root: "/mnt/home/.../preprocecessed/MenGrowth-2025"
  output_root: "/mnt/home/.../preprocecessed/synthseg"
  synthseg_repo: "/mnt/home/users/tic_163_uma/mpascual/fscratch/tools/SynthSeg"
  conda_env: "/mnt/home/users/tic_163_uma/mpascual/fscratch/tools/synthseg_env"

  # SynthSeg options
  input_modality: "t1n"          # Which modality to parcellate
  robust: true                    # --robust flag (MANDATORY for tumor cohort)
  threads: 1                      # CPU threads per job
  
  # Output filenames (written into each study directory or output_root)
  parcellation_filename: "synthseg_parc.nii.gz"
  volumes_filename: "synthseg_volumes.csv"
  qc_filename: "synthseg_qc.csv"
  posteriors_filename: null       # Optional: "synthseg_posteriors.nii.gz"
```

Create a corresponding `@dataclass` in `mengrowth/synthseg/config.py`.

#### 3.2 Refactor `mengrowth/synthseg/` Code

Read the existing files. The refactoring goals are:

1. **Adapt paths and discovery** to MenGrowth's `MenGrowth-{XXXX}/MenGrowth-{XXXX}-{YYY}/` structure.
2. **Ensure `--robust` is always passed.**
3. **Ensure outputs are written per-study** (parcellation NIfTI + volumes CSV + QC score) into a structured output directory:
   ```
   {output_root}/
     MenGrowth-{XXXX}/
       MenGrowth-{XXXX}-{YYY}/
         synthseg_parc.nii.gz       # 32-region label map
         synthseg_volumes.csv       # Per-region volumes
         synthseg_qc.csv            # QC score
   ```
4. **Add structured logging** consistent with the rest of the project.
5. **Handle missing T1n gracefully** — log a warning, skip the study, do not crash.
6. **Handle missing `seg.nii.gz` gracefully** — the analysis phase needs it, but SynthSeg execution does not.

The SynthSeg command being invoked should look like:

```bash
python {synthseg_repo}/scripts/commands/SynthSeg_predict.py \
  --i {input_t1n} \
  --o {output_parc} \
  --robust \
  --vol {output_volumes_csv} \
  --qc {output_qc_csv} \
  --threads {threads}
```

Verify this against the existing working code in `mengrowth/synthseg/` — do not guess the CLI interface; read `~/fscratch/tools/SynthSeg/scripts/commands/SynthSeg_predict.py` argument parser if needed.

#### 3.3 SLURM Scripts (`slurm/synthseg/`)

**Architecture:** SLURM array job, one task per patient.

Read and refactor the existing launcher scripts. The target design:

**`slurm/synthseg/synthseg_launcher.sh`** (run from login node):
- Discovers all patient directories in `input_root`.
- Creates a patient list file (`/tmp/synthseg_patients_XXXX.txt` or in a work directory).
- Submits a SLURM array job with `--array=0-{N_patients-1}`.
- Prints job ID and monitoring instructions.
- Accepts optional `--depends-on JOB_ID` for dependency chaining.

**`slurm/synthseg/synthseg_worker.sh`** (runs on compute node):
- Reads patient ID from the patient list file using `$SLURM_ARRAY_TASK_ID`.
- Activates the SynthSeg conda environment.
- Iterates over all study directories for that patient.
- Runs SynthSeg on each `t1n.nii.gz`.
- Logs timing and success/failure per study.

SLURM resource requirements (per array task):
```bash
#SBATCH --time=0-01:00:00       # ~2-5 min per study, patients have 2-10 studies
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1            # SynthSeg uses GPU if available
#SBATCH --constraint=dgx        # Or appropriate GPU partition
```

**Important:** SynthSeg can run on CPU (slower, ~1-2 min per volume) or GPU (~10-20 sec per volume). The existing implementation should already handle this. GPU is strongly preferred for 58 patients × ~3 studies each ≈ 174 volumes.

#### 3.4 Verification After Phase 1

Before proceeding to analysis, verify:

```python
# Quick check script (not part of the pipeline — for manual verification)
# Count expected vs actual outputs
import glob
expected = glob.glob("{input_root}/MenGrowth-*/MenGrowth-*/t1n.nii.gz")
produced_parc = glob.glob("{output_root}/MenGrowth-*/MenGrowth-*/synthseg_parc.nii.gz")
produced_qc = glob.glob("{output_root}/MenGrowth-*/MenGrowth-*/synthseg_qc.csv")
print(f"Expected: {len(expected)}, Parcellations: {len(produced_parc)}, QC: {len(produced_qc)}")
# All three counts should match
```

Also verify for 2-3 random studies:
- Parcellation is in the same geometric space as `t1n.nii.gz` (same affine, same shape).
- QC score is in $[0, 1]$.
- Volume CSV has ~32 rows with plausible volumes (total brain ~1000-1500 cm³).

---

### Phase 2: Downstream Analysis Pipeline

#### 4.1 Architecture

Create a standalone analysis module:

```
mengrowth/synthseg/
  __init__.py
  config.py              # @dataclass config (Phase 1)
  runner.py              # SynthSeg execution logic (Phase 1, refactored)
  analysis/
    __init__.py
    collector.py         # Aggregate all per-study outputs into cohort-level DataFrames
    region_metrics.py    # Per-region quality metrics using parcellation as ROI template
    qc_analysis.py       # QC score vs acquisition characteristics analysis
    longitudinal.py      # Longitudinal consistency analysis
    figures.py           # All visualization functions
    report.py            # Generate summary report (HTML or markdown)
```

Create a CLI entry point following the project pattern:

```
mengrowth/cli/synthseg_analysis.py   # or extend mengrowth/cli/segment.py
```

With a config file:

```yaml
# configs/picasso/synthseg_analysis.yaml
synthseg_analysis:
  synthseg_output_root: "/mnt/home/.../synthseg"
  preprocessed_root: "/mnt/home/.../preprocecessed/MenGrowth-2025"
  output_dir: "/mnt/home/.../synthseg/analysis"
  
  # Original acquisition metadata (pre-preprocessing spacing, etc.)
  # This should be a CSV/JSON from the curation quality analysis step.
  # If not available, the analysis module should extract spacing from
  # the quality_metrics.json or re-compute from curated (pre-preprocessing) data.
  original_metadata_path: null  # Path to CSV with original per-study, per-modality spacing
  
  # Tumor segmentation
  tumor_seg_filename: "seg.nii.gz"
  tumor_proximity_threshold_mm: 20.0   # δ for tumor-distant vs tumor-adjacent
  
  # Modalities for ROI-based analysis
  modalities_for_roi_analysis: ["t1n", "t1c", "t2w", "t2f"]
  
  # Figures
  figure_format: "png"
  figure_dpi: 300
```

#### 4.2 Data Collection (`collector.py`)

Scan all SynthSeg outputs and build cohort-level DataFrames:

1. **`volumes_df`**: Columns = `[patient_id, study_id, timepoint_index, region_id, region_name, volume_mm3]`.
   - Parse from per-study `synthseg_volumes.csv`.
   - Add `timepoint_index` by sorting studies within each patient lexicographically (the `MenGrowth-XXXX-YYY` naming encodes temporal order).

2. **`qc_df`**: Columns = `[patient_id, study_id, timepoint_index, qc_score]`.
   - Parse from per-study `synthseg_qc.csv`.

3. **`metadata_df`**: Columns = `[patient_id, study_id, modality, original_spacing_x, original_spacing_y, original_spacing_z, original_max_spacing, ...]`.
   - **Critical:** This requires original (pre-preprocessing) acquisition metadata. Check if it exists in:
     - `{curation_output}/quality/quality_metrics.json`
     - `{curation_output}/quality/qc_analysis/*.csv`
     - The quality analysis module's outputs.
   - If no pre-existing source, the agent should implement extraction from the curated (pre-preprocessing) NIfTI/NRRD headers. This is a fallback — check the existing outputs first.

Save all aggregated DataFrames to `{output_dir}/cohort_volumes.csv`, `{output_dir}/cohort_qc.csv`, `{output_dir}/cohort_metadata.csv`.

#### 4.3 Analysis 1: SynthSeg QC Score vs. Original Acquisition Features (`qc_analysis.py`)

**Goal:** Test whether post-preprocessing quality (measured by SynthSeg QC) depends on original acquisition characteristics. Correct preprocessing should yield approximately constant $q$ regardless of input quality.

**Metrics to plot $q$ against:**
- Original T1n inter-slice spacing (most relevant — thicker slices → more interpolation).
- Original T1n in-plane resolution.
- Original T1n anisotropy ratio: $\rho = s_{\max} / s_{\min}$ where $s$ are the three spacing components.
- (If available) Original T2w and T2-FLAIR spacing — using region-level metrics from Analysis 3 rather than $q$ directly.

**Statistical test:** Fit a linear mixed-effects model:
$$q_{i,t} = \beta_0 + \beta_1 \cdot s_{i,t}^{\text{orig}} + u_i + \epsilon_{i,t}, \quad u_i \sim \mathcal{N}(0, \sigma_u^2)$$

where $u_i$ is a patient-level random intercept (absorbs inter-subject baseline differences including tumor effects on $q$). Report $\beta_1$, its $p$-value, and the marginal $R^2$.

Use `statsmodels.formula.api.mixedlm` or equivalent.

**Figures:**
- Scatter plot: $q$ vs. original T1n max spacing, colored by patient, with regression line and confidence band.
- Box plot: $q$ grouped by original spacing bins (e.g., $< 1.5$mm, $1.5$–$3$mm, $> 3$mm).

#### 4.4 Analysis 2: Per-Region Longitudinal Consistency (`longitudinal.py`)

**Goal:** For each brain region, compute within-subject volume stability across timepoints. Low variability in tumor-distant regions validates preprocessing; higher variability in tumor-adjacent regions reflects genuine biology (mass effect progression).

**Metric:** Within-subject coefficient of variation per region:
$$\text{CV}_{i,r} = \frac{\sigma_{i,r}}{\mu_{i,r}}, \quad \text{over timepoints } t = 1, \ldots, T_i$$

where $V_{i,r,t}$ is the SynthSeg volume of region $r$ for subject $i$ at timepoint $t$.

**Tumor proximity stratification:**

For each subject $i$ and region $r$, compute the minimum distance from the region boundary to the tumor mask boundary:

```python
from scipy.ndimage import distance_transform_edt

def compute_region_tumor_distance(
    parcellation: np.ndarray,
    tumor_mask: np.ndarray,
    region_id: int,
    voxel_spacing: Tuple[float, float, float],
) -> float:
    """Minimum distance (mm) from region boundary to tumor surface."""
    tumor_dist_map = distance_transform_edt(~(tumor_mask > 0), sampling=voxel_spacing)
    region_mask = parcellation == region_id
    if not np.any(region_mask):
        return float('inf')
    return float(np.min(tumor_dist_map[region_mask]))
```

Classify regions:
- **Tumor-distant:** $d_{i,r} > \delta$ (default $\delta = 20$ mm).
- **Tumor-adjacent:** $d_{i,r} \leq \delta$.

For subjects without `seg.nii.gz`, all regions are treated as tumor-distant (conservative).

**Additional control:** For lateralized regions (e.g., left/right putamen, thalamus), compute:
$$\Delta\text{CV}_{i,r} = \text{CV}_{i,r}^{\text{ipsilateral}} - \text{CV}_{i,r}^{\text{contralateral}}$$

This isolates tumor-driven variability from preprocessing-driven variability. The ipsilateral side is determined by the tumor centroid's hemisphere.

**Figures:**
- Violin plot: distribution of $\text{CV}_{i,r}$ across subjects, faceted by region, colored by tumor-distant vs. tumor-adjacent.
- Heatmap: regions × subjects, cell color = $\text{CV}$, with tumor-adjacent cells marked.
- Bar plot: mean $\text{CV}$ for tumor-distant regions (this is the headline QC number).

#### 4.5 Analysis 3: ROI-Based Cross-Modal Quality Metrics (`region_metrics.py`)

**Goal:** Use the T1n parcellation as a structural template to assess quality of *all* co-registered modalities, including the problematic T2w and T2-FLAIR that we cannot run SynthSeg on directly.

**Metrics per modality $m$, region $r$, study $(i, t)$:**

1. **Within-region intensity CV** (tissue homogeneity):
$$\text{CV}_{i,r,t}^{(m)} = \frac{\sigma\bigl(I^{(m)}[\mathcal{L}_i = r]\bigr)}{\mu\bigl(I^{(m)}[\mathcal{L}_i = r]\bigr)}$$

For well-preprocessed data, homogeneous tissue regions (deep grey matter nuclei: putamen, caudate, thalamus) should have low CV. High CV in these regions on T2w after resampling from 5mm slices indicates interpolation artifacts or partial volume contamination.

2. **Boundary gradient energy** at region interfaces:
$$G_{i,r,t}^{(m)} = \frac{1}{|\partial r|} \sum_{\mathbf{x} \in \partial r} \|\nabla I^{(m)}(\mathbf{x})\|$$

Sharp boundaries indicate good effective resolution; blurred boundaries indicate information loss from thick-slice resampling. The boundary $\partial r$ is computed via morphological erosion/dilation of the region mask.

3. **Longitudinal within-region intensity stability:**
$$\text{CV}_{i,r}^{(m), \text{long}} = \frac{\sigma_t\bigl(\bar{I}_{i,t}^{(m)}[r]\bigr)}{\mu_t\bigl(\bar{I}_{i,t}^{(m)}[r]\bigr)}$$

where $\bar{I}_{i,t}^{(m)}[r]$ is the mean intensity in region $r$. After intensity normalization, this should be stable across timepoints. Instability flags normalization failures.

**Cross-modal validation plots:**
- Scatter: $G^{(\text{T2w})}$ (boundary gradient in thalamus) vs. original T2w inter-slice spacing. **This directly tests whether BSpline+ECLARE resampling preserves boundary definition in the challenging modality.**
- Scatter: $\text{CV}^{(\text{T2f})}$ (white matter region CV) vs. original T2-FLAIR spacing. Tests tissue homogeneity preservation.
- These plots answer: "did preprocessing successfully normalize quality across heterogeneous inputs?" without requiring SynthSeg to work on T2.

**Implementation note:** Only compute these metrics for a curated set of "reliable" regions — large subcortical structures (thalamus, putamen, caudate, pallidum, hippocampus, amygdala, lateral ventricle, cerebral white matter, cerebellum). Skip small cortical parcels where SynthSeg boundary accuracy is lower.

---

## 5. SLURM Execution Plan

### Job 1: SynthSeg Array (GPU)

```bash
# From login node:
bash slurm/synthseg/synthseg_launcher.sh --config configs/picasso/synthseg.yaml
# Submits array job: 58 tasks (one per patient), each processing all studies for that patient
# Expected runtime: ~5-15 min per task (2-10 studies × ~30 sec/study on GPU)
```

### Job 2: Analysis (CPU, depends on Job 1)

```bash
# After Job 1 completes:
bash slurm/synthseg/synthseg_analysis_launcher.sh \
  --config configs/picasso/synthseg_analysis.yaml \
  --depends-on <JOB1_ID>
# Single job, CPU only, ~10-30 min for full cohort analysis
# Alternatively: run interactively if small enough
```

Or create a single launcher that chains both:

```bash
bash slurm/synthseg/synthseg_full.sh --config configs/picasso/synthseg.yaml
# Submits Job 1, captures job ID, submits Job 2 with --dependency=afterok:<JOB1_ID>
```

---

## 6. Output Structure

```
{output_root}/synthseg/
  MenGrowth-{XXXX}/
    MenGrowth-{XXXX}-{YYY}/
      synthseg_parc.nii.gz           # 32-region label map (NIfTI, same geometry as t1n)
      synthseg_volumes.csv           # Per-region volumes
      synthseg_qc.csv                # QC score
  analysis/
    cohort_volumes.csv               # Aggregated volumes (all subjects × regions × timepoints)
    cohort_qc.csv                    # Aggregated QC scores
    cohort_metadata.csv              # Original acquisition metadata joined with QC
    region_tumor_distances.csv       # Per-subject, per-region distance to tumor
    roi_metrics/
      within_region_cv.csv           # CV per modality, region, study
      boundary_gradient.csv          # Gradient energy per modality, region, study
      longitudinal_stability.csv     # Longitudinal CV per modality, region, subject
    figures/
      qc_vs_original_spacing.png     # Analysis 1: QC vs T1n spacing scatter
      qc_vs_spacing_boxplot.png      # Analysis 1: QC by spacing bin
      region_cv_violin.png           # Analysis 2: Longitudinal CV violin by region
      region_cv_heatmap.png          # Analysis 2: CV heatmap (regions × subjects)
      t2w_gradient_vs_spacing.png    # Analysis 3: T2w boundary gradient vs original spacing
      t2f_cv_vs_spacing.png          # Analysis 3: T2-FLAIR region CV vs original spacing
      contralateral_delta_cv.png     # Ipsi vs contra CV comparison
    summary_report.md                # or .html — cohort-level QC summary with embedded figures
```

---

## 7. Testing & Verification Checklist

### Phase 1 (SynthSeg Execution)

- [ ] All `t1n.nii.gz` inputs have corresponding `synthseg_parc.nii.gz` outputs (count match).
- [ ] Parcellation geometry matches T1n geometry: `nib.load(parc).shape == nib.load(t1n).shape` and `np.allclose(parc.affine, t1n.affine)`.
- [ ] QC scores are in $[0, 1]$; flag any $q < 0.3$ for manual review (likely severe pathology or preprocessing failure).
- [ ] Volume CSV contains expected ~32 regions with non-zero volumes summing to plausible total brain volume.
- [ ] Visual spot-check: overlay `synthseg_parc.nii.gz` on `t1n.nii.gz` for 3 random studies in ITK-SNAP or equivalent. Boundaries should align with tissue interfaces.

### Phase 2 (Analysis)

- [ ] `cohort_volumes.csv` has $N_{\text{studies}} \times 32$ rows (one per study-region pair).
- [ ] Longitudinal CV for tumor-distant deep grey matter regions (putamen, thalamus, caudate) should be $\text{CV} < 0.05$ (5%) for well-preprocessed data — this is the key QC pass/fail criterion.
- [ ] QC-vs-spacing scatter should show weak or no correlation ($|\beta_1|$ small, $p > 0.05$ ideally) if preprocessing compensates for input heterogeneity.
- [ ] Boundary gradient on T2w should show a negative trend vs. original T2w slice thickness (thicker original slices → lower gradient energy after resampling — this is expected information loss, not a preprocessing failure).
- [ ] No NaN or Inf values in any output CSV.
- [ ] Figures render correctly and are publication-quality (300 DPI, proper axis labels, legends).

---

## 8. Dependencies

### Python (in the MenGrowth `growth` conda environment)

```
numpy, scipy, pandas, nibabel, matplotlib, seaborn, statsmodels
```

`statsmodels` is needed for the mixed-effects model. Verify it is installed; if not, `pip install statsmodels`.

### SynthSeg Environment

```
~/fscratch/tools/synthseg_env/    # Pre-existing, tested
~/fscratch/tools/SynthSeg/        # Repo with scripts and models
```

The worker script activates this environment before calling SynthSeg. Do not mix environments.

---

## 9. Scope Boundaries

**In scope:**
- Refactoring existing `mengrowth/synthseg/` and `slurm/synthseg/` code for MenGrowth conventions.
- SynthSeg execution on T1n with `--robust`.
- The three downstream analyses described above.
- SLURM launcher/worker scripts.
- YAML configs and `@dataclass` definitions.
- Verification scripts.

**Out of scope:**
- Modifying the preprocessing pipeline.
- Re-running preprocessing.
- Training or fine-tuning SynthSeg.
- Integrating parcellation into the HDF5 archive (future task — can be done analogously to `cmd_attach_to_archive` in `mengrowth/cli/segment.py`).

---

## 10. Key References

1. Billot, B., et al. (2023). "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining." *Medical Image Analysis*, 86, 102789. DOI: `10.1016/j.media.2023.102789`
2. Billot, B., et al. (2023). "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets." *PNAS*, 120(9), e2216399120. DOI: `10.1073/pnas.2216399120`
3. Fischl, B. (2012). "FreeSurfer." *NeuroImage*, 62(2), 774–781. (DKT atlas reference)
