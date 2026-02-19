# Quality Filtering: Formal Specification

This document formalizes the automated quality filtering stage applied to the MenGrowth cohort. The pipeline evaluates raw clinical MRI acquisitions in NRRD format across 15 quality checks organized into five hierarchical categories. Each check produces a binary decision (pass/fail) with an associated action (`block` or `warn`). Studies containing any blocking failure are removed from the dataset; warnings are logged for manual review but do not trigger exclusion.

---

## 1. Notation

| Symbol | Description |
|--------|-------------|
| $I \in \mathbb{R}^{X \times Y \times Z}$ | 3D image volume |
| $\mathbf{s} = (s_x, s_y, s_z)$ | Voxel spacing vector (mm) |
| $\mathbf{d} = (d_x, d_y, d_z)$ | Volume dimensions (voxels) |
| $\mathbf{A} \in \mathbb{R}^{3\times3}$ | Affine rotation/scaling matrix (from `space_directions`) |
| $m \in \{$`t1c`, `t1n`, `t2w`, `t2f`$\}$ | MRI modality |
| $\tau_m$ | Modality-specific threshold |
| $p_k(I)$ | $k$-th percentile of voxel intensities in $I$ |

---

## 2. Category A — Data Validation (Header-Only)

These checks operate exclusively on NRRD header metadata and require no image data loading, enabling fast rejection of structurally invalid files.

### A1. NRRD Header Validation

**Purpose.** Reject files that are not valid 3D volumetric images with spatial orientation metadata, which is required for all downstream spatial operations (registration, resampling, atlas alignment).

**Formulation.** A file passes if and only if:

$$\text{dimension}(I) = 3 \quad \wedge \quad (\texttt{space} \neq \varnothing \;\lor\; \texttt{space\_directions} \neq \varnothing)$$

The first condition rejects 2D single-slice images and 4D timeseries (e.g., dynamic contrast-enhanced or diffusion-weighted acquisitions stored as multi-frame volumes). The second condition ensures spatial orientation metadata is present; without it, the voxel-to-world coordinate mapping is undefined, making registration impossible.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| `require_3d` | `true` |
| `require_space_field` | `true` |
| Action | `block` |

---

### A2. Scout/Localizer Detection

**Purpose.** Reject low-resolution scout and localizer images that are routinely acquired at the start of an MRI session for scan planning but contain no diagnostic information and would introduce artifacts in the preprocessing pipeline.

**Formulation.** A file is classified as a scout/localizer if either condition holds:

$$\min_{i \in \{x,y,z\}} d_i < \delta_{\min} \qquad \text{or} \qquad \max_{i \in \{x,y,z\}} s_i > \sigma_{\max}$$

where $\delta_{\min}$ is the minimum acceptable number of voxels along any axis and $\sigma_{\max}$ is the maximum acceptable slice thickness.

**Rationale.** Scout images typically have very few slices (3–10) along one axis or extremely thick slices (>8 mm). The thresholds are set permissively to accommodate legitimate 2D multi-slice clinical acquisitions (T2-weighted and FLAIR sequences commonly have 15–50 slices with 5–6.5 mm slice spacing), while still rejecting true localizers.

**Configuration.**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $\delta_{\min}$ (`min_dimension_voxels`) | 10 | Lowered from default 64 for 2D multi-slice compatibility |
| $\sigma_{\max}$ (`max_slice_thickness_mm`) | 8.0 mm | Raised from default 5.0 to accommodate thick-slice FLAIR |
| Action | `block` | |

---

### A3. Voxel Spacing Validation

**Purpose.** Flag acquisitions with implausible voxel geometry, which may indicate corrupted headers, non-standard reconstruction, or acquisitions that would require extreme resampling.

**Formulation.** Three conditions are evaluated:

$$s_{\min} = \min_{i} s_i, \quad s_{\max} = \max_{i} s_i, \quad \alpha = \frac{s_{\max}}{s_{\min}}$$

| Condition | Criterion | Interpretation |
|-----------|-----------|----------------|
| Sub-voxel spacing | $s_{\min} < 0.2$ mm | Implausibly high resolution (possible header error) |
| Excessive spacing | $s_{\max} > 7.5$ mm | Slice gap too large for volumetric analysis |
| Extreme anisotropy | $\alpha > 20.0$ | Ratio of largest to smallest spacing exceeds 20:1 |

**Rationale.** Clinical brain MRI acquisitions span approximately 0.3–1.0 mm in-plane resolution with 1–6.5 mm slice spacing. The permissive bounds ($s_{\min} = 0.2$, $s_{\max} = 7.5$, $\alpha = 20$) accommodate the full range of 2D and 3D acquisitions in the MenGrowth cohort, including high-resolution T1-weighted (0.24 mm in-plane) and thick-slice 2D FLAIR (0.4 mm in-plane, 6.5 mm slice spacing, yielding anisotropy $\approx 16:1$).

**Configuration.**

| Parameter | Value |
|-----------|-------|
| $s_{\min}$ (`min_spacing_mm`) | 0.2 mm |
| $s_{\max}$ (`max_spacing_mm`) | 7.5 mm |
| $\alpha_{\max}$ (`max_anisotropy_ratio`) | 20.0 |
| Action | `warn` |

---

## 3. Category B — Image Quality (Data-Dependent)

These checks require loading image voxel data and compute intensity-based quality metrics.

### B1. Signal-to-Noise Ratio (SNR)

**Purpose.** Reject images with insufficient signal-to-noise ratio, which would compromise the accuracy of intensity-based registration (mutual information, cross-correlation) and histogram-based normalization methods.

**Formulation (corner-based method).** Background noise is estimated from eight corner cubes of size $c \times c \times c$ voxels ($c = 10$ by default) positioned at the vertices of the volume:

$$\mathcal{C} = \bigcup_{k=1}^{8} I[\text{corner}_k(c)]$$

The raw standard deviation of the corner samples is corrected for the Rayleigh distribution of magnitude MRI noise:

$$\sigma_{\text{noise}} = \text{std}(\mathcal{C}) \cdot \sqrt{\frac{2}{\pi}} \approx \text{std}(\mathcal{C}) \cdot 0.7979$$

The signal level is estimated as the 75th percentile of the foreground (voxels above the 10th percentile of all positive voxels):

$$I_{\text{fg}} = \{v \in I : v > p_{10}(I_{>0})\}, \quad \text{signal} = p_{75}(I_{\text{fg}})$$

$$\text{SNR} = \frac{\text{signal}}{\sigma_{\text{noise}}}$$

**Rayleigh correction.** In magnitude-reconstructed MRI, background noise follows a Rayleigh distribution rather than a Gaussian. The mean of a Rayleigh distribution with parameter $\sigma$ is $\sigma\sqrt{\pi/2}$, so the standard deviation of the measured noise overestimates the true noise parameter by a factor of $\sqrt{\pi/2} \approx 1.253$. The correction factor $\sqrt{2/\pi}$ compensates for this, preventing SNR underestimation by approximately 26%.

**Modality-specific thresholds.**

| Modality | Threshold ($\tau_m$) | Rationale |
|----------|---------------------|-----------|
| T1c | 8.0 | Gadolinium contrast enhances signal; higher baseline SNR expected |
| T1n | 6.0 | Native T1 without contrast agent |
| T2w | 5.0 | Higher tissue contrast but lower baseline SNR than T1 |
| T2-FLAIR | 4.0 | Inversion recovery reduces overall signal intensity; lower SNR inherent |

**Configuration.**

| Parameter | Value |
|-----------|-------|
| Method | `corner` |
| $c$ (`corner_cube_size`) | 10 voxels |
| Fallback threshold | 5.0 |
| Action | `block` |

---

### B2. Contrast Detection

**Purpose.** Reject images that are effectively uniform (all-black, all-white, or constant-value), indicating acquisition failure, incomplete reconstruction, or data corruption.

**Formulation.** Two complementary statistics are computed:

1. **Coefficient of variation (CV):** measures the relative spread of intensities.

$$\text{CV} = \frac{\text{std}(I)}{|\text{mean}(I)|}$$

If $\text{mean}(I) = 0$, the image is immediately rejected (all-zero). If $\text{CV} < \rho_{\min}$, the image has insufficient contrast.

2. **Uniform fraction:** measures the dominance of a single intensity value.

$$f_u = \frac{\max_v \text{count}(I = v)}{|I|}$$

If $f_u > \phi_{\max}$, more than $\phi_{\max}$ of all voxels share the same value, indicating a near-uniform image.

**Configuration.**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $\rho_{\min}$ (`min_std_ratio`) | 0.10 | CV below 10% indicates insufficient tissue contrast |
| $\phi_{\max}$ (`max_uniform_fraction`) | 0.95 | >95% uniform voxels indicates corrupted/blank image |
| Action | `block` | |

---

### B3. Intensity Outlier Detection

**Purpose.** Reject images containing numerical corruption (NaN, Inf) or extreme intensity spikes that would corrupt histogram-based preprocessing steps (WhiteStripe normalization, z-score standardization, KDE-based methods). Extreme dynamic range compresses useful signal into less than 10% of the intensity scale, making tissue segmentation and registration unreliable.

**Formulation.** Two sequential checks:

1. **Numerical validity:** If `reject_nan_inf = true` and $(|\{v \in I : v = \text{NaN}\}| > 0 \;\lor\; |\{v \in I : v = \text{Inf}\}| > 0)$, the file is immediately rejected.

2. **Outlier ratio:** For finite voxels $I_f$:

$$R = \frac{\max(I_f)}{p_{99}(I_f)}$$

If $R > \tau_m$, the maximum intensity is disproportionately large relative to the 99th percentile, indicating isolated extreme spikes (e.g., from RF artifacts, partial volume with metallic implants, or reconstruction errors).

**Rationale for the metric.** The ratio $\max / p_{99}$ quantifies how far the tail of the intensity distribution extends beyond the practical dynamic range. In well-behaved acquisitions, this ratio is typically 1.5–5.0. Values exceeding the threshold indicate that a small number of voxels ($< 1\%$) dominate the intensity range, which distorts histogram equalization and z-score normalization.

**Modality-specific thresholds.**

| Modality | Threshold ($\tau_m$) | Rationale |
|----------|---------------------|-----------|
| T1c | 10.0 | Gadolinium-enhanced regions create moderate tails |
| T1n | 15.0 | Native T1 may have fat-related bright voxels |
| T2w | 12.0 | CSF signal creates a moderate upper tail |
| T2-FLAIR | 20.0 | Inversion recovery can produce high variability at fluid boundaries; P95 in MenGrowth cohort = 17.4 |

**Configuration.**

| Parameter | Value |
|-----------|-------|
| `reject_nan_inf` | `true` |
| Fallback threshold | 10.0 |
| Action | `block` |

---

### B4. Motion Artifact Detection via Gradient Entropy

**Purpose.** Reject images exhibiting severe motion blur, ghosting from patient movement, or flow artifacts, which reduce the spatial frequency content and compromise intensity-based registration accuracy.

**Formulation.** The method quantifies image sharpness through the entropy of the gradient magnitude distribution:

1. **Compute 3D gradient magnitude** using Sobel operators along each axis:

$$G(x,y,z) = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2 + \left(\frac{\partial I}{\partial z}\right)^2}$$

where $\partial I / \partial x$ denotes the Sobel-filtered image along axis $x$.

2. **Construct gradient histogram** over nonzero gradient values with 256 bins:

$$h_k = |\{(x,y,z) : G(x,y,z) \in \text{bin}_k, \; G(x,y,z) > 0\}|, \quad k = 1, \ldots, 256$$

3. **Compute Shannon entropy** (base 2) of the normalized histogram:

$$p_k = \frac{h_k}{\sum_{j} h_j}, \qquad H_G = -\sum_{k=1}^{256} p_k \log_2 p_k$$

**Interpretation.** A sharp image with well-defined tissue boundaries produces a wide distribution of gradient magnitudes (high entropy). Motion blur smooths out edges, concentrating gradients near zero and collapsing the histogram into fewer effective bins (low entropy). An entropy below the threshold implies fewer than $2^{H_G}$ effective gradient bins, which is insufficient for reliable mutual-information or cross-correlation registration.

**Threshold derivation.** Thresholds were set at approximately 2 standard deviations below the empirical mean of each modality distribution in the MenGrowth cohort:

| Modality | Mean $H_G$ | $\sigma$ | Threshold ($\mu - 2\sigma$) | Adopted |
|----------|------------|----------|------------------------------|---------|
| T1c | 4.63 | 0.68 | 3.27 | 3.3 |
| T1n | 4.62 | 0.83 | 2.96 | 3.0 |
| T2w | 5.22 | 0.76 | 3.70 | 3.7 |
| T2-FLAIR | 4.67 | 0.97 | 2.73 | 2.7 |

**Note.** T2-FLAIR has a lower threshold because the fluid-attenuated inversion recovery technique suppresses CSF signal, reducing the overall dynamic range and consequently the gradient entropy, even in artifact-free acquisitions.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| Fallback threshold | 3.0 bits |
| GPU acceleration | CuPy Sobel (single-worker only) |
| Action | `block` |

---

### B5. Ghosting Detection

**Purpose.** Detect phase-encoding ghosting artifacts, where signal from the subject appears as displaced copies in the corners of the field of view due to phase errors during acquisition.

**Formulation.** Signal intensity in the eight corner cubes (same geometry as the SNR check, $c = 10$ voxels) is compared to the foreground intensity:

$$\bar{I}_{\text{corner}} = \text{mean}(|\mathcal{C}|), \qquad \bar{I}_{\text{fg}} = \text{mean}(\{v \in I : v > p_{10}(I_{>0})\})$$

$$\rho_g = \frac{\bar{I}_{\text{corner}}}{\bar{I}_{\text{fg}}}$$

If $\rho_g > 0.15$, ghosting is suspected: more than 15% of the foreground signal level is present in what should be empty background corners.

**Rationale.** In a well-acquired image, the corners of the volume should contain only noise (near-zero signal). Ghosting artifacts propagate replicated signal into these regions, elevating the corner mean. This metric is complementary to the SNR check: a low-SNR image may have high noise in the corners, but ghosting produces structured signal (not random noise) in the corners.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| $\rho_{g,\max}$ (`max_corner_to_foreground_ratio`) | 0.15 |
| $c$ (`corner_cube_size`) | 10 voxels |
| Action | `warn` |

---

## 4. Category C — Geometric Validation

These checks validate the spatial geometry of the acquisition.

### C1. Affine Matrix Validation

**Purpose.** Reject images with degenerate affine transformations that would cause numerical instability in registration and resampling.

**Formulation.** The affine rotation/scaling matrix $\mathbf{A}$ is extracted from the `space_directions` header field. Two conditions are checked:

1. **No NaN/Inf entries:** all elements of $\mathbf{A}$ must be finite.
2. **Determinant bounds:** the absolute determinant must satisfy:

$$\delta_{\min} \leq |\det(\mathbf{A})| \leq \delta_{\max}$$

**Interpretation.** The determinant $|\det(\mathbf{A})|$ equals the voxel volume in mm³. A near-zero determinant indicates a degenerate (collapsed) coordinate system; an excessively large determinant indicates implausible voxel sizes.

**Configuration.**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $\delta_{\min}$ (`min_det`) | 0.01 mm³ | Lowered from 0.1 to accommodate high-res 0.24 mm isotropic data ($0.24^3 \approx 0.014$) |
| $\delta_{\max}$ (`max_det`) | 100.0 mm³ | Upper bound for reasonable clinical acquisitions |
| Action | `block` | |

---

### C2. Field-of-View Consistency

**Purpose.** Detect acquisitions with highly asymmetric spatial coverage, which would introduce excessive zero-padding during the cubic padding preprocessing step and waste computational resources.

**Formulation.** The field of view along each axis is:

$$\text{FOV}_i = d_i \cdot s_i \quad (mm), \qquad r = \frac{\max_i \text{FOV}_i}{\min_i \text{FOV}_i}$$

| Condition | Action |
|-----------|--------|
| $r > 5.0$ | `block` |
| $3.0 < r \leq 5.0$ | `warn` |
| $r \leq 3.0$ | pass |

**Rationale.** A FOV ratio exceeding 5:1 indicates that one axis covers substantially less anatomy than the others (e.g., a thin axial slab covering only 33 mm in the superior-inferior direction versus 170 mm in-plane). Such acquisitions provide insufficient brain coverage for volumetric analysis and would require impractical amounts of zero-padding.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| `warn_ratio` | 3.0 |
| `block_ratio` | 5.0 |

---

### C3. Orientation Consistency (Study-Level)

**Purpose.** Verify that all modalities within a single imaging study share the same spatial orientation, ensuring consistent voxel-to-world coordinate mapping for multi-modal registration.

**Formulation.** For each study containing modalities $\{m_1, m_2, \ldots, m_n\}$, extract the `space` header field from each modality:

$$\mathcal{O} = \{\text{space}(m_i) : i = 1, \ldots, n\}$$

The check passes if and only if $|\mathcal{O}| = 1$ (all modalities share the same orientation, e.g., all `left-posterior-superior` or all `right-anterior-superior`).

**Configuration.**

| Parameter | Value |
|-----------|-------|
| Action | `warn` |

---

### C4. Brain Coverage Validation

**Purpose.** Reject acquisitions where the physical extent along any axis is too small to capture the full brain, ensuring that downstream atlas registration has sufficient anatomical context.

**Formulation.** The physical extent along each axis is:

$$e_i = d_i \cdot s_i \quad (mm)$$

The check fails if:

$$\min_{i \in \{x,y,z\}} e_i < e_{\min}$$

**Rationale.** The adult human brain measures approximately 140 mm (anterior-posterior) × 170 mm (left-right) × 120 mm (superior-inferior). A minimum extent of 100 mm allows for some variation while rejecting acquisitions that cover less than the full brain (e.g., targeted slab acquisitions of 30–70 mm extent). This threshold is critical because atlas-based registration (e.g., SRI24) assumes approximately full brain coverage; partial coverage would cause registration failure or extreme deformation artifacts.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| $e_{\min}$ (`min_extent_mm`) | 100.0 mm |
| Action | `block` |

---

## 5. Category D — Longitudinal Validation (Patient-Level)

These checks operate across all studies for a given patient.

### D1. Temporal Ordering

**Purpose.** Verify that the study identifiers within a patient reflect the expected temporal ordering, catching potential data-entry or reorganization errors.

**Formulation.** Parse numeric study indices from study identifiers. Let $\{n_1, n_2, \ldots, n_k\}$ be the extracted indices. The check passes if:

$$(n_1, n_2, \ldots, n_k) = \text{sort}(n_1, n_2, \ldots, n_k)$$

**Configuration.**

| Parameter | Value |
|-----------|-------|
| Action | `warn` |

---

### D2. Modality Consistency Across Timepoints

**Purpose.** Verify that all longitudinal timepoints for a patient were acquired with the same set of MRI modalities, ensuring comparable measurements across the growth trajectory.

**Formulation.** For each study $j$ of patient $p$, let $\mathcal{M}_j$ denote the set of acquired modalities. The check passes if:

$$\mathcal{M}_1 = \mathcal{M}_2 = \cdots = \mathcal{M}_k$$

**Configuration.**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Enabled | `false` | Disabled by default; overly strict for clinical data where protocol changes between visits are common |
| Action | `warn` | |

---

## 6. Category E — Preprocessing Requirements

### E1. Registration Reference Availability

**Purpose.** Ensure that at least one modality suitable for registration exists in the study. The preprocessing pipeline requires a reference modality (typically T1-weighted) for both intra-study co-registration and inter-study atlas registration.

**Formulation.** A priority ordering is defined over modalities:

$$\text{priority}: \texttt{t1n} > \texttt{t1c} > \texttt{t2f} > \texttt{t2w}$$

The check passes if at least one modality from the priority list is present in the study:

$$\exists \; m \in \text{priority} : m \in \mathcal{M}_{\text{study}}$$

**Rationale.** T1-weighted sequences (native or contrast-enhanced) provide the best anatomical contrast for rigid and affine registration to standard brain atlases. T1-native is preferred over T1-contrast because gadolinium enhancement alters tissue intensities and can bias similarity metrics. T2-weighted sequences serve as fallbacks when T1 is unavailable.

**Configuration.**

| Parameter | Value |
|-----------|-------|
| Priority | `t1n > t1c > t2f > t2w` |
| Action | `block` |

---

## 7. Pipeline Execution

### 7.1 Validation Phase (Parallelized)

All patients are processed in parallel using `ProcessPoolExecutor`. For each patient:

1. **File-level checks** (A1–A3, B1–B5, C1–C2, C4) are applied to every NRRD file.
2. **Study-level checks** (C3, E1) are applied to each study.
3. **Patient-level checks** (D1, D2) are applied across all studies.

Each check returns a `ValidationResult` with fields: `passed` (bool), `check_name` (str), `message` (str), `action` ∈ {`warn`, `block`}, and `details` (dict of metric values).

### 7.2 Removal Phase (Sequential)

For each patient with at least one blocking failure:

- Count the number of clean studies $n_{\text{clean}}$ (no blocking issues).
- If $n_{\text{clean}} \geq$ `min_studies_per_patient` (default: 2): remove only the blocked studies, preserving the patient.
- Otherwise: remove the entire patient (insufficient longitudinal data).

This strategy maximizes data retention: individual corrupted timepoints are excised without discarding entire longitudinal trajectories.

### 7.3 Outputs

| File | Content |
|------|---------|
| `quality/rejected_files.csv` | One row per rejected file with patient ID, study, modality, rejection reason, and pipeline stage |
| `quality/quality_issues.csv` | One row per (file, check) pair for all flagged issues (block and warn), including raw metric values as JSON |
| `quality/quality_metrics.json` | Nested JSON with full check results for all evaluated files (pass and fail), organized by patient → study → modality → check |

---

## 8. Summary of All Quality Checks

| ID | Check | Level | Action | Key Metric | Threshold |
|----|-------|-------|--------|------------|-----------|
| A1 | NRRD validation | file | block | dimension, space field | dim = 3, space ≠ ∅ |
| A2 | Scout detection | file | block | min dimension, max spacing | ≥ 10 vox, ≤ 8.0 mm |
| A3 | Voxel spacing | file | warn | spacing range, anisotropy | [0.2, 7.5] mm, α ≤ 20 |
| B1 | SNR | file | block | corner-based SNR | t1c: 8, t1n: 6, t2w: 5, t2f: 4 |
| B2 | Contrast | file | block | CV, uniform fraction | CV ≥ 0.10, $f_u$ < 0.95 |
| B3 | Intensity outliers | file | block | max / p99 ratio | t1c: 10, t1n: 15, t2w: 12, t2f: 20 |
| B4 | Motion (gradient entropy) | file | block | $H_G$ (bits) | t1c: 3.3, t1n: 3.0, t2w: 3.7, t2f: 2.7 |
| B5 | Ghosting | file | warn | corner/foreground ratio | ≤ 0.15 |
| C1 | Affine validation | file | block | $|\det(\mathbf{A})|$ | [0.01, 100] mm³ |
| C2 | FOV consistency | file | block/warn | FOV axis ratio | warn > 3, block > 5 |
| C3 | Orientation consistency | study | warn | unique orientations | = 1 |
| C4 | Brain coverage | file | block | min physical extent | ≥ 100 mm |
| D1 | Temporal ordering | patient | warn | study index sequence | sorted |
| D2 | Modality consistency | patient | warn | modality set equality | disabled |
| E1 | Registration reference | study | block | priority modality present | t1n > t1c > t2f > t2w |
