# preprocessing  
This sub-module implements the image-preprocessing pipeline for the multi-centre, multi-modal, longitudinal meningioma growth-prediction project (MenGrowth). The aim is to produce for each study and each modality a volume that is bias-field corrected, co-registered (intra-modal, atlas space, longitudinal), skull-stripped, resampled to 1 mm³ isotropic, and intensity-harmonised — ready for downstream feature extraction, super-resolution training, and classification.

---

## Overview  
The pipeline is structured in two layers:  
- **Per-study processing**: handle each time-point (pre-op or follow-up) independently, across modalities \{T1c, T1n, T2w, T2-FLAIR\}. Each study may be missing up to one sequence.   
- **Longitudinal alignment**: for each patient, align follow-up to baseline in atlas space so that volumes are geometrically comparable across time.

This design is consistent with growth-modelling platforms such as PREDICT-GBM, which convert DICOM to NIfTI, normalise intensities, and co-register all MRI modalities to the SRI24 atlas space.

---

## Per-study processing pipeline  
For each study $S_i$ (time-point) of each patient, execute the following steps:

1. **DICOM → NIfTI, metadata harmonisation**  
   - Convert all sequences (T1c, T1n, T2w, T2-FLAIR) to NIfTI.  
   - Standardise orientation (RAS), ensure consistent naming and modality labels.
   - Optionally remove background voxels and keep only the head.

2. **Bias-field correction (N4)**  
   - Apply N4 Bias Field Correction to each modality in its native space.  

3. **Resampling to Isotropic Resolution**
    - Resample the volumes to match 1x1x1 mm³ resolution. This part will leverage either interpolator or deep-learning algorithms through an easy-to-use interface.
4. **Light per-volume normalisation (before registration).**
   - For each modality (m), and using all in-head voxels (or a coarse brain mask if available), apply a simple robust scaling, e.g.
      $$
      I_m'(\mathbf{x}) = \frac{I_m(\mathbf{x}) - P_{m,,p_1}}{P_{m,,p_2} - P_{m,,p_1}},
      $$
      where $P_{m,,p_1}$ and $P_{m,,p_2}$ are, for instance, the 0.5th and 99.5th percentiles of in-head intensities. This maps intensities into a comparable dynamic range across scans, while preserving monotonicity and tissue ordering. This step is cheap, monotonic, and primarily serves to:
      * reduce the influence of outliers on MI/NMI;
      * stabilise the joint histograms across scanners and time-points.

5. **Intra-study rigid registration to T1c (native space)**  
   - Choose T1c as the reference modality due to typically higher resolution and tumour-contrast. (but could be tunable in the `configs/preprocessing.yaml` file) 
   - Register T1n → T1c, T2w → T1c, T2-FLAIR → T1c using rigid (6 d.o.f) transform with mutual-information metric (parameters tunable in the `configs/preprocessing.yaml` file, but we only contemplate rigid registration).  
   - Store transform matrices (artifacts folder).

6. **Transform into atlas space**  
   - Register T1c (native) → SRI24-atlas (or chosen standard) with a rigid/affine transform (ATLAS path tunable in the `configs/preprocessing.yaml`).  
   - For each other modality, compute composed transform:  
     $$
       T_{M\to\text{atlas}} = T_{\mathrm{T1c}\to\text{atlas}} \circ T_{M\to\mathrm{T1c}}
     $$  
   - Ensures uniform geometry across subjects and studies.

7. **Skull-stripping in atlas space + mask propagation**  
   - On the resampled T1c in atlas space (or not T1C, but the reference modality set in the `configs/preprocessing.yaml` file), apply a skull-stripping algorithm to extract intracranial brain mask.  
   - Propagate this mask to T1n, T2w, T2-FLAIR of the same study.  
   - Set voxels outside brain mask to zero.
   - At this point we have geometry harmonised and brain-extracted, with only a *light* per-volume scaling having influenced the registration.

8. **Intensity normalisation (multi-modal)**  
   - Per study (S_i), now in atlas space and brain-only:
      1. Global inter-subject intensity standardisation (Nyúl/WhiteStripe). For each modality (m), apply Nyúl-style standardisation or WhiteStripe-based mapping using landmarks learned from a training cohort, so that similar tissues (e.g. NAWM, CSF) occupy similar positions in the intensity scale across subjects and sites.([PubMed][1])
      2. Per-volume robust z-score normalisation. Within the brain mask, compute robust mean and standard deviation (e.g. after clipping at 0.5–99.5% percentiles) and map
      $$
      \tilde{I}_m(\mathbf{x}) = \frac{I_m^{\text{std}}(\mathbf{x}) - \mu_m}{\sigma_m}.
      $$


This ordering is aligned with many multi-centre pipelines where images are first bias-corrected and registered to a standard space, and only then undergo intensity normalisation and harmonisation for quantitative analysis.([PubMed Central][4])

---

## Longitudinal alignment pipeline  
For each patient with baseline study $S_0$ and follow-up studies $S_i$ ($i\ge1$):

8. **Longitudinal registration (atlas space)**  
   - Using the T1c volumes of each follow-up $S_i$ and baseline $S_0$ (both in atlas space), compute a rigid or affine (optionally deformable) transform  
     $$
       T^{\mathrm{long}}_{S_i\to S_0}: T1c_{S_i} \to T1c_{S_0}
     $$  
   - Apply $T^{\mathrm{long}}_{S_i\to S_0}$ to the follow-up T1n, T2w, T2-FLAIR and any tumour/tissue/mask volumes of $S_i$.  
   - As result, all time-points are aligned to the baseline T1c geometry.

Longitudinal registration will operate on the fully standardised $\tilde{I}_m$ volumes, since we want maximum robustness to scanner changes between time-points. Leveraging the *fully standardised* atlas-space T1c for estimating the longitudinal transform
$$
T^{\text{long}}*{S_i \to S_0}: \ T1c*{S_i} \to T1c_{S_0},
$$
reduces the chance that differences in scanner protocol are mistaken for anatomical changes. This is conceptually similar to using intensity standardisation to improve longitudinal atrophy quantification, as explored for SIENA-like pipelines.([MDPI][6])

---

## Notes & caveats  
- The pipeline is designed for a multi-centre, multi-modal, longitudinal meningioma dataset: uniform geometry and intensity spaces are critical for growth prediction and classification.  
- The code must follow the defined OOP techniques, where we define a `base.py` class with the most important functions that the specific algorithms (that inherit from that class) must implement.
- All steps should log visualizations.
- If any step can log metadata about the quality of it, it should store it in `artifacts/` folder specified in `configs/preprocessing.yaml`
- All steps must be robust to variations in scanner, vendor, sequence parameters.


[1]: https://pubmed.ncbi.nlm.nih.gov/10571928/?utm_source=chatgpt.com "On standardizing the MR image intensity scale - PubMed - NIH"
[2]: https://ulasbagci.files.wordpress.com/2010/11/printed_prl_effectofintnsitystandardization-0-s0167865509002384-m.pdf?utm_source=chatgpt.com "The role of intensity standardization in medical image registration"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4215426/?utm_source=chatgpt.com "Statistical normalization techniques for magnetic ..."
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6758567/?utm_source=chatgpt.com "Evaluating the Impact of Intensity Normalization on MR ..."
[5]: https://esmed.org/MRA/bme/article/download/1550/1255/?utm_source=chatgpt.com "A Review of Methods for Bias Correction in Medical Images"
[6]: https://www.mdpi.com/2076-3417/11/4/1773?utm_source=chatgpt.com "Evaluating the Effect of Intensity Standardisation on ..."
