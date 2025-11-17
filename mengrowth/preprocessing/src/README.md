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

4. **Intra-study rigid registration to T1c (native space)**  
   - Choose T1c as the reference modality due to typically higher resolution and tumour-contrast. (but could be tunable in the `configs/preprocessing.yaml` file) 
   - Register T1n → T1c, T2w → T1c, T2-FLAIR → T1c using rigid (6 d.o.f) transform with mutual-information metric (parameters tunable in the `configs/preprocessing.yaml` file, but we only contemplate rigid registration).  
   - Store transform matrices (artifacts folder).

5. **Transform into atlas space + resample to 1 mm³ isotropic**  
   - Register T1c (native) → SRI24-atlas (or chosen standard) with a rigid/affine transform (ATLAS path tunable in the `configs/preprocessing.yaml`).  
   - For each other modality, compute composed transform:  
     $$
       T_{M\to\text{atlas}} = T_{\mathrm{T1c}\to\text{atlas}} \circ T_{M\to\mathrm{T1c}}
     $$  
   - Ensures uniform geometry across subjects and studies.

6. **Skull-stripping in atlas space + mask propagation**  
   - On the resampled T1c in atlas space (or not T1C, but the reference modality set in the `configs/preprocessing.yaml` file), apply a skull-stripping algorithm to extract intracranial brain mask.  
   - Propagate this mask to T1n, T2w, T2-FLAIR of the same study.  
   - Set voxels outside brain mask to zero.

7. **Intensity normalisation (multi-modal)**  
   - For each modality $m$, apply a two-step normalisation:  
     a) Global inter-subject scale normalisation (e.g., Nyúl or WhiteStripe) using training-set landmarks.  
     b) Per-volume robust z-score normalisation (using brain-mask voxels, clipping e.g. at 0.5–99.5 percentiles).  
   - Produce harmonised intensity volumes ready for downstream processing.

---

## Longitudinal alignment pipeline  
For each patient with baseline study $S_0$ and follow-up studies $S_i$ ($i\ge1$):

7. **Longitudinal registration (atlas space)**  
   - Using the T1c volumes of each follow-up $S_i$ and baseline $S_0$ (both in atlas space), compute a rigid or affine (optionally deformable) transform  
     $$
       T^{\mathrm{long}}_{S_i\to S_0}: T1c_{S_i} \to T1c_{S_0}
     $$  
   - Apply $T^{\mathrm{long}}_{S_i\to S_0}$ to the follow-up T1n, T2w, T2-FLAIR and any tumour/tissue/mask volumes of $S_i$.  
   - As result, all time-points are aligned to the baseline T1c geometry.

---

## Notes & caveats  
- The pipeline is designed for a multi-centre, multi-modal, longitudinal meningioma dataset: uniform geometry and intensity spaces are critical for growth prediction and classification.  
- The code must follow the defined OOP techniques, where we define a `base.py` class with the most important functions that the specific algorithms (that inherit from that class) must implement.
- All steps should log visualizations.
- If any step can log metadata about the quality of it, it should store it in `artifacts/` folder specified in `configs/preprocessing.yaml`
- All steps must be robust to variations in scanner, vendor, sequence parameters.

---

## References  
- Bielak et al., “Impact of image preprocessing methods on reproducibility of radiomics” Med. Phys., 2020.  
- Dorfner F.J. et al., “A review of deep learning for brain tumour analysis in MRI” npj Precision Oncology, 2025.  
- PREDICT-GBM: “Platform for Robust Evaluation and Development of Individualized Computational Tumor Models in Glioblastoma” arXiv 2025. 

---

