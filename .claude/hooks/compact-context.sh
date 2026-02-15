#!/usr/bin/env bash
# compact-context.sh
# Generates a compact project context summary for LLM consumption.
# Outputs a single text block to stdout (pipe to file or clipboard).
#
# Usage:
#   ./compact-context.sh                    # Print to stdout
#   ./compact-context.sh > context.txt      # Save to file
#   ./compact-context.sh | pbcopy           # macOS clipboard
#   ./compact-context.sh | xclip -sel clip  # Linux clipboard
#
# Options:
#   --full      Include CLAUDE.md verbatim instead of the compact summary
#   --tree      Include directory tree scan of the project
#   --config    Include sample configuration excerpts

set -euo pipefail

# ── Resolve project root (directory containing this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ── Parse arguments ──────────────────────────────────────────────────────────
INCLUDE_FULL=false
INCLUDE_TREE=false
INCLUDE_CONFIG=false

for arg in "$@"; do
    case "$arg" in
        --full)    INCLUDE_FULL=true ;;
        --tree)    INCLUDE_TREE=true ;;
        --config)  INCLUDE_CONFIG=true ;;
        --help|-h)
            echo "Usage: $0 [--full] [--tree] [--config]"
            echo "  --full    Include full CLAUDE.md instead of compact summary"
            echo "  --tree    Include live directory tree of the project"
            echo "  --config  Include sample configuration excerpts"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# ── Header ───────────────────────────────────────────────────────────────────
cat <<'HEADER'
<project name="MenGrowth" type="biomedical-imaging-pipeline" lang="python>=3.11">
HEADER

# ── Full or compact mode ────────────────────────────────────────────────────
if [ "$INCLUDE_FULL" = true ] && [ -f "${PROJECT_ROOT}/CLAUDE.md" ]; then
    echo ""
    echo "<claude_md>"
    cat "${PROJECT_ROOT}/CLAUDE.md"
    echo "</claude_md>"
    echo ""
else
    # Compact summary (fits ~2k tokens)
    cat <<'COMPACT'

<summary>
MenGrowth: Automated MRI pipeline for longitudinal meningioma growth prediction.
Two stages: (1) Data Curation — transforms heterogeneous raw NRRD through 6 phases
(reorganize, filter, quality filter, re-ID, analyze, visualize) into a standardized
dataset. (2) BraTS-like Preprocessing — transforms curated NRRD through 8 configurable
steps into analysis-ready NIfTI (harmonization, N4, resampling, padding, registration,
skull stripping, intensity normalization, longitudinal registration).
</summary>

<architecture>
mengrowth/
  cli/
    curate_dataset.py                  # Curation entry: mengrowth-curate-dataset
    preprocess.py                      # Preprocessing entry: mengrowth-preprocess
  preprocessing/
    config.py                          # Curation @dataclass config + YAML parsing
    utils/
      reorganize_raw_data.py           # Phase 1: standardize directory layout
      filter_raw_data.py               # Phase 2: completeness + Phase 4: re-ID
      quality_filtering.py             # Phase 3: 15-check quality validation
      metadata.py                      # Clinical metadata (XLSX→JSON→dataclass)
    quality_analysis/
      analyzer.py                      # Phase 5: population metrics (SimpleITK)
      metrics.py                       # Per-image metric functions
      visualize.py                     # Phase 6: plots + HTML report
    src/
      config.py                        # Preprocessing @dataclass config + StepRegistry
      preprocess.py                    # PreprocessingOrchestrator (step engine)
      base.py                          # BasePreprocessingStep ABC
      steps/                           # Step handlers (8 steps)
      data_harmonization/              # NRRD→NIfTI, reorient, head masking
      bias_field_correction/           # N4 via SimpleITK
      resampling/                      # BSpline, ECLARE, Composite
      registration/                    # ANTs (nipype + antspyx)
      skull_stripping/                 # HD-BET, SynthStrip
      normalization/                   # Z-score, KDE, FCM, WhiteStripe, PercentileMinMax, LSQ

Curation: Raw → Reorganize → Filter → Quality → Re-ID → Analyze → Visualize
Preprocessing: Curated NRRD → 8 configurable steps → Analysis-Ready NIfTI
Config: YAML → typed @dataclass trees. Every threshold has code defaults.
</architecture>

<identifiers>
Raw patient: P{N} (e.g., P1, P42)
Anon patient: MenGrowth-{XXXX} (e.g., MenGrowth-0001)
Anon study: MenGrowth-{XXXX}-{YYY} (e.g., MenGrowth-0001-000)
Modalities: t1c (contrast T1), t1n (native T1), t2w (T2-weighted), t2f (FLAIR)
</identifiers>

<quality_checks count="15">
Block: A1(NRRD header), A2(scout), B1(SNR), B2(contrast), C1(affine),
       C2-extreme(FOV>5), C4(brain coverage≥100mm), E1(registration ref)
Warn:  A3(spacing), B3(intensity outliers), B4(motion entropy≥3.0),
       B5(ghosting≤0.15), C2-moderate(FOV>3), C3(orientation), D1(temporal order)
Disabled: D3(modality consistency)
</quality_checks>

<preprocessing_steps>
1. Data Harmonization (modality): NRRD→NIfTI, RAS reorientation, background zeroing
2. Bias Field Correction (modality): N4ITK, shrink_factor=4, 4-level multi-resolution
3. Resampling (modality): BSpline/ECLARE/Composite → isotropic 1mm³
4. Cubic Padding (study): Symmetric zero-pad to cubic FOV
5. Registration (study): Intra-study coregistration + atlas alignment (ANTs)
6. Skull Stripping (study): HD-BET or SynthStrip brain extraction
7. Intensity Normalization (modality): z-score/KDE/FCM/WhiteStripe/PercentileMinMax/LSQ
8. Longitudinal Registration (patient): Cross-timepoint alignment (quality-based ref selection)
</preprocessing_steps>

ENVIRONMENT:
  Conda: ~/.conda/envs/growth/bin
  Tests: ~/.conda/envs/growth/bin/python -m pytest tests/ -v
  Curation CLI: mengrowth-curate-dataset --config configs/raw_data.yaml ...
  Preprocessing CLI: mengrowth-preprocess --config configs/icaiserver/preprocessing_icai.yaml ...

<conventions>
- Non-destructive (curation copies; preprocessing uses temp-file-then-rename)
- Reproducible (sorted outputs, YAML configs with code defaults)
- Traceable (rejected_files.csv, quality_issues.csv, quality_metrics.json)
- Full typing, @dataclass configs (picklable), Python logging
- Quality filtering before anonymization → gap-free IDs
- Preprocessing steps configurable via YAML list + StepRegistry pattern matching
</conventions>

COMPACT
fi

# ── Optional: live directory tree ────────────────────────────────────────────
if [ "$INCLUDE_TREE" = true ]; then
    echo "<directory_tree>"
    if command -v tree &>/dev/null; then
        tree -L 3 -I '__pycache__|*.pyc|.git|node_modules|.eggs|*.egg-info' \
            "${PROJECT_ROOT}/mengrowth" 2>/dev/null || echo "(mengrowth/ not found)"
    else
        # Fallback: find-based tree
        find "${PROJECT_ROOT}/mengrowth" -maxdepth 3 \
            -not -path '*/__pycache__/*' \
            -not -path '*/.git/*' \
            -not -name '*.pyc' \
            2>/dev/null | sort | head -80 || echo "(mengrowth/ not found)"
    fi
    echo "</directory_tree>"
    echo ""
fi

# ── Optional: config excerpts ────────────────────────────────────────────────
if [ "$INCLUDE_CONFIG" = true ]; then
    echo "<config_excerpts>"
    for cfg in "${PROJECT_ROOT}/configs/raw_data.yaml" \
               "${PROJECT_ROOT}/configs/templates/quality_analysis.yaml" \
               "${PROJECT_ROOT}/configs/icaiserver/preprocessing_icai.yaml"; do
        if [ -f "$cfg" ]; then
            echo ""
            echo "<!-- $(basename "$cfg") (first 60 lines) -->"
            head -60 "$cfg"
            echo "..."
        fi
    done
    echo "</config_excerpts>"
    echo ""
fi

# ── Footer ───────────────────────────────────────────────────────────────────
cat <<'FOOTER'

<usage_hints>
- To add a quality check: quality_filtering.py + config dataclass in config.py
- To change thresholds: configs/raw_data.yaml → quality_filtering section
- To debug rejections: quality/rejected_files.csv (filter by stage)
- To re-run analysis only: --skip-reorganize --skip-filter --skip-quality-filter
- To add a preprocessing step: src/steps/ handler + StepMetadata + config in src/config.py
- To change preprocessing step order: modify steps: list in preprocessing YAML
- To debug preprocessing: check viz PNGs in {viz_root}/, artifacts in {artifacts}/
- Curation entry: mengrowth-curate-dataset --config ... --qa-config ... --input-root ... --output-root ...
- Preprocessing entry: mengrowth-preprocess --config ... [--patient ...] [--verbose]
</usage_hints>

</project>
FOOTER
