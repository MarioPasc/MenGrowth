"""Pipeline attrition diagram for the quality HTML report.

Computes patient/study/volume counts at each filtering stage from
rejected_files.csv and the current dataset state, and generates a
pure HTML/CSS flow diagram.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Stage labels in pipeline order
STAGE_LABELS: Dict[int, str] = {
    0: "Reorganization",
    1: "Completeness Filter",
    2: "Quality Filter",
    3: "Manual Curation",
}


def compute_attrition_data(
    quality_dir: Path,
    mengrowth_dir: Path,
) -> Optional[List[Dict]]:
    """Compute pipeline attrition data from rejection records and current state.

    Reads rejected_files.csv to determine how many patients/studies/files
    were removed at each stage, then combines with the current dataset
    counts to produce a forward-flowing attrition table.

    Args:
        quality_dir: Path to quality output directory containing rejected_files.csv.
        mengrowth_dir: Path to MenGrowth-2025 dataset directory.

    Returns:
        List of dicts with keys: stage, label, patients, studies, volumes.
        Returns None if rejected_files.csv is not found.
    """
    rejected_csv = quality_dir / "rejected_files.csv"
    if not rejected_csv.exists():
        logger.warning(
            f"rejected_files.csv not found at {rejected_csv}, skipping attrition"
        )
        return None

    # ── Parse rejected files by stage ──
    stage_patients: Dict[int, set] = defaultdict(set)
    stage_studies: Dict[int, set] = defaultdict(set)
    stage_files: Dict[int, int] = defaultdict(int)

    with open(rejected_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stage = int(row.get("stage", 0))
            except (ValueError, TypeError):
                stage = 0
            patient_id = row.get("patient_id", "")
            study_name = row.get("study_name", "")

            stage_patients[stage].add(patient_id)
            stage_studies[stage].add((patient_id, study_name))
            stage_files[stage] += 1

    # ── Count current dataset ──
    current_patients = (
        sorted(
            [d for d in mengrowth_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if mengrowth_dir.exists()
        else []
    )

    final_patients = len(current_patients)
    final_studies = sum(
        len([s for s in p.iterdir() if s.is_dir()]) for p in current_patients
    )
    final_volumes = sum(
        len(list(s.glob("*.nrrd")) + list(s.glob("*.nii.gz")))
        for p in current_patients
        for s in p.iterdir()
        if s.is_dir()
    )

    # ── Determine which stages are present ──
    present_stages = sorted(stage_files.keys())
    if not present_stages:
        logger.info("No rejections found in rejected_files.csv")
        return None

    # ── Build attrition flow (working backwards from final) ──
    # Total removed across all stages
    total_removed_patients = sum(len(v) for v in stage_patients.values())
    total_removed_studies = sum(len(v) for v in stage_studies.values())
    total_removed_files = sum(stage_files.values())

    # Note: patient/study counts from rejected CSV may overlap across stages
    # (a patient removed in stage 1 won't appear in stage 2). The cumulative
    # counts work forward from the starting total.
    starting_patients = final_patients + total_removed_patients
    starting_studies = final_studies + total_removed_studies
    starting_volumes = final_volumes + total_removed_files

    attrition: List[Dict] = []

    # Starting point
    attrition.append(
        {
            "stage": -1,
            "label": "Raw Dataset",
            "patients": starting_patients,
            "studies": starting_studies,
            "volumes": starting_volumes,
            "removed_patients": 0,
            "removed_studies": 0,
            "removed_volumes": 0,
        }
    )

    # Each filtering stage
    running_patients = starting_patients
    running_studies = starting_studies
    running_volumes = starting_volumes

    for stage_num in sorted(STAGE_LABELS.keys()):
        if stage_num not in stage_files:
            continue

        removed_p = len(stage_patients[stage_num])
        removed_s = len(stage_studies[stage_num])
        removed_v = stage_files[stage_num]

        running_patients -= removed_p
        running_studies -= removed_s
        running_volumes -= removed_v

        attrition.append(
            {
                "stage": stage_num,
                "label": STAGE_LABELS[stage_num],
                "patients": running_patients,
                "studies": running_studies,
                "volumes": running_volumes,
                "removed_patients": removed_p,
                "removed_studies": removed_s,
                "removed_volumes": removed_v,
            }
        )

    logger.info(
        f"Attrition: {starting_patients} → {final_patients} patients, "
        f"{starting_studies} → {final_studies} studies across "
        f"{len(present_stages)} filtering stages"
    )

    return attrition


def generate_attrition_html(attrition_data: List[Dict]) -> str:
    """Generate pure HTML/CSS attrition flow diagram.

    Args:
        attrition_data: Attrition data from compute_attrition_data().

    Returns:
        HTML string to embed in the quality report.
    """
    if not attrition_data:
        return ""

    html_parts = [
        "<h2>Pipeline Attrition</h2>\n",
        '<div class="attrition-flow">\n',
    ]

    for i, entry in enumerate(attrition_data):
        is_first = i == 0
        is_last = i == len(attrition_data) - 1

        # Arrow with removal counts (between boxes)
        if not is_first:
            removed_p = entry["removed_patients"]
            removed_s = entry["removed_studies"]
            removed_v = entry["removed_volumes"]
            html_parts.append(
                f'  <div class="attrition-arrow">\n'
                f'    <div class="arrow-line">&#x2192;</div>\n'
                f'    <div class="arrow-removed">'
                f"-{removed_p}P / -{removed_s}S / -{removed_v}V"
                f"</div>\n"
                f"  </div>\n"
            )

        # Box
        if is_first:
            box_class = "attrition-box attrition-start"
        elif is_last:
            box_class = "attrition-box attrition-final"
        else:
            box_class = "attrition-box attrition-intermediate"

        html_parts.append(
            f'  <div class="{box_class}">\n'
            f'    <div class="attrition-label">{entry["label"]}</div>\n'
            f'    <div class="attrition-counts">\n'
            f"      <span>{entry['patients']} patients</span>\n"
            f"      <span>{entry['studies']} studies</span>\n"
            f"      <span>{entry['volumes']} volumes</span>\n"
            f"    </div>\n"
            f"  </div>\n"
        )

    html_parts.append("</div>\n")
    return "".join(html_parts)


# CSS to inject into the HTML report <style> block
ATTRITION_CSS = """
        .attrition-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 4px;
            margin: 20px 0 30px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .attrition-box {
            padding: 12px 16px;
            border-radius: 6px;
            text-align: center;
            min-width: 120px;
            border: 2px solid;
        }
        .attrition-start {
            background-color: #e8f5e9;
            border-color: #4caf50;
        }
        .attrition-intermediate {
            background-color: #e3f2fd;
            border-color: #2196f3;
        }
        .attrition-final {
            background-color: #e0f2f1;
            border-color: #009688;
        }
        .attrition-label {
            font-weight: bold;
            font-size: 13px;
            margin-bottom: 6px;
            color: #2c3e50;
        }
        .attrition-counts {
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-size: 12px;
            color: #555;
        }
        .attrition-arrow {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 80px;
        }
        .arrow-line {
            font-size: 24px;
            color: #999;
            line-height: 1;
        }
        .arrow-removed {
            font-size: 10px;
            color: #e74c3c;
            font-weight: bold;
            white-space: nowrap;
        }
"""
