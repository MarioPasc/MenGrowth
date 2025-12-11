"""Parser for ANTs diagnostic output."""

import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def parse_ants_diagnostic_output(stdout: str) -> Dict[str, Any]:
    """Parse ANTs registration diagnostic output.

    Extracts iteration-level convergence metrics, timing information,
    and registration stage details from ANTs stdout.

    Args:
        stdout: Captured stdout from ANTs registration

    Returns:
        Dictionary with parsed timing, convergence, and stage information

    Example output format from ANTs:
        DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST
         2DIAGNOSTIC,     1, -1.090614080429e+00, inf, 1.2849e-01, 1.2849e-01,
         2DIAGNOSTIC,     2, -1.118562936783e+00, inf, 1.4995e-01, 2.1458e-02,
        ...
          Elapsed time (stage 0): 1.1270e+01

        Total elapsed time: 1.1271e+01
    """
    parsed = {
        "stages": [],
        "total_elapsed_time_seconds": None,
        "command_lines_ok": False,
        "full_output": stdout
    }

    # Check for successful command lines
    if "All_Command_lines_OK" in stdout:
        parsed["command_lines_ok"] = True

    # Parse total elapsed time
    total_time_match = re.search(r'Total elapsed time:\s+([\d.e+-]+)', stdout)
    if total_time_match:
        try:
            parsed["total_elapsed_time_seconds"] = float(total_time_match.group(1))
        except ValueError:
            logger.warning("Failed to parse total elapsed time")

    # Split by stage (each stage starts with "DIAGNOSTIC,Iteration,metricValue...")
    # This pattern matches the header line and all subsequent DIAGNOSTIC iteration lines
    stage_pattern = r'DIAGNOSTIC,Iteration,metricValue,convergenceValue.*?\n((?:\s*\d+DIAGNOSTIC,.*?\n)+)'
    stage_matches = re.finditer(stage_pattern, stdout, re.MULTILINE)

    for stage_idx, stage_match in enumerate(stage_matches):
        stage_data = {
            "stage_index": stage_idx,
            "iterations": [],
            "elapsed_time_seconds": None
        }

        # Parse iterations within this stage
        # Format: 2DIAGNOSTIC,     1, -1.090614080429e+00, inf, 1.2849e-01, 1.2849e-01,
        iteration_pattern = r'\s*\d+DIAGNOSTIC,\s*(\d+),\s*([\d.e+-]+),\s*([\w.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+)'
        for iter_match in re.finditer(iteration_pattern, stage_match.group(1)):
            try:
                iteration = int(iter_match.group(1))
                metric_value = float(iter_match.group(2))
                convergence_str = iter_match.group(3)
                iteration_time = float(iter_match.group(4))
                time_since_last = float(iter_match.group(5))

                # Convert "inf" to proper representation
                if convergence_str == "inf":
                    convergence_value = float('inf')
                else:
                    convergence_value = float(convergence_str)

                iteration_data = {
                    "iteration": iteration,
                    "metric_value": metric_value,
                    "convergence_value": convergence_value,
                    "iteration_time_seconds": iteration_time,
                    "time_since_last_seconds": time_since_last
                }

                stage_data["iterations"].append(iteration_data)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse iteration: {e}")
                continue

        # Parse stage elapsed time
        # Format: Elapsed time (stage 0): 1.1270e+01
        stage_time_pattern = rf'Elapsed time \(stage {stage_idx}\):\s+([\d.e+-]+)'
        stage_time_match = re.search(stage_time_pattern, stdout)
        if stage_time_match:
            try:
                stage_data["elapsed_time_seconds"] = float(stage_time_match.group(1))
            except ValueError:
                logger.debug(f"Failed to parse stage {stage_idx} elapsed time")

        # Add final metrics if we have iterations
        if stage_data["iterations"]:
            last_iter = stage_data["iterations"][-1]
            stage_data["final_metric_value"] = last_iter["metric_value"]
            stage_data["final_convergence_value"] = last_iter["convergence_value"]
            # Converged if convergence value is finite (not inf)
            stage_data["converged"] = (
                isinstance(last_iter["convergence_value"], float) and
                not (last_iter["convergence_value"] == float('inf'))
            )

            parsed["stages"].append(stage_data)

    return parsed


def extract_transform_types(stdout: str) -> List[str]:
    """Extract transform types from ANTs output.

    Looks for lines describing the composite transform composition like:
      "The composite transform comprises the following transforms (in order):"
      "1. Center of mass alignment using fixed image: ... (type = Euler3DTransform)"
      "2. Rigid"
      "3. Affine"

    Args:
        stdout: Captured stdout from ANTs registration

    Returns:
        List of transform type strings (e.g., ["Rigid", "Affine"])
    """
    transform_types = []

    # Look for the composite transform section
    in_transform_section = False

    for line in stdout.split('\n'):
        # Check if we're entering the transform listing section
        if "composite transform comprises" in line.lower():
            in_transform_section = True
            continue

        if in_transform_section:
            # Look for numbered transform lines like "1. Rigid" or "2. Affine"
            # Also handle lines with type info like "(type = Euler3DTransform)"
            match = re.match(r'^\s*\d+\.\s+(.+?)(?:\s+using|\s*\(type\s*=|\s*$)', line)
            if match:
                transform_type = match.group(1).strip()
                # Extract just the transform name, not full descriptions
                # E.g., "Center of mass alignment" -> skip
                # "Rigid" -> keep
                # "Affine" -> keep
                if transform_type and len(transform_type.split()) <= 2:
                    # Simple heuristic: if it's 1-2 words, it's likely a transform type
                    if transform_type not in ["Center of", "Center", "mass", "alignment"]:
                        transform_types.append(transform_type)
            elif line.strip() and not line.startswith(' '):
                # End of transform section (non-indented, non-empty line)
                break
            elif not line.strip():
                # Empty line might indicate end of section
                continue

    return transform_types
