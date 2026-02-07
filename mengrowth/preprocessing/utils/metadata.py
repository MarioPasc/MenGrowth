import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

HARDCODED_COLUMNS_DICT: Dict[str, str] = {
    "Unnamed: 0": "general",
    "0 nada; 1 cirugía; 2 atrofia; 3: infarto; 4 hemorragia; 5 aneurisma; 6 cáncer; 7 trauma": "general",
    "Unnamed: 2": "general",
    "0: hombre 1:mujer": "general",
    "Unnamed: 4": "first_study/rm",
    "Unnamed: 5": "first_study/rm",
    "Unnamed: 6": "first_study/tc",
    "Unnamed: 7": "first_study/tc",
    "MEDIDAS REFERENCIA  PRIMER ESTUDIO ": "first_study/measurements",
    "Unnamed: 9": "first_study/measurements",
    "Unnamed: 10": "first_study/measurements",
    "VOLUMEN ": "first_study/measurements",
    "Unnamed: 12": "first_study/attributes",
    "PRIMER  ESTUDIO": "first_study/attributes",
    "Unnamed: 14": "first_study/measurements",
    "0: NO 1:<25% 2:25-50% 3:>50% 4:completo": "first_study/attributes",
    "Unnamed: 16": "first_study/attributes",
    "Unnamed: 17": "first_study/attributes",
    "0: fosa posterior.\n1: ala esfenoidal.\n2: hoz cerebral.\n3: plano etmoidal / oflatorio 4 convexidad 5 seno cavern": "first_study/attributes",
    "0 no, 1 periférica, 2 central, 3 mixta, 4 total": "first_study/measurements",
    "1ER CONTROL": "c1",
    "Unnamed: 21": "c1",
    "Unnamed: 22": "c1",
    "Unnamed: 23": "c1",
    "Unnamed: 24": "c1",
    "Unnamed: 25": "c1",
    "Unnamed: 26": "c1",
    "2DO": "c2",
    "Unnamed: 28": "c2",
    "Unnamed: 29": "c2",
    "Unnamed: 30": "c2",
    "Unnamed: 31": "c2",
    "Unnamed: 32": "c2",
    "Unnamed: 33": "c2",
    "TERCERO": "c3",
    "Unnamed: 35": "c3",
    "Unnamed: 36": "c3",
    "Unnamed: 37": "c3",
    "Unnamed: 38": "c3",
    "Unnamed: 39": "c3",
    "Unnamed: 40": "c3",
    "Unnamed: 41": "c3",
    "CUARTO": "c4",
    "Unnamed: 43": "c4",
    "Unnamed: 44": "c4",
    "Unnamed: 45": "c4",
    "Unnamed: 46": "c4",
    "Unnamed: 47": "c4",
    "Unnamed: 48": "c4",
    "QUINTO": "c5",
    "Unnamed: 50": "c5",
    "Unnamed: 51": "c5",
    "Unnamed: 52": "c5",
    "Unnamed: 53": "c5",
    "Unnamed: 54": "c5",
    "Unnamed: 55": "c5",
    "PROGR CALCIF": "groundtruth",
    "CRECE": "groundtruth",
}

HARCODED_SUBCOLUMNS_DIC: Dict[str, str] = {
    "paciente": "ID",
    "antec": "medical_history",
    "edad": "age",
    "sexo": "sex",
    "fecha rm": "date",
    "equipo": "machine",
    "fecha tc": "date",
    "tec tc": "tec",
    "cc": "cc",
    "ll": "ll",
    "ap": "ap",
    "volumen": "vol",
    "lobulado si(1)/no(0)": "lobed",
    "hiperostosis si(1)/no(0)": "hiperostosis",
    "edema si(1)/no (0)": "edema",
    "escala visual de calcificacion": "visual_calcification_scale",
    "señal t2 (0 isointenso, 1 heterogeneo, 2 hipo, 3 hiper)": "t2_signal",
    "patron realce (0 homogeneo fuerte, 1 heterogéneo, 2 hipocaptante)": "enhancement_pattern",
    "localizacion": "loc",
    "tipo calcif": "calcif",
    "fecha": "date",
    "vol": "vol",
    "calcificacion": "calcif",
    "edema": "edema",
    "progr calcif": "progr_calcif",
    "crece": "growth",
}


def normalize_text(text: Any) -> Any:
    """
    Cleans text values:
    - Strips leading/trailing spaces
    - Converts to lowercase
    - Replaces multiple spaces with a single space
    - Removes special hidden characters (non-breaking spaces)
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\xa0", " ")  # Handle non-breaking spaces
        return text.lower()
    return text


def apply_hardcoded_codification(xlsx_path: str, output_csv_path: str) -> None:
    """
    Apply the hardcoded codification to the first-grade and second-grade columns of the
    xlsx file, in order to generate a first-stage clean csv file.
    """
    df = pd.read_excel(xlsx_path)
    # Rename top-level columns
    df.rename(columns=HARDCODED_COLUMNS_DICT, inplace=True)

    # Prepare normalized subcolumn keys
    hardcoded_subcolumns_clean = {
        normalize_text(k): v for k, v in HARCODED_SUBCOLUMNS_DIC.items()
    }

    # Normalize the row (index=0) that contains subcolumn names
    df.iloc[0] = df.iloc[0].apply(normalize_text)

    # Replace them using the dictionary
    df.iloc[0] = df.iloc[0].replace(hardcoded_subcolumns_clean, regex=False)

    df.to_csv(path_or_buf=output_csv_path, index=False)


def is_zero_or_none(value: Any) -> bool:
    """
    Returns True if 'value' is None, or zero, or the string "0", else False.
    """
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip() == "0"
    return False


def is_control_block_empty(c_dict: Dict[str, Any]) -> bool:
    """
    Returns True if:
      - The "vol" field is None or zero (int/float) or the string "0"
      AND
      - All other fields are None
    Otherwise returns False.
    """
    vol_val = c_dict.get("vol", None)

    # If vol is not zero/None, the block is definitely not empty
    if not is_zero_or_none(vol_val):
        return False

    # Now check that every other field (besides "vol") is None
    for k, v in c_dict.items():
        if k == "vol":
            continue
        if v is not None:
            return False

    # If we reached here, vol was zero/None, and every other field is None
    return True


def create_json_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads a CSV that has:
        - Row 0: top-level grouping (e.g., 'general', 'first_study/rm', 'c1', 'groundtruth', etc.)
        - Row 1: second-level column names (keys under those groupings)
        - Rows 2+: patient data rows

    The script constructs a JSON of the form:
    {
        "P{patient_id}": {
            "general": { ... },
            "first_study": {
                "rm": { ... },
                "tc": { ... },
                "attributes": { ... },
                "measurements": { ... }
            },
            "c1": { ... },
            "c2": { ... },
            ...
            "groundtruth": { ... }
        },
        ...
    }

    The patient_id is taken from the column where top_level == 'general'
    and second_level == 'ID'. Missing/NaN values become null in the JSON.

    :param csv_path: Path to the input CSV file.
    :return: A dictionary that can be serialized to JSON.
    """
    # Read the entire CSV without interpreting headers
    df = pd.read_csv(csv_path, header=None)

    # Row 0 -> top-level group names
    top_level_cols = df.iloc[0].tolist()
    # Row 1 -> second-level column names
    second_level_cols = df.iloc[1].tolist()

    # The data for patients starts from row 2 onward
    data = df.iloc[2:].reset_index(drop=True)

    # Identify 'general/ID' column index
    id_column_index = None
    for i, (top_val, second_val) in enumerate(zip(top_level_cols, second_level_cols)):
        if top_val == "general" and second_val == "ID":
            id_column_index = i
            break

    if id_column_index is None:
        raise ValueError(
            "Could not find the 'general/ID' column in the CSV headers. "
            "Ensure the CSV has a top-level 'general' with second-level 'ID'."
        )

    # Prepare a structure to map columns into the JSON hierarchy
    column_map = []
    for top_val, second_val in zip(top_level_cols, second_level_cols):
        if "/" in str(top_val):
            parts = top_val.split("/", 1)
            top_key = parts[0]
            sub_key = parts[1]
        else:
            top_key = top_val
            sub_key = None

        column_map.append((top_key, sub_key, second_val))

    result: Dict[str, Dict[str, Any]] = {}

    # Process each patient row
    for row_idx in range(data.shape[0]):
        row_data = data.iloc[row_idx]

        # Retrieve patient ID from 'general/ID'
        raw_pid = row_data[id_column_index]
        if pd.isnull(raw_pid):
            # If no ID, skip
            continue

        # Convert ID to e.g., "P1"
        patient_id_str = f"P{int(raw_pid)}"
        patient_dict: Dict[str, Any] = {}

        # Fill data
        for col_idx, (top_key, sub_key, second_key) in enumerate(column_map):
            if not top_key:
                # Skip if top_key is empty
                continue

            val = row_data[col_idx]
            if pd.isnull(val):
                val = None

            if sub_key is None:
                # E.g. "general" -> "ID": val
                if top_key not in patient_dict:
                    patient_dict[top_key] = {}
                patient_dict[top_key][second_key] = val
            else:
                # E.g. "first_study" -> "rm" -> "date": val
                if top_key not in patient_dict:
                    patient_dict[top_key] = {}
                if sub_key not in patient_dict[top_key]:
                    patient_dict[top_key][sub_key] = {}
                patient_dict[top_key][sub_key][second_key] = val

        # Check c1..c5 blocks; remove if empty
        for cx in ["c1", "c2", "c3", "c4", "c5"]:
            if cx in patient_dict and isinstance(patient_dict[cx], dict):
                if is_control_block_empty(patient_dict[cx]):
                    del patient_dict[cx]

        result[patient_id_str] = patient_dict

    return result


@dataclass
class PatientMetadata:
    """Clinical metadata for a single patient."""

    patient_id: str  # Original ID (e.g., "P1")
    age: Optional[int] = None
    sex: Optional[int] = None  # 0=male, 1=female
    medical_history: Optional[int] = None
    first_study: Optional[Dict[str, Any]] = None
    controls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    groundtruth: Optional[Dict[str, Any]] = None
    # Curation tracking fields
    included: bool = True
    exclusion_reason: Optional[str] = None
    mengrowth_id: Optional[str] = None

    @classmethod
    def from_json_dict(cls, patient_id: str, data: Dict[str, Any]) -> "PatientMetadata":
        """Create PatientMetadata from parsed JSON dictionary."""
        general = data.get("general", {})
        first_study = data.get("first_study", {})
        groundtruth = data.get("groundtruth", {})

        # Extract control studies
        controls = {}
        for key in ["c1", "c2", "c3", "c4", "c5"]:
            if key in data:
                controls[key] = data[key]

        # Helper function to safely parse integer values
        def safe_int(val: Any) -> Optional[int]:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                # Handle comma-separated values by taking the first one
                val = val.strip()
                if "," in val:
                    val = val.split(",")[0].strip()
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return None
            return None

        # Parse age safely
        age = safe_int(general.get("age"))

        # Parse sex safely
        sex = safe_int(general.get("sex"))

        # Parse medical history safely (can be comma-separated like "1, 5")
        medical_history = safe_int(general.get("medical_history"))

        return cls(
            patient_id=patient_id,
            age=age,
            sex=sex,
            medical_history=medical_history,
            first_study=first_study if first_study else None,
            controls=controls,
            groundtruth=groundtruth if groundtruth else None,
        )

    def get_first_study_volume(self) -> Optional[float]:
        """Get initial tumor volume from first study."""
        if not self.first_study:
            return None
        measurements = self.first_study.get("measurements", {})
        vol = measurements.get("vol")
        if vol is not None:
            try:
                return float(vol)
            except (ValueError, TypeError):
                return None
        return None

    def get_growth_status(self) -> Optional[bool]:
        """Get whether tumor is growing (True) or stable (False)."""
        if not self.groundtruth:
            return None
        growth = self.groundtruth.get("growth")
        if growth is not None:
            return bool(int(growth))
        return None

    def get_num_controls(self) -> int:
        """Get number of control/follow-up studies."""
        return len(self.controls)


class MetadataManager:
    """Manages clinical metadata loading, ID mapping, and export."""

    def __init__(self) -> None:
        self._patients: Dict[str, PatientMetadata] = {}
        self._id_map: Dict[str, str] = {}  # original_id -> mengrowth_id
        self._reverse_id_map: Dict[str, str] = {}  # mengrowth_id -> original_id
        self._raw_json_data: Dict[str, Dict[str, Any]] = {}

    def load_from_xlsx(
        self, xlsx_path: Path, temp_csv_path: Optional[Path] = None
    ) -> None:
        """Load xlsx via existing apply_hardcoded_codification + create_json_from_csv.

        Args:
            xlsx_path: Path to the xlsx file (will not be modified).
            temp_csv_path: Optional path for intermediate CSV. If None, uses temp file.
        """
        xlsx_path = Path(xlsx_path)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Metadata xlsx file not found: {xlsx_path}")

        # Use temp file if no path specified
        if temp_csv_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                temp_csv_path = Path(tmp.name)

        try:
            # Convert xlsx to CSV
            apply_hardcoded_codification(str(xlsx_path), str(temp_csv_path))

            # Parse CSV to JSON structure
            self._raw_json_data = create_json_from_csv(str(temp_csv_path))

            # Convert to PatientMetadata objects
            self._patients = {}
            for patient_id, data in self._raw_json_data.items():
                self._patients[patient_id] = PatientMetadata.from_json_dict(
                    patient_id, data
                )

            logger.info(f"Loaded metadata for {len(self._patients)} patients from {xlsx_path}")

        finally:
            # Clean up temp file if we created it
            if temp_csv_path and temp_csv_path.exists():
                try:
                    temp_csv_path.unlink()
                except OSError:
                    pass

    def load_from_json(self, json_path: Path) -> None:
        """Load from previously exported JSON."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            self._raw_json_data = json.load(f)

        self._patients = {}
        for patient_id, data in self._raw_json_data.items():
            self._patients[patient_id] = PatientMetadata.from_json_dict(
                patient_id, data
            )

        logger.info(f"Loaded metadata for {len(self._patients)} patients from {json_path}")

    def load_from_enriched_csv(self, csv_path: Path) -> None:
        """Load from previously exported enriched CSV."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Enriched CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        self._patients = {}
        for _, row in df.iterrows():
            patient_id = row.get("patient_id", row.get("original_id"))
            if pd.isna(patient_id):
                continue

            patient_id = str(patient_id)
            if not patient_id.startswith("P"):
                patient_id = f"P{patient_id}"

            patient = PatientMetadata(
                patient_id=patient_id,
                age=int(row["age"]) if pd.notna(row.get("age")) else None,
                sex=int(row["sex"]) if pd.notna(row.get("sex")) else None,
                included=bool(row["included"]) if pd.notna(row.get("included")) else True,
                exclusion_reason=str(row["exclusion_reason"]) if pd.notna(row.get("exclusion_reason")) else None,
                mengrowth_id=str(row["MenGrowth_ID"]) if pd.notna(row.get("MenGrowth_ID")) else None,
            )
            self._patients[patient_id] = patient

            if patient.mengrowth_id:
                self._id_map[patient_id] = patient.mengrowth_id
                self._reverse_id_map[patient.mengrowth_id] = patient_id

        logger.info(f"Loaded enriched metadata for {len(self._patients)} patients")

    def _normalize_patient_id(self, patient_id: str) -> str:
        """Normalize patient ID to standard P{num} format."""
        patient_id = str(patient_id).strip()

        # Check if it's a MenGrowth ID
        if patient_id in self._reverse_id_map:
            return self._reverse_id_map[patient_id]

        # Remove leading P if present, then add it back
        if patient_id.upper().startswith("P"):
            patient_id = patient_id[1:]

        # Remove leading zeros and convert to int, then back to string
        try:
            num = int(patient_id)
            return f"P{num}"
        except ValueError:
            return f"P{patient_id}"

    def get_patient(self, patient_id: str) -> Optional[PatientMetadata]:
        """Get patient by ID (handles P1, 1, MenGrowth-0001 formats)."""
        normalized = self._normalize_patient_id(patient_id)
        return self._patients.get(normalized)

    def get_all_patients(self) -> Dict[str, PatientMetadata]:
        """Return all patient metadata."""
        return self._patients.copy()

    def get_patient_ids(self) -> Set[str]:
        """Return set of all patient IDs."""
        return set(self._patients.keys())

    def ensure_patient_exists(self, patient_id: str) -> None:
        """Create a stub entry if patient doesn't exist in metadata."""
        normalized = self._normalize_patient_id(patient_id)
        if normalized not in self._patients:
            self._patients[normalized] = PatientMetadata(
                patient_id=normalized,
                included=True,
            )
            logger.debug(f"Created metadata stub for patient {normalized}")

    def mark_excluded(self, patient_id: str, reason: str) -> None:
        """Mark a patient as excluded with reason."""
        normalized = self._normalize_patient_id(patient_id)
        if normalized not in self._patients:
            self._patients[normalized] = PatientMetadata(patient_id=normalized)
        self._patients[normalized].included = False
        self._patients[normalized].exclusion_reason = reason
        logger.debug(f"Marked patient {normalized} as excluded: {reason}")

    def mark_included(self, patient_id: str) -> None:
        """Mark a patient as included (clearing any previous exclusion)."""
        normalized = self._normalize_patient_id(patient_id)
        if normalized not in self._patients:
            self._patients[normalized] = PatientMetadata(patient_id=normalized)
        self._patients[normalized].included = True
        self._patients[normalized].exclusion_reason = None

    def set_mengrowth_id(self, original_id: str, mengrowth_id: str) -> None:
        """Map original ID to MenGrowth ID."""
        normalized = self._normalize_patient_id(original_id)
        if normalized not in self._patients:
            self._patients[normalized] = PatientMetadata(
                patient_id=normalized, included=True
            )
            logger.debug(f"Created metadata stub for patient {normalized}")
        self._patients[normalized].mengrowth_id = mengrowth_id
        self._id_map[normalized] = mengrowth_id
        self._reverse_id_map[mengrowth_id] = normalized
        logger.debug(f"Set MenGrowth ID for {normalized}: {mengrowth_id}")

    def apply_id_mapping(self, id_mapping: Dict[str, Dict[str, Any]]) -> None:
        """Apply ID mapping from filter's id_mapping.json.

        Expected format:
        {
            "MenGrowth-0001": {"original_id": "P1", "studies": {...}},
            ...
        }
        """
        for mengrowth_id, info in id_mapping.items():
            original_id = info.get("original_id")
            if original_id:
                self.set_mengrowth_id(original_id, mengrowth_id)

    def export_enriched_csv(self, output_path: Path) -> None:
        """Export CSV with columns: original data + included, exclusion_reason, MenGrowth_ID."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for patient_id, patient in sorted(self._patients.items(), key=lambda x: x[0]):
            row = {
                "patient_id": patient.patient_id,
                "age": patient.age,
                "sex": patient.sex,
                "medical_history": patient.medical_history,
                "first_study_volume": patient.get_first_study_volume(),
                "growth": patient.get_growth_status(),
                "num_controls": patient.get_num_controls(),
                "included": patient.included,
                "exclusion_reason": patient.exclusion_reason,
                "MenGrowth_ID": patient.mengrowth_id,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported enriched metadata to {output_path}")

    def export_json(self, output_path: Path) -> None:
        """Export full metadata as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build output structure with curation tracking
        output = {}
        for patient_id, patient in self._patients.items():
            # Start with raw data if available
            if patient_id in self._raw_json_data:
                patient_data = self._raw_json_data[patient_id].copy()
            else:
                patient_data = {}

            # Add curation tracking fields
            patient_data["_curation"] = {
                "included": patient.included,
                "exclusion_reason": patient.exclusion_reason,
                "mengrowth_id": patient.mengrowth_id,
            }

            output[patient_id] = patient_data

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported metadata JSON to {output_path}")

    def get_clinical_summary(self) -> Dict[str, Any]:
        """Return aggregate stats for visualization."""
        included_patients = [p for p in self._patients.values() if p.included]
        excluded_patients = [p for p in self._patients.values() if not p.included]

        # Age statistics
        ages = [p.age for p in included_patients if p.age is not None]
        age_stats = {}
        if ages:
            import statistics
            age_stats = {
                "min": min(ages),
                "max": max(ages),
                "mean": statistics.mean(ages),
                "median": statistics.median(ages),
                "std": statistics.stdev(ages) if len(ages) > 1 else 0,
            }

        # Sex distribution
        sex_counts = {"male": 0, "female": 0, "unknown": 0}
        for p in included_patients:
            if p.sex == 0:
                sex_counts["male"] += 1
            elif p.sex == 1:
                sex_counts["female"] += 1
            else:
                sex_counts["unknown"] += 1

        # Growth statistics
        growth_counts = {"growing": 0, "stable": 0, "unknown": 0}
        for p in included_patients:
            growth = p.get_growth_status()
            if growth is True:
                growth_counts["growing"] += 1
            elif growth is False:
                growth_counts["stable"] += 1
            else:
                growth_counts["unknown"] += 1

        # Volume statistics
        volumes = [
            p.get_first_study_volume()
            for p in included_patients
            if p.get_first_study_volume() is not None
        ]
        volume_stats = {}
        if volumes:
            import statistics
            volume_stats = {
                "min": min(volumes),
                "max": max(volumes),
                "mean": statistics.mean(volumes),
                "median": statistics.median(volumes),
            }

        # Controls distribution
        controls_counts = [p.get_num_controls() for p in included_patients]
        controls_stats = {}
        if controls_counts:
            import statistics
            controls_stats = {
                "min": min(controls_counts),
                "max": max(controls_counts),
                "mean": statistics.mean(controls_counts),
                "total": sum(controls_counts),
            }

        # Exclusion reasons
        exclusion_reasons: Dict[str, int] = {}
        for p in excluded_patients:
            reason = p.exclusion_reason or "unknown"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

        return {
            "total_patients": len(self._patients),
            "included_patients": len(included_patients),
            "excluded_patients": len(excluded_patients),
            "age_stats": age_stats,
            "sex_distribution": sex_counts,
            "growth_distribution": growth_counts,
            "volume_stats": volume_stats,
            "controls_stats": controls_stats,
            "exclusion_reasons": exclusion_reasons,
        }

    def get_volume_progression_data(self) -> List[Dict[str, Any]]:
        """Get tumor volume progression data for visualization."""
        progressions = []

        for patient_id, patient in self._patients.items():
            if not patient.included:
                continue

            volumes = []

            # First study volume
            first_vol = patient.get_first_study_volume()
            if first_vol is not None:
                volumes.append({"timepoint": 0, "volume": first_vol, "label": "baseline"})

            # Control volumes
            for i, (ctrl_key, ctrl_data) in enumerate(
                sorted(patient.controls.items()), start=1
            ):
                vol = ctrl_data.get("vol")
                if vol is not None:
                    try:
                        volumes.append({
                            "timepoint": i,
                            "volume": float(vol),
                            "label": ctrl_key,
                        })
                    except (ValueError, TypeError):
                        pass

            if len(volumes) >= 2:
                progressions.append({
                    "patient_id": patient_id,
                    "mengrowth_id": patient.mengrowth_id,
                    "growth_status": patient.get_growth_status(),
                    "volumes": volumes,
                })

        return progressions


def main() -> None:
    """
    Main function to run when calling this script from the command line.
    It reads an XLSX file, applies codification, writes an intermediate CSV,
    then reads that CSV to build a final JSON structure.
    """
    xlsx_path = "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/raw/processed/baseline/metadata.xlsx"
    output_folder = (
        "/media/mpascual/PortableSSD/Meningiomas/MenGrowth/preprocessed/"
    )
    csv_path = os.path.join(output_folder, "metadata_recodified.csv")

    # 1. Apply the hardcoded codification
    apply_hardcoded_codification(
        xlsx_path=xlsx_path,
        output_csv_path=csv_path,
    )

    # 2. Create JSON from the CSV
    json_data = create_json_from_csv(csv_path)

    # 3. Write the JSON structure to file
    output_json = os.path.join(output_folder, "metadata_clean.json")
    with open(output_json, "w") as outfile:
        json.dump(json_data, outfile, indent=2)


if __name__ == "__main__":
    main()
