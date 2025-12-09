from intensity_normalization.domain.models import Modality
from typing import Dict

def modality_str_to_enum(modality_str: str) -> str:
    """Convert modality string to Modality enum.

    Args:
        modality_str (str): Modality as string ('t1', 't2', 'flair').

    Returns:
        Modality: Corresponding Modality enum.
    """
    modality_map: Dict[str, str] = {
        "t1c": Modality.T1,
        "t1n": Modality.T1,
        "t2": Modality.T2,
        "flair": Modality.FLAIR,
    }
    try:
        mod: str = modality_map[modality_str.lower()]
        return mod
    except KeyError:
        raise ValueError(f"Unsupported modality string: {modality_str}")
    
    
def infer_modality_from_filename(filename: str) -> str:
    """Infer modality from filename.

    Args:
        filename (str): Filename containing modality information.
    Returns:
        Modality: Inferred Modality enum.
    """
    # Convert Path object to string if needed
    filename_str = str(filename)
    # Convert Path object to string if needed
    filename_str = str(filename)
    filename_lower = filename_str.lower()
    if "t1c" in filename_lower:
        return modality_str_to_enum("t1c")
    elif "t1" in filename_lower:
        return modality_str_to_enum("t1n")
    elif "t2" in filename_lower:
        return modality_str_to_enum("t2")
    elif "flair" in filename_lower:
        return modality_str_to_enum("flair")
    else:
        raise ValueError(f"Could not infer modality from filename: {filename_str}")