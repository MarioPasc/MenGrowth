"""Registration module constants and defaults.

This module defines default values and valid options for registration engines.
"""

# Default registration engine
# Options: "nipype" | "antspyx"
DEFAULT_REGISTRATION_ENGINE = "nipype"

# Valid engine types
VALID_ENGINES = ["nipype", "antspyx"]

# Interpolation method mapping between engines
INTERPOLATION_MAP = {
    "nipype_to_antspyx": {
        "Linear": "linear",
        "BSpline": "bSpline",
        "NearestNeighbor": "nearestNeighbor",
        "Gaussian": "gaussian",
        "MultiLabel": "multiLabel",
    },
    "antspyx_to_nipype": {
        "linear": "Linear",
        "bSpline": "BSpline",
        "nearestNeighbor": "NearestNeighbor",
        "gaussian": "Gaussian",
        "multiLabel": "MultiLabel",
    }
}
