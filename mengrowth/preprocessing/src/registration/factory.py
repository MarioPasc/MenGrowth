"""Factory functions for creating registration instances based on engine."""

from typing import Dict, Any
import logging

from mengrowth.preprocessing.src.registration.constants import (
    DEFAULT_REGISTRATION_ENGINE,
    VALID_ENGINES
)
from mengrowth.preprocessing.src.registration.base import BaseRegistrator

logger = logging.getLogger(__name__)


def create_multi_modal_coregistration(
    config: Dict[str, Any],
    verbose: bool = False
) -> BaseRegistrator:
    """Create multi-modal coregistration instance based on engine.

    Args:
        config: Configuration dictionary (must contain all registration params)
        verbose: Enable verbose logging

    Returns:
        BaseRegistrator instance (either MultiModalCoregistration or AntsPyXMultiModalCoregistration)

    Raises:
        ValueError: If engine is invalid or dependencies missing
    """
    # Determine engine
    engine = config.get("engine", DEFAULT_REGISTRATION_ENGINE)

    if engine not in VALID_ENGINES:
        raise ValueError(
            f"Invalid registration engine: {engine}. Must be one of {VALID_ENGINES}"
        )

    logger.info(f"Creating multi-modal coregistration with engine: {engine}")

    if engine == "nipype":
        from mengrowth.preprocessing.src.registration.multi_modal_coregistration import MultiModalCoregistration
        return MultiModalCoregistration(config=config, verbose=verbose)

    elif engine == "antspyx":
        # Lazy import to avoid requiring antspyx if not used
        try:
            from mengrowth.preprocessing.src.registration.antspyx_multi_modal_coregistration import AntsPyXMultiModalCoregistration
            return AntsPyXMultiModalCoregistration(config=config, verbose=verbose)
        except ImportError as e:
            raise ValueError(
                f"AntsPyX engine requested but antspyx is not installed. "
                f"Install with: pip install antspyx"
            ) from e

    else:
        raise ValueError(f"Unsupported engine: {engine}")


def create_intra_study_to_atlas(
    config: Dict[str, Any],
    reference_modality: str,
    verbose: bool = False
) -> BaseRegistrator:
    """Create intra-study to atlas registration instance based on engine.

    Args:
        config: Configuration dictionary
        reference_modality: Reference modality selected in step 3a
        verbose: Enable verbose logging

    Returns:
        BaseRegistrator instance (either IntraStudyToAtlas or AntsPyXIntraStudyToAtlas)

    Raises:
        ValueError: If engine is invalid or dependencies missing
    """
    # Determine engine
    engine = config.get("engine", DEFAULT_REGISTRATION_ENGINE)

    if engine not in VALID_ENGINES:
        raise ValueError(
            f"Invalid registration engine: {engine}. Must be one of {VALID_ENGINES}"
        )

    logger.info(f"Creating intra-study to atlas registration with engine: {engine}")

    if engine == "nipype":
        from mengrowth.preprocessing.src.registration.intra_study_to_atlas import IntraStudyToAtlas
        return IntraStudyToAtlas(
            config=config,
            reference_modality=reference_modality,
            verbose=verbose
        )

    elif engine == "antspyx":
        try:
            from mengrowth.preprocessing.src.registration.antspyx_intra_study_to_atlas import AntsPyXIntraStudyToAtlas
            return AntsPyXIntraStudyToAtlas(
                config=config,
                reference_modality=reference_modality,
                verbose=verbose
            )
        except ImportError as e:
            raise ValueError(
                f"AntsPyX engine requested but antspyx is not installed. "
                f"Install with: pip install antspyx"
            ) from e

    else:
        raise ValueError(f"Unsupported engine: {engine}")
