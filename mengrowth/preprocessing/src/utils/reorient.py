#!/usr/bin/env python3
"""
reorient.py
Re-orients a volume and a mask (both sitk.Image objects) to a specified orientation (e.g. RAS).
"""

from typing import Tuple
import SimpleITK as sitk


def reorient_images(
    volume_img: sitk.Image, mask_img: sitk.Image, orientation: str = "LPS"
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Re-orients a volume and a corresponding mask into the specified orientation.

    Args:
        volume_img (sitk.Image):
            The input volume as a SimpleITK Image.
        mask_img (sitk.Image):
            The corresponding mask as a SimpleITK Image.
        orientation (str, optional):
            Desired orientation string, e.g. 'RAS', 'LPS', 'RAI', etc.
            Defaults to "LPS".

    Returns:
        Tuple[sitk.Image, sitk.Image]:
            (reoriented_volume, reoriented_mask), both SimpleITK Images.

    Raises:
        ValueError:
            If the specified orientation string is not 3 characters (e.g., 'RAS').

    Comments:
        More info on https://theaisummer.com/medical-image-coordinates/#the-coordinate-systems-in-medical-imaging
    """
    # Check orientation string
    if len(orientation) != 3:
        raise ValueError(
            f"Orientation '{orientation}' must be a 3-letter string (e.g. 'RAS')."
        )

    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)

    # Apply re-orientation
    reoriented_volume = orient_filter.Execute(volume_img)
    reoriented_mask = orient_filter.Execute(mask_img)

    return reoriented_volume, reoriented_mask


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 4:
        print("Usage: python reorient.py <volume.nii.gz> <mask.nrrd> <orientation>")
        sys.exit(1)

    vol_path = sys.argv[1]
    mask_path = sys.argv[2]
    orientation_code = sys.argv[3]

    try:
        volume_in = sitk.ReadImage(vol_path)
        mask_in = sitk.ReadImage(mask_path)

        vol_out, mask_out = reorient_images(volume_in, mask_in, orientation_code)

        sitk.WriteImage(vol_out, "reoriented_volume.nii.gz")
        sitk.WriteImage(mask_out, "reoriented_mask.nrrd")

        print("Reorientation completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
