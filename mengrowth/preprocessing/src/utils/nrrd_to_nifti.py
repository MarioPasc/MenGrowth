#!/usr/bin/env python

"""
This script has been heavily inspired by github repository pnlbwh,
which can be found in https://github.com/pnlbwh/conversion/blob/master/conversion/nifti_write.py

The modifications to the original code were made in order to ignore 4D image data, since
we are only dealing with sMRI 3D data, and to ensure a better conversion of the nrrd header
to the nifti header through the numpy converter.
"""

import collections
import nrrd  
import numpy as np
import argparse
import os
import json
import nibabel as nib
from mengrowth.preprocessing.src.utils.parse_nrrd_header import numpy_converter
from typing import Union, Tuple
import SimpleITK as sitk

PRECISION = 17
np.set_printoptions(precision=PRECISION, suppress=True, floatmode="maxprec")


def _space2ras(space):
    """
    Return a 4x4 diagonal flip matrix to transform the specified NRRD 'space'
    to the conventional RAS orientation.

    Examples of NRRD space strings:
        'left-posterior-inferior'
        'left-posterior-superior'
        'right-anterior-superior'
        etc.

    For a 3-character string like 'LPI', we assume those map to:
        'left'  -> flips X
        'posterior' -> flips Y
        'inferior'  -> flips Z
    """
    # Handle either short or long forms
    if len(space) == 3:
        # short definition e.g. 'LPI'
        positive = [space[0], space[1], space[2]]
    else:
        # e.g. 'left-posterior-inferior'
        positive = space.split("-")

    flips = []
    # X axis
    if positive[0][0].lower() == "l":  # left => flip
        flips.append(-1)
    else:
        flips.append(1)
    # Y axis
    if positive[1][0].lower() == "p":  # posterior => flip
        flips.append(-1)
    else:
        flips.append(1)
    # Z axis
    if positive[2][0].lower() == "i":  # inferior => flip
        flips.append(-1)
    else:
        flips.append(1)

    # Final element for homogeneous coordinates
    flips.append(1)

    return np.diag(flips)


def nifti_write_3d(
    volume: Union[str, Tuple[sitk.Image, collections.OrderedDict]],
    out_file: str = "default",
    verbose: bool = False,
) -> str:
    """
    Convert a 3D NRRD image to NIfTI, preserving as much metadata as possible:
      - space directions (voxel spacing & orientation)
      - space origin
      - flips to RAS if the NRRD space is recognized
      - stores the entire NRRD header in a NIfTI extension

    :param volume: path to input NRRD
    :param prefix: output file prefix (if None, uses input filename stem)

    :return path to output niigz
    """

    if isinstance(volume, (str, os.PathLike)):
        data, hdr = nrrd.read(volume)
    else:
        data, hdr = volume
        data = sitk.GetArrayFromImage(data)

    if "dimension" not in hdr or hdr["dimension"] != 3:
        raise ValueError("This script only supports 3D NRRD data.")

    # Build the affine
    # ----------------------------------------------------------------------
    # 1) Flip from the NRRD-defined 'space' to RAS
    if "space" in hdr:
        space_to_ras = _space2ras(hdr["space"])
    else:
        # If no known space, assume 'left-posterior-superior' (LPS) by default
        space_to_ras = _space2ras("left-posterior-superior")

    # 2) Parse 'space directions' (the 3x3)
    #    shape => (3,3). Each row is a direction vector in the real world
    if "space directions" not in hdr:
        raise ValueError("NRRD header missing 'space directions' for 3D data.")

    rotation = hdr["space directions"]  # shape (3, 3)

    # 3) Parse 'space origin' (the translation)
    if "space origin" not in hdr:
        # If missing, set origin to [0,0,0]
        translation = np.zeros(3)
    else:
        translation = hdr["space origin"]  # shape (3,)

    # Combine rotation & translation into a 4x4
    affine_nhdr = np.eye(4)
    affine_nhdr[:3, :3] = rotation.T
    affine_nhdr[:3, 3] = translation

    # 4) Combine with the flips to RAS
    xfrm_nifti = space_to_ras @ affine_nhdr

    # Create the nibabel NIfTI image
    img_nifti = nib.nifti1.Nifti1Image(data, affine=xfrm_nifti)
    hdr_nifti = img_nifti.header

    # Set codes for coordinate frames, units, etc.
    #    xyz units = 2 means 'mm'
    #    t units   = 0 means 'unknown' (we have only 3D here)
    hdr_nifti.set_xyzt_units(xyz=2, t=0)
    hdr_nifti["qform_code"] = 2
    hdr_nifti["sform_code"] = 2
    hdr_nifti["descrip"] = (
        "Original code from pnlbwh, Modified by Mario Pascual Gonzalez, NRRD to NIFTI converter for sMRI"
    )

    # Optionally store the entire NRRD header as a JSON extension
    # so that we preserve that metadata if needed later.
    #
    # 'comment' is the typical code for a text-based extension,
    # though nibabel won't parse it automatically. It's just stored.
    extension_data = json.dumps(hdr, indent=2, default=numpy_converter)
    extension = nib.nifti1.Nifti1Extension("comment", extension_data.encode("utf-8"))
    hdr_nifti.extensions.append(extension)

    # Finally save the .nii.gz
    if not out_file.endswith("nii.gz"):
        out_file = out_file + ".nii.gz"
    nib.save(img_nifti, out_file)
    if verbose:
        print(f"Saved 3D NIfTI to: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 3D NRRD/NHDR to NIFTI, preserving header info."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input 3D NRRD/NHDR file"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file prefix"
    )
    args = parser.parse_args()

    nifti_write_3d(args.input, args.output, verbose=True)


if __name__ == "__main__":
    main()
