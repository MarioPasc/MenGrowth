#!/usr/bin/env python

"""
Simple nrrd header parse of np.ndarray object to json-serializable
objects. Used in various scripts such as nrrd_to_nifti.py or planner.py 
"""

import numpy as np


# Add or modify the numpy_converter function to handle NaN values

def numpy_converter(obj):
    """
    Custom JSON converter that handles numpy arrays and datatypes.
    Also converts NaN values to null for JSON compatibility.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj):
            return None  # Convert NaN to null for JSON
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.void)):
        return None
    
    # Handle NaN values in any other format
    if hasattr(obj, "__float__"):
        try:
            if np.isnan(float(obj)):
                return None
        except:
            pass
            
    return obj