"""
Core library for arc detection and data processing.

This package provides the fundamental components for processing electrical arc data,
including data loading, processing, metrics calculation, and visualization.
"""

__version__ = "0.1.0"

# Import main modules for easy access
from . import loader
from . import processor
from . import transient
from . import transient_center
from . import metrics
from . import utils
from . import visualization
from . import output

__all__ = [
    "loader",
    "processor", 
    "transient",
    "transient_center",
    "metrics",
    "utils",
    "visualization",
    "output"
]