"""Population dynamics of taste responses in cortical spike data.

Conventions used across the package
-----------------------------------
* All times are in **seconds** (spike times, event times, windows, bin sizes,
  smoothing widths).
* Spike data are a ``dict`` mapping ``unit_id -> sorted 1-D array of spike times``.
* Events are a ``pandas.DataFrame`` with columns
  ``session``, ``trial``, ``taste``, ``time_s`` (one row per taste delivery).
"""

from taste_dynamics.io import load_dataset, save_dataset
from taste_dynamics.synthetic import Dataset, make_dataset

__all__ = ["Dataset", "make_dataset", "load_dataset", "save_dataset"]
__version__ = "0.1.0"
