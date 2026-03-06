"""Class legend for Decadal LULC India (1985/1995/2005).

Sources:
- ORNL DAAC guide: https://daac.ornl.gov/VEGETATION/guides/Decadal_LULC_India.html
- Remote Sensing 2015, 7(3), 2401-2430: https://www.mdpi.com/2072-4292/7/3/2401
"""

from __future__ import annotations

from typing import Dict, Tuple

# Pixel value -> (Level-I class, Level-II class)
# Source: ORNL DAAC DS-1336 (Decadal LULC India) Table 2 pixel-value legend.
LULC_CLASS_MAP: Dict[int, Tuple[str, str]] = {
    0: ("No Data", "No Data"),
    1: ("Forest", "Deciduous Broadleaf Forest"),
    2: ("Agricultural Land", "Crop land"),
    3: ("Built-up Land", "Built-up Land"),
    4: ("Forest", "Mixed Forest"),
    5: ("Shrubland", "Shrubland"),
    6: ("Barren Land", "Barren Land"),
    7: ("Agricultural Land", "Fallow land"),
    8: ("Barren Land", "Wasteland"),
    9: ("Water", "Water bodies"),
    10: ("Plantation", "Plantation"),
    11: ("Water", "Aquaculture"),
    12: ("Forest", "Mangrove Forest"),
    13: ("Barren Land", "Salt Pan"),
    14: ("Grassland", "Grassland"),
    15: ("Forest", "Evergreen Broadleaf Forest"),
    16: ("Forest", "Deciduous Needleleaf Forest"),
    17: ("Water", "Permanent wetland"),
    18: ("Snow and Ice", "Snow and Ice"),
    19: ("Forest", "Evergreen Needleleaf Forest"),
}

LULC_VALID_CLASS_IDS = tuple(sorted(LULC_CLASS_MAP.keys()))


# Standardized NRSC-style colors for the 20 classes
class_colors = [
    "#000000", "#E60000", "#FFFF00", "#DAA520", "#228B22", 
    "#006400", "#00FF00", "#32CD32", "#6B8E23", "#A52A2A", 
    "#D2B48C", "#F4A460", "#8B4513", "#000080", "#4169E1", 
    "#00BFFF", "#FFFFFF", "#D3D3D3", "#808000", "#90EE90"
]
