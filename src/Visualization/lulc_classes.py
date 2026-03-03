"""Class legend for Decadal LULC India (1985/1995/2005).

Sources:
- ORNL DAAC guide: https://daac.ornl.gov/VEGETATION/guides/Decadal_LULC_India.html
- Remote Sensing 2015, 7(3), 2401-2430: https://www.mdpi.com/2072-4292/7/3/2401
"""

from __future__ import annotations

from typing import Dict, Tuple

# Pixel value -> (Level-I class, Level-II class)
LULC_CLASS_MAP: Dict[int, Tuple[str, str]] = {
    0: ("No Data", "No Data"),
    1: ("Built-up Land", "Built-up Land"),
    2: ("Agricultural Land", "Crop land"),
    3: ("Agricultural Land", "Fallow land"),
    4: ("Forest", "Deciduous Broadleaf Forest"),
    5: ("Forest", "Deciduous Needleleaf Forest"),
    6: ("Forest", "Evergreen Broadleaf Forest"),
    7: ("Forest", "Evergreen Needleleaf Forest"),
    8: ("Forest", "Mixed Forest"),
    9: ("Barren Land", "Barren rocky"),
    10: ("Barren Land", "Scrub land"),
    11: ("Barren Land", "Sandy area"),
    12: ("Barren Land", "Barren/unculturable/wasteland"),
    13: ("Water", "Permanent wetland"),
    14: ("Water", "Seasonal wetland"),
    15: ("Water", "Water bodies"),
    16: ("Snow and Ice", "Snow and Ice"),
    17: ("Open Area", "Open Area"),
    18: ("Plantation", "Plantation"),
    19: ("Grassland", "Grassland"),
}

LULC_VALID_CLASS_IDS = tuple(sorted(LULC_CLASS_MAP.keys()))


# Standardized NRSC-style colors for the 20 classes
class_colors = [
    "#000000", "#E60000", "#FFFF00", "#DAA520", "#228B22", 
    "#006400", "#00FF00", "#32CD32", "#6B8E23", "#A52A2A", 
    "#D2B48C", "#F4A460", "#8B4513", "#000080", "#4169E1", 
    "#00BFFF", "#FFFFFF", "#D3D3D3", "#808000", "#90EE90"
]
