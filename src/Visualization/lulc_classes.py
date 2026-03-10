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


# Paper-aligned generalized palette (Roy et al., Remote Sensing 2015, Table 3 classes).
# The paper defines class taxonomy but does not provide a fixed HEX legend table;
# colors below follow standard thematic conventions by Level-I/Level-II class meaning.
# Index = class ID (0..19).
class_colors = [
    "#000000",  # 0  No Data
    "#2E8B57",  # 1  Deciduous Broadleaf Forest
    "#FFF200",  # 2  Crop land
    "#E31A1C",  # 3  Built-up Land
    "#4CAF50",  # 4  Mixed Forest
    "#7CB342",  # 5  Shrubland
    "#C2B280",  # 6  Barren Land
    "#D4A017",  # 7  Fallow land
    "#BCAAA4",  # 8  Wasteland
    "#1E88E5",  # 9  Water bodies
    "#66BB6A",  # 10 Plantation
    "#26C6DA",  # 11 Aquaculture
    "#00695C",  # 12 Mangrove Forest
    "#F48FB1",  # 13 Salt Pan
    "#8BC34A",  # 14 Grassland
    "#1B5E20",  # 15 Evergreen Broadleaf Forest
    "#33691E",  # 16 Deciduous Needleleaf Forest
    "#00ACC1",  # 17 Permanent wetland
    "#FFFFFF",  # 18 Snow and Ice
    "#2E7D32",  # 19 Evergreen Needleleaf Forest
]
