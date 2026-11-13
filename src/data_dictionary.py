"""Feature dictionary for the Kaggle House Prices (Ames, Iowa) dataset.

The competition ships ``data_description.txt``. When that file is present in
``data/raw`` it is parsed and used directly (see :func:`load_feature_descriptions`).
The built-in short descriptions below follow the official dictionary and are a
fallback when the text file is not available (the automated download source used
in this project only mirrors ``train.csv`` and ``test.csv``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "MSSubClass": "Type of dwelling involved in the sale (coded class, categorical)",
    "MSZoning": "General zoning classification of the sale",
    "LotFrontage": "Linear feet of street connected to property",
    "LotArea": "Lot size in square feet",
    "Street": "Type of road access to property",
    "Alley": "Type of alley access (NA = no alley access)",
    "LotShape": "General shape of property",
    "LandContour": "Flatness of the property",
    "Utilities": "Type of utilities available",
    "LotConfig": "Lot configuration",
    "LandSlope": "Slope of property",
    "Neighborhood": "Physical location within Ames city limits",
    "Condition1": "Proximity to various conditions",
    "Condition2": "Proximity to various conditions (if more than one is present)",
    "BldgType": "Type of dwelling",
    "HouseStyle": "Style of dwelling",
    "OverallQual": "Overall material and finish quality (1 = very poor, 10 = very excellent)",
    "OverallCond": "Overall condition rating (1 = very poor, 10 = very excellent)",
    "YearBuilt": "Original construction year",
    "YearRemodAdd": "Remodel year (same as construction year if no remodelling or additions)",
    "RoofStyle": "Type of roof",
    "RoofMatl": "Roof material",
    "Exterior1st": "Exterior covering on house",
    "Exterior2nd": "Exterior covering on house (if more than one material)",
    "MasVnrType": "Masonry veneer type",
    "MasVnrArea": "Masonry veneer area in square feet",
    "ExterQual": "Quality of the exterior material",
    "ExterCond": "Present condition of the exterior material",
    "Foundation": "Type of foundation",
    "BsmtQual": "Height of the basement (NA = no basement)",
    "BsmtCond": "General condition of the basement (NA = no basement)",
    "BsmtExposure": "Walkout or garden level basement walls (NA = no basement)",
    "BsmtFinType1": "Quality of basement finished area (NA = no basement)",
    "BsmtFinSF1": "Type 1 finished square feet",
    "BsmtFinType2": "Quality of second finished area, if present (NA = no basement)",
    "BsmtFinSF2": "Type 2 finished square feet",
    "BsmtUnfSF": "Unfinished square feet of basement area",
    "TotalBsmtSF": "Total square feet of basement area",
    "Heating": "Type of heating",
    "HeatingQC": "Heating quality and condition",
    "CentralAir": "Central air conditioning (Y/N)",
    "Electrical": "Electrical system",
    "1stFlrSF": "First floor square feet",
    "2ndFlrSF": "Second floor square feet",
    "LowQualFinSF": "Low quality finished square feet (all floors)",
    "GrLivArea": "Above grade (ground) living area square feet",
    "BsmtFullBath": "Basement full bathrooms",
    "BsmtHalfBath": "Basement half bathrooms",
    "FullBath": "Full bathrooms above grade",
    "HalfBath": "Half baths above grade",
    "BedroomAbvGr": "Bedrooms above grade (does not include basement bedrooms)",
    "KitchenAbvGr": "Kitchens above grade",
    "KitchenQual": "Kitchen quality",
    "TotRmsAbvGrd": "Total rooms above grade (does not include bathrooms)",
    "Functional": "Home functionality (assume typical unless deductions are warranted)",
    "Fireplaces": "Number of fireplaces",
    "FireplaceQu": "Fireplace quality (NA = no fireplace)",
    "GarageType": "Garage location (NA = no garage)",
    "GarageYrBlt": "Year garage was built",
    "GarageFinish": "Interior finish of the garage (NA = no garage)",
    "GarageCars": "Size of garage in car capacity",
    "GarageArea": "Size of garage in square feet",
    "GarageQual": "Garage quality (NA = no garage)",
    "GarageCond": "Garage condition (NA = no garage)",
    "PavedDrive": "Paved driveway",
    "WoodDeckSF": "Wood deck area in square feet",
    "OpenPorchSF": "Open porch area in square feet",
    "EnclosedPorch": "Enclosed porch area in square feet",
    "3SsnPorch": "Three season porch area in square feet",
    "ScreenPorch": "Screen porch area in square feet",
    "PoolArea": "Pool area in square feet",
    "PoolQC": "Pool quality (NA = no pool)",
    "Fence": "Fence quality (NA = no fence)",
    "MiscFeature": "Miscellaneous feature not covered in other categories (NA = none)",
    "MiscVal": "Dollar value of miscellaneous feature",
    "MoSold": "Month sold (MM)",
    "YrSold": "Year sold (YYYY)",
    "SaleType": "Type of sale",
    "SaleCondition": "Condition of sale",
}

# Columns whose missing value means "the feature is absent" according to the
# data dictionary (e.g. NA in BsmtQual means "No Basement").
NA_MEANS_ABSENT_CATEGORICAL: list[str] = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Numeric columns where a missing value accompanies an absent feature, so 0 is
# the correct value (e.g. basement square footage when there is no basement).
NA_MEANS_ZERO_NUMERIC: list[str] = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# Numerically coded columns that are really categories.
NUMERIC_CODED_CATEGORICAL: list[str] = ["MSSubClass"]

# Human readable labels for the most user-facing fields.
FRIENDLY_NAMES: dict[str, str] = {
    "OverallQual": "Overall quality",
    "OverallCond": "Overall condition",
    "GrLivArea": "Living area (sq ft)",
    "LotArea": "Lot area (sq ft)",
    "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodeled",
    "Neighborhood": "Neighborhood",
    "BldgType": "Building type",
    "BedroomAbvGr": "Bedrooms",
    "FullBath": "Full bathrooms",
    "HalfBath": "Half bathrooms",
    "KitchenAbvGr": "Kitchens",
    "TotRmsAbvGrd": "Total rooms",
    "TotalBsmtSF": "Basement area (sq ft)",
    "GarageCars": "Garage capacity (cars)",
    "GarageArea": "Garage area (sq ft)",
    "Fireplaces": "Fireplaces",
    "CentralAir": "Central air",
    "PoolArea": "Pool area (sq ft)",
    "KitchenQual": "Kitchen quality",
    "ExterQual": "Exterior quality",
    "BsmtQual": "Basement quality",
    "TotalSF": "Total area (sq ft)",
    "TotalBathrooms": "Total bathrooms",
    "HouseAge": "Age at sale (years)",
    "YearsSinceRemodel": "Years since remodel",
    "QualityArea": "Quality × living area",
    "QualityCondition": "Quality × condition",
    "GarageAge": "Garage age (years)",
    "TotalPorchSF": "Porch & deck area (sq ft)",
    "LivingLotRatio": "Living / lot area ratio",
    "AreaPerGarageCar": "Garage area per car",
    "MSSubClass": "Dwelling class",
    "MSZoning": "Zoning",
    "LotFrontage": "Lot frontage (ft)",
    "1stFlrSF": "First floor (sq ft)",
    "2ndFlrSF": "Second floor (sq ft)",
    "BsmtFinSF1": "Finished basement (sq ft)",
    "SaleCondition": "Sale condition",
    "Functional": "Functionality",
    "HeatingQC": "Heating quality",
    "GarageFinish": "Garage finish",
    "GarageType": "Garage type",
    "FireplaceQu": "Fireplace quality",
    "MasVnrArea": "Masonry veneer area",
    "BsmtExposure": "Basement exposure",
    "BsmtFinType1": "Basement finish type",
    "Foundation": "Foundation",
    "HouseStyle": "House style",
    "YrSold": "Year sold",
    "HasGarage": "Has garage",
    "HasBasement": "Has basement",
    "HasFireplace": "Has fireplace",
    "HasPool": "Has pool",
    "HasCentralAir": "Has central air",
    "Has2ndFloor": "Has second floor",
    "WasRemodeled": "Was remodeled",
    "QualityBand": "Quality band",
    "ConditionBand": "Condition band",
}


def friendly(name: str) -> str:
    return FRIENDLY_NAMES.get(name, name)


def parse_data_description(path: Path) -> dict[str, str]:
    """Parse the official data_description.txt into {column: description}."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9]+):\s+(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


@lru_cache(maxsize=1)
def load_feature_descriptions() -> dict[str, str]:
    from .config import PATHS

    descriptions = dict(FEATURE_DESCRIPTIONS)
    if PATHS.description_txt.exists():
        descriptions.update(parse_data_description(PATHS.description_txt))
    return descriptions
