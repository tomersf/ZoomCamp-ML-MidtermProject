from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_HANDLE = "rtatman/188-million-us-wildfires"
DATASET_FILE = "FPA_FOD_20170508.sqlite"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

SAMPLE_PATH = DATA_DIR / "wildfires_sample.parquet"
FULL_PATH = DATA_DIR / "wildfires_full.parquet"

SQL_QUERY = """
    SELECT
        FOD_ID,

        -- Time
        FIRE_YEAR,
        DISCOVERY_DATE,
        DISCOVERY_DOY,
        DISCOVERY_TIME,

        -- Location
        LATITUDE,
        LONGITUDE,
        STATE,
        FIPS_CODE,
        FIPS_NAME,

        -- Cause
        STAT_CAUSE_DESCR,

        -- Fire severity
        FIRE_SIZE,
        FIRE_SIZE_CLASS

    FROM Fires

    WHERE FIRE_SIZE_CLASS IS NOT NULL
      AND LATITUDE IS NOT NULL
      AND LONGITUDE IS NOT NULL
"""


def load_data(full: bool = False) -> pd.DataFrame:
    """Load the wildfire data.

    full=False -> fixed, reproducible sample.
    full=True  -> the entire dataset.

    Each variant is cached as its own parquet file under <project>/data/processed.
    """
    path = FULL_PATH if full else SAMPLE_PATH

    if path.exists():
        print(f"Loading existing {'full dataset' if full else 'sample'}...")
        return pd.read_parquet(path)

    print("Downloading/loading wildfire dataset from Kaggle...")

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        DATASET_FILE,
        sql_query=SQL_QUERY,
    )

    print(f"Full dataset shape: {df.shape}")

    if not full:
        df = df.sample(
            n=min(SAMPLE_SIZE, len(df)),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    df.to_parquet(path, index=False)

    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_data(full=False)
    print(df.head())
