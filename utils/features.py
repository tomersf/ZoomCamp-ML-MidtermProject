"""Feature engineering shared by training (train.py) and serving, so both see identical inputs."""
import pandas as pd

CATEGORICAL = ["state", "stat_cause_descr"]
FEATURES = [
    "fire_year", "discovery_doy", "latitude", "longitude",
    "state", "stat_cause_descr", "discovery_hour", "month", "day_of_week",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive discovery_hour, month and day_of_week from discovery_time / discovery_date."""
    df = df.copy()
    df["discovery_hour"] = pd.to_numeric(df["discovery_time"], errors="coerce") // 100
    d = pd.to_datetime(df["discovery_date"], unit="D", origin="julian")
    df["month"] = d.dt.month
    df["day_of_week"] = d.dt.day_of_week
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw wildfire table -> (X, y) where y = 1 for fire size classes C-G (10+ acres)."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    y = df["fire_size_class"].isin(list("CDEFG")).astype(int)
    return add_time_features(df)[FEATURES], y
