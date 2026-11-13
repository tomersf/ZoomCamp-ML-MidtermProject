# ZoomCamp ML Midterm Project: Wildfire Severity Prediction

Midterm project for the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course.

## Problem description

Wildfires cause large-scale damage to ecosystems, property and human life, and
the resources needed to fight a fire depend heavily on how big it becomes.
Knowing early whether a fire is likely to stay small or grow into a major
event helps with prioritising response and allocating resources.

This project uses the **[1.88 Million US Wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires)**
dataset (Short, 2017; US wildfire records from 1992 to 2015, stored as a SQLite
database on Kaggle) to frame this as a **multi-class classification problem**:

> Given when and where a fire was discovered, and what caused it, predict its
> **size class** (`FIRE_SIZE_CLASS`, from `A` for the smallest to `G` for the largest).

### Features and target

| Group    | Columns                                                                            |
|----------|------------------------------------------------------------------------------------|
| Time     | `FIRE_YEAR`, `DISCOVERY_DATE`, `DISCOVERY_DOY`, `DISCOVERY_TIME`                   |
| Location | `LATITUDE`, `LONGITUDE`, `STATE`, `FIPS_CODE`, `FIPS_NAME`                         |
| Cause    | `STAT_CAUSE_DESCR`                                                                 |
| Target   | `FIRE_SIZE_CLASS` (`A`-`G`); `FIRE_SIZE` (acres) is the underlying continuous value |

Rows with a missing size class, latitude or longitude are dropped when loading.
