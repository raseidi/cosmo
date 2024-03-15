import numpy as np
import pandas as pd
from functools import wraps


def common_preprocessing(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        log = fn(*args, **kwargs)
        log = _common_preprocessing(log)

        return log

    return wrapper


def _common_preprocessing(log):
    """Common preprocessing for all logs

    Args:
        log (pd.DataFrame): log dataframe

    Returns:
        pd.DataFrame: log dataframe with common preprocessing
    """

    log = (
        log.rename(
            columns={
                "concept:name": "activity",
                "time:timestamp": "timestamp",
                "case:concept:name": "case_id",
            }
        )
        .assign(
            timestamp=lambda df: pd.to_datetime(
                df["timestamp"], infer_datetime_format=True
            )
        )
        .sort_values(by=["case_id", "timestamp"])
        .loc[:, ["case_id", "activity", "timestamp", "split"]]
    )

    # 0. drop nan cases
    log = log.dropna(subset=["case_id"])
    log.case_id = log.case_id.astype(str)

    # 1. drop activities with frequency less than 5%
    # activities with frequencies less than 5% are dropped
    # activities = log.activity.value_counts()
    # activities = activities[activities >= 0.05 * len(log)].index.values
    # log = log[log.activity.isin(activities)]

    # 2. drop cases if
    # len(case) > quantile(.9)
    # len(case) < 3
    quantile = log.groupby("case_id").size().quantile(0.9)
    valid_cases = log.groupby("case_id").size().le(quantile)
    valid_cases = valid_cases[valid_cases].index.values
    log = log[log["case_id"].isin(valid_cases)]

    valid_cases = log.groupby("case_id").size().ge(2)
    valid_cases = valid_cases[valid_cases].index.values
    log = log[log["case_id"].isin(valid_cases)]
    # log = log.groupby("case_id").size().gt(avg + std)#filter(lambda x: len(x) < avg + std)

    # 3. drop incomplete traces based on the last activity frequency
    # traces where the last activities have frequencies less than 5% are dropped
    last_activities = log.groupby("case_id")["activity"].last().value_counts() / (
        log["case_id"].nunique()
    )
    last_activities = (last_activities[last_activities > 0.05]).index.values

    valid_cases = log.groupby("case_id")["activity"].last().isin(last_activities)
    valid_cases = valid_cases[valid_cases].index.values

    log = log[log["case_id"].isin(valid_cases)]

    # 4. preparing groupby for augumentation
    group_cols = ["case_id", "split"]

    # Augmenting the log with time features
    log = log.reset_index(drop=True).copy()
    log = time_feature_engineering(log, group_cols)
    if "remaining_time" in log.columns:
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        sc.fit(log.loc[log.split == "train", ["remaining_time"]])
        log.loc[log.split == "train", "remaining_time_norm"] = sc.transform(
            log.loc[log.split == "train", ["remaining_time"]]
        )
        log.loc[log.split == "test", "remaining_time_norm"] = sc.transform(
            log.loc[log.split == "test", ["remaining_time"]].values
        )

    # Augmenting the log with EOS rows
    # log = log.reset_index(drop=True)
    log = add_eos(log, group_cols=group_cols, static_cols=group_cols + ["timestamp"])

    # log = log[log.columns.drop("timestamp")]
    return log


def time_feature_engineering(log, group_cols):
    # grouped = log.groupby(group_cols)
    # log["time_accumulated"] = (
    #     grouped["timestamp"]
    #     .transform(lambda s: (s - s.min()).dt.total_seconds())
    #     .astype(int)
    # )
    # log["time_since_last_event"] = (
    #     grouped["timestamp"].diff().dt.total_seconds().fillna(0).astype(int)
    # )
    # log["remaining_time"] = (
    #     log
    #     .groupby(group_cols, observed=True, as_index=False, group_keys=False)
    #     .timestamp
    #     .apply(lambda x: x.max() - x)
    #     .dt.total_seconds()
    # )
    log["remaining_time"] = log.groupby(
        ["case_id", "split"], observed=True, as_index=False, group_keys=False
    ).timestamp.transform("max")
    log["remaining_time"] = (log["remaining_time"] - log["timestamp"]).astype(
        int
    ) // 10 ** 9

    # log = (
    # .assign(year=lambda df: df["timestamp"].dt.year)
    # .assign(month=lambda df: df["timestamp"].dt.month)
    # .assign(day=lambda df: df["timestamp"].dt.day)
    # .assign(weekday=lambda df: df["timestamp"].dt.weekday)
    # .assign(hour=lambda df: df["timestamp"].dt.hour)
    # .assign(minute=lambda df: df["timestamp"].dt.minute)
    # .assign(second=lambda df: df["timestamp"].dt.second)
    # .assign(
    #     seconds_from_start_of_day=lambda df: df["hour"] * 3600
    #     + df["minute"] * 60
    #     + df["second"]
    # )
    # .assign(
    #     seconds_from_start_of_week=lambda df: df["weekday"] * 86400
    #     + df["seconds_from_start_of_day"]
    # )
    # .assign(
    #     seconds_from_start_of_month=lambda df: df["day"] * 86400
    #     + df["seconds_from_start_of_day"]
    # )
    # .assign(
    #     seconds_from_start_of_year=lambda df: df["month"] * 86400
    #     + df["seconds_from_start_of_month"]
    # )
    # .assign(
    #     total_seconds_iso_8601=lambda df: df["timestamp"].astype(int) // 10**9
    # )
    # .assign(
    #     periodic=lambda df: np.sin(
    #         2 * np.pi * df["total_seconds_iso_8601"] / (24 * 60 * 60)
    #     )
    # )
    # )
    # .assign(
    #     periodic_weekly=lambda df: np.sin(
    #         2 * np.pi * df["total_seconds_iso_8601"] / (7 * 24 * 60 * 60)
    #     )
    # )
    # )
    return log


def add_eos(log, group_cols, static_cols):
    log = log.reset_index(drop=True)
    grouped = log.groupby(group_cols, observed=True, as_index=False, group_keys=False)
    cat_cols = log.select_dtypes(include=["object", "category"]).columns.drop(
        static_cols, errors="ignore"
    )
    num_cols = log.select_dtypes(exclude=["object", "category"]).columns.drop(
        static_cols, errors="ignore"
    )
    last = grouped.last().copy()
    last[num_cols] = 0
    last[cat_cols] = "<EOS>"
    last["timestamp"] = last["timestamp"] + pd.Timedelta(seconds=1)

    log = pd.concat((log, last), ignore_index=True, sort=False)
    log = log.sort_values(by=["case_id", "timestamp"])
    return log.copy()
