def execution_time(x):
    """
    returns
    Args:
        x (ndarray, pd.Series): array of timestamp values

    Returns:
        _type_: _description_
    """
    from pandas import NaT

    x = x - x.shift(1, fill_value=NaT)
    return x


def accumulated_time(x):
    """
    returns x_{i} - x_{0} given an array of dt values
    Args:
        x (ndarray, pd.Series): array of timestamp values

    Returns:
        _type_: _description_
    """
    x = x - x.min()
    return x


def remaining_time(x):
    """Returns the remaining time of a case

    Args:
        x (_type_): _description_
    """
    x = x.max() - x
    return x


def within_day(x):
    """Returns the time within the day
    x_i - midnight
    Args:
        x (_type_): _description_
    """
    x = x - x.replace(hour=0, minute=0, second=0, microsecond=0)
    return x


def within_week(x):
    """Returns the time within the week w.r.t last sunday

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    from datetime import timedelta

    ls = x - timedelta(x.day_of_week + 1) if x.day_of_week != 6 else x
    ls = ls.replace(hour=0, minute=0, second=0, microsecond=0)
    return x - ls


# df = pd.read_csv(
#     "/home/seidi/Repositores/ConditionedBPS/data/trace_time/RequestForPayment/log_preprocessed.csv"
# )
# df = df.loc[
#     :,
#     [
#         "case:concept:name",
#         "concept:name",
#         "org:resource",
#         "time:timestampr_",
#         "tf2",
#         "tf1",
#     ],
# ]
# df = df.rename(
#     columns={
#         "case:concept:name": "case_id",
#         "concept:name": "activity",
#         "org:resource": "resource",
#         "time:timestampr_": "time",
#     }
# )
# df["time"] = pd.to_datetime(df["time"], infer_datetime_format=True).dt.tz_localize(None)
# df["tf2"] = (
#     df.groupby("case_id")["time"].transform(accumulated_time)
# )
# df["tf1"] = df.groupby("case_id")["time"].transform(execution_time)
# df["tf3"] = df["time"].transform(within_day)
# df["tf4"] = df["time"].transform(within_week)


# df[["time", "tf1", "tf2", "tf3", "tf4"]]
# df[["tf1", "tf2", "tf3", "tf4"]].apply(lambda x: x.dt.total_seconds())
