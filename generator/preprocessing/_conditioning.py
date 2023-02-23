import numpy as np


def trace_time(df, threshold="mean"):
    """This functions labels the event log in a binary fashion.
    It will classify traces below or above the threshold.

    Params
    ------------------
    df: pandas.DataFrame represent the event log
    threshold: ["mean"]: mean sets the mean elapsed time to end process executions

    Pseudo algo:
    ------------------
    1. specify and convert time (min, hours, days)
    2. group by case id and get the process execution time
    3. apply the threshold to label the variants
    Returns:
        ndarray: array containing conditions extracted from the DF
    """
    total_time = df.groupby("case_id")["time"].apply(lambda x: x.max() - x.min())
    labeled_cases = total_time >= total_time.mean()
    labeled_cases.name = "trace_time"
    df = df.merge(labeled_cases, on="case_id")
    df["trace_time"] = np.where(0, 1, df.trace_time)
    return df["trace_time"].values


def resource_usage(df):
    """This function labels cases that use the second most frequent resource.

    This is due to the fact most event logs have the the most frequent resource present
    in all cases.
    Args:
        df (pd.DataFrame): event log dataframe
    """
    most_frequent_resource = df["resource"].value_counts().nlargest(2).idxmin()
    df["resource_usage"] = df.groupby("case_id")["resource"].transform(
        lambda group: most_frequent_resource in group.unique()
    )
    df["resource_usage"] = df["resource_usage"].astype(int)
    return df["resource_usage"].values
