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
    labeled_cases.name = "target"
    df = df.merge(labeled_cases, on="case_id")
    df["target"] = np.where(0, 1, df.target)
    return df["target"].values
