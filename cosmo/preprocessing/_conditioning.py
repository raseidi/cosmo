import numpy as np


def trace_time(log, threshold="mean"):
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
    df = log.copy()
    total_time = df.groupby("case_id")["time"].apply(lambda x: x.max() - x.min())
    labeled_cases = total_time >= total_time.mean()
    labeled_cases.name = "trace_time"
    df = df.merge(labeled_cases, on="case_id")
    df["trace_time"] = np.where(0, 1, df.trace_time)
    return df["trace_time"].values


def resource_usage(log):
    """This function labels cases that use the second most frequent resource.

    This is due to the fact most event logs have the the most frequent resource present
    in all cases.
    Args:
        df (pd.DataFrame): event log dataframe
    Returns:
        ndarray: array containing conditions extracted from the DF
    """
    df = log.copy()
    most_frequent_resource = df["resource"].value_counts().nlargest(2).idxmin()
    df["resource_usage"] = df.groupby("case_id")["resource"].transform(
        lambda group: most_frequent_resource in group.unique()
    )
    df["resource_usage"] = df["resource_usage"].astype(int)
    return df["resource_usage"].values

def bpi17(log):
    """BPI17 label from paper [1].
    
    Original implementation in [2].

    [1] Outcome-Oriented Predictive Process Monitoring: Review and Benchmark. \
        Irene Teinemaa, Marlon Dumas, Marcello La Rosa, Fabrizio Maria Maggi. \
        ACM Trans. KDD 2019.
    [2] https://github.com/irhete/predictive-monitoring-benchmark
    Args:
        df (pd.DataFrame): event log dataframe with a new column containing the label
    """
    df = log.copy()
    relevant_offer_events = ["O_Cancelled", "O_Accepted", "O_Refused"]
    last_o_events = (
        df
        .loc[df["EventOrigin"] == "Offer"]
        .sort_values(by="time", ascending=True, kind="mergesort")
        .groupby("case_id")
        .last()["activity"]
        .loc[lambda x: x.isin(relevant_offer_events)]
    )
    last_o_events.name = "last_o_event"
    df = df.merge(last_o_events, left_on=["case_id"], right_index=True)
    for label in relevant_offer_events:
        df[f"binary_{label}"] = np.where(df["last_o_event"] == label, 1, 0)

    return df

def label_variants(df):
    # refactor to return variants; maybe this also should change the path
    if "variant" in df.columns:
        return df

    from sklearn.preprocessing import LabelEncoder

    variants = df.groupby(["case_id"])["activity"].apply(
        list
    )  # transform groupby into list
    variants = variants.apply(
        lambda x: ",".join(map(str, x))
    )  # transform list into a unique string
    variants = variants.to_frame()  # encode each string (trace)
    variants.loc[:, "encoded_variants"] = LabelEncoder().fit_transform(
        variants.values.reshape(
            -1,
        )
    )
    variants.loc[:, "variant"] = variants["encoded_variants"].apply(
        lambda x: "Variant " + str(x)
    )  # formating

    df = df.join(
        variants.loc[:, ["variant"]],
        on="case_id",
    )
    return df
