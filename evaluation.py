import wandb
import pandas as pd
from scipy.stats import wasserstein_distance as emd


def optimal_string_alignment_distance(s1, s2):
    # Create a table to store the results of subproblems
    dp = [[0 for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]

    # Initialize the table
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Populate the table using dynamic programming
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Return the edit distance
    return dp[len(s1)][len(s2)]


def get_variants(df):
    variants = df.groupby(["case_id"])["activity"].apply(
        list
    )  # transform groupby into list
    variants = variants.apply(
        lambda x: ",".join(map(str, x))
    )  # transfor list into a unique string
    return sorted(set(variants))


def get_results(project="multi-task-trace-time", best_step_run=True):
    api = wandb.Api()
    runs = api.runs(project)

    cols = None
    results = pd.DataFrame()
    for r in runs:
        if r.state != "finished":
            continue

        hist = r.history()
        if cols is None:
            cols = [
                c
                for c in r.history().columns
                if not c.startswith("param") and not c.startswith("grad")
            ]

        hist = hist[cols]
        hist["run_name"] = r.name
        if best_step_run:
            best_step = hist["test_loss"].idxmin()
            hist = hist.loc[hist._step == best_step, cols]
            hist = hist.join(pd.DataFrame([r.config], index=[best_step]))

        results = pd.concat((results, hist.iloc[[-1], :]))

    results.reset_index(inplace=True, drop=True)
    return results


# results = get_results()
# top = results.iloc[results.groupby(["dataset", "condition"]).test_loss.idxmin()]

df = pd.read_csv("data/RequestForPayment/trace_time/log.csv")

"""
Earth Mover's Distance (EMD): minimum amount of “work” required to transform into , where “work” is measured as the amount of distribution weight that must be moved, multiplied by the distance it has to be moved.
"""
# emd of the mean durations of the activities
# 0: identidcal, 1: completly different
df.groupby("activity").apply(lambda x: emd(x["remaining_time"], x["remaining_time"]))

"""
https://github.com/AdaptiveBProcess/DDSvsDL/blob/f85e41bb06357d04d493ae1cb4f0d24843c06fc6/analyzers/sim_evaluator.py#L90
1. pair each trace (original and generated)
2. select the pair with maximum similarity
3. final score is the average similarity of optimal paired traces
4. this can be optimized by running variants only

import jellyfish as jf
(1 - (jf.damerau_levenshtein_distance(x, y) / np.max([len(x), len(y)])))
"""
variants = get_variants(df)

""" 
CFLS
"""
# import itertools
# log=df

# log['duration'] = 0
# log = log.to_dict('records')
# log = sorted(log, key=lambda x: x['case_id'])
# for _, group in itertools.groupby(log, key=lambda x: x['case_id']):
#     events = list(group)
#     break
#     ordk = 'end_timestamp' if self.one_timestamp else 'start_timestamp'
#     events = sorted(events, key=itemgetter(ordk))
#     for i in range(0, len(events)):
#         # In one-timestamp approach the first activity of the trace
#         # is taken as instant since there is no previous timestamp
#         if self.one_timestamp:
#             if i == 0:
#                 dur = 0
#             else:
#                 dur = (events[i]['end_timestamp'] -
#                         events[i-1]['end_timestamp']).total_seconds()
#         else:
#             dur = (events[i]['end_timestamp'] -
#                     events[i]['start_timestamp']).total_seconds()
#             if i == 0:
#                 wit = 0
#             else:
#                 wit = (events[i]['start_timestamp'] -
#                         events[i-1]['end_timestamp']).total_seconds()
#             events[i]['waiting'] = wit
#         events[i]['duration'] = dur
# return pd.DataFrame.from_dict(log)
