import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import jellyfish as jf
import matplotlib.pyplot as plt

from pm4py import discover_petri_net_heuristics as heuristics
from pm4py import discover_petri_net_inductive as inductive_miner
from pm4py import fitness_token_based_replay as fitness_token_replay
# from pm4py import fitness_alignments
from pm4py.conformance import fitness_alignments

from itertools import product
from sklearn.preprocessing import LabelEncoder
from scipy.stats import wasserstein_distance


def discover_pm(log: pd.DataFrame, algorithm="inductive_miner"):
    if algorithm == "inductive_miner":
        algo = inductive_miner
    elif algorithm == "heuristics":
        algo = heuristics
    net, im, fm = algo(
        log,
        activity_key="activity",
        timestamp_key="time",
        case_id_key="case_id",
        multi_processing=True,
    )
    return net, im, fm


def get_variants(log, column="activity"):
    variants = log.groupby(["case_id"])[column].apply(
        list
    )  # transform groupby into list
    variants = variants.apply(
        lambda x: "".join(map(str, x))
    )  # transfor list into a unique string
    return sorted(set(variants))


def cfls(variant_true, variant_pred):

    """
    CFLS
    https://github.com/AdaptiveBProcess/DDSvsDL/blob/f85e41bb06357d04d493ae1cb4f0d24843c06fc6/analyzers/sim_evaluator.py#L90
    1. pair each trace (original and generated)
    2. select the pair with maximum similarity
    3. final score is the average similarity of optimal paired traces
    4. this can be optimized by running variants only?

    the higher the better (1 is the best)
    """
    # 1 = perfect
    # ToDo: weighted average using the variant frequencies?
    scores = []
    for var_pred in variant_pred:
        best_sim = 0
        for var_true in variant_true:
            sim_score = 1 - (
                jf.damerau_levenshtein_distance(var_true, var_pred)
                / np.max([len(var_true), len(var_pred)])
            )

            if sim_score > best_sim:
                best_sim = sim_score

        scores.append(best_sim)
    return scores


def emd(log: pd.DataFrame, simulated_log: pd.DataFrame):
    """Earth Mover's Distance

    Calculating the EMD score in three different ways
    'histo': score between entire logs
    'histo_by_ac': score between histograms of each activity
    'histo_by_avg_ac': score between vector of RT averages of each activity
        i.e. true=[a1_rt.mean(), a2_rt.mean()]
    the lower the better (0 is the best)
    """
    # average of emd grouped by activities
    scores = dict(emd=0, emd_by_ac=0, emd_by_avg_ac=0)
    for ac in simulated_log.activity.unique():
        # comparting the test set with the simulated log has some troubles:
        # the simulated log has activities that the test set does not
        # this can be seen as a concept drift example, since activities in the
        # training set are not seen in the test set anymore;
        # comparing the simulated set with the train set is not a fair evaluation tho
        if ac not in log.activity.unique():
            continue
        true = log[log.activity == ac]["remaining_time"]
        pred = simulated_log[simulated_log.activity == ac]["remaining_time"]
        # wasserstein_distance
        score = np.exp(wasserstein_distance(true, pred)) / (24 * 60 * 60)
        scores["emd_by_ac"] += score
        # print(ac, score)
    scores["emd_by_ac"] /= simulated_log.activity.nunique()

    # average of average emd by grouped acs
    true = log.groupby("activity")["remaining_time"].mean().to_frame()
    pred = sim.groupby("activity")["remaining_time"].mean().to_frame()
    true = true.join(pred, rsuffix="_pred").dropna()
    scores["emd_by_avg_ac"] = wasserstein_distance(
        true["remaining_time"], true["remaining_time_pred"]
    )

    # emd of entire logs regardless acs
    scores["emd"] = wasserstein_distance(log["remaining_time"], sim["remaining_time"])

    scores = {k: np.exp(v) / (24 * 60 * 60) for k, v in scores.items()}
    return scores


def read_sim_scores(path="results/sim_evaluation.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()


def minimum_performance(dataset, condition, wandb_performances):
    if (
        wandb_performances[
            (wandb_performances.dataset == dataset)
            & (wandb_performances.condition == condition)
        ]["test_loss_rt(days)"].values[0]
        < 1
    ):
        return True
    return False


def trace_time_boxplot(log, sim, dataset, condition):
    df = pd.concat([log, sim], ignore_index=True)
    plt.figure()
    sns.boxplot(x="condition", y="duration", hue="type_set", data=df)
    plt.savefig(f"results/plots/{dataset}_{condition}.png")
    plt.close()


def positioning_resource_scores(log, sim, most_freq_res, model_arc, constraint="resource_usage"):
    position_scores = {}
    for case in log[log.type_set == "test"].case_id.unique():
        trace = log[log.case_id == case].reset_index(drop=True)
        trace_len = len(trace)

        condition = trace.resource_usage.unique()[0]
        resources = trace.resource.values
        # most_freq_res in resources
        sim_trace = sim[sim.case_id.str.startswith(case)]
        for case_position in range(0, trace_len - 1, 2):
            on_going = resources[: case_position + 1]
            simulated = sim_trace.loc[
                sim_trace.case_id.str.endswith(f"={case_position}"), "resource"
            ].values

            if case_position not in position_scores:
                position_scores[case_position] = dict(
                    class_0=dict(correct=0, total=0), class_1=dict(correct=0, total=0)
                )

            if condition:
                position_scores[case_position]["class_1"]["total"] += 1
                if most_freq_res in np.append(on_going, simulated):
                    position_scores[case_position]["class_1"]["correct"] += 1
            else:
                position_scores[case_position]["class_0"]["total"] += 1
                if most_freq_res not in np.append(on_going, simulated):
                    position_scores[case_position]["class_0"]["correct"] += 1

    final = pd.DataFrame()
    for key in position_scores.keys():
        if position_scores[key]["class_0"]["total"]:
            c0 = (
                position_scores[key]["class_0"]["correct"]
                / position_scores[key]["class_0"]["total"]
            )
        else:
            c0 = 1
        if position_scores[key]["class_1"]["total"]:
            c1 = (
                position_scores[key]["class_1"]["correct"]
                / position_scores[key]["class_1"]["total"]
            )
        else:
            c1 = 1
        final = pd.concat(
            (final, pd.DataFrame([{"class_0": c0, "class_1": c1, "pos": key}])),
            ignore_index=True,
        )

    final = pd.melt(
        final,
        id_vars=["pos"],
        value_vars=["class_0", "class_1"],
        var_name="resource_usage",
        value_name="accuracy",
    )
    final.to_csv(
        f"results/datasets/resource_usage/{dataset}{model_arc}_positioning_accuracy.csv",
        index=False,
    )
    plt.figure()
    sns.lineplot(x="pos", y="accuracy", hue="resource_usage", data=final)
    plt.savefig(f"results/plots/{dataset}_{constraint}{model_arc}.png")
    plt.close()


def filter_sim_from_scratch(log, sim):
    """Temporary auxiliar method

    During the simulation I saved only the remaining simulated trace
    instead of concatenating the ongoing case + simulated. Thus,
    this method simply appends the first event of each case to the
    begining of each simulated case.

    Args:
        log (_type_): _description_
        sim (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_pos_0 = log[log.type_set == "test"].drop_duplicates("case_id", keep="first")
    log_pos_0["time"] = pd.Timestamp.min

    from_scratch = sim[sim.case_id.str.endswith("=0")].copy()
    from_scratch.case_id = from_scratch.case_id.apply(lambda x: x.replace("_pos=0", ""))
    from_scratch = pd.concat((from_scratch, log_pos_0)).sort_values(
        by=["case_id", "time"]
    )
    ix = from_scratch.case_id.value_counts() > 1
    ix = ix[ix].index
    from_scratch = from_scratch[from_scratch.case_id.isin(ix)]
    return from_scratch


if __name__ == "__main__":
    # basic settings
    sim_path = "results/simulations"
    log_path = "data/"
    wandb_performances = pd.read_csv("results/best_runs.csv")
    final_scores = read_sim_scores("results/sim_evaluation_v3.csv")
    final_accumulated_scores = read_sim_scores("results/sim_evaluation_accum_v3.csv")

    datasets = pd.read_csv("results/datasets_statistics.csv").dataset.unique()
    sym_types = ["on_going"]
    conditions = [
        "resource_usage",
    ]  # resource_usage"]
    ignore_datasets = [
        "BPI_Challenge_2012_W",
        "BPI_Challenge_2012_Complete",
        "BPI_Challenge_2012_A",
        "BPI_Challenge_2012_O",
        "BPI_Challenge_2012",
        "BPI_Challenge_2012_W_Complete",
    ]
    models = ["DG"]
    prods = product(datasets, sym_types, conditions, models)

    for dataset, sim_type, condition, model_arc in prods:
        # condition = "resource_usage"
        # sim_type = "on_going"
        # if os.path.exists(f"results/plots/{dataset}_{condition}.png"):
        #     continue

        if dataset in ignore_datasets and condition == "resource_usage":
            print("ignore")
            continue

        if condition == "trace_time" and not minimum_performance(
            dataset, condition, wandb_performances
        ):
            print("skipping since experiments went bad")
            continue

        if not os.path.exists(os.path.join("data", dataset, "log.csv")):
            print(os.path.join("data", dataset, "log.csv"), "Doest not exist.")
            continue

        scores = dict(dataset=dataset, sim_type=sim_type, condition=condition, model=model_arc)
        accumulated_scores = dict()
        """ reading and preprocessing data """
        log = pd.read_csv(os.path.join("data", dataset, "log.csv"))
        log["condition"] = log[condition]
        log["time"] = pd.to_datetime(log["time"])
        log["case_id"] = log["case_id"].astype(str)
        log["duration"] = log.groupby("case_id").remaining_time.transform(
            lambda x: np.exp(x.max()) / (24 * 60 * 60)
        )
        most_freq_res = log["resource"].value_counts().nlargest(2).idxmin()
        resource_is_numerical = log.resource.dtype == float
        if resource_is_numerical:
            try:
                mean = log[log.type_set == "train"].resource.mean()
                std = log[log.type_set == "train"].resource.std()
            except:
                print("categorical resource")
                continue

        sim_data = os.path.join(
            sim_path, "_".join([dataset, condition+model_arc, sim_type]) + ".csv"
        )
        if not os.path.exists(sim_data):
            print("simulated log doesnt exist")
            continue
        sim = pd.read_csv(sim_data)
        drop_cases = sim[sim.remaining_time > 20].case_id.unique()
        sim = sim[~sim.case_id.isin(drop_cases)]
        sim["duration"] = sim.groupby("case_id").remaining_time.transform(
            lambda x: np.exp(x.max()) / (24 * 60 * 60)
        )
        sim["case_id"] = sim["case_id"].astype(str)

        if resource_is_numerical:
            sim["resource"] = sim["resource"] * std + mean
        sim = sim[(sim.activity != "<eos>") & (sim.activity != "<pad>")]

        # sim = label_variants(sim)
        # log = label_variants(log)

        le = LabelEncoder().fit(log.activity.values)
        log["enc_ac"] = le.transform(log.activity.values)
        sim["enc_ac"] = le.transform(sim.activity.values)

        def rt_to_datetime(x):
            days = np.exp(x.remaining_time) / (24 * 60 * 60)
            new_date = x.time + datetime.timedelta(days=days)
            return new_date

        sim["time"] = pd.Timestamp.min
        sim["time"] = sim.loc[:, ["remaining_time", "time"]].apply(
            rt_to_datetime, axis=1
        )
        # break
        sim["time"] = pd.to_datetime(sim["time"])
        from_scratch = filter_sim_from_scratch(log, sim)
        print(dataset, sim_type, condition)

        """ EMD """
        print("[+] EMD SCORE")
        scores.update(emd(log, from_scratch))

        """ CLFS """
        print("[+] CLFS SCORE")
        true_vars = get_variants(log, column="enc_ac")
        pred_vars = get_variants(from_scratch, column="enc_ac")
        cfls_score = cfls(true_vars, pred_vars)
        scores.update(
            {"cfls_mean": np.mean(cfls_score), "cfls_std": np.std(cfls_score)}
        )
        accumulated_scores["cfls"] = cfls_score

        # # TodO
        # # if i can plot bar plot with mean+std only, without the whole array
        # # the following lines are not needed
        accumulated_scores = pd.DataFrame(accumulated_scores)
        accumulated_scores["dataset"] = dataset
        accumulated_scores["condition"] = condition
        accumulated_scores["sim_type"] = sim_type
        accumulated_scores["model"] = model_arc

        """ Process model """
        print("[+] PM FITNESS")
        net, im, fm = discover_pm(
            log[log.type_set == "train"], algorithm="inductive_miner"
        )
        fitness_config = {
            "petri_net": net,
            "initial_marking": im,
            "final_marking": fm,
            "activity_key": "activity",
            "timestamp_key": "time",
            "case_id_key": "case_id",
        }

        # fit = fitness_token_replay(log[log.type_set == "train"], **fitness_config)
        # # for some unkown reason the pm4py multiplies this key by 100
        # # and duplicates it
        # # update: multiplies by 100 cause it is the percentage, not
        # # the fitness score itself
        # fit["perc_fit_traces"] /= 100
        # _ = fit.pop("percentage_of_fitting_traces")
        # fit = {"gr_tr_" + k: v for k, v in fit.items()}
        # scores.update(fit)

        fit = fitness_token_replay(from_scratch, **fitness_config)
        fit["perc_fit_traces"] /= 100
        _ = fit.pop("percentage_of_fitting_traces")
        fit = {"sim_tr_" + k: v for k, v in fit.items()}
        scores.update(fit)

        # fit = fitness_alignments(log=log, multi_processing=True, **fitness_config)
        # fit = {"gr_al_" + k: v for k, v in fit.items()}
        # scores.update(fit)

        # sizes = from_scratch.groupby("case_id").size()
        # sizes = sizes[sizes < 50]
        # small_log = from_scratch[from_scratch.case_id.isin(sizes.index)]
        
        fit = fitness_alignments(from_scratch, multi_processing=True, **fitness_config)
        fit = {"sim_al_" + k: v for k, v in fit.items()}
        scores.update(fit)

        # break
        final_scores = pd.concat((final_scores, pd.DataFrame([scores])))
        final_scores.to_csv("results/sim_evaluation_v3.csv", index=False)
        final_accumulated_scores = pd.concat(
            (final_accumulated_scores, accumulated_scores)
        )
        final_accumulated_scores.to_csv("results/sim_evaluation_accum_v3.csv", index=False)
        # """ what-if analysis """
        # print("[+] WHAT-IF ANALYSIS")
        # if condition == "trace_time":
        #     a = (
        #         log[log.type_set == "test"]
        #         .drop_duplicates("case_id")
        #         .reset_index(drop=True)
        #     )
        #     b = (
        #         from_scratch[from_scratch.type_set.isna()]
        #         .drop_duplicates("case_id")
        #         .reset_index(drop=True)
        #     )
        #     b["type_set"] = "sim"

        #     trace_time_boxplot(
        #         log=a,
        #         sim=b,
        #         dataset=dataset,
        #         condition=condition,
        #     )
        # elif condition == "resource_usage":
        #     if model_arc == "DG":
        #         positioning_resource_scores(log, sim, most_freq_res, model_arc)
