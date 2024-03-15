import os
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import classification_report
# if A exists in the trace
def existence(log, activity_to_simulate):
    a = (
        log.groupby(["case_id"])["activity"]
        .apply(lambda x: activity_to_simulate in x.values)
    )
    
    return a

# for exclusive choice:

# A or B have to occur at least
# once but not both.

# (A and not B) OR (not A and B)
# (higher the better)
def choice(log, act1, act2):
    cases = (
        log.groupby(["case_id"])
        .activity
        .apply(
            lambda x: (
                act1 in x.values
                and act2 not in x.values
            )
            or (
                act1 not in x.values
                and act2 in x.values
            )
        )
    )
    return cases # / log.case_id.nunique()

# for chain response (if a, b is next)
# if SATISFY RULE = 1, the rate should be 0
# if SATISFY RULE = 0, the rate should be 1

def check_rule(activities, act1, act2):
    try:
        a = [ix for ix, e in enumerate(activities) if e == act1]
    except ValueError:
        return False
    try:
        b = [ix for ix, e in enumerate(activities) if e == act2]
    except ValueError:
        return False

    for i in a:
        if i + 1 in b:
            return True
    return False

def positive_relations(log, act1, act2):
    cases = log.groupby(["case_id"]).activity.apply(check_rule, act1, act2)
    return cases #/ len(log.case_id.unique())

from log_distance_measures.config import DEFAULT_CSV_IDS, DEFAULT_XES_IDS
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.config import EventLogIDs

event_log_ids = EventLogIDs(
    case="case_id",
    activity="activity",
    start_time="timestamp",
    end_time="timestamp"
)


DATASETS=["sepsis", "bpi12", "bpi13_incidents", "bpi13_problems", "bpi17", "bpi20_permit", "bpi20_prepaid", "bpi20_req4pay"]
SIMULATION_STRATEGIES=["invert_subset", "original"]#, "invert_all", "ones", "zeros"]
SAMPLING_STRATEGIES=["argmax", "multinomial"]
TEMPLATES=["existence", "choice", "positive relations"]

out_loop = list(product(DATASETS, TEMPLATES))
inner_loop = list(product(SIMULATION_STRATEGIES, SAMPLING_STRATEGIES))

# act1='W_Call incomplete files'
# act2=None
# sim_strat='invert_subset'
# sampling_strat='argmax'
try:
    final_scores = pd.read_csv("scores.csv")
except:
    final_scores = pd.DataFrame()

for dataset, template in out_loop:
    # dataset="sepsis"
    # tempalte = "positive relations"
    # sim_strat="invert_subset"
    # sampling_strat="multinomial"
    
    original_log = pd.read_pickle(f'data/{dataset}/cached_log.pkl')
    original_log = original_log[original_log.split=="test"]
    original_log = original_log[original_log.activity != "<EOS>"]
    
    variants = original_log.groupby("case_id").activity.apply(list).values
    ov = len(set([tuple(variant) for variant in variants]))

    for sim_strat, sampling_strat in inner_loop:
        sim_log_path = f"data/simulation/crnn/dataset={dataset}-template={template}-sim_strat={sim_strat}-sampling_strat={sampling_strat}.pkl"
        if not os.path.exists(sim_log_path):
            continue
        sim_log = pd.read_pickle(sim_log_path)
        if sim_log.empty:
            continue

        prods = list(product(sim_log.activity_1.unique(), sim_log.activity_2.unique()))
        # arr = np.stack([sim_log.activity_1.values, sim_log.activity_2.values], axis=1)
        # prods = set(map(tuple, arr))
        
        for act1, act2 in prods:

            scores_ = dict()
            for ix in sim_log[(sim_log.activity_1==act1) & (sim_log.activity_2==act2)].sim_id.unique():
                if not final_scores.empty and final_scores[
                    (final_scores["dataset"] == dataset) &
                    (final_scores["template"] == template) &
                    (final_scores["sim_strat"] == sim_strat) &
                    (final_scores["sampling_strat"] ==  sampling_strat) &
                    (final_scores["act1"] == act1) &
                    ((final_scores["act2"] == act2) |  (final_scores["act2"].isna())) &
                    (final_scores["sim_id"] == ix)
                ].shape[0] > 0:
                    print(f"skipping {dataset}")
                    continue
                s_log = sim_log[(sim_log.activity_1==act1) & (sim_log.activity_2==act2) & (sim_log.sim_id == ix)].copy()
                if s_log.empty:
                    continue 
                
                
                # Number of unique variants
                if sampling_strat == "argmax":
                    variants = s_log.groupby("case_id").activity.apply(list).values
                    sv_mean = len(set([tuple(variant) for variant in variants]))
                    sv_std = 0
                else: # multinomial we gotta average the number of variants through all the simulated logs
                    sv = []
                    variants = s_log.groupby("case_id").activity.apply(list).values
                    sv.append(len(set([tuple(variant) for variant in variants])))
                    sv_mean = np.mean(sv)
                    sv_std = np.std(sv)
                    
                scores_["dataset"] = dataset
                scores_["template"] = template
                scores_["sim_strat"] = sim_strat
                scores_["sampling_strat"] = sampling_strat
                scores_["sim_id"] = ix
                scores_["original_variants"] = ov
                scores_["sim_variants"] = sv_mean
                scores_["sim_variants_std"] = sv_std
                
                # Rule satisfaction scores
                if s_log.empty:
                    # not all combinations of activities are simulated
                    continue
                if template == "existence":
                    # existence only
                    # second statement is to avoid duplicate activities for `existence`
                    ori_rate = existence(original_log, act1)
                    sim_rate = existence(s_log, act1)
                    act2 = ""
                elif template == "choice":
                    # choice only
                    ori_rate = choice(original_log, act1, act2)
                    sim_rate = choice(s_log, act1, act2)
                elif template == "positive relations":
                    # positive relations only
                    ori_rate = positive_relations(original_log, act1, act2)
                    sim_rate = positive_relations(s_log, act1, act2)
                
                pred = sim_rate.to_frame("simulated")
                true = ori_rate.to_frame("original")
                
                if ori_rate.sum() == 0:
                    continue
                
                df = true.join(pred, how="inner")
                
                # when inverting rules, we want the opposite of the original rule (i.e. not satisfied)
                # otherwise, when using the original rules we want the same as the original rule
                if sim_strat == "invert_subset":
                    df["target"] = ~df.simulated
                else:
                    df["target"] = df.simulated
                # df["target"] = df["target"].apply(lambda x: "satisfied" if x else "not_satisfied")
                clf_report = classification_report(df.original, df.target, output_dict=True, labels=[True, False], target_names=["satisfied", "not_satisfied"])

                scores_["act1"] = act1
                scores_["act2"] = act2
                
                scores_["satisfied_precision"] =        clf_report["satisfied"]["precision"]
                scores_["satisfied_recall"] =           clf_report["satisfied"]["recall"]
                scores_["satisfied_f1"] =               clf_report["satisfied"]["f1-score"]
                scores_["satisfied_support"] =          clf_report["satisfied"]["support"]
                
                scores_["not_satisfied_precision"] =    clf_report["not_satisfied"]["precision"]
                scores_["not_satisfied_recall"] =       clf_report["not_satisfied"]["recall"]
                scores_["not_satisfied_f1"] =           clf_report["not_satisfied"]["f1-score"]
                scores_["not_satisfied_support"] =      clf_report["not_satisfied"]["support"]

                scores_["total_samples"] =              clf_report["macro avg"]["support"]
                scores_["accuracy"] =                   clf_report.get("accuracy", sum(df.original == df.target) / len(df))

                scores_["macro_precision"] =            clf_report["macro avg"]["precision"]
                scores_["macro_recall"] =               clf_report["macro avg"]["recall"]
                scores_["macro_f1"] =                   clf_report["macro avg"]["f1-score"]

                scores_["weighted_precision"] =         clf_report["weighted avg"]["precision"]
                scores_["weighted_recall"] =            clf_report["weighted avg"]["recall"]
                scores_["weighted_f1"] =                clf_report["weighted avg"]["f1-score"]
                            
                final_scores = pd.concat((final_scores, pd.DataFrame([scores_])))
                final_scores.to_csv("scores.csv", index=False)
            