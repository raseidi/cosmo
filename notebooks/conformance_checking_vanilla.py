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


DATASETS=["sepsis", "bpi12", "bpi13_problems", "bpi20_permit", "bpi17"]
SIMULATION_STRATEGIES=["original"]#, "invert_all", "ones", "zeros"]
SAMPLING_STRATEGIES=["multinomial"]
TEMPLATES=["existence", "choice", "positive relations"]

out_loop = list(product(DATASETS, TEMPLATES))
inner_loop = list(product(SIMULATION_STRATEGIES, SAMPLING_STRATEGIES))

try:
    final_scores = pd.read_csv("scores_vanilla.csv")
except:
    final_scores = pd.DataFrame()

keys = [['bpi12', 'choice', 'W_Completeren aanvraag',
        'W_Nabellen offertes'],
       ['bpi12', 'existence', 'A_DECLINED', ""],
       ['bpi12', 'positive relations', 'A_PARTLYSUBMITTED',
        'W_Completeren aanvraag'],
       ['bpi13_problems', 'choice', 'Accepted-Assigned_complete',
        'Queued-Awaiting Assignment_complete'],
       ['bpi13_problems', 'existence',
        'Queued-Awaiting Assignment_complete', ""],
       ['bpi13_problems', 'positive relations',
        'Accepted-In Progress_complete',
        'Queued-Awaiting Assignment_complete'],
       ['bpi17', 'choice', 'A_Submitted', 'A_Cancelled'],
       ['bpi17', 'existence', 'W_Handle leads', ""],
       ['bpi17', 'positive relations', 'O_Accepted', 'A_Pending'],
       ['bpi20_permit', 'choice', 'Declaration APPROVED by BUDGET OWNER',
        'Declaration SUBMITTED by EMPLOYEE'],
       ['bpi20_permit', 'existence', 'Permit APPROVED by BUDGET OWNER',
        ""],
       ['bpi20_permit', 'positive relations', 'End trip',
        'Send Reminder'],
       ['sepsis', 'choice', 'Admission NC', 'LacticAcid'],
       ['sepsis', 'existence', 'LacticAcid', ""],
       ['sepsis', 'positive relations', 'Admission NC', 'CRP']]

cfg = dict()
for k in keys:
    if k[0] not in cfg:
        cfg[k[0]] = dict()
    
    if k[1] not in cfg[k[0]]:
        cfg[k[0]][k[1]] = dict()
        
    cfg[k[0]][k[1]]["act1"] = k[2]
    cfg[k[0]][k[1]]["act2"] = k[3]

for dataset, template in out_loop:
    original_log = pd.read_pickle(f'data/{dataset}/cached_log.pkl')
    original_log = original_log[original_log.split=="test"]
    original_log = original_log[original_log.activity != "<EOS>"]
    
    variants = original_log.groupby("case_id").activity.apply(list).values
    ov = len(set([tuple(variant) for variant in variants]))

    for sim_strat, sampling_strat in inner_loop:
        if not final_scores.empty and final_scores[
            (final_scores["dataset"] == dataset) &
            (final_scores["template"] == template) &
            (final_scores["sim_strat"] == sim_strat) &
            (final_scores["sampling_strat"] ==  sampling_strat) 
            # (final_scores["act1"] == act1) &
            # ((final_scores["act2"] == act2) | (final_scores["act2"].isna()))
        ].shape[0] > 0:
            continue
        
        sim_log_path = f"data/simulation/vanilla/dataset={dataset}-template=all-sim_strat={sim_strat}.pkl"
        if not os.path.exists(sim_log_path):
            continue
        sim_log = pd.read_pickle(sim_log_path)
        if sim_log.empty:
            continue

        for ix in sim_log.sim_id.unique():
            scores_ = dict()
            s_log = sim_log[sim_log.sim_id == ix].copy()
            
            # Number of unique variants
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
            act1 = cfg[dataset][template]["act1"]
            act2 = cfg[dataset][template]["act2"]
            
            # s_log = s_log[(s_log.activity_1==act1) & (s_log.activity_2==act2) & (s_log.sim_id == ix)].copy()
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
            
            df["target"] = df.simulated

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
            scores_["accuracy"] =                  clf_report.get("accuracy", sum(df.original == df.target) / len(df))

            scores_["macro_precision"] =            clf_report["macro avg"]["precision"]
            scores_["macro_recall"] =               clf_report["macro avg"]["recall"]
            scores_["macro_f1"] =                   clf_report["macro avg"]["f1-score"]

            scores_["weighted_precision"] =         clf_report["weighted avg"]["precision"]
            scores_["weighted_recall"] =            clf_report["weighted avg"]["recall"]
            scores_["weighted_f1"] =                clf_report["weighted avg"]["f1-score"]
                            
            final_scores = pd.concat((final_scores, pd.DataFrame([scores_])))
            final_scores.to_csv("scores_vanilla.csv", index=False)
        
