import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

def load_data(dataset, template, sampling_strat, sim_strat):
    original_log_path = f'data/{dataset}/cached_log.pkl'
    sim_log_path = f"data/simulation/crnn/dataset={dataset}-template={template}-sim_strat={sim_strat}-sampling_strat={sampling_strat}.pkl"
    
    if not os.path.exists(sim_log_path):
        print(f"File {sim_log_path} does not exist")
        return None, None
    
    original_log = pd.read_pickle(original_log_path)
    original_log = original_log[original_log.split=="test"]
    original_log = original_log[original_log.activity!="<EOS>"]

    sim_log = pd.read_pickle(sim_log_path)
    if sim_log.empty:
        return None, None

    sim_log = sim_log[sim_log.sim_id==1]
    original_log["event_ix"] = original_log.groupby(["case_id"]).cumcount()
    sim_log["event_ix"] = sim_log.groupby(["case_id", "activity_1", "activity_2"]).cumcount()

    return original_log, sim_log

DATASETS=["sepsis", "bpi12", "bpi13_incidents", "bpi13_problems", "bpi17", "bpi19", "bpi20_permit", "bpi20_prepaid", "bpi20_req4pay"]
TEMPLATES=["existence", "choice", "positive relations"]
# Remaining time is only evaluated for multinomial sampling and invert_subset simulation strategy

sampling_strat = "multinomial"
sim_strat = "invert_subset"

for dataset, template in product(DATASETS, TEMPLATES):
    original_log, sim_log = load_data(dataset, template, sampling_strat, sim_strat)
    if original_log is None or sim_log is None:
        print(f"Skipping {dataset} - {template}")
        continue

    prods = list(product(sim_log.activity_1.unique(), sim_log.activity_2.unique()))

    for act1, act2 in prods:
        sl = sim_log[(sim_log.activity_1==act1) & (sim_log.activity_2==act2)]
        if sl.empty:
            continue  
        
        fig, ax = plt.subplots()
        sns.lineplot(
            x="event_ix", 
            y="remaining_time", 
            data=original_log, 
            label="Original", 
            ax=ax
        )
        sns.lineplot(
            x="event_ix", 
            y="remaining_time", 
            data=sl, 
            label="Simulated", 
            ax=ax
        )
        
        fig.savefig(f"/home/seidi/Repositores/pm_projects/cosmo/data/simulation/crnn/plots/rt_{dataset}_{template}_{act1}_{act2}.png")
        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close()