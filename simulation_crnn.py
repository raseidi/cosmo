import os
import torch
import numpy as np
import pandas as pd
from itertools import product

from cosmo.simulation import constrained_simulation
from cosmo.simulation.utils import load_stuff, get_constraints, posthoc_formatting

torch.random.manual_seed(42)
np.random.seed(42)

SIMULATION_STRATEGIES = ["invert_subset", "original"]#, "invert_all", "ones", "zeros"]
SAMPLING_STRATEGIES = ["multinomial"]

def main(dataset="sepsis", template="existence"):
    inner_loop = list(product(SIMULATION_STRATEGIES, SAMPLING_STRATEGIES))
    model, event_dataset, log, declare = load_stuff(
        dataset=dataset, template=template, backbone="crnn"
    )
    log = log[log.activity != "<EOS>"]
    max_trace_length = log[log.split == "train"].groupby("case_id").size().max() #+ log[log.split == "train"].groupby("case_id").size().std()
    # max_trace_length = int(max_trace_length)
    
    for sim_strat, sampling_strat in inner_loop:
        simulated_log = pd.DataFrame()
        output_file_name = f"data/simulation/crnn/dataset={dataset}-template={template}-sim_strat={sim_strat}-sampling_strat={sampling_strat}.pkl"
        if os.path.exists(output_file_name):
            print("[+] Already simulated for", dataset, template, sim_strat, sampling_strat)
            continue
        
        print("[+] Simulating for", dataset, template, sim_strat, sampling_strat)
        
        dist = declare.drop("case_id", axis=1).mean().sort_values()
        
        if template == "positive relations":
            ix = list(filter(lambda x: x.startswith("Chain Response["), dist.index))
            dist = dist[ix]
        elif template == "choice":
            ix = list(filter(lambda x: x.startswith("Choice["), dist.index))
            dist = dist[ix]
        
        q1 = dist.quantile(0.25)
        q3 = dist.quantile(0.75)
        dist = dist[(dist > q1) & (dist < q3)]
        if len(dist) >= 3:
            dist = dist.sample(3, random_state=42)

        dist = dist.index
        fn = lambda x: x.split("[")[1].split("]")[0]
        activities = list(set(map(fn, dist)))
            
        for a in activities:
            a = a.split(", ")
            if len(a) == 1:
                act1 = a[0]
                act2 = ""
            else:
                act1, act2 = a
            
            rules = get_constraints(
                template=template, activity_to_simulate=act1, activity_to_simulate_2=act2
            )
            
            sim = constrained_simulation(
                model=model,
                event_dataset=event_dataset,
                rules=rules,
                sim_strat=sim_strat,
                max_trace_length=max_trace_length,
                n_simulated_logs=2,
                sampling_strat=sampling_strat,
            )
            sim = posthoc_formatting(
                sim, log[log.split == "train"], event_dataset
            )
            sim.loc[:, ["activity_1"]] = act1
            sim.loc[:, ["activity_2"]] = act2
            simulated_log = pd.concat([simulated_log, sim], axis=0)

        simulated_log.to_pickle(output_file_name)
        
        
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sepsis")
    parser.add_argument("--template", type=str, default="positive relations")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_args()
    main(args.dataset, args.template)
