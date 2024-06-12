import os
import torch
import numpy as np
import pandas as pd
from itertools import product

from cosmo.simulation import sthocastic_simulation
from cosmo.simulation.utils import load_stuff, get_constraints, posthoc_formatting

torch.random.manual_seed(42)
np.random.seed(42)

SIMULATION_STRATEGIES = ["original"]#, "invert_all", "ones", "zeros"]
SAMPLING_STRATEGIES = ["multinomial"]

def main(dataset="sepsis", template="all", backbone="vanilla"):
    inner_loop = list(product(SIMULATION_STRATEGIES, SAMPLING_STRATEGIES))
    model, event_dataset, log, declare = load_stuff(
        dataset=dataset, template=template, backbone=backbone
    )
    log = log[log.activity != "<EOS>"]
    max_trace_length = log[log.split == "train"].groupby("case_id").size().mean()
    max_trace_length = int(max_trace_length)
    
    for sim_strat, sampling_strat in inner_loop:
        simulated_log = pd.DataFrame()
        output_file_name = f"data/simulation/{backbone}/{dataset}-template={template}-sim_strat={sim_strat}.pkl"
        if os.path.exists(output_file_name):
            continue
        
        print("[+] Simulating for", dataset, template, sim_strat)
            
        sim = sthocastic_simulation(
            model=model,
            event_dataset=event_dataset,
            max_trace_length=max_trace_length,
            n_simulated_logs=2,
            sampling_strat=sampling_strat,
        )
        sim = posthoc_formatting(
            sim, log[log.split == "train"], event_dataset
        )
        simulated_log = pd.concat([simulated_log, sim], axis=0)

        simulated_log.to_pickle(output_file_name)
        
        
def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sepsis")
    parser.add_argument("--template", type=str, default="all")
    parser.add_argument("--backbone", type=str, default="vanilla")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_args()
    main(args.dataset, args.template)
