import os
import torch
import pandas as pd

def simulate(
    model, 
    data_config,
    n_simulations: int = 20,
    temperature: float = 0.1,
    ):
    rules                   = data_config["rules"]
    cases                   = data_config["cases"]
    event_dataset           = data_config["event_dataset"]
    template                = data_config["template"]
    activity_to_simulate    = data_config["activity_to_simulate"]
    activity_to_simulate_2  = data_config["activity_to_simulate_2"]
    
    simulated_log = pd.DataFrame()
    for case_id in cases:
        trace = event_dataset.get_case(case_id)

        # get constraints from that trace
        constraints = event_dataset.constraints[event_dataset.constraints.index == case_id]


        # get a mask array with the rules in constraints.columns
        mask = constraints.columns.isin(rules)

        # invert the rules, if 0 then 1 and vice versa
        # trace["constraints"][:, :] = 1
        trace["constraints"][:,mask] = 1 - trace["constraints"][:,mask]
        # constraints.loc[:, constraints.columns[mask]]
        min_remaining_time = event_dataset.log.remaining_time_norm.min().round(2)

        for iteration in range(n_simulations):
            # generate a new trace given the new constraints
            # it must start from the first activity in trace["cat_activity"]
            cat, num = trace["cat_activity"][:1], trace["num_remaining_time_norm"][:1]
            constraints = trace["constraints"] # this has been modified above
            with torch.no_grad():
                iters = 0
                while True:
                    if cat.dim() == 1: # add batch dimension and seq dimension
                        cat = cat.unsqueeze(0).unsqueeze(-1)
                        num = num.unsqueeze(0).unsqueeze(-1)
                    logits, reg, _ = model((cat, num), constraints)
                    # get only the last prediction from seq dimension
                    logits = logits[:, -1, :].squeeze()

                    next_activity = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
                    
                    next_remaining_time = reg.squeeze()
                    next_activity = next_activity.unsqueeze(0).unsqueeze(0)
                    
                    
                    cat = torch.cat([cat, next_activity], dim=1)
                    num = torch.cat([num, reg[:, -1, :].unsqueeze(0)], dim=1)
                    
                    if next_activity.item() == event_dataset.feature2idx["activity"]["<EOS>"] or iters > 50 or reg[:, -1, :].squeeze().item() <= min_remaining_time:
                        break
                    iters+=1

            activities = cat.detach().cpu().numpy().squeeze()
            remaining_time = num.detach().cpu().numpy().squeeze()
            
            # concat to the simulated_log dataframe
            df = pd.DataFrame({
                "case_id": case_id + "_" + str(iteration),
                "activity": [event_dataset.idx2feature["activity"][a] for a in activities],
                "remaining_time": remaining_time,
                "condition": template +"_"+ activity_to_simulate + "_" + str(activity_to_simulate_2 if activity_to_simulate_2 is not None else ""),
            })
            simulated_log = pd.concat([simulated_log, df], axis=0)

    simulated_log["template"] = template
    return simulated_log.reset_index(drop=True)