import torch
import pandas as pd
from cosmo.event_logs.utils import collate_fn
from torch.utils.data import DataLoader
from einops import rearrange


def _as_frame(cat, num, event_dataset):
    simulated_log = pd.DataFrame()
    log = event_dataset.log
    for ix, case in enumerate(event_dataset.cases):
        df = pd.DataFrame(
            {
                "case_id": [case] * len(cat[ix]),
                "timestamp": log[log.case_id == case].timestamp.values[0],
                "activity": cat[ix],
                "remaining_time": num[ix],
            }
        )
        simulated_log = pd.concat([simulated_log, df], axis=0)
    return simulated_log


def _simulate_constraints(constraints, mask, strategy="invert"):
    if strategy == "original":
        pass
    elif strategy == "invert_subset":
        constraints[:, :, mask] = 1 - constraints[:, :, mask]
    elif strategy == "invert_all":
        constraints = 1 - constraints
    elif strategy == "ones":
        constraints = torch.ones_like(constraints)
    elif strategy == "zeros":
        constraints = torch.zeros_like(constraints)
    else:
        raise ValueError(
            f"Strategy {strategy} not found. Use 'original', 'invert_subset', 'invert_all', 'ones' or 'zeros'"
        )
    return constraints


def _sample(logits, strategy="argmax", temperature=0.1):
    if strategy == "argmax":
        next_act = torch.argmax(logits, dim=-1, keepdim=True)
    elif strategy == "multinomial":
        next_act = torch.multinomial(
            torch.softmax(logits.squeeze() / temperature, dim=-1), num_samples=1
        )
        next_act = rearrange(next_act, "b t -> b t ()")
    else:
        raise ValueError(
            f"Strategy {strategy} not found. Use 'argmax' or 'multinomial'"
        )

    return next_act


def _constrained_simulation(
    model,
    loader,
    mask_constraints,
    sim_strat,
    sampling_strat,
    temperature,
    max_trace_length,
):
    model.eval()
    final_cat = []
    final_num = []
    for batch in loader:
        cat, num, c = batch["cat"], batch["num"], batch["constraints"]
        cat, num = cat[:, :1, :], num[:, :1, :]

        c = _simulate_constraints(c, mask_constraints, strategy=sim_strat)

        for _ in range(max_trace_length):
            with torch.no_grad():
                logits, reg, _ = model((cat, num), c)
                logits = logits[:, -1:, :]

                next_act = _sample(
                    logits=logits, strategy=sampling_strat, temperature=temperature
                )

                cat = torch.cat([cat, next_act], dim=1)
                num = torch.cat([num, reg[:, -1:, :]], dim=1)
        final_cat.append(cat.detach().cpu())
        final_num.append(num.detach().cpu())

    final_cat = torch.cat(final_cat, dim=0)
    final_num = torch.cat(final_num, dim=0)
    # shapes are (batch, seq_len)
    final_cat = final_cat.squeeze().detach().cpu().numpy()
    final_num = final_num.squeeze().detach().cpu().numpy()
    return final_cat, final_num


def constrained_simulation(
    model: torch.nn.Module,
    event_dataset: torch.utils.data.Dataset,
    rules: list[str],
    sim_strat: str = "invert_subset",
    max_trace_length: int = 50,
    n_simulated_logs: int = 1,
    sampling_strat: str = "argmax",
    temperature: float = 0.1,
):
    mask_constraints = event_dataset.constraints.columns.isin(rules)

    loader = DataLoader(
        event_dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=collate_fn,
    )
    logs = pd.DataFrame()
    if sampling_strat == "argmax":
        n_simulated_logs = 1
    for i in range(n_simulated_logs):
        cat, num = _constrained_simulation(
            model=model,
            loader=loader,
            mask_constraints=mask_constraints,
            sim_strat=sim_strat,
            sampling_strat=sampling_strat,
            temperature=temperature,
            max_trace_length=max_trace_length,
        )
        sim_log = _as_frame(cat, num, event_dataset)
        sim_log["sim_id"] = i
        logs = pd.concat([logs, sim_log], axis=0)
    return logs

def _sthocastic_simulation(
    model,
    loader,
    sampling_strat,
    temperature,
    max_trace_length,
):
    model.eval()
    final_cat = []
    final_num = []
    for batch in loader:
        cat, num, c = batch["cat"], batch["num"], batch["constraints"]
        cat, num = cat[:, :1, :], num[:, :1, :]

        for _ in range(max_trace_length):
            with torch.no_grad():
                logits, reg, _ = model((cat, num), constraints=c)
                logits = logits[:, -1:, :]

                next_act = _sample(
                    logits=logits, strategy=sampling_strat, temperature=temperature
                )

                cat = torch.cat([cat, next_act], dim=1)
                num = torch.cat([num, reg[:, -1:, :]], dim=1)
        final_cat.append(cat.detach().cpu())
        final_num.append(num.detach().cpu())

    final_cat = torch.cat(final_cat, dim=0)
    final_num = torch.cat(final_num, dim=0)
    # shapes are (batch, seq_len)
    final_cat = final_cat.squeeze().detach().cpu().numpy()
    final_num = final_num.squeeze().detach().cpu().numpy()
    return final_cat, final_num


def sthocastic_simulation(
    model: torch.nn.Module,
    event_dataset: torch.utils.data.Dataset,
    max_trace_length: int = 50,
    n_simulated_logs: int = 1,
    sampling_strat: str = "multinomial",
    temperature: float = 1.0, # logits / 1 = logits (no change, classic approach)
):
    """Generation of logs from non-constrained models."""
    loader = DataLoader(
        event_dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=collate_fn,
    )
    logs = pd.DataFrame()
    if sampling_strat == "argmax":
        n_simulated_logs = 1
    for i in range(n_simulated_logs):
        cat, num = _sthocastic_simulation(
            model=model,
            loader=loader,
            sampling_strat=sampling_strat,
            temperature=temperature,
            max_trace_length=max_trace_length,
        )
        sim_log = _as_frame(cat, num, event_dataset)
        sim_log["sim_id"] = i
        logs = pd.concat([logs, sim_log], axis=0)
    return logs


# def simulate(
#     model,
#     data_config,
#     n_simulations: int = 20,
#     temperature: float = 0.1,
# ):
#     rules = data_config["rules"]
#     cases = data_config["cases"]
#     event_dataset = data_config["event_dataset"]
#     template = data_config["template"]
#     activity_to_simulate = data_config["activity_to_simulate"]
#     activity_to_simulate_2 = data_config["activity_to_simulate_2"]
#     model.eval()
#     simulated_log = pd.DataFrame()
#     for case_id in cases:
#         trace = event_dataset.get_case(case_id)

#         # get constraints from that trace
#         constraints = event_dataset.constraints[
#             event_dataset.constraints.index == case_id
#         ]

#         # get a mask array with the rules in constraints.columns
#         mask = constraints.columns.isin(rules)

#         # invert the rules, if 0 then 1 and vice versa
#         # trace["constraints"][:, :] = 1
#         trace["constraints"][:, mask] = 1 - trace["constraints"][:, mask]
#         # constraints.loc[:, constraints.columns[mask]]
#         min_remaining_time = event_dataset.log.remaining_time_norm.min().round(2)

#         for iteration in range(n_simulations):
#             # generate a new trace given the new constraints
#             # it must start from the first activity in trace["cat_activity"]
#             cat, num = trace["cat_activity"][:1], trace["num_remaining_time_norm"][:1]
#             constraints = trace["constraints"]  # this has been modified above
#             with torch.no_grad():
#                 iters = 0
#                 while True:
#                     if cat.dim() == 1:  # add batch dimension and seq dimension
#                         cat = cat.unsqueeze(0).unsqueeze(-1)
#                         num = num.unsqueeze(0).unsqueeze(-1)
#                     logits, reg, _ = model((cat, num), constraints)
#                     # get only the last prediction from seq dimension
#                     logits = logits[:, -1, :].squeeze()

#                     next_activity = torch.multinomial(
#                         torch.softmax(logits / temperature, dim=-1), num_samples=1
#                     )

#                     next_remaining_time = reg.squeeze()
#                     next_activity = next_activity.unsqueeze(0).unsqueeze(0)

#                     cat = torch.cat([cat, next_activity], dim=1)
#                     num = torch.cat([num, reg[:, -1, :].unsqueeze(0)], dim=1)

#                     if (
#                         next_activity.item()
#                         == event_dataset.feature2idx["activity"]["<EOS>"]
#                         or iters > 50
#                         or reg[:, -1, :].squeeze().item() <= min_remaining_time
#                     ):
#                         break
#                     iters += 1

#             activities = cat.detach().cpu().numpy().squeeze()
#             remaining_time = num.detach().cpu().numpy().squeeze()

#             # concat to the simulated_log dataframe
#             df = pd.DataFrame(
#                 {
#                     "case_id": case_id + "_" + str(iteration),
#                     "activity": [
#                         event_dataset.idx2feature["activity"][a] for a in activities
#                     ],
#                     "remaining_time": remaining_time,
#                     "condition": template
#                     + "_"
#                     + activity_to_simulate
#                     + "_"
#                     + str(
#                         activity_to_simulate_2
#                         if activity_to_simulate_2 is not None
#                         else ""
#                     ),
#                 }
#             )
#             simulated_log = pd.concat([simulated_log, df], axis=0)

#     simulated_log["template"] = template
#     return simulated_log.reset_index(drop=True)
