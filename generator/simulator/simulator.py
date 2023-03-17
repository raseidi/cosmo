import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F


def _prepare_model_input(sim_trace, resource_is_cat):
    na_prefix = torch.tensor(sim_trace["activity"][-5:], dtype=torch.long).unsqueeze(0)
    res_type = torch.long if resource_is_cat else torch.float
    res_prefix = torch.tensor(sim_trace["resource"][-5:], dtype=res_type).unsqueeze(0)
    rt_prefix = torch.tensor(
        sim_trace["remaining_time"][-5:], dtype=torch.float
    ).unsqueeze(0)

    x = torch.cat((na_prefix, res_prefix, rt_prefix))
    x = x.transpose(0, 1).unsqueeze(0)
    return x


def _simulate_trace(model, sim_trace, cond, max_len, device="cuda", states=None):
    g = torch.Generator().manual_seed(13)  # for reproducibility
    for _ in range(max_len):  # max trace len
        x = _prepare_model_input(sim_trace, "resource" in model.vocabs)
        x = [x.to(device), cond.to(device)]
        ac, res, rt, states = model(x, states)

        # appending next activity
        probs = F.softmax(ac, dim=1)
        ix = torch.multinomial(probs.cpu(), num_samples=1, generator=g).item()
        # ix = torch.argmax(torch.softmax(ac.cpu(), dim=1), dim=1).item()
        sim_trace["activity"].append(ix)

        # appending next resource; this is needed to distinguish cat/num resource
        if "resource" in model.vocabs.keys():
            res_ix = torch.argmax(torch.softmax(res.cpu(), dim=1), dim=1).item()
            # res_ix = torch.multinomial(F.softmax(res, dim=1).cpu(), num_samples=1, generator=g).item()
            sim_trace["resource"].append(res_ix)
        else:
            sim_trace["resource"].append(res.item())

        # appending next remaining time
        sim_trace["remaining_time"].append(rt.item())
        if ix == model.vocabs["activity"]["stoi"]["<eos>"] or rt.item() <= 0:
            break
    return sim_trace


def simulate_from_scratch(
    model, n_traces=100, max_len=100, conditions=[0, 1], device="cuda"
):
    simulations = pd.DataFrame()
    # simulations = dict(activity=[], resource=[], remaining_time=[], condition=[])
    case_id = 0
    for c in conditions:
        cond = torch.tensor([c])
        for _ in range(n_traces):
            sim_trace = dict(
                activity=[0, 0, 0, 0, 0],
                resource=[0, 0, 0, 0, 0],
                remaining_time=[0, 0, 0, 0, 0],
            )
            sim_trace = _simulate_trace(model, sim_trace, cond, max_len)
            _sim = pd.DataFrame(sim_trace)
            _sim["condition"] = c
            _sim["case_id"] = case_id
            simulations = pd.concat((simulations, _sim))
            case_id += 1

    # removing <pad> tokens
    simulations = simulations[
        simulations.activity != model.vocabs["activity"]["stoi"]["<pad>"]
    ]
    return simulations


def simulate_remaining_case(model, data, device="cuda", window_slide=2):
    def get_attributes(df, case_position):
        ac, res, rt, cond = df.loc[
            :case_position, ["activity", "resource", "remaining_time", "target"]
        ].values.T
        ac = df.loc[:case_position, "activity"].values
        res = df.loc[:case_position, "resource"].values
        rt = df.loc[:case_position, "remaining_time"].values

        cond = torch.tensor(df.target.unique(), dtype=torch.long)

        # padding the attributes so we can slide throughout
        # each prefix containing 5 events
        pads = np.array([0, 0, 0, 0])
        ac = np.concatenate((pads, ac)).tolist()
        res = np.concatenate((pads, res)).tolist()
        rt = np.concatenate((pads, rt)).tolist()

        return ac, res, rt, cond

    simulations = pd.DataFrame()
    for case in data["case_id"].unique():
        df = data[data["case_id"] == case].reset_index(drop=True)
        df = df[df.activity != model.vocabs["activity"]["stoi"]["<eos>"]]
        case_len = df["case_id"].value_counts().max()
        case = str(case)

        for case_position in range(0, case_len - 1, window_slide):
            ac, res, rt, cond = get_attributes(df, case_position)
            sim_trace = dict(activity=ac, resource=res, remaining_time=rt)
            states = None
            if len(cond) > 1:
                continue
            # len(trace)-4 means we are going until the last 5 events
            # i.e. the last prefix
            # for each case, we simulate the remaining case from each
            # position, i.e. if case_len = 10, we will simulate 10 traces
            # starting from case[0:1], case[0:2], case[0:3], etc.
            for ix in range(0, len(sim_trace["activity"]) - 4, 1):
                curr_sim_trace = dict(
                    activity=sim_trace["activity"][ix : ix + 5],
                    resource=sim_trace["resource"][ix : ix + 5],
                    remaining_time=sim_trace["remaining_time"][ix : ix + 5],
                )
                # print(curr_sim_trace)
                x = _prepare_model_input(curr_sim_trace, "resource" in model.vocabs)
                x = [x.to(device), cond.to(device)]
                # we iteratively run throughout the ongoing case
                # to generate states since the first event;
                _, _, _, states = model(x, states)
            curr_sim_trace = _simulate_trace(
                model, curr_sim_trace, cond, case_len, states=states
            )
            # filtering the simulated trace in order to remove the true events
            # we include only the simulated ones
            curr_sim_trace = dict(
                activity=curr_sim_trace["activity"][5:],
                resource=curr_sim_trace["resource"][5:],
                remaining_time=curr_sim_trace["remaining_time"][5:],
            )
            _sim = pd.DataFrame(curr_sim_trace)
            _sim["condition"] = cond.item()
            _sim["case_id"] = case + f"_pos={case_position}"
            simulations = pd.concat((simulations, _sim))
    simulations = simulations[
        (simulations.activity != model.vocabs["activity"]["stoi"]["<pad>"])
        & (simulations.remaining_time >= 0)
    ]
    return simulations
