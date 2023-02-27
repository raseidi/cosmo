import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F


def simulate_from_scratch(model, meta, n_traces=100, device="cuda"):
    _sim = pd.DataFrame()
    g = torch.Generator().manual_seed(2147483647)  # for reproducibility
    for c in [0, 1]:
        cond = torch.tensor([[c]])
        sim, rt_sim = [], []
        with torch.inference_mode():
            for _ in range(n_traces):
                na_trace = [0, 0, 0, 0, 0]
                rt_trace = [0, 0, 0, 0, 0]
                for _ in range(100):  # max prefix len == 100
                    na_prefix = torch.tensor([na_trace[-5:]])
                    rt_prefix = torch.tensor([rt_trace[-5:]], dtype=torch.float)

                    x = [na_prefix.to(device), rt_prefix.to(device), cond.to(device)]
                    na, rt = model(x)
                    probs = F.softmax(na, dim=1)
                    ix = torch.multinomial(
                        probs.cpu(), num_samples=1, generator=g
                    ).item()
                    # ix = torch.argmax(torch.softmax(na.cpu(), dim=1), dim=1).item()
                    if ix == 0:
                        break
                    na_trace.append(ix)
                    rt_trace.append(rt.item())

                sim.append(na_trace[4:])
                rt_sim.append(rt_trace[4:])

        sim = {k: v for k, v in enumerate(sim)}
        sim = (
            pd.concat({k: pd.Series(v) for k, v in sim.items()})
            .reset_index()
            .drop("level_1", axis=1)
            .rename(columns={"level_0": "case_id", 0: "activity"})
        )
        sim["remaining_time"] = [np.exp(item) for sublist in rt_sim for item in sublist]

        meta_dict = dict(zip(meta.encoded.values, meta.activities.values))
        sim["activity"] = sim["activity"].apply(lambda x: meta_dict[x])

        sim = sim[sim["activity"] != "<.>"]
        sim["condition"] = c
        _sim = pd.concat((_sim, sim))
    return _sim


def simulate_remaining_case(model, data, device="cuda"):
    def pad_data(ac, res, rt):
        pad_ac, pad_res, pad_rt = np.zeros(5), np.zeros(5)
        pad_ac[5 - ac.shape[0] :] = ac
        pad_res[5 - ac.shape[0] :] = res
        pad_rt[5 - ac.shape[0] :] = rt

        return pad_ac, pad_rt

    def get_remaining(initial_prefix, device="cuda"):
        na_prefix, rt_prefix, cond = initial_prefix
        na_trace, rt_trace = na_prefix.tolist(), rt_prefix.tolist()
        for _ in range(100):  # max prefix len == 100
            x = [
                na_prefix.unsqueeze(0).to(device),
                rt_prefix.unsqueeze(0).to(device),
                cond.unsqueeze(0).to(device),
            ]
            na, rt = model(x)
            # probs = F.softmax(na, dim=1)
            # ix = torch.multinomial(
            #     probs.cpu(), num_samples=1, generator=g
            # ).item()
            ix = torch.argmax(torch.softmax(na.cpu(), dim=1), dim=1).item()
            if ix == 0:
                break
            na_trace.append(ix)
            rt_trace.append(rt.item())
            na_prefix = torch.tensor(na_trace[-5:])
            rt_prefix = torch.tensor(rt_trace[-5:])

        return na_trace, rt_trace

    sim = {}
    for case in data["case_id"].unique():
        c = data[data["case_id"] == case].reset_index(drop=True)
        case_len = len(c)
        case = str(case)
        for input in range(1, case_len, 1):
            ac, res, rt, cond = c.loc[
                :input, ["activity", "resource", "remaining_time", "target"]
            ].values.T
            cond = cond[0]
            if len(ac) < 5:
                ac, rt = pad_data(ac, res, rt)

            x = (
                torch.tensor(ac[-5:]).long(),  # we take only the last 5
                torch.tensor(rt[-5:]),
                torch.tensor(cond).long(),
            )
            activities, remaining_time = get_remaining(x)
            if case not in sim:
                sim[case] = {}
                sim[case]["activities"] = {}
                sim[case]["remaining_time"] = {}

            sim[case]["activities"][str(input)] = activities
            sim[case]["remaining_time"][str(input)] = remaining_time

    return sim
