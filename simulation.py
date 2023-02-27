import torch
import pandas as pd

from generator import MTCondLSTM
from generator.data_loader import get_loader
from generator.meld import prepare_log, vectorize_log
from generator.simulator.simulator import simulate_remaining_case
from generator.utils import get_runs, load_checkpoint, get_vocabs, read_data

run_name = "rosy-sweep-9"
runs = get_runs("multi-task")
config = runs[runs.run_name == run_name]
log = read_data(f"data/PrepaidTravelCost/log.csv")
log["target"] = log["resource_usage"]
log.drop(["trace_time", "resource_usage"], axis=1, inplace=True)
log = prepare_log(log)
vocabs = get_vocabs(log=log)

for f in vocabs:
    log.loc[:, f] = log.loc[:, f].transform(lambda x: vocabs[f]["stoi"][x])

# ToDo how to track cat features? here we have (act, res, rt)
_, data_test = vectorize_log(log)
test_loader = get_loader(data_test, batch_size=1024, shuffle=False)

model = MTCondLSTM(vocabs=vocabs, batch_size=config.batch_size.values[0])
checkpoint = load_checkpoint(
    ckpt_dir_or_file=f"models/PrepaidTravelCost/trace_time/{run_name}/best_model.ckpt"
)
model.load_state_dict(checkpoint["net"])
model.cuda()
model.eval()

# sim = simulate_from_scratch(
#     model,
# )
sim = simulate_remaining_case(model, log)

# sim.to_csv(f"evaluation/simulation/{dataset}/from_scratch.csv", index=False)
