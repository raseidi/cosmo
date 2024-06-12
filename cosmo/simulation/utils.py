import torch
import pandas as pd
from cosmo.event_logs import LOG_READERS
from cosmo.event_logs.reader import get_declare
from cosmo.models import Cosmo
from cosmo.event_logs import ConstrainedContinuousTraces
from cosmo.utils import get_existing_experiments


def load_stuff(
    dataset: str = "sepsis", template: str = "existence", backbone="vanilla"
):
    """Loading and setting up dataset and model"""

    if backbone == "vanilla":
        template = "all"
    runs = get_existing_experiments(project="cosmo-bpm-sim", force_fetch=True)
    runs = runs[
        (runs.dataset == dataset)
        & (runs.template == template)
        & (runs.backbone == backbone)
    ]
    config = runs.to_dict("records")[0]
    config["grad_clip"] = None  # from csv it comes as nan as it ruins the model loading

    log = LOG_READERS[config["dataset"]]()
    declare_constraints = get_declare(config["dataset"], templates=config["template"])
    log = log.sort_values(by=["case_id", "timestamp", "split"])
    not_found_constraints = set(log.case_id.unique()) - set(
        declare_constraints.case_id.unique()
    )
    log = log[~log.case_id.isin(not_found_constraints)]

    t, test_set = log[log["split"] == "train"], log[log["split"] == "test"]
    t = ConstrainedContinuousTraces(
        log=t,
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=True,
        device=config["device"],
    )
    event_dataset = ConstrainedContinuousTraces(
        log=test_set,
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=False,
        vocab=t.get_vocabs(),
        device=config["device"],
    )

    run_name = f"backbone={config['backbone']}-templates={config['template']}-lr={config['lr']}-bs={config['batch_size']}-hidden={config['hidden_size']}-input={config['input_size']}"

    try:
        state = torch.load(f"models/{config['dataset']}/{run_name}.pth")
    except FileNotFoundError as e:
        message = f"Model not found, please train it first. Run `python train.py --dataset {config['dataset']} --template '{config['template']}' --backbone {config['backbone']} --lr {config['lr']} --batch-size {config['batch_size']} --hidden-size {config['hidden_size']} --input-size {config['input_size']} --n-layers {config['n_layers']} --epochs {config['epochs']}` to train the model."
        raise FileNotFoundError(message) from e

    # model
    model = Cosmo(
        vocabs=event_dataset.feature2idx,
        n_continuous=event_dataset.num_cont_features,
        n_constraints=event_dataset.num_constraints,
        backbone_model=config["backbone"],
        embedding_size=config["input_size"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        lora=True,
        r_rank=config["r_rank"],
        lora_alpha=config["lora_alpha"],
    )
    model.load_state_dict(state["net"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, event_dataset, log, declare_constraints


def get_constraints(template, activity_to_simulate, activity_to_simulate_2=None):
    rules = {
        "existence": [
            f"Existence1[{activity_to_simulate}] | |",
            f"Exactly1[{activity_to_simulate}] | |",
            f"Absence1[{activity_to_simulate}] | |",
        ],
        "choice": [
            f"Choice[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Exclusive Choice[{activity_to_simulate}, {activity_to_simulate_2}] | |",
        ],
        "positive relations": [
            f"Chain Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            # f"Alternate Precedence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Alternate Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Chain Precedence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            # f"Chain Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Precedence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            # f"Responded Existence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
        ],
        "negative relations": [
            f"Not Responded Existence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Not Chain Precedence[{activity_to_simulate}, {activity_to_simulate_2} ] | |",
            f"Not Chain Response[{activity_to_simulate}, {activity_to_simulate_2} ] | |",
            f"Not Precedence[{activity_to_simulate}, {activity_to_simulate_2} ] | |",
            # f"Not Responded Existence[{activity_to_simulate}, {activity_to_simulate_2} ] | |",
            f"Not Response[{activity_to_simulate}, {activity_to_simulate_2} ] | |",
        ],
    }

    return rules[template]

def remove_eos(group):
    condition_met_index = group[group == "<EOS>"].index.min()
    if pd.isna(condition_met_index):  # Check if the condition is never met
        return group
    else:
        # Keep rows up to and including the row where the condition is met
        return group.loc[:condition_met_index-1]

def posthoc_formatting(simulated_log, original_log, event_dataset, skip_eos=True):
    """Inverse transform the simulated log to the original scale and format.

    Tokenized activities are transformed back to their original form. Remaining time is also transformed back to the original scale."""
    from sklearn.preprocessing import StandardScaler

    simulated_log["activity"] = simulated_log["activity"].apply(
        lambda x: event_dataset.idx2feature["activity"][x]
    )
    simulated_log = simulated_log.reset_index(drop=True)
    
    # bottleneck 1
    x = simulated_log.groupby("case_id").activity.apply(remove_eos).reset_index(level=0).index
    simulated_log = simulated_log.iloc[x]

    sc = StandardScaler()
    sc.fit(original_log.loc[:, ["remaining_time"]])
    simulated_log.loc[:, "remaining_time"] = sc.inverse_transform(
        simulated_log.loc[:, ["remaining_time"]]
    )
    
    # TODO: infer timestamp according to the remaining time
    # currently we only replicate the first timestamp (for each trace) from the test set
    return simulated_log
